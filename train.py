import os
import random
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
import torchvision.transforms as transforms
from demo.model import Hashnet
from demo.read_data import ISICDataSet
from cal_map_mult import calculate_top_map, calculate_map, compress
from gme import GmeQwen2VL
from peft import get_peft_model, LoraConfig, TaskType

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def adjust_learning_rate(optimizer, epoch, lr):
    lr = lr * (0.1 ** (epoch // 60))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(emb_model, train_loader, test_loader, batch_size, epochs, device, lr, hash_dim):
    # 获取GME特征维度（只需要执行一次）
    # 使用iter创建新迭代器，避免消耗原始数据加载器
    temp_loader = iter(train_loader)
    sample_images, _ = next(temp_loader)
    with torch.no_grad():
        sample_gme_emb = emb_model.get_image_embeddings(sample_images.to(device))
    gme_dim = sample_gme_emb.size(1)  # 获取特征维度（第二个维度）
    print(f"GME特征维度: {gme_dim}")

    # 初始化Hashnet模型（只初始化一次）
    hashnet = Hashnet(hash_dim=hash_dim, gme_dim=gme_dim).to(device)

    # 合并参数并创建优化器（只创建一次）
    optimizer = Adam(
        list(emb_model.base.parameters()) + list(hashnet.parameters()),
        lr=lr
    )

    for epoch in range(epochs):
        # 同时设置两个模型为训练模式
        emb_model.base.train()
        hashnet.train()

        # 学习率衰减（建议启用）
        # adjust_learning_rate(optimizer, epoch, lr)

        total_loss = 0.0

        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad()  # 清零梯度
            images, labels = data[0], data[1]

            # 获取图像嵌入并移至设备
            gme_emb = emb_model.get_image_embeddings(images)
            gme_emb = gme_emb.to(device)
            labels = labels.to(device)

            # 前向传播
            fc_emb, hash_code = hashnet(gme_emb)

            # 计算损失 - 确保批次大小为偶数
            assert labels.size(0) % 2 == 0, "批次大小必须为偶数"
            half_size = int(labels.size(0) / 2)
            target_b = F.cosine_similarity(hash_code[:half_size], hash_code[half_size:])
            target_x = F.cosine_similarity(gme_emb[:half_size], gme_emb[half_size:])

            loss = F.mse_loss(target_b, target_x)

            # 反向传播与优化
            loss.backward()  # 梯度回传
            optimizer.step()  # 更新参数

            total_loss += loss.item() * batch_size

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
                )

        # 计算 epoch 平均损失
        avg_loss = total_loss / (len(train_loader) * batch_size)
        print(f"Epoch [{epoch + 1}/{epochs}] | 平均训练损失: {avg_loss:.4f}\n")

        # 每个epoch结束后评估mAP（建议启用）
        emb_model.base.eval()
        hashnet.eval()
        with torch.inference_mode():
            retrievalB, retrievalL, queryB, queryL = compress(
                train_loader, test_loader, emb_model, hashnet, device, classes=3
            )
            map_score = calculate_map(
                qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL
            )
            print(f"Epoch [{epoch + 1}/{epochs}] | mAP@All: {map_score:.4f}\n")


def main(args):

    random.seed(0)
    torch.manual_seed(0)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 初始化GME模型
    emb_model = GmeQwen2VL(model_path="../gme-Qwen2-VL-2B-Instruct", device=device)

    # 配置LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # 应用LoRA适配器
    emb_model.base = get_peft_model(emb_model.base, peft_config)
    print("可训练参数:")
    emb_model.base.print_trainable_parameters()

    # 图像预处理
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.Resize(256),
        transforms.RandomResizedCrop(224) if hasattr(args,
                                                     'rand_resize') and args.rand_resize else transforms.CenterCrop(
            224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    # 加载数据集
    if args.dataset == 'isic':
        train_dataset = ISICDataSet(
            data_dir=os.path.join(args.dataset_dir, 'ISIC-2017_Training_Data'),
            image_list_file=args.train_image_list,
            transform=train_transform
        )
        test_dataset = ISICDataSet(
            data_dir=os.path.join(args.dataset_dir, 'ISIC-2017_Test_v2_Data'),
            image_list_file=args.test_image_list,
            transform=test_transform
        )
    elif args.dataset == 'covid':
        train_dataset = ISICDataSet(
            data_dir=os.path.join(args.dataset_dir, 'trainandval'),
            image_list_file=args.train_image_list,
            transform=train_transform
        )
        test_dataset = ISICDataSet(
            data_dir=os.path.join(args.dataset_dir, 'test'),
            image_list_file=args.test_image_list,
            transform=test_transform
        )
    else:
        raise NotImplementedError('Dataset not supported!')

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_worker,
        pin_memory=True,
        drop_last=True  # 确保批次大小一致且为偶数
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_worker,
        pin_memory=True,
        drop_last=True
    )

    # 打印数据信息
    print(f"数据加载完成：")
    print(f"- 训练集：{len(train_dataset)}张")
    print(f"- 测试集：{len(test_dataset)}张")

    # 开始训练
    train(
        emb_model,
        train_loader,
        test_loader,
        args.batch_size,
        args.epochs,
        device,
        args.lr,
        args.hash_dim
    )


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Embedding Learning')

    parser.add_argument('--dataset', default='isic', help='Dataset to use (covid or isic)')
    parser.add_argument('--dataset-dir', default='./data/isic', help='Dataset directory path')
    parser.add_argument('--train-image-list', default='./ISIC-2017_Training_Part3_GroundTruth.csv',
                        help='Train image list')
    parser.add_argument('--test-image-list', default='./ISIC-2017_Test_v2_Part3_GroundTruth_balanced.csv',
                        help='Test image list')
    parser.add_argument('--batch-size', default=4, type=int, help='Number of images in each mini-batch (必须为偶数)')
    parser.add_argument('--hash-dim', default=64, type=int, help='hashing dimension')
    parser.add_argument('--epochs', default=5, type=int, metavar='N', help='Number of training epochs to run')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--num-worker', default=4, type=int, help='Number of dataloader workers')
    parser.add_argument('--rand-resize', action='store_true', help='Use random resized crop')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
