import os
import random

import torch
from torch.utils.data import DataLoader

from demo.model import Gmenet
from demo.read_data import ISICDataSet
import torch.nn.functional as F
from cal_map_mult import calculate_top_map, calculate_map, compress
import torchvision.transforms as transforms


def adjust_learning_rate(optimizer, epoch, lr):
    lr = lr * (0.1 ** (epoch // 60))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(model,train_loader,test_loader,batch_size,epochs,device,optimizer,lr):
    for epoch in range(epochs):
        model.train()
        # adjust_learning_rate(optimizer, epoch,lr)  # 学习率衰减
        total_loss = 0.0

        for batch_idx, data in enumerate(train_loader):
            images, labels = data[0], data[1]

            # 正确转换：批次张量 → PIL列表
            to_pil = transforms.ToPILImage()
            images = [to_pil(tensor) for tensor in images]  # 逐个处理

            # 标签移到设备
            labels = labels.to(device)

            # 前向传播
            optimizer.zero_grad()
            gme_emb, fc_emb, hash_code = model(images, training=True)

            target_b = F.cosine_similarity(hash_code[:int(labels.size(0) / 2)], hash_code[int(labels.size(0) / 2):])
            target_x = F.cosine_similarity(gme_emb[:int(labels.size(0) / 2)], gme_emb[int(labels.size(0) / 2):])

            loss = F.mse_loss(target_b, target_x)
            # 反向传播与优化
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size
            if (batch_idx + 1) % 10 == 0:  # 每10个批次打印一次
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # 计算 epoch 平均损失
        avg_loss = total_loss / (len(train_loader) * batch_size)
        print(f"Epoch [{epoch + 1}/{epochs}] | 平均训练损失: {avg_loss:.4f}")


        model.save_lora_ckpt(f"./lora_ckpt/epoch_{epoch + 1}")

        # 每个epoch结束后评估mAP
        model.eval()
        with torch.inference_mode():
            retrievalB, retrievalL, queryB, queryL = compress(
                train_loader, test_loader, model, classes=3
            )
            map_score = calculate_map(
                qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL
            )
            print(f"Epoch [{epoch + 1}/{epochs}] | mAP@All: {map_score:.4f}\n")

def main(args):

    random.seed(0)
    torch.manual_seed(0)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    gme_config_path = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
    gme_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "cuda"

    }

    # 初始化可微调模型
    model = Gmenet(
        gme_config_path=gme_config_path,
        hash_dim=args.hash_dim,
        gme_kwargs=gme_kwargs,
        lora_r = args.lora_r,
        lora_alpha = args.lora_alpha,
    ).to(device)


    train_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()
                                          ])

    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor()
                                         ])
    if args.dataset == 'isic':
        train_dataset = ISICDataSet(data_dir= os.path.join(args.dataset_dir,'ISIC-2017_Training_Data'),
                                    image_list_file=args.train_image_list,transform=train_transform)
        test_dataset = ISICDataSet(data_dir=os.path.join(args.dataset_dir,'ISIC-2017_Test_v2_Data'),
                                   image_list_file=args.test_image_list,transform=test_transform)
    elif args.dataset == 'covid':
        train_dataset = ISICDataSet(data_dir= os.path.join(args.dataset_dir,'trainandval'),
                                    image_list_file=args.train_image_list,transform=train_transform)
        test_dataset = ISICDataSet(data_dir=os.path.join(args.dataset_dir,'test'),
                                   image_list_file=args.test_image_list,transform=test_transform)
    else:
        raise NotImplementedError('Dataset not supported!')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_worker,
        pin_memory=True,
        drop_last=True
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

    trainable_params = model.get_trainable_params()  # 模型中已定义的分模块参数

    optimizer = torch.optim.AdamW([
        {"params": trainable_params["lora_params"], "lr": args.lr},
        {"params": trainable_params["fc_params"], "lr": 10 * args.lr},
    ], weight_decay=5e-4)

    # model.eval()
    # with torch.inference_mode():
    #     for data in train_loader:
    #         images, labels = data[0], data[1]
    #
    #         to_pil = transforms.ToPILImage()
    #         images = [to_pil(tensor) for tensor in images]
    #
    #         print(f"输入图像类型：{type(images[0])}（应为PIL.Image）")
    #         print(f"批次大小：{len(images)}（应为偶数）")
    #
    #         emb, emb_fc, code = model(images, training=False)
    #
    #         print(f"Gme嵌入形状：{emb.shape}（应为[{args.batch_size}, 1536]）")
    #         print(f"哈希码形状：{code.shape}（应为[{args.batch_size}, {args.hash_dim}]）")
    #         print(f"哈希码示例：{code[:2]}（应为1/-1）")
    #         break

    train(model,train_loader,test_loader,args.batch_size,args.epochs,device,optimizer,args.lr)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Embedding Learning')

    parser.add_argument('--dataset', default='isic',help='Dataset to use (covid or isic)')

    parser.add_argument('--dataset-dir', default='./data/isic',help='Dataset directory path')

    parser.add_argument('--train-image-list', default='./ISIC-2017_Training_Part3_GroundTruth.csv', help='Train image list')

    parser.add_argument('--test-image-list', default='./ISIC-2017_Test_v2_Part3_GroundTruth_balanced.csv', help='Test image list')

    parser.add_argument('--batch-size', default=4, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--hash-dim', default=64, type=int, help='hashing dimension')

    parser.add_argument('--epochs', default=1, type=int, metavar='N', help='Number of training epochs to run')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')

    parser.add_argument('--lora-r', default=8, type=int, help='lora r')
    parser.add_argument('--lora-alpha', default=16, type=int, help='lora alpha')

    parser.add_argument('--num-worker', default=4, type=int, help='Number of dataloader workers')


    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)