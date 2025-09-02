import random

import torch
from torch.utils.data import DataLoader

from demo.model import Gmenet
from demo.read_data import ISICDataSet
import torch.nn.functional as F
from cal_map_mult import calculate_top_map, calculate_map, compress
import torchvision.transforms as transforms

# 超参数配置
learning_rate = 0.001
epoch_lr_decrease = 60  # 每60个epoch学习率衰减10倍
hash_dim = 16
batch_size = 8  # 必须为偶数（双半哈希要求）
epochs = 5
lora_r = 64  # 推荐8-32，根据显存调整
lora_alpha = 16

def adjust_learning_rate(optimizer, epoch):
    """学习率衰减策略"""
    lr = learning_rate * (0.1 ** (epoch // epoch_lr_decrease))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():

    random.seed(0)
    torch.manual_seed(0)

    # 配置参数（2B模型建议用FP16节省显存）
    gme_config_path = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
    hash_dim = 16  # 与FC输出维度一致
    gme_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto"  # 自动分配设备（优先GPU）

    }


    # 初始化可微调模型
    model = Gmenet(
        gme_config_path=gme_config_path,
        hash_dim=hash_dim,
        gme_kwargs=gme_kwargs,
        lora_r = lora_r,
        lora_alpha = lora_alpha
    )

    device = next(model.parameters()).device

    train_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()
                                          ])

    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor()
                                         ])
    train_dataset = ISICDataSet(
        data_dir="./data/isic/ISIC-2017_Training_Data",
        image_list_file="./ISIC-2017_Training_Part3_GroundTruth.csv",
        transform=train_transform
    )
    test_dataset = ISICDataSet(
        data_dir="./data/isic/ISIC-2017_Test_v2_Data",
        image_list_file="./ISIC-2017_Test_v2_Part3_GroundTruth_balanced.csv",
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True  # 关键：确保批次大小为偶数（双半哈希要求）
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试集不打乱
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # 打印数据信息
    print(f"数据加载完成：")
    print(f"- 训练集：{len(train_dataset)}张，批次大小{batch_size}（丢弃最后非偶数批次）")
    print(f"- 测试集：{len(test_dataset)}张")

    trainable_params = model.get_trainable_params()  # 模型中已定义的分模块参数
    optimizer = torch.optim.AdamW([
        {"params": trainable_params["lora_params"], "lr": 2e-4},  # Gme低学习率
        {"params": trainable_params["fc_params"], "lr": 1e-3}  # FC层高学习率
    ], weight_decay=5e-4)

    model.eval()
    with torch.inference_mode():
        for data in train_loader:
            images, labels = data[0], data[1]
            # 正确转换：批次张量 → PIL列表
            to_pil = transforms.ToPILImage()
            images = [to_pil(tensor.cpu()) for tensor in images]  # 逐个处理

            # 验证输入类型：images应为PIL.Image列表
            print(f"输入图像类型：{type(images[0])}（应为PIL.Image）")
            print(f"批次大小：{len(images)}（应为偶数）")
            # images = images.to(dtype=torch.float32)
            # 模型前向传播（直接传入PIL列表）
            emb, emb_fc, code = model(images, training=False)

            # 验证输出格式
            print(f"Gme嵌入形状：{emb.shape}（应为[{batch_size}, 1536]）")
            print(f"哈希码形状：{code.shape}（应为[{batch_size}, {hash_dim}]）")
            print(f"哈希码示例：{code[:2]}（应为1/-1）")
            break  # 仅验证一个批次

        # 8. 训练循环
    # for epoch in range(epochs):
    #     model.train()
    #     # adjust_learning_rate(optimizer, epoch)  # 学习率衰减
    #     total_loss = 0.0
    #
    #     for batch_idx, data in enumerate(train_loader):
    #         images, labels = data[0], data[1]
    #
    #         # 正确转换：批次张量 → PIL列表
    #         to_pil = transforms.ToPILImage()
    #         images = [to_pil(tensor.cpu()) for tensor in images]  # 逐个处理
    #
    #         # 标签移到设备
    #         labels = labels.to(device)
    #
    #         # 前向传播（传入PIL图像列表，training=True启用梯度）
    #         optimizer.zero_grad()
    #         gme_emb, fc_emb, hash_code = model(images, training=True)
    #
    #         # 对比损失计算（基于标签的同类/异类约束）
    #         # 分割批次为两半，计算相似性损失
    #         half_batch = batch_size // 2
    #         target_sim = F.cosine_similarity(
    #             labels[:half_batch], labels[half_batch:], dim=1
    #         ).unsqueeze(1)  # 标签语义相似度
    #         hash_sim = F.cosine_similarity(
    #             hash_code[:half_batch], hash_code[half_batch:], dim=1
    #         ).unsqueeze(1)  # 哈希码相似度
    #         loss = F.mse_loss(hash_sim, target_sim)  # 最小化哈希相似度与标签相似度的差异
    #
    #         # 反向传播与优化
    #         loss.backward()
    #         optimizer.step()
    #
    #         total_loss += loss.item() * batch_size
    #         if (batch_idx + 1) % 10 == 0:  # 每10个批次打印一次
    #             print(
    #                 f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    #
    #     # 计算 epoch 平均损失
    #     avg_loss = total_loss / (len(train_loader) * batch_size)
    #     print(f"Epoch [{epoch + 1}/{epochs}] | 平均训练损失: {avg_loss:.4f}")
    #     # 保存LoRA权重（仅几MB，无需保存整个模型）
    #     model.save_lora_ckpt(f"./lora_ckpt/epoch_{epoch + 1}")
    #
    #     # 每个epoch结束后评估mAP
    #     model.eval()
    #     with torch.inference_mode():
    #         retrievalB, retrievalL, queryB, queryL = compress(
    #             train_loader, test_loader, model, classes=3
    #         )
    #         map_score = calculate_map(
    #             qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL
    #         )
    #         print(f"Epoch [{epoch + 1}/{epochs}] | mAP@All: {map_score:.4f}\n")

if __name__ == '__main__':
    main()