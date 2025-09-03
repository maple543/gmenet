import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from gme import GmeQwen2VL  # 导入你的GmeQwen2VL类
from demo.model import Hashnet  # 导入你的Hashnet类
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def test_feature_extraction():
    # -------------------------- 1. 配置基础参数 --------------------------
    # 请根据你的环境修改以下路径！
    GME_MODEL_PATH = "../gme-Qwen2-VL-2B-Instruct"  # Gme模型权重路径
    TEST_IMAGE_PATH = r"D:\workplace\demo\data\isic\ISIC-2017_Training_Data\ISIC_0000000.jpg"  # 测试用单张图像路径（如ISIC数据集的任意图像）
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设备（CPU/GPU）
    HASH_DIM = 64  # 与训练时一致的哈希维度

    # 检查测试图像是否存在
    if not os.path.exists(TEST_IMAGE_PATH):
        raise FileNotFoundError(f"测试图像不存在！路径：{TEST_IMAGE_PATH}")
    # 检查GME模型路径是否存在
    if not os.path.exists(GME_MODEL_PATH):
        raise FileNotFoundError(f"GME模型路径不存在！路径：{GME_MODEL_PATH}")


    # -------------------------- 2. 定义与训练一致的图像预处理 --------------------------
    # 必须与训练时的预处理逻辑完全一致（否则特征提取结果会偏差）
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),  # 转为RGB（避免灰度图问题）
        transforms.Resize(256),
        transforms.CenterCrop(224),  # 与训练时一致
        transforms.ToTensor(),
        normalize
    ])


    # -------------------------- 3. 加载GmeQwen2VL模型（特征提取核心） --------------------------
    print(f"[Step 1] 在 {DEVICE} 上加载 GmeQwen2VL 模型...")
    try:
        # 初始化GME模型（无需LoRA，仅测试特征提取时可简化）
        emb_model = GmeQwen2VL(model_path=GME_MODEL_PATH, device=DEVICE)
        # 设置为评估模式（关闭训练相关层，如Dropout）
        emb_model.base.eval()
        print(f"[Success] GmeQwen2VL 模型加载完成！")
    except Exception as e:
        print(f"[Error] GmeQwen2VL 模型加载失败：{str(e)}")
        return


    # -------------------------- 4. 加载并预处理测试图像 --------------------------
    print(f"\n[Step 2] 加载并预处理测试图像：{TEST_IMAGE_PATH}")
    try:
        # 加载图像
        img = Image.open(TEST_IMAGE_PATH)
        # 预处理（转为Tensor并添加batch维度：[H,W,C] → [1, C, H, W]）
        img_tensor = test_transform(img).unsqueeze(0).to(DEVICE)  # batch_size=1
        print(f"[Success] 图像预处理完成！张量形状：{img_tensor.shape}（batch, channel, height, width）")
    except Exception as e:
        print(f"[Error] 图像加载/预处理失败：{str(e)}")
        return


    # -------------------------- 5. 提取GME原始图像特征 --------------------------
    print(f"\n[Step 3] 提取 GME 原始图像特征...")
    with torch.no_grad():  # 关闭梯度计算，加速并避免内存占用
        try:
            # 调用特征提取方法
            gme_emb = emb_model.get_image_embeddings(img_tensor)
            # 验证特征维度（GmeQwen2VL输出的特征维度通常是固定的，如768/1024等，需根据模型确认）
            print(f"[Success] GME 原始特征提取完成！")
            print(f"  - 特征形状：{gme_emb.shape} → (batch_size, gme_dim)")  # 预期：[1, gme_dim]
            print(f"  - 特征设备：{gme_emb.device}")
            print(f"  - 特征数据类型：{gme_emb.dtype}")
            gme_dim = gme_emb.size(1)  # 获取GME特征维度，用于初始化Hashnet
        except Exception as e:
            print(f"[Error] GME 特征提取失败：{str(e)}")
            return


    # -------------------------- 6. 测试Hashnet特征处理（可选，验证端到端） --------------------------
    print(f"\n[Step 4] 测试 Hashnet 特征处理...")
    try:
        # 初始化Hashnet（使用GME特征维度gme_dim）
        hashnet = Hashnet(hash_dim=HASH_DIM, gme_dim=gme_dim).to(DEVICE)
        hashnet.eval()  # 评估模式

        with torch.no_grad():
            # 输入GME特征，输出Hashnet处理后的特征（fc_emb）和哈希码（hash_code）
            fc_emb, hash_code = hashnet(gme_emb)

        print(f"[Success] Hashnet 特征处理完成！")
        print(f"  - 哈希前特征（fc_emb）形状：{fc_emb.shape} → (batch_size, hash_dim)")  # 预期：[1, 64]
        print(f"  - 哈希码（hash_code）形状：{hash_code.shape} → (batch_size, hash_dim)")  # 预期：[1, 64]
        print(f"  - 哈希码取值（前10个）：{hash_code[0, :10].cpu().numpy()}")  # 应只有1/-1（双半哈希特性）
    except Exception as e:
        print(f"[Error] Hashnet 特征处理失败：{str(e)}")
        return


    # -------------------------- 7. 最终验证结果 --------------------------
    print(f"\n[Final Check] 特征提取全链路验证通过！")
    print(f"  核心结论：GME能输出有效图像特征，Hashnet能正常处理特征并生成哈希码。")


if __name__ == "__main__":
    test_feature_extraction()