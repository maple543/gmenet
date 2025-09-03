from __future__ import annotations

import base64
import logging
import math
import os
import requests
from io import BytesIO
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch import nn
from torch.autograd import Function
from transformers import AutoModelForVision2Seq, AutoProcessor


# 双半哈希函数实现
class bihalfHash(Function):
    @staticmethod
    def forward(ctx, U: torch.Tensor) -> torch.Tensor:
        """前向传播：生成二值哈希码（批次大小必须为偶数）"""
        assert U.shape[0] % 2 == 0, f"双半哈希要求批次大小为偶数，当前为{U.shape[0]}"
        _, index = U.sort(0, descending=True)
        N, D = U.shape
        half_n = int(N / 2)

        # 生成正负1矩阵并跟随输入设备
        B_creat = torch.cat((
            torch.ones([half_n, D], device=U.device, dtype=U.dtype),
            -torch.ones([N - half_n, D], device=U.device, dtype=U.dtype)
        ), dim=0)
        B = torch.zeros_like(U).scatter_(0, index, B_creat)
        ctx.save_for_backward(U, B)
        return B

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """反向传播：计算梯度（降低放大系数避免爆炸）"""
        U, B = ctx.saved_tensors
        add_grad = (U - B) / B.numel()  # 归一化梯度
        total_grad = grad_output + 0.5 * add_grad  # 温和的梯度放大
        return total_grad


def bihalfHash_layer(input: torch.Tensor) -> torch.Tensor:
    return bihalfHash.apply(input)


# GME+哈希融合模型
class GmeQwen2VLWithHash(nn.Module):
    def __init__(
            self,
            model_name: str = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
            model_path: Optional[str] = None,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            min_image_tokens: int = 256,
            max_image_tokens: int = 512,
            max_length: int = 768,
            hash_dim: int = 64, **kwargs,
    ) -> None:
        super().__init__()
        self.device = device
        self.hash_dim = hash_dim
        self.normalize = True

        # 初始化GME基础模型
        model_name = model_path or model_name
        self.gme_base = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )

        # 初始化图像处理器
        min_pixels = min_image_tokens * 28 * 28
        max_pixels = max_image_tokens * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels, **kwargs
        )
        self.processor.tokenizer.padding_side = 'right'
        self.default_instruction = 'You are a helpful assistant.'
        self.max_length = max_length

        # 特征降维与哈希相关层
        self.fc_down = nn.Linear(1536, self.hash_dim)  # 1536→64降维
        self.fc_bn = nn.BatchNorm1d(self.hash_dim, eps=1e-3)  # 增强数值稳定性

        # 初始化fc层参数
        nn.init.xavier_normal_(self.fc_down.weight)
        nn.init.constant_(self.fc_down.bias, 0.0)

        # 移动模型到目标设备
        self.to(self.device)

    def get_image_embeddings(self, images: List[str | Image.Image]) -> torch.Tensor:
        """提取GME的1536维图像特征"""
        # 处理输入图像（路径→PIL Image）
        input_images = []
        for img in images:
            if isinstance(img, str):
                img = fetch_image(img)  # 路径转PIL Image
            input_images.append(img)

        # 构造模型输入文本（仅包含图像标记）
        input_texts = [
            f'<|im_start|>system\n{self.default_instruction}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n<|endoftext|>'
            for _ in input_images
        ]

        # 预处理（获取grid_thw参数）
        inputs = self.processor(
            text=input_texts,
            images=input_images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 提取GME特征
        with torch.set_grad_enabled(self.training):
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            pixel_values = inputs.get('pixel_values')
            # 新增：获取grid_thw参数（视觉模型必需）
            grid_thw = inputs.get('image_grid_thw', None)  # 从processor输出中获取

            # 生成输入嵌入（文本+图像）
            inputs_embeds = self.gme_base.model.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.gme_base.visual.get_dtype())
                # 修复：传入grid_thw参数
                image_embeds = self.gme_base.visual(
                    pixel_values,
                    grid_thw=grid_thw  # 补充视觉模型必需的网格信息
                ).to(inputs_embeds.device)
                image_mask = input_ids == self.gme_base.config.image_token_id
                inputs_embeds[image_mask] = image_embeds

            # 前向传播获取隐藏状态
            outputs = self.gme_base.model(
                input_ids=None,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds
            )
            hidden_states = outputs.hidden_states[-1]

            # 池化获取句级特征
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            gme_emb = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]

            # 特征归一化
            if self.normalize:
                gme_emb = torch.nn.functional.normalize(gme_emb, p=2, dim=1)

        return gme_emb

    def forward(self, images: List[str | Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        """端到端前向传播：图像→GME特征→哈希码"""
        # 1. 获取GME特征（1536维）
        gme_emb = self.get_image_embeddings(images)

        # 2. 降维到hash_dim（64维）并稳定数值
        fc_emb = self.fc_down(gme_emb)
        fc_emb = self.fc_bn(fc_emb)
        fc_emb = torch.clamp(fc_emb, min=-2.0, max=2.0)  # 防止数值溢出

        # 3. 生成哈希码
        hash_code = bihalfHash_layer(fc_emb)

        return fc_emb, hash_code

    def get_trainable_params(self) -> List[nn.Parameter]:
        """获取所有可训练参数（GME LoRA参数 + 降维层参数）"""
        trainable_params = []
        # GME模型的可训练参数（含LoRA）
        for param in self.gme_base.parameters():
            if param.requires_grad:
                trainable_params.append(param)
        # 降维层和BN层参数
        trainable_params.extend(self.fc_down.parameters())
        trainable_params.extend(self.fc_bn.parameters())
        return trainable_params


# 图像处理辅助函数
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
        height: int,
        width: int,
        factor: int = IMAGE_FACTOR,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    if max(h_bar, w_bar) / min(h_bar, w_bar) > MAX_RATIO:
        logging.warning(f"图像宽高比超过阈值{MAX_RATIO}，已自动调整")
        if h_bar > w_bar:
            h_bar = w_bar * MAX_RATIO
        else:
            w_bar = h_bar * MAX_RATIO

    return h_bar, w_bar


def fetch_image(image: str, size_factor: int = IMAGE_FACTOR) -> Image.Image:
    """将图像路径（本地/网络/Base64）转换为PIL Image并调整大小"""
    try:
        if image.startswith(("http://", "https://")):
            image_obj = Image.open(requests.get(image, stream=True, timeout=10).raw)
        elif image.startswith("file://"):
            image_obj = Image.open(image[7:])
        elif image.startswith("data:image"):
            _, base64_data = image.split("base64,", 1)
            image_obj = Image.open(BytesIO(base64.b64decode(base64_data)))
        else:  # 本地路径
            if not os.path.exists(image):
                raise FileNotFoundError(f"图像文件不存在: {image}")
            image_obj = Image.open(image)
    except Exception as e:
        logging.warning(f"加载图像失败: {e}，使用空白图像替代")
        image_obj = Image.new('RGB', (224, 224), color=(255, 255, 255))

    # 统一转为RGB并调整大小
    image_obj = image_obj.convert("RGB")
    width, height = image_obj.size
    resized_height, resized_width = smart_resize(height, width, factor=size_factor)
    return image_obj.resize((resized_width, resized_height))
