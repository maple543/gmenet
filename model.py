from __future__ import annotations

import base64
import logging
import math
import os
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Function
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from transformers import AutoProcessor, PreTrainedModel
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
    Qwen2VLConfig,
    Qwen2VLModel,
)

# 新增：PEFT库用于LoRA
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from transformers.utils.versions import require_version



require_version(
    "transformers<4.52.0",
    "This code has some issues with transformers>=4.52.0, please downgrade: pip install transformers==4.51.3"
)


class GmeQwen2VLConfig(Qwen2VLConfig):
    def __init__(
        self,
        min_image_tokens: int = 256,
        max_image_tokens: int = 1280,
        max_length: int = 1800,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.min_image_tokens = min_image_tokens
        self.max_image_tokens = max_image_tokens
        self.max_length = max_length


class GmeQwen2VL(PreTrainedModel):
    config_class = GmeQwen2VLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2VLDecoderLayer", "Qwen2VLVisionBlock"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_static_cache = False

    def __init__(self, config: GmeQwen2VLConfig,** kwargs: Any) -> None:
        super().__init__(config)
        self.visual = Qwen2VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.model = Qwen2VLModel(config)
        self.vocab_size = config.vocab_size
        self.rope_deltas = None  # cache rope_deltas here

        min_pixels: int = config.min_image_tokens * 28 * 28
        max_pixels: int = config.max_image_tokens * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            config._name_or_path, min_pixels=min_pixels, max_pixels=max_pixels, **kwargs
        )
        self.max_length: int = config.max_length
        self.normalize: bool = True
        self.processor.tokenizer.padding_side = "right"
        self.default_instruction: str = "You are a helpful assistant."
        self.sep: str = " "

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        pooling_mask: Optional[torch.LongTensor] = None,** kwargs
    ) -> torch.Tensor:

        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw).to(inputs_embeds.device)
                image_mask = input_ids == self.config.image_token_id
                inputs_embeds[image_mask] = image_embeds
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )

        pooling_mask = attention_mask if pooling_mask is None else pooling_mask
        left_padding = (pooling_mask[:, -1].sum() == pooling_mask.shape[0])
        if left_padding:
            embeddings = outputs.last_hidden_state[:, -1]
        else:
            sequence_lengths = pooling_mask.sum(dim=1) - 1
            batch_size = outputs.last_hidden_state.shape[0]
            embeddings = outputs.last_hidden_state[torch.arange(
                batch_size, device=outputs.last_hidden_state.device
            ), sequence_lengths]
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.contiguous()

    def embed(self, texts: list[str], images: list[Image.Image], is_query=True, instruction=None, **kwargs):

        self.eval()
        input_texts, input_images = list(), list()
        for t, i in zip(texts, images):
            if not is_query or instruction is None:
                instruction = self.default_instruction
            input_str = ''
            if i is None:
                input_images = None
            else:
                input_str += '<|vision_start|><|image_pad|><|vision_end|>'
                i = fetch_image(i)
                input_images.append(i)
            if t is not None:
                input_str += t

            msg = f'<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input_str}<|im_end|>\n<|im_start|>assistant\n<|endoftext|>'
            input_texts.append(msg)

        inputs = self.processor(
            text=input_texts,
            images=input_images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            embeddings = self.forward(** inputs)
        return embeddings


    def encode(self, sentences: list[str], *, prompt_name=None, **kwargs):
        return self.get_fused_embeddings(texts=sentences, prompt_name=prompt_name, **kwargs)

    def encode_queries(self, queries: List[str], **kwargs):
        embeddings = self.encode(queries, **kwargs)
        return embeddings

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs):
        if type(corpus) is dict:
            sentences = [
                (corpus["title"][i] + self.sep + corpus["text"][i]).strip()
                if "title" in corpus
                else corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        else:
            sentences = [
                (doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip()
                for doc in corpus
            ]
        embeddings = self.encode(sentences, is_query=False, **kwargs)
        return embeddings

    def get_image_embeddings(self, images: list[Image.Image] | DataLoader, **kwargs):
        return self.get_fused_embeddings(images=images, **kwargs)

    def get_text_embeddings(self, texts: list[str], **kwargs):
        return self.get_fused_embeddings(texts=texts, **kwargs)

    def get_fused_embeddings(self, texts: list[str] = None, images: list[Image.Image] | DataLoader = None, **kwargs):
        if isinstance(images, DataLoader):
            image_loader = images
            batch_size = image_loader.batch_size
            image_loader.dataset.transform = None
        else:
            batch_size = kwargs.pop('batch_size', 32)
            if images is None:
                image_loader = None
            else:
                image_loader = DataLoader(
                    images,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=custom_collate_fn,
                    num_workers=min(math.floor(os.cpu_count() / 2), 8),
                )

        if texts is None:
            assert image_loader is not None
            n_batch = len(image_loader)
        else:
            n_batch = len(texts) // batch_size + int(len(texts) % batch_size > 0)
            image_loader = image_loader or [None] * n_batch

        all_embeddings = list()
        none_batch = [None] * batch_size
        show_progress_bar = kwargs.pop('show_progress_bar', False)
        pbar = tqdm(total=n_batch, disable=not show_progress_bar, mininterval=1, miniters=10, desc='encode')
        for n, img_batch in zip(range(0, n_batch * batch_size, batch_size), image_loader):
            text_batch = none_batch if texts is None else texts[n: n+batch_size]
            img_batch = none_batch if img_batch is None else img_batch
            embeddings = self.embed(texts=text_batch, images=img_batch, **kwargs)
            pbar.update(1)
            all_embeddings.append(embeddings.cpu())
        pbar.close()
        all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings



def custom_collate_fn(batch):
    return batch


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
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
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
        logging.warning(
            f"Absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(h_bar, w_bar) / min(h_bar, w_bar)}"
        )
        if h_bar > w_bar:
            h_bar = w_bar * MAX_RATIO
        else:
            w_bar = h_bar * MAX_RATIO
    return h_bar, w_bar


def fetch_image(image: str | Image.Image, size_factor: int = IMAGE_FACTOR) -> Image.Image:
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = Image.open(requests.get(image, stream=True).raw)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = image_obj.convert("RGB")
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=size_factor,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    image = image.resize((resized_width, resized_height))
    return image


class bihalfHash(Function):
    @staticmethod
    def forward(ctx, U: torch.Tensor) -> torch.Tensor:
        assert U.shape[0] % 2 == 0, f"双半哈希要求批次大小为偶数，当前为{U.shape[0]}"
        _, index = U.sort(0, descending=True)
        N, D = U.shape
        half_n = int(N / 2)

        # 动态设备：跟随输入U（兼容原生Gme的设备）
        B_creat = torch.cat((
            torch.ones([half_n, D], device=U.device, dtype=U.dtype),
            -torch.ones([N - half_n, D], device=U.device, dtype=U.dtype)
        ), dim=0)
        B = torch.zeros_like(U).scatter_(0, index, B_creat)

        ctx.save_for_backward(U, B)
        return B

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        U, B = ctx.saved_tensors
        add_grad = (U - B) / B.numel()  # 归一化梯度，避免量级问题
        total_grad = grad_output + 6 * add_grad  # 梯度=上游梯度+正则化梯度
        return total_grad


def bihalfHash_layer(input: torch.Tensor) -> torch.Tensor:
    return bihalfHash.apply(input)



class Gmenet(nn.Module):

    def __init__(self, gme_config_path, hash_dim, lora_r, lora_alpha, gme_kwargs: Optional[dict] = None):
        super().__init__()
        gme_kwargs = gme_kwargs or {}

        self.gme = self._init_gme(gme_config_path, **gme_kwargs)

        self.gme_dim = self.gme.config.hidden_size
        self.hash_dim = hash_dim

        # print(self.gme)

        num_visual_blocks = len(self.gme.visual.blocks)
        target_modules = []
        for block_idx in range(num_visual_blocks):
            target_modules.extend([
                f"visual.blocks.{block_idx}.attn.qkv",
                f"visual.blocks.{block_idx}.attn.proj"
            ])

        # num_decoder_layers = len(self.gme.model.layers)
        # for layer_idx in range(num_decoder_layers):
        #     target_modules.extend([
        #         f"model.layers.{layer_idx}.self_attn.q_proj",
        #         f"model.layers.{layer_idx}.self_attn.k_proj",
        #         f"model.layers.{layer_idx}.self_attn.v_proj",
        #     ])

        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )

        #  应用LoRA适配器（自动冻结非目标层）
        self.gme = get_peft_model(self.gme, self.lora_config)
        self.gme.print_trainable_parameters()


        self.fc_layer = nn.Linear(self.gme_dim, self.hash_dim)
        self.fc_bn = nn.BatchNorm1d(self.hash_dim)

    def _init_gme(self, config_path: str, **kwargs) -> GmeQwen2VL:

        config = GmeQwen2VLConfig.from_pretrained(config_path, **kwargs)

        model = GmeQwen2VL(config, **kwargs)
        if os.path.exists(config_path):
            state_dict = torch.load(os.path.join(config_path, "pytorch_model.bin"), map_location="cpu")
        else:
            state_dict = model.from_pretrained(config_path, **kwargs).state_dict()
        model.load_state_dict(state_dict, strict=False)

        return model

    def forward(self, images: list[Image.Image], training: bool = False) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """前向传播：PIL图像→Gme嵌入（LoRA微调）→FC→哈希码"""
        # 1. 控制Gme的训练状态（原生默认eval，训练时切换为train）
        if training:
            self.gme.train()
            # 禁用原生embed的inference_mode，启用梯度
            gme_emb = self._gme_embed_with_grad(images)
        else:
            self.gme.eval()
            # 推理时用原生embed（含inference_mode，禁用梯度）
            gme_emb = self.gme.embed(texts=[None]*len(images), images=images)

        # 2. FC层+BatchNorm（训练时更新统计量）
        fc_emb = self.fc_layer(gme_emb)
        fc_emb = self.fc_bn(fc_emb)  # 训练时自动启用更新，推理时固定

        # 3. 双半哈希生成二值码
        hash_code = bihalfHash_layer(fc_emb)

        return gme_emb, fc_emb, hash_code

    def _gme_embed_with_grad(self, images: list[Image.Image]) -> torch.Tensor:
        """训练专用：绕过原生embed的inference_mode，保留梯度"""
        # 复用原生embed的预处理逻辑（生成input_ids/pixel_values）
        input_texts = []
        input_images = []
        instruction = self.gme.default_instruction
        for img in images:
            # 原生文本模板：确保格式正确
            input_str = '<|vision_start|><|image_pad|><|vision_end|>'  # 仅图像，无额外文本
            msg = f'<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input_str}<|im_end|>\n<|im_start|>assistant\n<|endoftext|>'
            input_texts.append(msg)
            # 原生图像预处理
            input_images.append(fetch_image(img))

        # 原生processor调用：生成模型输入
        inputs = self.gme.processor(
            text=input_texts,
            images=input_images,
            padding=True,
            truncation=True,
            max_length=self.gme.max_length,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.gme.device) for k, v in inputs.items()}

        # 启用梯度，调用原生forward（无inference_mode）
        gme_emb = self.gme.forward(** inputs)
        return gme_emb

    def get_trainable_params(self) -> dict:
        """分模块返回可训练参数（支持分学习率）"""
        return {
            "lora_params": list(self.gme.parameters()),  # lora参数：低学习率
            "fc_params": list(self.fc_layer.parameters()) + list(self.fc_bn.parameters())  # FC参数：高学习率
        }

    def save_lora_ckpt(self, save_path: str):
        """仅保存LoRA适配器权重（体积小）"""
        self.gme.save_pretrained(save_path)

    def load_lora_ckpt(self, ckpt_path: str):
        """加载LoRA适配器权重"""
        from peft import PeftModel
        self.gme = PeftModel.from_pretrained(self.gme, ckpt_path)
        return self

    def to(self, device: Union[str, torch.device], **kwargs) -> "Gmenet":
        self.gme = self.gme.to(device,** kwargs)
        self.fc_layer = self.fc_layer.to(device, **kwargs)
        self.fc_bn = self.fc_bn.to(device,** kwargs)
        return self