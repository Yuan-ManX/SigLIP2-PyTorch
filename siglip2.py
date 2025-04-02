import math
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.init import _calculate_fan_in_and_fan_out

from configuration_siglip2 import Siglip2Config, Siglip2TextConfig, Siglip2VisionConfig


@dataclass
class Siglip2VisionOutput():
    """
    视觉模型的输出基类，包含最后一层隐藏状态的池化结果得到的图像嵌入。

    Args:
        image_embeds (torch.FloatTensor, 可选, 形状为 `(batch_size, output_dim)`):
            当模型初始化时设置 `with_projection=True`，则返回通过投影层应用于 `pooler_output` 得到的图像嵌入。
            - **类型**: `torch.FloatTensor`
            - **形状**: `(batch_size, output_dim)`
            - **说明**: 图像嵌入，用于表示图像的特征。
        
        last_hidden_state (torch.FloatTensor, 必填, 形状为 `(batch_size, sequence_length, hidden_size)`):
            模型最后一层输出的隐藏状态序列。
            - **类型**: `torch.FloatTensor`
            - **形状**: `(batch_size, sequence_length, hidden_size)`
            - **说明**: 包含模型最后一层输出的隐藏状态，用于后续的任务处理。
        
        hidden_states (tuple(torch.FloatTensor), 可选):
            当 `output_hidden_states=True` 被传递或 `config.output_hidden_states=True` 时返回。
            - **类型**: `tuple(torch.FloatTensor)`
            - **说明**: 包含模型每一层的隐藏状态输出，以及可选的初始嵌入输出。
            - **形状**: `(batch_size, sequence_length, hidden_size)` 每个元组元素。
        
        attentions (tuple(torch.FloatTensor), 可选):
            当 `output_attentions=True` 被传递或 `config.output_attentions=True` 时返回。
            - **类型**: `tuple(torch.FloatTensor)`
            - **说明**: 包含每一层的注意力权重，经过 softmax 后的注意力权重，用于计算自注意力头中的加权平均。
            - **形状**: `(batch_size, num_heads, sequence_length, sequence_length)` 每个元组元素。
    """

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class Siglip2TextOutput():
    """
    文本模型的输出基类，包含最后一层隐藏状态的池化结果得到的文本嵌入。

    Args:
        text_embeds (torch.FloatTensor, 可选, 形状为 `(batch_size, output_dim)`):
            当模型初始化时设置 `with_projection=True`，则返回通过投影层应用于 `pooler_output` 得到的文本嵌入。
            - **类型**: `torch.FloatTensor`
            - **形状**: `(batch_size, output_dim)`
            - **说明**: 文本嵌入，用于表示文本的特征。
        
        last_hidden_state (torch.FloatTensor, 必填, 形状为 `(batch_size, sequence_length, hidden_size)`):
            模型最后一层输出的隐藏状态序列。
            - **类型**: `torch.FloatTensor`
            - **形状**: `(batch_size, sequence_length, hidden_size)`
            - **说明**: 包含模型最后一层输出的隐藏状态，用于后续的任务处理。
        
        hidden_states (tuple(torch.FloatTensor), 可选):
            当 `output_hidden_states=True` 被传递或 `config.output_hidden_states=True` 时返回。
            - **类型**: `tuple(torch.FloatTensor)`
            - **说明**: 包含模型每一层的隐藏状态输出，以及可选的初始嵌入输出。
            - **形状**: `(batch_size, sequence_length, hidden_size)` 每个元组元素。
        
        attentions (tuple(torch.FloatTensor), 可选):
            当 `output_attentions=True` 被传递或 `config.output_attentions=True` 时返回。
            - **类型**: `tuple(torch.FloatTensor)`
            - **说明**: 包含每一层的注意力权重，经过 softmax 后的注意力权重，用于计算自注意力头中的加权平均。
            - **形状**: `(batch_size, num_heads, sequence_length, sequence_length)` 每个元组元素。
    """

    text_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class Siglip2Output():
    """
    Siglip2 模型的输出，包含图像-文本对比损失、相似度分数、嵌入以及子模型的输出。

    Args:
        loss (torch.FloatTensor, 可选, 形状为 `(1,)`):
            当 `return_loss=True` 时返回，用于图像-文本相似度的对比损失。
            - **类型**: `torch.FloatTensor`
            - **形状**: `(1,)`
            - **说明**: 对比损失，用于衡量图像和文本之间的相似度。
        
        logits_per_image (torch.FloatTensor, 必填, 形状为 `(image_batch_size, text_batch_size)`):
            `image_embeds` 和 `text_embeds` 之间的缩放点积分数，表示图像-文本相似度分数。
            - **类型**: `torch.FloatTensor`
            - **形状**: `(image_batch_size, text_batch_size)`
            - **说明**: 图像-文本相似度分数，用于评估图像和文本之间的匹配程度。
        
        logits_per_text (torch.FloatTensor, 必填, 形状为 `(text_batch_size, image_batch_size)`):
            `text_embeds` 和 `image_embeds` 之间的缩放点积分数，表示文本-图像相似度分数。
            - **类型**: `torch.FloatTensor`
            - **形状**: `(text_batch_size, image_batch_size)`
            - **说明**: 文本-图像相似度分数，用于评估文本和图像之间的匹配程度。
        
        text_embeds (torch.FloatTensor, 必填, 形状为 `(batch_size, output_dim)`):
            通过投影层应用于 [`Siglip2TextModel`] 的池化输出得到的文本嵌入。
            - **类型**: `torch.FloatTensor`
            - **形状**: `(batch_size, output_dim)`
            - **说明**: 文本嵌入，用于表示文本的特征。
        
        image_embeds (torch.FloatTensor, 必填, 形状为 `(batch_size, output_dim)`):
            通过投影层应用于 [`Siglip2VisionModel`] 的池化输出得到的图像嵌入。
            - **类型**: `torch.FloatTensor`
            - **形状**: `(batch_size, output_dim)`
            - **说明**: 图像嵌入，用于表示图像的特征。
        
        text_model_output (BaseModelOutputWithPooling):
            [`Siglip2TextModel`] 的输出。
            - **类型**: `BaseModelOutputWithPooling`
            - **说明**: 包含文本模型的详细输出信息，如隐藏状态等。
        
        vision_model_output (BaseModelOutputWithPooling):
            [`Siglip2VisionModel`] 的输出。
            - **类型**: `BaseModelOutputWithPooling`
            - **说明**: 包含视觉模型的详细输出信息，如隐藏状态等。
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        """
        将 Siglip2Output 对象转换为元组。

        Returns:
            Tuple[Any]: 包含 Siglip2Output 对象的各个属性值的元组。
        """
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class Siglip2VisionEmbeddings(nn.Module):
    """
    Siglip2 视觉嵌入模块，用于将图像像素值转换为嵌入向量，并添加位置嵌入。

    Args:
        config (Siglip2VisionConfig): 
            视觉模型的配置对象，包含模型的各种配置参数，如隐藏层大小、patch 大小等。
    """
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        # 嵌入维度，通常与隐藏层大小相同
        self.embed_dim = config.hidden_size
        # patch 大小，表示每个图像块的高度和宽度
        self.patch_size = config.patch_size

        # 定义一个线性层，用于将每个图像 patch（像素块）映射到嵌入向量
        self.patch_embedding = nn.Linear(
            # 输入特征数：通道数 * patch大小平方
            in_features=config.num_channels * self.patch_size * self.patch_size,
            # 输出特征数：嵌入维度
            out_features=self.embed_dim,
        )

        # 图像被分割成的总patch数量
        self.num_patches = config.num_patches
        # 计算位置嵌入的网格大小（假设图像是正方形）
        self.position_embedding_size = int(self.num_patches**0.5)
        # 定义一个嵌入层，用于存储每个patch的位置嵌入
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    @staticmethod
    def resize_positional_embeddings(
        positional_embeddings: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        max_length: int,
    ) -> torch.Tensor:
        """
        调整位置嵌入的大小以适应图像的特定尺寸，并填充到固定大小。

        Args:
            positional_embeddings (`torch.Tensor`):
                位置嵌入张量，形状为 (高度, 宽度, 嵌入维度)。
            spatial_shapes (`torch.LongTensor`):
                空间形状张量，形状为 (batch_size, 2)，用于调整位置嵌入的大小。
                每个元素包含 [目标高度, 目标宽度]。
            max_length (`int`):
                填充后的最大长度，用于确保所有批次的位置嵌入具有相同的长度。

        Returns:
            `torch.Tensor`: 
                调整大小并填充后的嵌入张量，形状为 (batch_size, max_length, 嵌入维度)。
        """
        # 获取批次大小
        batch_size = spatial_shapes.shape[0]
        # 获取嵌入维度
        embed_dim = positional_embeddings.shape[-1]
        # 记录原始数据类型
        source_dtype = positional_embeddings.dtype

        # 创建一个空的张量，用于存储调整后的位置嵌入
        resulted_positional_embeddings = torch.empty(
            (batch_size, max_length, embed_dim),
            device=positional_embeddings.device,
            dtype=source_dtype,
        )

        # 将位置嵌入的维度顺序从 (高度, 宽度, 嵌入维度) 转换为 (嵌入维度, 高度, 宽度) 以便进行插值
        positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)

        # 如果设备是 CPU，则将数据类型上转换为 float32，因为 CPU 不支持 bfloat16/float16 的 antialias
        if positional_embeddings.device.type == "cpu":
            positional_embeddings = positional_embeddings.to(torch.float32)

        for i in range(batch_size):
            # 获取当前批次的目标高度和宽度
            # (1, dim, height, width) -> (1, dim, target_height, target_width)
            height, width = spatial_shapes[i]
            # 对位置嵌入进行双线性插值，调整到目标尺寸
            resized_embeddings = F.interpolate(
                positional_embeddings,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

            # 将调整后的嵌入形状从 (1, 嵌入维度, 高度, 宽度) 转换为 (高度 * 宽度, 嵌入维度)
            # (1, dim, target_height, target_width) -> (target_height * target_width, dim)
            resized_embeddings = resized_embeddings.reshape(embed_dim, height * width).transpose(0, 1)

            # 将数据类型转换回原始类型
            resized_embeddings = resized_embeddings.to(source_dtype)

            # 将调整后的嵌入填充到结果张量中
            resulted_positional_embeddings[i, : height * width] = resized_embeddings
            # 对于不足的部分，用第一个位置的嵌入填充
            resulted_positional_embeddings[i, height * width :] = resized_embeddings[0]

        return resulted_positional_embeddings

    def forward(self, pixel_values: torch.FloatTensor, spatial_shapes: torch.LongTensor) -> torch.Tensor:
        """
        前向传播方法，用于生成图像嵌入。

        Args:
            pixel_values (`torch.FloatTensor`):
                像素值张量，形状为 (batch_size, 最大patch数量, 通道数 * patch大小平方)。
            spatial_shapes (`List[Tuple[int, int]]`):
                空间形状列表，形状为 (batch_size, 2)，用于调整位置嵌入的大小。
                每个元素包含 [高度, 宽度]。

        Returns:
            `torch.Tensor`: 
                生成的图像嵌入张量，形状为 (batch_size, 最大patch数量, 嵌入维度)。
        """
        # 将像素值张量转换为与patch_embedding权重相同的类型
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))

        # 获取位置嵌入，并调整其形状为 (高度, 宽度, 嵌入维度)
        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size, self.position_embedding_size, -1
        )

        # 调整位置嵌入的大小以适应图像的特定尺寸，并填充到固定大小
        resized_positional_embeddings = self.resize_positional_embeddings(
            positional_embeddings, spatial_shapes, max_length=pixel_values.shape[1]
        )

        # 将位置嵌入添加到patch嵌入中，得到最终的图像嵌入
        embeddings = patch_embeds + resized_positional_embeddings
        return embeddings


class Siglip2Attention(nn.Module):
    """
    多头注意力机制，源自论文 'Attention Is All You Need'。

    Args:
        config: 
            模型配置对象，包含以下属性：
            - hidden_size (int): 隐藏层大小，也是注意力机制的嵌入维度。
            - num_attention_heads (int): 注意力头的数量。
            - attention_dropout (float): 注意力权重在 dropout 时的丢弃概率。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        # 注意力机制的嵌入维度，通常与隐藏层大小相同
        self.embed_dim = config.hidden_size
        # 注意力头的数量
        self.num_heads = config.num_attention_heads
        # 每个注意力头的维度
        self.head_dim = self.embed_dim // self.num_heads

        # 检查 embed_dim 是否能被 num_heads 整除
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        
        # 缩放因子，用于缩放注意力得分
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        # 定义线性层，用于计算查询 (query)、键 (key) 和值 (value) 的投影
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # 输出投影层
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播方法，计算多头注意力。

        Args:
            hidden_states (`torch.Tensor`):
                输入张量，形状为 (batch_size, 时间步长度, 通道数)。
            attention_mask (`torch.Tensor`, 可选):
                注意力掩码张量，用于屏蔽某些位置，形状为 (batch_size, 1, 时间步长度, 时间步长度)。
            output_attentions (`bool`, 可选):
                是否返回注意力权重。

        Returns:
            `Tuple[torch.Tensor, Optional[torch.Tensor]]`:
                - `attn_output`: 注意力输出，形状为 (batch_size, 时间步长度, 隐藏层大小)。
                - `attn_weights`: 注意力权重，如果 `output_attentions` 为 `True`，则返回，形状为 (batch_size, num_heads, 时间步长度, 时间步长度)。
        """
        # 获取批次大小和时间步长度
        batch_size, q_len, _ = hidden_states.size()

        # 计算查询 (query)、键 (key) 和值 (value) 的投影
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 重塑查询、键和值张量，以适应多头注意力的计算
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 获取键和值序列的长度
        k_v_seq_len = key_states.shape[-2]
        # 计算原始的注意力得分
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        # 检查注意力权重的形状是否正确
        if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # 如果提供了注意力掩码，则将其添加到注意力得分中
        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, q_len, k_v_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # 将注意力得分转换为 float32 以进行 softmax 计算，然后转换回原始数据类型
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

         # 计算最终的注意力输出
        attn_output = torch.matmul(attn_weights, value_states)

        # 检查注意力输出的形状是否正确
        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # 重塑注意力输出张量，以适应后续的处理
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        # 应用输出投影层
        attn_output = self.out_proj(attn_output)

        # 根据需要返回注意力权重
        return attn_output, attn_weights


class Siglip2FlashAttention2(Siglip2Attention):
    """
    Siglip2Attention 的 Flash Attention 模块。该模块继承自 `Siglip2Attention`，因此模型的权重保持不变。
    唯一需要修改的是前向传播方法，需要正确调用 Flash Attention 的公共 API，并处理输入中可能存在的填充 token。

    Attributes:
        is_causal (bool): 
            是否为因果注意力。如果为 `True`，则只允许对当前和之前的 token 进行注意力计算。
    """

    is_causal = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Adapted from transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        前向传播方法，计算 Flash Attention。

        Args:
            hidden_states (`torch.Tensor`):
                输入张量，形状为 (batch_size, 时间步长度, 通道数)。
            attention_mask (`torch.LongTensor`, 可选):
                注意力掩码张量，用于屏蔽某些位置，形状为 (batch_size, 1, 时间步长度, 时间步长度)。
            output_attentions (`bool`):
                是否输出注意力权重。

        Returns:
            `Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]`:
                - `attn_output`: 注意力输出，形状为 (batch_size, 时间步长度, 隐藏层大小)。
                - `attn_weights`: 注意力权重，如果 `output_attentions` 为 `True`，则返回，形状为 (batch_size, num_heads, 时间步长度, 时间步长度)。
                - `attn_weights_tuple`: 其他注意力权重信息（可选）。
        """
        output_attentions = False

        # 获取批次大小和时间步长度
        batch_size, q_len, _ = hidden_states.size()

        # 计算查询 (query)、键 (key) 和值 (value) 的投影
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash Attention 要求输入的形状为 (batch_size, seq_length, head_dim, hidden_dim)
        # 因此我们保持原始形状不变
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim)

        dropout_rate = self.dropout if self.training else 0.0

        # 在 PEFT 中，通常我们将层归一化层转换为 float32 以提高训练稳定性
        # 因此，输入的隐藏状态会被静默地转换为 float32。因此，我们需要将其转换回正确的类型，以确保一切按预期工作。
        # 这种转换可能会减慢训练和推理速度，因此建议不要将 LayerNorms 转换为 fp32。

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # 处理模型量化的情形
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # 调用 Flash Attention 的前向传播方法
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        # 重塑注意力输出张量
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        # 返回注意力输出和可选的注意力权重
        return attn_output, attn_weights


class Siglip2SdpaAttention(Siglip2Attention):
    """
    使用 torch.nn.functional.scaled_dot_product_attention 的 Siglip2 注意力模块。该模块继承自 `Siglip2Attention`，因为模块的权重保持不变。
    唯一的变化是在前向传播方法中，以适应 SDPA API。

    Attributes:
        is_causal (bool): 
            是否为因果注意力。如果为 `True`，则只允许对当前和之前的 token 进行注意力计算。
    """

    is_causal = False

    # Adapted from Siglip2Attention.forward and transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播方法，计算使用 SDPA 的注意力。

        Args:
            hidden_states (`torch.Tensor`):
                输入张量，形状为 (batch_size, 时间步长度, 通道数)。
            attention_mask (`torch.Tensor`, 可选):
                注意力掩码张量，用于屏蔽某些位置，形状为 (batch_size, 1, 时间步长度, 时间步长度)。
            output_attentions (`bool`, 可选):
                是否输出注意力权重。

        Returns:
            `Tuple[torch.Tensor, Optional[torch.Tensor]]`:
                - `attn_output`: 注意力输出，形状为 (batch_size, 时间步长度, 隐藏层大小)。
                - `attn_weights`: 注意力权重，如果 `output_attentions` 为 `True`，则返回，形状为 (batch_size, num_heads, 时间步长度, 时间步长度)。
        """
        if output_attentions:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

        # 获取批次大小和时间步长度
        batch_size, q_len, _ = hidden_states.size()

        # 计算查询 (query)、键 (key) 和值 (value) 的投影
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 重塑张量以适应多头注意力的计算
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # 我们通过 `is_causal` 语句而不是 SDPA 中的内联条件分配来调度到 SDPA 的 Flash Attention 或 Efficient 内核，
        # 以支持 torch.compile 的动态形状和完整图选项。内联条件会阻止动态形状的编译。
        is_causal = True if self.is_causal and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # 重塑注意力输出张量
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, q_len, self.embed_dim)

        # 应用输出投影层
        attn_output = self.out_proj(attn_output)

        # 返回注意力输出，不返回注意力权重
        return attn_output, None


class Siglip2MLP(nn.Module):
    """
    Siglip2 的多层感知机（MLP）模块，用于在注意力机制之后进行非线性变换。

    Args:
        config: 
            模型配置对象，包含以下属性：
            - hidden_size (int): 隐藏层大小。
            - intermediate_size (int): MLP 中间层的维度。
            - hidden_act (str): 激活函数的类型，如 "relu", "gelu" 等。
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 根据配置选择激活函数
        self.activation_fn = ACT2FN[config.hidden_act]
        # 定义第一个全连接层，将隐藏层大小映射到中间层大小
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # 定义第二个全连接层，将中间层大小映射回隐藏层大小
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法，应用 MLP 变换。

        Args:
            hidden_states (`torch.Tensor`):
                输入张量，形状为 `(batch_size, seq_length, hidden_size)`。

        Returns:
            `torch.Tensor`: 
                变换后的张量，形状为 `(batch_size, seq_length, hidden_size)`。
        """
        # 应用第一个全连接层
        hidden_states = self.fc1(hidden_states)
        # 应用激活函数
        hidden_states = self.activation_fn(hidden_states)
        # 应用第二个全连接层
        hidden_states = self.fc2(hidden_states)
        # 返回变换后的张量
        return hidden_states


# 定义注意力机制的实现类映射
SIGLIP2_ATTENTION_CLASSES = {
    "eager": Siglip2Attention,
    "flash_attention_2": Siglip2FlashAttention2,
    "sdpa": Siglip2SdpaAttention,
}


class Siglip2EncoderLayer(nn.Module):
    """
    Siglip2 的编码器层，包含自注意力机制和 MLP 模块。

    Args:
        config (Siglip2Config): 
            模型配置对象，包含以下属性：
            - hidden_size (int): 隐藏层大小。
            - intermediate_size (int): MLP 中间层的维度。
            - hidden_act (str): 激活函数的类型。
            - layer_norm_eps (float): 层归一化中的 epsilon 值。
            - _attn_implementation (str): 注意力机制的实现方式，如 "eager", "flash_attention_2", "sdpa"。
    """
    def __init__(self, config: Siglip2Config):
        super().__init__()

        # 嵌入维度，通常与隐藏层大小相同
        self.embed_dim = config.hidden_size
        # 根据配置选择注意力机制的实现类，并实例化
        self.self_attn = SIGLIP2_ATTENTION_CLASSES[config._attn_implementation](config=config)
        # 定义第一个层归一化层
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # 实例化 MLP 模块
        self.mlp = Siglip2MLP(config)
        # 定义第二个层归一化层
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        前向传播方法，处理输入张量通过自注意力和 MLP。

        Args:
            hidden_states (`torch.FloatTensor`):
                输入张量，形状为 `(batch_size, seq_len, embed_dim)`。
            attention_mask (`torch.FloatTensor`):
                注意力掩码张量，形状为 `(batch_size, 1, q_len, k_v_seq_len)`，其中填充元素由非常大的负值表示。
            output_attentions (`bool`, 可选, 默认值为 `False`):
                是否返回所有注意力层的注意力权重。

        Returns:
            `Tuple[torch.FloatTensor]`: 
                - `hidden_states`: 变换后的隐藏状态，形状为 `(batch_size, seq_len, embed_dim)`。
                - `attn_weights`: 注意力权重，如果 `output_attentions` 为 `True`，则返回，形状为 `(batch_size, num_heads, q_len, k_v_seq_len)`。
        """
        # 保存残差连接的输入
        residual = hidden_states

        # 应用第一个层归一化
        hidden_states = self.layer_norm1(hidden_states)
        # 应用自注意力机制
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        # 残差连接
        hidden_states = residual + hidden_states

        # 保存残差连接的输入
        residual = hidden_states

        # 应用第二个层归一化
        hidden_states = self.layer_norm2(hidden_states)
        # 应用 MLP
        hidden_states = self.mlp(hidden_states)
        # 残差连接
        hidden_states = residual + hidden_states
        # 打包输出
        outputs = (hidden_states,)

        if output_attentions:
            # 如果需要，添加注意力权重到输出
            outputs += (attn_weights,)

        return outputs


class Siglip2Encoder(nn.Module):
    """
    Transformer 编码器，由 `config.num_hidden_layers` 个自注意力层组成。每个层都是 [`Siglip2EncoderLayer`] 的实例。

    Args:
        config (Siglip2Config): 
            模型配置对象，包含以下属性：
            - num_hidden_layers (int): 编码器的层数。
            - 其他配置参数，如 hidden_size, intermediate_size, hidden_act, layer_norm_eps, output_attentions, output_hidden_states, use_return_dict 等。
    """

    def __init__(self, config: Siglip2Config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([Siglip2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # Ignore copy
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        前向传播方法，处理输入嵌入通过多个自注意力层。

        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                可选地，直接传递嵌入表示，而不是传递 `input_ids`。这在您希望对如何将 `input_ids` 索引转换为关联向量有更多控制时非常有用。
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                掩码张量，用于避免在填充 token 索引上执行注意力计算。掩码值选择 `[0, 1]`：

                - `1` 表示 **未掩码** 的 token，
                - `0` 表示 **掩码** 的 token。

                [什么是注意力掩码？](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力权重。详见返回的张量中的 `attentions`。
            output_hidden_states (`bool`, *optional*):
                是否返回所有层的隐藏状态。详见返回的张量中的 `hidden_states`。
            return_dict (`bool`, *optional*):
                是否返回一个 [`~utils.ModelOutput`] 而不是普通的元组。

        Returns:
            `Union[Tuple, BaseModelOutput]`: 
                - 如果 `return_dict=True`，返回 `BaseModelOutput` 对象。
                - 否则，返回包含 `hidden_states`, `encoder_states`, `all_attentions` 的元组。
        """
        # 根据配置或传入参数设置输出标志
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 初始化存储隐藏状态和注意力权重的容器
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # 设置初始隐藏状态为输入嵌入
        hidden_states = inputs_embeds

        # 遍历所有编码器层
        for encoder_layer in self.layers:
            if output_hidden_states:
                # 存储当前隐藏状态
                encoder_states = encoder_states + (hidden_states,)

            # 如果启用梯度检查点，则使用检查点机制
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                # 否则，正常调用编码器层的前向传播方法
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            # 更新隐藏状态
            hidden_states = layer_outputs[0]

            if output_attentions:
                # 存储注意力权重
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            # 存储最终隐藏状态
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


# 视觉输入参数的中文文档字符串
SIGLIP2_VISION_INPUTS_DOCSTRING = r"""
Args:
    pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
        像素值。默认情况下，如果提供填充，将被忽略。可以使用 [`AutoImageProcessor`] 获取像素值。详情见 [`CLIPImageProcessor.__call__`]。
    output_attentions (`bool`, *optional*):
        是否返回所有注意力层的注意力权重。详见返回的张量中的 `attentions`。
    output_hidden_states (`bool`, *optional*):
        是否返回所有层的隐藏状态。详见返回的张量中的 `hidden_states`。
    interpolate_pos_encoding (`bool`, *optional*, defaults to `False`):
        是否插值预训练的位置编码。
    return_dict (`bool`, *optional*):
        是否返回一个 [`~utils.ModelOutput`] 而不是普通的元组。
"""


class Siglip2VisionTransformer(nn.Module):
    """
    Siglip2 的视觉 Transformer 模型，包含图像嵌入、编码器、层归一化以及可选的头部模块。

    Args:
        config (Siglip2VisionConfig): 
            视觉模型的配置对象，包含以下属性：
            - hidden_size (int): 隐藏层大小。
            - layer_norm_eps (float): 层归一化中的 epsilon 值。
            - vision_use_head (bool): 是否使用头部模块。
            - 其他配置参数，如 num_attention_heads, intermediate_size, hidden_act, attention_dropout, hidden_dropout_prob 等。
    """
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        # 嵌入维度，通常与隐藏层大小相同
        embed_dim = config.hidden_size

        # 实例化视觉嵌入模块
        self.embeddings = Siglip2VisionEmbeddings(config)
        # 实例化编码器模块
        self.encoder = Siglip2Encoder(config)
        # 实例化后置层归一化层
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        # 判断是否使用头部模块
        self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head
        if self.use_head:
            self.head = Siglip2MultiheadAttentionPoolingHead(config)

        # 判断是否使用 Flash Attention 2
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        前向传播方法，处理输入图像像素值通过视觉 Transformer 模型。

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                像素值张量。默认情况下，如果提供填充，将被忽略。可以使用 [`AutoImageProcessor`] 获取像素值。
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                注意力掩码张量，用于避免在填充 token 索引上执行注意力计算。
            spatial_shapes (`torch.LongTensor` of shape `(batch_size, 2)`):
                空间形状张量，用于调整位置嵌入的大小。
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力权重。
            output_hidden_states (`bool`, *optional*):
                是否返回所有层的隐藏状态。
            return_dict (`bool`, *optional*):
                是否返回一个 [`~utils.ModelOutput`] 而不是普通的元组。

        Returns:
            `Union[Tuple, BaseModelOutputWithPooling]`: 
                - 如果 `return_dict=True`，返回 `BaseModelOutputWithPooling` 对象。
                - 否则，返回包含 `last_hidden_state`, `pooler_output`, `hidden_states`, `attentions` 的元组。
        """
        # 根据配置或传入参数设置输出标志
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 应用视觉嵌入模块，将像素值转换为嵌入向量
        hidden_states = self.embeddings(pixel_values, spatial_shapes)

        # 处理注意力掩码
        if attention_mask is not None and not self._use_flash_attention_2:
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
        else:
            encoder_attention_mask = attention_mask

        # 应用编码器模块
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取最后一层的隐藏状态
        last_hidden_state = encoder_outputs[0]
        # 应用后置层归一化
        last_hidden_state = self.post_layernorm(last_hidden_state)

        # 应用头部模块（如果使用）
        pooler_output = self.head(last_hidden_state, attention_mask) if self.use_head else None
        
        # 根据 return_dict 参数返回不同的输出格式
        if not return_dict:
            return (last_hidden_state, pooler_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class Siglip2TextEmbeddings(nn.Module):
    """
    Siglip2 的文本嵌入模块，用于将输入的 token ID 转换为嵌入向量，并添加位置嵌入。

    Args:
        config (Siglip2TextConfig): 
            文本模型的配置对象，包含以下属性：
            - vocab_size (int): 词汇表大小。
            - hidden_size (int): 隐藏层大小。
            - max_position_embeddings (int): 最大位置嵌入长度。
    """
    def __init__(self, config: Siglip2TextConfig):
        super().__init__()
        # 嵌入维度，通常与隐藏层大小相同
        embed_dim = config.hidden_size

        # 定义 token 嵌入层，将 token ID 转换为嵌入向量
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        # 定义位置嵌入层，为每个位置生成位置嵌入
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # 注册一个缓冲区，存储位置 ID 张量，并在模型保存时不进行序列化
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        """
        前向传播方法，将输入的 token ID 或嵌入向量转换为包含位置嵌入的嵌入向量。

        Args:
            input_ids (`torch.LongTensor`, *optional*):
                输入的 token ID 张量，形状为 `(batch_size, sequence_length)`。
            position_ids (`torch.LongTensor`, *optional*):
                位置 ID 张量，形状为 `(batch_size, sequence_length)`。如果未提供，则根据 `input_ids` 自动生成。
            inputs_embeds (`torch.FloatTensor`, *optional*):
                预先计算的输入嵌入向量，形状为 `(batch_size, sequence_length, hidden_size)`。如果提供，则忽略 `input_ids`。

        Returns:
            `torch.Tensor`: 
                包含位置嵌入的嵌入向量，形状为 `(batch_size, sequence_length, hidden_size)`。
        """
        # 获取序列长度
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        # 获取最大位置嵌入长度
        max_position_embedding = self.position_embedding.weight.shape[0]

        # 检查序列长度是否超过最大位置嵌入长度
        if seq_length > max_position_embedding:
            raise ValueError(
                f"Sequence length must be less than max_position_embeddings (got `sequence length`: "
                f"{seq_length} and max_position_embeddings: {max_position_embedding}"
            )

        # 如果未提供位置 ID，则自动生成
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 如果未提供输入嵌入向量，则使用 token 嵌入层进行嵌入
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        # 获取位置嵌入
        position_embeddings = self.position_embedding(position_ids)

        # 将 token 嵌入和位置嵌入相加，得到最终的嵌入向量
        embeddings = inputs_embeds + position_embeddings

        return embeddings


def _trunc_normal_(tensor, mean, std, a, b):
    """
    对输入张量进行截断正态分布初始化。

    Args:
        tensor (torch.Tensor): 需要初始化的张量。
        mean (float): 正态分布的均值。
        std (float): 正态分布的标准差。
        a (float): 截断的下界。
        b (float): 截断的上界。
    """
    def norm_cdf(x):
        """
        计算标准正态分布的累积分布函数（CDF）。

        Args:
            x (float): 输入值。

        Returns:
            float: 标准正态分布的 CDF 值。
        """
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    
    # 检查均值是否在截断范围内超过2个标准差
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    # 通过使用截断的均匀分布，然后使用正态分布的逆 CDF 来生成值。
    # 获取上下 CDF 值
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # 将张量均匀地填充为 [2l-1, 2u-1] 范围内的值
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # 使用逆误差函数（erfinv）进行逆 CDF 变换，得到截断的标准正态分布
    tensor.erfinv_()

    # 将分布转换为指定的均值和标准差
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)

    # 截断以确保值在正确的范围内
    tensor.clamp_(min=a, max=b)


def trunc_normal_tf_(
    tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0
) -> torch.Tensor:
    """
    使用截断正态分布填充输入张量。值实际上是从正态分布 :math:`\\mathcal{N}(\\text{mean}, \\text{std}^2)` 中抽取的，
    超出 :math:`[a, b]` 的值将被重新抽取，直到它们在边界内。该方法在 :math:`a \\leq \\text{mean} \\leq b` 时效果最佳。

    注意：这个 'tf' 变体更接近于 TensorFlow / JAX 的实现，其中边界 [a, b] 在采样正态分布（均值为0，标准差为1.0）时应用，
    然后结果通过均值和标准差参数进行缩放和平移。

    Args:
        tensor (torch.Tensor): 一个 n 维的 `torch.Tensor`。
        mean (float): 正态分布的均值，默认为0.0。
        std (float): 正态分布的标准差，默认为1.0。
        a (float): 最小截断值，默认为-2.0。
        b (float): 最大截断值，默认为2.0。
    """
    with torch.no_grad():
        # 使用标准正态分布（均值为0，标准差为1.0）进行截断初始化
        _trunc_normal_(tensor, 0, 1.0, a, b)
        # 将分布缩放和平移到指定的均值和标准差
        tensor.mul_(std).add_(mean)


def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    """
    方差缩放初始化方法，用于根据指定的模式和分布初始化张量。

    Args:
        tensor (torch.Tensor): 需要初始化的张量。
        scale (float, optional): 缩放因子，默认为1.0。
        mode (str, optional): 缩放模式，可以是 'fan_in', 'fan_out' 或 'fan_avg'，默认为 'fan_in'。
        distribution (str, optional): 分布类型，可以是 'truncated_normal', 'normal' 或 'uniform'，默认为 'normal'。
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        # 标准正态分布截断到 (-2, 2) 的标准差常数
        trunc_normal_tf_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        with torch.no_grad():
            tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        with torch.no_grad():
            tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    """
    LeCun 正态分布初始化方法。

    使用方差缩放初始化方法，模式为 'fan_in'，分布为 'truncated_normal'。

    Args:
        tensor (torch.Tensor): 需要初始化的张量。
    """
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


def default_flax_embed_init(tensor):
    """
    默认的 Flax 嵌入初始化方法。

    使用方差缩放初始化方法，模式为 'fan_in'，分布为 'normal'。

    Args:
        tensor (torch.Tensor): 需要初始化的张量。
    """
    variance_scaling_(tensor, mode="fan_in", distribution="normal")


class Siglip2TextTransformer(nn.Module):
    """
    Siglip2 的文本 Transformer 模型，包含文本嵌入、编码器、最终层归一化以及线性头。

    Args:
        config (Siglip2TextConfig): 
            文本模型的配置对象，包含以下属性：
            - hidden_size (int): 隐藏层大小。
            - projection_size (int): 线性头的输出维度。
            - layer_norm_eps (float): 层归一化中的 epsilon 值。
            - _attn_implementation (str): 注意力机制的实现方式，如 "eager", "flash_attention_2", "sdpa"。
            - 其他配置参数，如 num_attention_heads, intermediate_size, hidden_act, attention_dropout, hidden_dropout_prob 等。
    """
    def __init__(self, config: Siglip2TextConfig):
        super().__init__()
        self.config = config
        # 嵌入维度，通常与隐藏层大小相同
        embed_dim = config.hidden_size
        # 实例化文本嵌入模块
        self.embeddings = Siglip2TextEmbeddings(config)
        # 实例化编码器模块
        self.encoder = Siglip2Encoder(config)
        # 实例化最终层归一化层
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        # 定义线性头，将隐藏状态映射到指定的投影大小
        self.head = nn.Linear(embed_dim, config.projection_size)
        # 判断是否使用 Flash Attention 2
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        前向传播方法，处理输入文本通过文本 Transformer 模型。

        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                输入序列 token 的索引。如果未提供，则必须提供 `inputs_embeds`。
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                注意力掩码张量，用于避免在填充 token 索引上执行注意力计算。
            position_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                每个输入序列 token 在位置嵌入中的位置索引。如果未提供，则自动生成。
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力权重。
            output_hidden_states (`bool`, *optional*):
                是否返回所有层的隐藏状态。
            return_dict (`bool`, *optional*):
                是否返回一个 [`~utils.ModelOutput`] 而不是普通的元组。

        Returns:
            `Union[Tuple, BaseModelOutputWithPooling]`: 
                - 如果 `return_dict=True`，返回 `BaseModelOutputWithPooling` 对象。
                - 否则，返回包含 `last_hidden_state`, `pooler_output`, `hidden_states`, `attentions` 的元组。
        """
        # 根据配置或传入参数设置输出标志
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        # 获取输入形状
        input_shape = input_ids.size()
        # 重塑张量
        input_ids = input_ids.view(-1, input_shape[-1])

        # 应用文本嵌入模块，将 token ID 转换为嵌入向量
        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # 注意：Siglip2 的文本模型不像原始的 CLIP 模型那样使用因果掩码。
        # 扩展注意力掩码
        if attention_mask is not None and not self._use_flash_attention_2:
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        # 应用编码器模块
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取最后一层的隐藏状态
        last_hidden_state = encoder_outputs[0]
        # 应用最终层归一化
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # 假设使用 "sticky" EOS tokenization，最后一个 token 始终是 EOS。
        # 取最后一个 token 的隐藏状态作为池化输出
        pooled_output = last_hidden_state[:, -1, :]
        # 应用线性头
        pooled_output = self.head(pooled_output)

        if not return_dict:
            # 返回元组形式的输出
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class Siglip2PreTrainedModel(PreTrainedModel):
    """
    一个抽象类，用于处理权重初始化以及提供一个简单的接口用于下载和加载预训练模型。

    Attributes:
        config_class (Siglip2Config): 配置类。
        base_model_prefix (str): 模型前缀，用于保存和加载模型时的命名。
        supports_gradient_checkpointing (bool): 是否支持梯度检查点。
    """

    config_class = Siglip2Config
    base_model_prefix = "siglip2"
    supports_gradient_checkpointing = True

    # 不需要分割的模块列表
    _no_split_modules = [
        "Siglip2TextEmbeddings",
        "Siglip2EncoderLayer",
        "Siglip2VisionEmbeddings",
        "Siglip2EncoderLayer",
        "Siglip2MultiheadAttentionPoolingHead",
    ]
    # 是否支持 Flash Attention 2
    _supports_flash_attn_2 = True
    # 是否支持 SDPA
    _supports_sdpa = True

    def _init_weights(self, module):
        """
        初始化模型权重。

        Args:
            module (nn.Module): 需要初始化的模块。
        """
        if isinstance(module, Siglip2VisionEmbeddings):
            width = (
                self.config.vision_config.hidden_size
                if isinstance(self.config, Siglip2Config)
                else self.config.hidden_size
            )
            # 初始化位置嵌入权重
            nn.init.normal_(module.position_embedding.weight, std=1 / np.sqrt(width))
        elif isinstance(module, nn.Embedding):
            default_flax_embed_init(module.weight)  # 使用默认的 Flax 嵌入初始化方法
        elif isinstance(module, Siglip2Attention):
            nn.init.xavier_uniform_(module.q_proj.weight)  # 初始化查询投影权重
            nn.init.xavier_uniform_(module.k_proj.weight)  # 初始化键投影权重
            nn.init.xavier_uniform_(module.v_proj.weight)  # 初始化值投影权重
            nn.init.xavier_uniform_(module.out_proj.weight)  # 初始化输出投影权重
            nn.init.zeros_(module.q_proj.bias)  # 初始化查询投影偏置
            nn.init.zeros_(module.k_proj.bias)  # 初始化键投影偏置
            nn.init.zeros_(module.v_proj.bias)  # 初始化值投影偏置
            nn.init.zeros_(module.out_proj.bias)  # 初始化输出投影偏置
        elif isinstance(module, Siglip2MLP):
            nn.init.xavier_uniform_(module.fc1.weight)  # 初始化第一个全连接层权重
            nn.init.xavier_uniform_(module.fc2.weight)  # 初始化第二个全连接层权重
            nn.init.normal_(module.fc1.bias, std=1e-6)  # 初始化第一个全连接层偏置
            nn.init.normal_(module.fc2.bias, std=1e-6)  # 初始化第二个全连接层偏置
        elif isinstance(module, Siglip2MultiheadAttentionPoolingHead):
            nn.init.xavier_uniform_(module.probe.data)  # 初始化探测数据
            nn.init.xavier_uniform_(module.attention.in_proj_weight.data)  # 初始化注意力输入投影权重
            nn.init.zeros_(module.attention.in_proj_bias.data)  # 初始化注意力输入投影偏置
        elif isinstance(module, Siglip2Model):
            logit_scale_init = torch.log(torch.tensor(1.0))  # 初始化 logit scale
            module.logit_scale.data.fill_(logit_scale_init)
            module.logit_bias.data.zero_()  # 初始化 logit 偏置
        elif isinstance(module, Siglip2ForImageClassification):
            # 初始化分类器权重
            nn.init.normal_(
                module.classifier.weight,
                std=self.config.vision_config.hidden_size**-0.5 * self.config.initializer_factor,
            )
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            # 使用 LeCun 正态分布初始化线性层或卷积层权重
            lecun_normal_(module.weight)
            if module.bias is not None:
                # 初始化偏置为零
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            # 初始化层归一化偏置为零
            module.bias.data.zero_()
            # 初始化层归一化权重为1.0
            module.weight.data.fill_(1.0)


class Siglip2TextModel(Siglip2PreTrainedModel):
    """
    Siglip2 的文本模型，基于 `Siglip2PreTrainedModel` 抽象类实现。

    Args:
        config (Siglip2TextConfig): 
            文本模型的配置对象，包含以下属性：
            - hidden_size (int): 隐藏层大小。
            - projection_size (int): 线性头的输出维度。
            - layer_norm_eps (float): 层归一化中的 epsilon 值。
            - use_return_dict (bool): 是否使用 `ModelOutput` 对象返回结果。
            - 其他配置参数，如 num_attention_heads, intermediate_size, hidden_act, attention_dropout, hidden_dropout_prob 等。
    """
    config_class = Siglip2TextConfig

    def __init__(self, config: Siglip2TextConfig):
        super().__init__(config)
        self.text_model = Siglip2TextTransformer(config)
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        """
        获取输入嵌入层。

        Returns:
            nn.Module: 输入嵌入层，通常是 `token_embedding`。
        """
        # 返回文本嵌入模块中的 token 嵌入层
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        """
        设置输入嵌入层。

        Args:
            value (nn.Module): 要设置的输入嵌入层。
        """
        # 设置文本嵌入模块中的 token 嵌入层
        self.text_model.embeddings.token_embedding = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        前向传播方法，处理输入文本通过文本模型。

        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                输入序列 token 的索引。如果未提供，则必须提供 `inputs_embeds`。
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                注意力掩码张量，用于避免在填充 token 索引上执行注意力计算。
            position_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                每个输入序列 token 在位置嵌入中的位置索引。如果未提供，则自动生成。
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力权重。
            output_hidden_states (`bool`, *optional*):
                是否返回所有层的隐藏状态。
            return_dict (`bool`, *optional*):
                是否返回一个 [`~utils.ModelOutput`] 而不是普通的元组。

        Returns:
            `Union[Tuple, BaseModelOutputWithPooling]`: 
                - 如果 `return_dict=True`，返回 `BaseModelOutputWithPooling` 对象。
                - 否则，返回包含 `last_hidden_state`, `pooler_output`, `hidden_states`, `attentions` 的元组。
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用文本 Transformer 模型的前向传播方法
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class Siglip2MultiheadAttentionPoolingHead(nn.Module):
    """
    多头注意力池化头。

    Args:
        config (Siglip2VisionConfig): 
            视觉模型的配置对象，包含以下属性：
            - hidden_size (int): 隐藏层大小。
            - num_attention_heads (int): 注意力头的数量。
            - layer_norm_eps (float): 层归一化中的 epsilon 值。
            - 其他配置参数，如 intermediate_size, hidden_act, attention_dropout, hidden_dropout_prob 等。
    """

    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()

        # 初始化探测参数，形状为 (1, 1, hidden_size)
        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        # 实例化多头注意力层
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        # 实例化层归一化层
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 实例化 MLP 模块
        self.mlp = Siglip2MLP(config)
        # 注意力头的数量
        self.num_heads = config.num_attention_heads

    def forward(self, hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        前向传播方法，应用多头注意力池化。

        Args:
            hidden_state (`torch.Tensor`):
                输入隐藏状态，形状为 `(batch_size, sequence_length, hidden_size)`。
            attention_mask (`torch.Tensor`, *optional*):
                注意力掩码张量，形状为 `(batch_size, sequence_length)`，用于屏蔽某些位置。

        Returns:
            `torch.Tensor`: 
                池化后的隐藏状态，形状为 `(batch_size, hidden_size)`。
        """
        # 获取批次大小
        batch_size = hidden_state.shape[0]
        # 重复探测参数以匹配批次大小
        probe = self.probe.repeat(batch_size, 1, 1)

        if attention_mask is not None:
            # 获取目标长度和源长度
            target_len, source_len = probe.shape[1], hidden_state.shape[1]
            # 准备 4D 注意力掩码
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_state.dtype, target_len)
            # 重复掩码以匹配多头
            attention_mask = attention_mask.repeat(1, self.num_heads, target_len, 1)
            # 重塑掩码形状
            attention_mask = attention_mask.reshape(-1, target_len, source_len)
        
        # 应用多头注意力层
        hidden_state = self.attention(probe, hidden_state, hidden_state, attn_mask=attention_mask)[0]

        # 保存残差连接的输入
        residual = hidden_state
        # 应用层归一化
        hidden_state = self.layernorm(hidden_state)
        # 应用 MLP 并进行残差连接
        hidden_state = residual + self.mlp(hidden_state)

        # 返回池化后的隐藏状态（第一个 token）
        return hidden_state[:, 0]


class Siglip2VisionModel(Siglip2PreTrainedModel):
    """
    Siglip2 的视觉模型，基于 `Siglip2PreTrainedModel` 抽象类实现。

    Args:
        config (Siglip2VisionConfig): 
            视觉模型的配置对象，包含以下属性：
            - hidden_size (int): 隐藏层大小。
            - num_attention_heads (int): 注意力头的数量。
            - intermediate_size (int): MLP 中间层的维度。
            - hidden_act (str): 激活函数的类型，如 "relu", "gelu" 等。
            - layer_norm_eps (float): 层归一化中的 epsilon 值。
            - 其他配置参数，如 num_hidden_layers, attention_dropout, hidden_dropout_prob 等。
    """
    config_class = Siglip2VisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: Siglip2VisionConfig):
        super().__init__(config)

        # 实例化视觉 Transformer 模型
        self.vision_model = Siglip2VisionTransformer(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        """
        获取输入嵌入层。

        Returns:
            nn.Module: 输入嵌入层，通常是 `patch_embedding`。
        """
        # 返回视觉嵌入模块中的 patch 嵌入层
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_attention_mask: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        前向传播方法，处理输入像素值通过视觉模型。

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                输入像素值张量。
            pixel_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                注意力掩码张量，用于避免在填充 token 索引上执行注意力计算。
            spatial_shapes (`torch.LongTensor` of shape `(batch_size, 2)`):
                空间形状张量，用于调整位置嵌入的大小。
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力权重。
            output_hidden_states (`bool`, *optional*):
                是否返回所有层的隐藏状态。
            return_dict (`bool`, *optional*):
                是否返回一个 [`~utils.ModelOutput`] 而不是普通的元组。

        Returns:
            `Union[Tuple, BaseModelOutputWithPooling]`: 
                - 如果 `return_dict=True`，返回 `BaseModelOutputWithPooling` 对象。
                - 否则，返回包含 `last_hidden_state`, `pooler_output`, `hidden_states`, `attentions` 的元组。
        """
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用视觉 Transformer 模型的前向传播方法
        return self.vision_model(
            pixel_values=pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class Siglip2Model(Siglip2PreTrainedModel):
    """
    Siglip2 模型，结合了文本和视觉模型，基于 `Siglip2PreTrainedModel` 抽象类实现。

    Args:
        config (Siglip2Config): 
            模型的配置对象，包含以下属性：
            - text_config (Siglip2TextConfig): 文本模型的配置。
            - vision_config (Siglip2VisionConfig): 视觉模型的配置。
            - 其他配置参数，如 logit_scale, logit_bias 等。
    """
    config_class = Siglip2Config

    def __init__(self, config: Siglip2Config):
        super().__init__(config)

        # 检查 text_config 是否为 Siglip2TextConfig 类型
        if not isinstance(config.text_config, Siglip2TextConfig):
            raise TypeError(
                "config.text_config is expected to be of type Siglip2TextConfig but is of type"
                f" {type(config.text_config)}."
            )

        # 检查 vision_config 是否为 Siglip2VisionConfig 类型
        if not isinstance(config.vision_config, Siglip2VisionConfig):
            raise TypeError(
                "config.vision_config is expected to be of type Siglip2VisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        # 获取文本配置
        text_config = config.text_config
        # 获取视觉配置
        vision_config = config.vision_config

        # 首先，使用正确的注意力实现方式初始化文本和视觉模型
        text_model = Siglip2TextModel._from_config(text_config)
        vision_model = Siglip2VisionModel._from_config(vision_config)

        # 其次，获取文本和视觉子模块（为了向后兼容）
        # 获取文本模型的子模块
        self.text_model = text_model.text_model
        # 获取视觉模型的子模块
        self.vision_model = vision_model.vision_model

        self.logit_scale = nn.Parameter(torch.randn(1))
        self.logit_bias = nn.Parameter(torch.randn(1))

        # 初始化权重并应用最终处理
        self.post_init()

    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        """
        获取文本特征。

        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim)`):
                通过投影层应用于 [`Siglip2TextModel`] 的池化输出得到的文本嵌入。

        示例:

        ```python
        >>> from transformers import AutoTokenizer, AutoModel
        >>> import torch

        >>> model = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/siglip2-base-patch16-224")

        >>> # 重要：确保设置 padding="max_length"，因为这是模型训练的方式
        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding="max_length", return_tensors="pt")
        >>> with torch.no_grad():
        ...     text_features = model.get_text_features(**inputs)
        ```
        """
        # 使用 Siglip2 模型的配置（如果指定）而不是视觉和文本组件的配置。
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用文本模型的前向传播方法
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取池化输出
        pooled_output = text_outputs[1]

        # 返回文本特征
        return pooled_output

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        """
        获取图像特征。

        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim)`):
                通过投影层应用于 [`Siglip2VisionModel`] 的池化输出得到的图像嵌入。

        示例:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, AutoModel
        >>> import torch

        >>> model = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")

        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     image_features = model.get_image_features(**inputs)
        ```
        """
        # 使用 Siglip2Model 的配置（如果指定）而不是视觉和文本组件的配置。
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用视觉模型的前向传播方法
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取池化输出
        pooled_output = vision_outputs[1]

        # 返回图像特征
        return pooled_output

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Siglip2Output]:
        """
        前向传播方法，处理输入文本和图像通过 Siglip2 模型。

        Returns:

        示例:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, AutoModel
        >>> import torch

        >>> model = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")

        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> texts = ["a photo of 2 cats", "a photo of 2 dogs"]
        >>> # 重要：我们传递 `padding=max_length`，因为模型是使用这种方式训练的
        >>> inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> logits_per_image = outputs.logits_per_image
        >>> probs = torch.sigmoid(logits_per_image) # 这些是概率
        >>> print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")
        31.9% that image 0 is 'a photo of 2 cats'
        ```
        """
        # 使用 Siglip2 模型的配置（如果指定）而不是视觉和文本组件的配置。
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用视觉模型的前向传播方法
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 调用文本模型的前向传播方法
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取图像嵌入
        image_embeds = vision_outputs[1]
        # 获取文本嵌入
        text_embeds = text_outputs[1]

        # 归一化特征
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # 计算余弦相似度作为 logits
        logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device))

        logit_scale, logit_bias = self.logit_scale.to(text_embeds.device), self.logit_bias.to(text_embeds.device)
        logits_per_text = logits_per_text * logit_scale.exp() + logit_bias

        # 转置得到图像-文本的 logits
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            eye = torch.eye(logits_per_text.size(0), device=logits_per_text.device)
            m1_diag1 = -torch.ones_like(logits_per_text) + 2 * eye
            loglik = torch.nn.functional.logsigmoid(m1_diag1 * logits_per_text)
            nll = -torch.sum(loglik, dim=-1)
            loss = nll.mean()

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        # 返回 Siglip2Output 对象
        return Siglip2Output(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


class Siglip2ForImageClassification(Siglip2PreTrainedModel):
    """
    Siglip2 的图像分类模型，基于 `Siglip2PreTrainedModel` 抽象类实现。

    Args:
        config (Siglip2Config): 
            模型的配置对象，包含以下属性：
            - vision_config (Siglip2VisionConfig): 视觉模型的配置。
            - num_labels (int): 分类标签的数量。
            - 其他配置参数，如 problem_type 等。
    """
    main_input_name = "pixel_values"

    def __init__(self, config: Siglip2Config) -> None:
        super().__init__(config)

        # 分类标签的数量
        self.num_labels = config.num_labels

        # 使用正确的注意力实现方式创建视觉模型，并仅获取视觉模型的子模块（为了向后兼容）
        vision_model = Siglip2VisionModel._from_config(config.vision_config)
        self.vision_model = vision_model.vision_model

        # 分类器头
        self.classifier = (
            # 如果 num_labels > 0，则使用线性层作为分类器，否则使用恒等映射
            nn.Linear(config.vision_config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # 初始化权重并应用最终处理
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.LongTensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        前向传播方法，处理输入图像通过图像分类模型。

        Args:
            pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`, *optional*):
                输入像素值张量。
            pixel_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                注意力掩码张量，用于避免在填充 token 索引上执行注意力计算。
            spatial_shapes (`torch.LongTensor` of shape `(batch_size, 2)`, *optional*):
                空间形状张量，用于调整位置嵌入的大小。
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                标签，用于计算图像分类/回归损失。索引应在 `[0, ..., config.num_labels - 1]`。
                如果 `config.num_labels == 1`，则计算回归损失（均方损失），如果 `config.num_labels > 1`，则计算分类损失（交叉熵）。
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力权重。
            output_hidden_states (`bool`, *optional*):
                是否返回所有层的隐藏状态。
            return_dict (`bool`, *optional*):
                是否返回一个 [`~utils.ModelOutput`] 而不是普通的元组。

        Returns:
            `Union[Tuple, ImageClassifierOutput]`: 
                - 如果 `return_dict=True`，返回 `ImageClassifierOutput` 对象。
                - 否则，返回包含 `logits`, `hidden_states`, `attentions` 的元组。
        """
        # 设置是否输出注意力
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置是否输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用视觉模型的前向传播方法
        outputs = self.vision_model(
            pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出
        sequence_output = outputs[0]

        # 对 patch tokens 进行平均池化
        if pixel_attention_mask is not None:
            # 准备池化掩码
            pool_mask = pixel_attention_mask[..., None].to(sequence_output.device)
            # 应用掩码进行池化
            sequence_output = torch.sum(sequence_output * pool_mask, dim=1) / torch.sum(pool_mask, dim=1)
        else:
            # 否则，直接进行平均池化
            sequence_output = torch.mean(sequence_output, dim=1)

        # 应用分类器
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # 将标签移动到正确的设备以启用模型并行
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    # 设置问题类型为回归
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    # 设置问题类型为单标签分类
                    self.config.problem_type = "single_label_classification"
                else:
                    # 设置问题类型为多标签分类
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                # 使用均方损失
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    # 计算回归损失
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 使用交叉熵损失
                loss_fct = CrossEntropyLoss()
                # 计算分类损失
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 使用二元交叉熵损失
                loss_fct = BCEWithLogitsLoss()
                # 计算多标签分类损失
                loss = loss_fct(logits, labels)

        if not return_dict:
            # 返回元组形式的输出
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回封装后的模型输出
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
