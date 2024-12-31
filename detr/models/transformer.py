# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers

定义了一个名为 DETR Transformer 的模块，主要用于实现基于 Transformer 的深度学习模型
（如 DETR: End-to-End Object Detection with Transformers）。
它是对 PyTorch 自带的 torch.nn.Transformer 类的修改版本，针对目标检测任务进行了优化。
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

import IPython
e = IPython.embed

"""
这个 Transformer 类是一个定制的 Transformer 模块，
包含 Encoder 和 Decoder 两部分。该模块用于处理序列化特征数据，
并为任务（如目标检测、特征解码等）提供高级特征输出。
"""
class Transformer(nn.Module):

    """
    d_model: 特征嵌入的维度（默认 512）。所有输入和输出的特征维度必须与之匹配
    nhead: 多头注意力机制中的头数
    num_encoder_layers / num_decoder_layers: Encoder 和 Decoder 堆叠的层数
    dim_feedforward：前馈网络（FFN）中的隐藏层维度。
    dropout：Dropout 概率，防止过拟合。
    activation：激活函数类型（默认 relu）。
    normalize_before：是否在层前执行 LayerNorm。
    return_intermediate_dec：是否返回 Decoder 每一层的中间结果。
    """
    """
    多头注意力机制（Multi-Head Attention, MHA） 是 Transformer 模型中的核心组件，
    用于在序列数据（如文本、图像特征等）中建模不同位置之间的关系。
    相比单头注意力机制，多头注意力能够更好地捕获输入数据的多种特征模式。
    注意力机制的核心思想是：对于每个输入位置，计算它与其他位置的相关性分数，
    并根据这些分数对其他位置的信息进行加权汇总。
    """
    """
    Dropout 是一种常用的正则化技术，主要用于防止模型在训练过程中发生过拟合（overfitting）。
    它通过随机地将神经网络中的一部分神经元“关闭”（即将它们的输出置为 0），
    来减少神经元之间的相互依赖，增强模型的泛化能力。
    """
    # 初始化一个完整的 Transformer 模块，包括 Encoder 和 Decoder 两部分，以及必要的配置和参数初始化。
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__() # 调用父类 nn.Module 的初始化方法，确保子类能够正确继承父类的功能。
        """
        TransformerEncoderLayer：定义了单个 Encoder 层。
        包括：
            多头自注意力机制。
            前馈网络。
            归一化和 Dropout。
        """
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None  # 层归一化通过对每一层的输出进行归一化，使得激活值具有均匀的分布，从而稳定梯度更新。
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        

        """
        TransformerDecoderLayer：定义了单个 Decoder 层。
                包括：
                    目标查询的多头自注意力机制。
                    跨注意力机制（结合 Encoder 的输出）。
                    前馈网络。
                    归一化和 Dropout。
        """
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters() # 重置模型中的参数，确保训练开始时权重初始化得当。
        # 存储模型的核心超参数,存储 d_model 和 nhead，在模型前向传播和其他计算中会用到。
        self.d_model = d_model
        self.nhead = nhead
    # 使用 Xavier 均匀分布初始化权重，确保模型数值稳定性。
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    """
    总结
        功能
            输入特征预处理：
            根据输入形状，展平和调整特征。
            添加位置编码和额外特征。
        Encoder：
            提取输入特征的上下文信息。
        Decoder：
            使用目标查询嵌入生成解码特征。
        适用场景
            目标检测：
                query_embed 表示目标查询，用于生成检测框和类别。
            序列到序列任务：
                如翻译、特征解码等。
    """
    # 接收输入特征、掩码和嵌入信息，经过 Encoder 和 Decoder 处理后，生成解码的特征输出。
    def forward(self, src, mask, query_embed, pos_embed, latent_input=None, proprio_input=None, additional_pos_embed=None):
        # TODO flatten only when input has H and W
        # 输入形状;展平与调整;位置编码调整;目标查询嵌入调整
        if len(src.shape) == 4: # has H and W
            # flatten NxCxHxW to HWxNxC
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
            # mask = mask.flatten(1)

            # 添加额外特征和位置编码
            # 额外位置编码：将 additional_pos_embed 添加到位置编码中，扩展位置信息的范围。
            # 额外特征输入：堆叠 latent_input 和 proprio_input，并与 src 组合
            additional_pos_embed = additional_pos_embed.unsqueeze(1).repeat(1, bs, 1) # seq, bs, dim
            pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=0)

            addition_input = torch.stack([latent_input, proprio_input], axis=0)
            src = torch.cat([addition_input, src], axis=0)
        else:
            assert len(src.shape) == 3
            # flatten NxHWxC to HWxNxC
            bs, hw, c = src.shape
            src = src.permute(1, 0, 2)
            pos_embed = pos_embed.unsqueeze(1).repeat(1, bs, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        tgt = torch.zeros_like(query_embed) # Decoder 的目标输入张量（tgt）初始化为全零，形状与 query_embed 一致。
        # 接收输入特征 src 和位置编码 pos_embed。输出记忆张量 memory，包含输入特征的上下文信息。
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        # self.decoder：接收目标张量 tgt 和 memory（Encoder 的输出）。使用目标查询嵌入 query_embed 引导解码。返回目标特征ℎ𝑠
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        hs = hs.transpose(1, 2) # 调整目标特征的维度以匹配预期格式。
        return hs


"""
这个 TransformerEncoder 类是 Transformer 模型的编码器模块。
它将输入特征通过多层 TransformerEncoderLayer 进行处理，生成上下文增强的特征表示。
工作流程总结
    输入特征：
        输入特征序列 src 依次通过多个 TransformerEncoderLayer。
        每一层都会增强特征的上下文表示。
    逐层堆叠：
        每层的输出作为下一层的输入，捕获序列中更高层次的依赖关系。
    最终归一化（可选）：
        如果提供了 norm，对最终输出特征进行归一化处理。
    返回结果：
        输出上下文增强的特征，供后续模块使用（如 Transformer 的 Decoder 或分类头）。
"""
class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        """
        遍历所有编码器层 self.layers。每一层调用 TransformerEncoderLayer 的前向传播方法
        每层处理后返回上下文增强的特征，赋值给 output。
        """
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

"""
TransformerDecoder 是 Transformer 模型的解码器部分，
负责将编码器生成的上下文信息（memory）和目标查询向量（tgt）结合起来，生成解码结果（output）。
解码器由多层 TransformerDecoderLayer 堆叠而成。
"""
class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []


        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        # 可选的层归一化
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)

"""
这个 TransformerEncoderLayer 类是 Transformer 模型中的单层编码器模块，包含以下核心组件：
多头自注意力机制（Self-Attention）。
前馈网络（Feedforward Network, FFN）。
归一化（Layer Normalization）。
可选的 Dropout。

工作流程总结
多头自注意力：
    捕获序列中各位置之间的关系。
    结果通过残差连接返回输入。
前馈网络：
    进一步处理注意力输出，提升特征表达能力。
归一化：
    通过 LayerNorm 稳定训练过程。
残差连接：
    保持梯度流动，避免深层网络的梯度消失。
"""
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    """
    如果位置编码 pos 存在，将其与输入张量相加。
    位置编码引入序列中每个位置的显式位置信息。
    """
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    # 前向传播
    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    
    """
    与 forward_post 的唯一区别是归一化的顺序。
    在 forward_pre 中，归一化发生在多头自注意力和前馈网络之前。
    """
    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    """
    根据 self.normalize_before 参数选择归一化方式：
    normalize_before=True：使用 forward_pre，归一化在操作之前执行。
    normalize_before=False：使用 forward_post，归一化在操作之后执行。
    """
    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

"""
TransformerDecoderLayer 是 Transformer 解码器的核心模块,单层decoderlayer，
负责对目标序列（tgt）进行解码，结合编码器的输出（memory）生成最终的解码特征。
核心作用：
    通过多头自注意力建模目标序列的依赖关系。
    通过交叉注意力从编码器输出中提取相关信息。
    利用前馈网络进一步处理特征。
    使用残差连接和归一化稳定训练。

pos 和 query_pos：位置编码，分别为编码器输出和目标查询的位置。

工作流程总结
    自注意力：
        建模目标序列中各位置的关系。
    交叉注意力：
        从编码器输出中提取目标相关的上下文信息。
    前馈网络：
        提升目标特征的表达能力。
    归一化和残差连接：
        提高训练稳定性，防止梯度消失或爆炸。
"""
class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        # 目标序列的自注意力
        """
        使用目标查询 query_pos 和目标特征 tgt，计算目标序列中位置间的关系。
        通过残差连接和归一化稳定训练。
        """
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # 交叉注意力
        """
        使用编码器的输出 memory 和目标查询，提取目标相关的上下文信息。
        通过残差连接和归一化，增强特征的稳定性。
        """
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        """
        对目标特征通过前馈网络进行进一步处理，提升特征表达能力。
        最后通过残差连接和归一化返回结果。
        """
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
