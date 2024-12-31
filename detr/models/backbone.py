# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
实现了一个可扩展的神经网络Backbone（骨干网络）模块，
并使用了特定的冻结批归一化（Frozen BatchNorm）。
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import IPython
e = IPython.embed

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    """
    这是 PyTorch 中的一种自定义模块，用于实现冻结的批归一化层，
    通过固定均值、方差、缩放因子和偏移量，实现更稳定的训练或迁移学习。
    你可以将它用于小批量训练或模型冻结场景，以提高模型的稳定性。
    """
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    """
    重写_load_from_state_dict方法, 删去num_batches_tracked量，避免引入多余参数
    num_batches_tracked 是动态批归一化中的统计变量，
    用于记录已处理的批次数，与冻结批归一化无关。
    删除它可以避免：
    在加载预训练模型时引发参数名不匹配的错误。
    误加载多余的动态批归一化参数。
    """
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    """
    定义了冻结批归一化层在前向传播中的具体计算逻辑。
    return x * scale + bias 执行批归一化
    """
    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


"""
定义了一个名为 BackboneBase 的类，它继承了 torch.nn.Module，
并用于将骨干网络（backbone）的特定层输出特征提取并返回

骨干网络（Backbone） 是深度学习中一种用于特征提取的核心网络架构，
通常作为整个模型的基础部分。它的作用是从输入数据（例如图像或序列）中提取有用的特征，
以供后续的任务模块使用，例如分类、检测、分割或生成任务。
"""
class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        # 截取网络中的特定层并返回其输出
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):# tensor 是一个 4D 张量，通常形状为 (𝑁,𝐶,𝐻,𝑊)(N,C,H,W)，表示一批图像
        """
        调用 self.body，将输入张量通过骨干网络的指定层。
        提取返回层的特征，存储在字典 xs 中。
        """
        xs = self.body(tensor)
        return xs
        """
        输入张量的掩码 是一种用于标记输入数据中有效部分和无效部分的机制，
        通常用于处理不规则形状的输入或存在缺失数据的场景。
        """
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str, # ResNet 模型的名称，如 resnet50
                 train_backbone: bool, # 控制骨干网络的参数是否需要训练
                 return_interm_layers: bool, # 控制是否返回中间层的特征
                 dilation: bool):# 扩张卷积能够增加特征图的感受野，适用于密集预测任务（如分割）。
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],# 控制 ResNet 中的步幅（stride）是否被扩张卷积替代。
            # is_main_process()一个辅助函数，通常在分布式训练中，只有主进程加载权重，其他进程共享权重。
            # norm_layer=FrozenBatchNorm2d：使用自定义的冻结批归一化代替标准批归一化。
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        """
        ResNet 通道数规则：
            resnet18 和 resnet34 是浅层网络，最后的特征图通道数为 512。
            resnet50 和更深的网络（如 resnet101）最后的特征图通道数为 2048。
        """
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048 # 设置通道数
        # num_channels 这个值会传递给父类 BackboneBase，以供后续模块匹配通道数。
        # 调用父类构造函数
        # 调用 BackboneBase 的构造函数，将处理后的 ResNet 模型及相关参数传递进去。
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


"""
定义了一个名为 Joiner 的类，它继承了 PyTorch 的 nn.Sequential，
用来组合骨干网络 (backbone) 和位置编码 (position_embedding)。
Joiner 的作用是：
    使用骨干网络提取输入特征。
    为提取到的特征图生成对应的位置信息编码。
    最终返回特征图和位置编码两部分信息。
"""
class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos

"""
用于构建一个完整的骨干网络模型（backbone），包括位置编码和特征提取器。
这个函数的设计意图是根据传入的参数配置，动态地构建和返回一个适合具体任务的骨干网络模块。
输入参数 args：包含配置的命名空间或字典，
通常包括以下内容：
    args.backbone：指定骨干网络的名称（如 resnet50）。
    args.lr_backbone：决定骨干网络的学习率，间接控制是否训练骨干网络。
    args.masks：决定是否返回中间层的特征。
    args.dilation：控制骨干网络中是否使用扩张卷积。
"""
def build_backbone(args):
    position_embedding = build_position_encoding(args)# 一个辅助函数，用于根据配置 args 创建一个位置编码模块。
    train_backbone = args.lr_backbone > 0 # train_backbone：布尔值，用于控制骨干网络参数是否冻结。
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    """
    Joiner 的作用：
        使用骨干网络提取输入特征。
        为特征图生成对应的位置信息编码。
    """
    model = Joiner(backbone, position_embedding)
    """
    从骨干网络中获取最后一层的输出通道数（num_channels）。
    将该值赋给 model，供后续模块（如分类头、检测头）使用。
    """
    model.num_channels = backbone.num_channels
    return model
