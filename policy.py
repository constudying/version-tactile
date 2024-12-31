"""
这个文件实现了两个策略模型：ACTPolicy 和 CNNMLPPolicy，以及它们的训练与推理逻辑，
同时包含了计算 KL 散度（kl_divergence）的辅助函数。

策略模型定义：
    ACTPolicy：基于 Transformer 的策略模型，支持条件变分自编码器（CVAE）结构。
    CNNMLPPolicy：基于 CNN 和 MLP 的策略模型。
训练与推理逻辑：
    模型在训练时输出损失字典（如 L1 损失、KL 散度）。
    推理时直接输出预测的动作（a_hat）。
辅助函数：
    kl_divergence：计算变分自编码器中的 KL 散度，用于约束潜变量的分布。
"""

import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed


class ACTPolicy(nn.Module):
    """
    通过 build_ACT_model_and_optimizer 初始化模型和优化器。
    kl_weight 用于控制 KL 散度的权重。
    """
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    """
    输入：
        qpos：关节状态。
        image：图像输入，归一化为标准分布。
        actions 和 is_pad：
            训练时提供真实动作和填充掩码。
            推理时为 None，从潜变量中采样动作。

    """
    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            """
            模型输出：
                a_hat：预测的动作。
                (mu, logvar)：潜变量的均值和方差。
            """
            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            """
            计算损失:
                L1 损失：预测动作与真实动作的差异。
                KL 损失：潜变量分布与标准正态分布的差异。
            """
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            # 从潜变量中采样动作
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat
    # 返回模型优化器
    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    # 功能类似于 ACTPolicy，使用 CNNMLP 模型构建函数。
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer
    """
    训练逻辑：
        使用均方误差（MSE）作为损失。
        不涉及潜变量或 KL 散度。
    推理逻辑：
        直接通过模型输出预测动作。
    """
    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

"""
输出：
    total_kld：KL 散度的总和。
    dimension_wise_kld：每个潜变量维度的 KL 散度。
    mean_kld：批量的平均 KL 散度。
"""
def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
