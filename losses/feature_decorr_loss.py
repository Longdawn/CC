"""
改进的正交约束损失函数
特征级的去相关性约束
"""
import torch
import torch.nn.functional as F


def feature_decorr_loss(feat_text, feat_bg):
    """
    特征级正交约束：
    让文本分支的特征和背景分支的特征尽量"正交"（无相关性）
    
    Args:
        feat_text: [B, C, H, W] 文本分支特征
        feat_bg: [B, C, H, W] 背景分支特征
    
    Returns:
        loss: 标量，越小越好
    """
    B, C, H, W = feat_text.shape
    
    # 展平为 [B, C, HW]
    feat_t = feat_text.reshape(B, C, -1)
    feat_b = feat_bg.reshape(B, C, -1)
    
    # 去均值和归一化
    feat_t = feat_t - feat_t.mean(dim=2, keepdim=True)
    feat_b = feat_b - feat_b.mean(dim=2, keepdim=True)
    
    feat_t = F.normalize(feat_t, dim=2, p=2)
    feat_b = F.normalize(feat_b, dim=2, p=2)
    
    # 计算交叉相关矩阵 [B, C, C]
    # corr[i, j, k] = 第i个样本，通道j和k的相关性
    cross_corr = torch.bmm(feat_t, feat_b.transpose(1, 2)) / (H * W)
    
    # 目标：让所有相关性都接近0
    loss = cross_corr.abs().mean()
    
    return loss
