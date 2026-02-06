"""
重设计的双流文本检测器
- 共享UNet骨干网络
- 两个独立的检测头（文本 + 背景）
- 只用文本头的输出做最终检测
- 背景头用于特征正交约束
"""
import torch
import torch.nn as nn
from models.unet import UNet
from models.head import SegHead


class DualHeadDetector(nn.Module):
    """
    双流检测器：
    - 文本分支：学习纯文本特征 → 文本logit
    - 背景分支：学习纯背景特征 → 背景logit（仅用于正交）
    - 通过正交约束解耦两个分支
    """
    
    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()
        
        # 共享骨干网络
        self.backbone = UNet(in_ch=in_ch, base_ch=base_ch)
        
        # 两个独立检测头
        self.text_head = SegHead(in_ch=self.backbone.out_ch)  # 文本检测
        self.bg_head = SegHead(in_ch=self.backbone.out_ch)    # 背景检测（辅助）
    
    def forward(self, img, img_bg=None, return_features=False):
        """
        Args:
            img: 原图 [B, 3, H, W]
            img_bg: BG-view [B, 3, H, W]（可选，训练时需要）
            return_features: 是否返回特征向量（用于正交约束）
        
        Returns:
            如果 img_bg is None（推理模式）:
                logit_text: [B, 1, H, W] 文本概率
            
            如果 img_bg is not None（训练模式）:
                logit_text: [B, 1, H, W] 文本概率
                logit_bg: [B, 1, H, W] 背景概率
                (optionally) feat_text, feat_bg
        """
        
        # 文本分支：处理原图
        feat_text = self.backbone(img)
        logit_text = self.text_head(feat_text)
        
        # 推理模式：只需文本分支
        if img_bg is None:
            return logit_text
        
        # 训练模式：背景分支
        feat_bg = self.backbone(img_bg)
        logit_bg = self.bg_head(feat_bg)
        
        if return_features:
            return logit_text, logit_bg, feat_text, feat_bg
        else:
            return logit_text, logit_bg
