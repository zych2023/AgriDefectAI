"""
ResNet50 模型定义 — 61类农作物病害分类
- ImageNet 预训练权重
- 自定义分类头（Dropout + 61类FC）
- 支持混合精度训练
"""
import torch
import torch.nn as nn
from torchvision import models


class CropDiseaseClassifier(nn.Module):
    """
    基于 ResNet50 的农作物病害分类器
    - 输入: [B, 3, 224, 224]
    - 输出: [B, 61] logits
    """

    def __init__(self, num_classes=61, pretrained=True, dropout=0.3):
        super().__init__()

        # 骨干网络
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # 替换全连接分类头
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

        # 方案备选：冻结浅层
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        # 只训练全连接层用上面；全量微调用下面

    def forward(self, x):
        return self.backbone(x)

    def extract_features(self, x):
        """提取分类头之前的 2048 维特征向量，用于相似图片检索"""
        original_fc = self.backbone.fc
        self.backbone.fc = nn.Identity()
        features = self.backbone(x)          # [B, 2048]
        self.backbone.fc = original_fc
        return features

    def freeze_backbone(self, freeze=True):
        """冻结/解冻骨干网络，仅训练分类头"""
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = not freeze

    def unfreeze_stage(self, stage_name):
        """逐层解冻: 'layer1' | 'layer2' | 'layer3' | 'layer4'"""
        for name, param in self.backbone.named_parameters():
            if stage_name in name or "fc" in name:
                param.requires_grad = True


if __name__ == "__main__":
    # 快速测试
    model = CropDiseaseClassifier(num_classes=61)
    dummy = torch.randn(8, 3, 224, 224)
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}  (应为 [8, 61])")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print("模型定义验证通过 [OK]")
