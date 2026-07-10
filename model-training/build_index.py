"""
构建相似图片检索的特征库
- 用训练好的模型提取所有参考图片的 2048 维特征向量
- 保存特征库供推理时做相似度检索

用法:
    python build_index.py

产出:
    checkpoints/feature_index.pt  — {features, image_paths, labels, class_names}
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from models.model import CropDiseaseClassifier


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class FeatureDataset(Dataset):
    """只读图片 + 标签，不做数据增强"""

    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform

        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}
        self.samples = []
        for f in sorted(self.image_dir.iterdir()):
            if f.suffix in valid_ext and " - 副本" not in f.name and " - 副本" not in f.stem:
                # 从文件名提取标签: {label}_{id}.ext
                prefix = f.name.split("_")[0]
                if prefix.isdigit():
                    self.samples.append((str(f), int(prefix)))
                else:
                    self.samples.append((str(f), -1))  # 无法解析标签的标记为 -1

        print(f"  Found {len(self.samples)} valid images in {image_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, path


def build_index(model_path="checkpoints/best_model.pth",
                config_path="config.yaml",
                output_path="checkpoints/feature_index.pt",
                batch_size=128):
    """构建特征库"""
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # 加载模型
    print(f"\n[Model] Loading: {model_path}")
    model = CropDiseaseClassifier(
        num_classes=cfg["data"]["num_classes"],
        pretrained=False,
        dropout=cfg["model"]["dropout"],
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  训练轮次: {checkpoint.get('epoch', '?') + 1}")
    print(f"  验证 Acc:  {checkpoint.get('val_acc', '?'):.2f}%")

    # 加载类别名 + 标签映射
    class_names = {int(k): v for k, v in cfg["class_names"].items()}
    label_mapping = cfg.get("label_mapping", None)

    # 参考图集：训练集图片
    print("\n[Data] Loading reference images...")
    aug = cfg.get("augmentation", {})
    transform = transforms.Compose([
        transforms.Resize(int(cfg["data"]["image_size"] * 256 / 224)),
        transforms.CenterCrop(cfg["data"]["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=aug.get("normalize_mean", [0.485, 0.456, 0.406]),
            std=aug.get("normalize_std", [0.229, 0.224, 0.225]),
        ),
    ])

    dataset = FeatureDataset(
        image_dir=cfg["data"]["train_images"],
        transform=transform,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=cfg["train"].get("num_workers", 4),
        pin_memory=True,
    )

    # 提取特征
    print(f"\n[Extracting] {len(dataset)} images, batch_size={batch_size}...")
    all_features = []
    all_paths = []
    all_labels = []

    with torch.no_grad():
        for images, labels, paths in tqdm(loader, desc="Extracting features"):
            images = images.to(device)
            features = model.extract_features(images)  # [B, 2048]
            all_features.append(features.cpu())
            all_paths.extend(paths)
            all_labels.extend(labels.tolist())

    # 合并 & 应用标签映射
    features = torch.cat(all_features, dim=0)  # [N, 2048]
    labels = all_labels

    if label_mapping is not None:
        labels = [label_mapping.get(lb, lb) for lb in labels]

    # L2 归一化（用归一化后的向量做余弦相似度 = 点积）
    features = torch.nn.functional.normalize(features, p=2, dim=1)

    # 保存
    output = {
        "features": features,           # [N, 2048] L2归一化
        "image_paths": all_paths,       # list of str
        "labels": labels,               # list of int (36类映射后)
        "class_names": class_names,     # {id: name}
        "image_size": cfg["data"]["image_size"],
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_file)

    size_mb = output_file.stat().st_size / 1024**2
    print(f"\n[OK] 特征库已保存: {output_file}")
    print(f"     特征矩阵: {features.shape}  ({features.shape[0]} 张 × {features.shape[1]} 维)")
    print(f"     文件大小:  {size_mb:.1f} MB")
    print(f"     覆盖类别:  {len(set(labels))} 类")


if __name__ == "__main__":
    build_index()
