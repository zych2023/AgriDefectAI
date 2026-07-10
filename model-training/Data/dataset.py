"""
农作物病害数据集模块
- AI Challenger 2018 农作物叶子图像及标注
- 支持训练/验证 DataLoader
- 核心: 标注文件与磁盘文件按位置对齐
"""
import os
import json
import yaml
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sorted_image_files(image_dir):
    """返回按字母排序的图片文件(过滤非图片+副本)"""
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}
    files = []
    for f in Path(image_dir).iterdir():
        if f.suffix in valid_ext:
            # 跳过 Windows 复制重复文件
            if " - 副本" in f.name or " - 副本" in f.stem:
                continue
            files.append(f.name)
    return sorted(files)


class CropDiseaseDataset(Dataset):
    """农作物病害数据集"""

    def __init__(self, image_dir, annotation_path, transform=None,
                 annotation_type="auto", label_mapping=None):
        """
        annotation_type: "json" | "txt" | "auto"
           - "json": AI Challenger JSON 标注格式
           - "txt":  train_list.txt 格式 (路径 标签)
           - "auto": 根据文件后缀自动判断
        """
        self.image_dir = Path(image_dir)
        self.transform = transform

        if annotation_type == "auto":
            if annotation_path.endswith(".json"):
                annotation_type = "json"
            elif annotation_path.endswith(".txt"):
                annotation_type = "filename"  # 用文件名提取标签
            else:
                raise ValueError(f"无法判断标注类型: {annotation_path}")

        self.label_mapping = label_mapping

        print(f"Loading: {annotation_path} (type={annotation_type}) ...")

        if annotation_type == "filename":
            self._load_from_filename()
        elif annotation_type == "txt":
            self._load_txt(annotation_path)
        else:
            self._load_json(annotation_path)

        # 应用标签映射 (61→36)
        if self.label_mapping is not None:
            for i, (path, label) in enumerate(self.samples):
                self.samples[i] = (path, self.label_mapping.get(label, label))

        print(f"  Loaded {len(self.samples)} samples  |  classes: {len(set(l for _,l in self.samples))}")

    def _load_from_filename(self):
        """从文件名提取标签: {label}_{id}.ext 例如 0_27417.jpg -> label=0"""
        disk_files = _sorted_image_files(self.image_dir)
        self.samples = []
        skipped = 0
        for fname in disk_files:
            prefix = fname.split("_")[0]
            if prefix.isdigit():
                label = int(prefix)
                self.samples.append((str(self.image_dir / fname), label))
            else:
                skipped += 1
        if skipped:
            print(f"  Skipped {skipped} files with non-numeric prefix")

    def _load_txt(self, path):
        """train_list.txt: 路径 标签"""
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        # 解析标签 (取每行最后一个数字)
        labels = []
        for line in lines:
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                labels.append(int(parts[1]))
            else:
                labels.append(0)

        disk_files = _sorted_image_files(self.image_dir)

        if len(labels) != len(disk_files):
            print(f"  [WARN] 标注行数({len(labels)}) != 文件数({len(disk_files)})，按较短的一方配对")

        self.samples = []
        for i in range(min(len(labels), len(disk_files))):
            fname = disk_files[i]
            label = labels[i]
            self.samples.append((str(self.image_dir / fname), label))

    def _load_json(self, path):
        """JSON 标注: [{"image_id":"xx.jpg","disease_class":0}, ...]"""
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        labels = [item["disease_class"] for item in raw]
        disk_files = _sorted_image_files(self.image_dir)

        if len(labels) != len(disk_files):
            print(f"  [WARN] JSON条目({len(labels)}) != 文件数({len(disk_files)})，按较短的一方配对")

        self.samples = []
        for i in range(min(len(labels), len(disk_files))):
            fname = disk_files[i]
            label = labels[i]
            self.samples.append((str(self.image_dir / fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(cfg, is_train=True):
    """根据配置生成数据增强流水线"""
    aug = cfg.get("augmentation", {})
    normalize = transforms.Normalize(
        mean=aug.get("normalize_mean", [0.485, 0.456, 0.406]),
        std=aug.get("normalize_std", [0.229, 0.224, 0.225]),
    )

    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                cfg["data"]["image_size"],
                scale=tuple(aug.get("random_crop_scale", [0.8, 1.0])),
                ratio=tuple(aug.get("random_crop_ratio", [0.9, 1.1])),
            ),
            transforms.RandomHorizontalFlip(p=aug.get("horizontal_flip_prob", 0.5)),
            transforms.RandomRotation(degrees=aug.get("rotation_degrees", 15)),
            transforms.ColorJitter(
                brightness=aug.get("color_jitter", {}).get("brightness", 0.2),
                contrast=aug.get("color_jitter", {}).get("contrast", 0.2),
                saturation=aug.get("color_jitter", {}).get("saturation", 0.2),
                hue=aug.get("color_jitter", {}).get("hue", 0.1),
            ),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(cfg["data"]["image_size"] * 256 / 224)),
            transforms.CenterCrop(cfg["data"]["image_size"]),
            transforms.ToTensor(),
            normalize,
        ])


def get_dataloaders(cfg):
    """构建训练和验证 DataLoader"""
    batch_size = cfg["train"]["batch_size"]
    num_workers = cfg["train"].get("num_workers", 4)

    train_transform = get_transforms(cfg, is_train=True)
    val_transform = get_transforms(cfg, is_train=False)

    label_mapping = cfg.get("label_mapping", None)

    train_dataset = CropDiseaseDataset(
        image_dir=cfg["data"]["train_images"],
        annotation_path=cfg["data"]["train_annotations"],
        transform=train_transform,
        annotation_type="filename",
        label_mapping=label_mapping,
    )
    val_dataset = CropDiseaseDataset(
        image_dir=cfg["data"]["val_images"],
        annotation_path=cfg["data"]["val_annotations"],
        transform=val_transform,
        annotation_type="filename",
        label_mapping=label_mapping,
    )

    # 类别分布
    train_labels = [lb for _, lb in train_dataset.samples]
    label_dist = Counter(train_labels)
    print(f"训练集: {len(train_dataset)}张, {len(label_dist)}类")
    print(f"验证集: {len(val_dataset)}张")
    print(f"类别分布(top5): {label_dist.most_common(5)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, label_dist


if __name__ == "__main__":
    cfg = load_config()
    train_loader, val_loader, dist = get_dataloaders(cfg)
    images, labels = next(iter(train_loader))
    print(f"\nBatch: {images.shape}, labels: {labels.shape}")
    print(f"标签范围: {labels.min().item()} ~ {labels.max().item()}")
    print("Data loading OK!")
