"""
农作物病害识别 — 模型训练脚本
支持: 本地调试 / 华为云 ModelArts 训练作业
在 ModelArts 训练作业中, 设置环境变量 MODELARTS=True 自动走OBS数据通路

用法:
  # 本地 (快速验证)
  python train.py

  # ModelArts 训练作业 (需设置环境变量)
  export D_MODELARTS=true
  export D_DATA_URL="s3://bucket/data/"
  export D_TRAIN_URL="s3://bucket/output/"
  python train.py
"""
import os
import sys
import time
import math
import warnings
from pathlib import Path
from datetime import datetime

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

# ---- 将项目根目录加入 path ----
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Data.dataset import get_dataloaders
from models.model import CropDiseaseClassifier


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class AverageMeter:
    """运行时均值统计"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """计算 top-k 准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, cfg):
    """训练一个epoch"""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    use_amp = cfg["train"].get("mixed_precision", True)

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 统计
        acc1, acc3 = accuracy(outputs, labels, topk=(1, 3))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))
        top3.update(acc3, images.size(0))

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(loader)}  "
                  f"Loss: {losses.val:.3f} ({losses.avg:.3f})  "
                  f"Acc@1: {top1.val:.1f}% ({top1.avg:.1f}%)")

    return losses.avg, top1.avg, top3.avg


@torch.no_grad()
def validate(model, loader, criterion, device):
    """验证集评估"""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        acc1, acc3 = accuracy(outputs, labels, topk=(1, 3))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))
        top3.update(acc3, images.size(0))

    return losses.avg, top1.avg, top3.avg


def cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, epoch, base_lr):
    """Cosine 学习率衰减 + Warmup"""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def main():
    cfg = load_config()

    # ========== 设备检测 ==========
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[GPU] 使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[WARN]  未检测到GPU, 使用CPU训练 (仅适合本地调试)")

    # ========== 数据 ==========
    train_loader, val_loader, label_dist = get_dataloaders(cfg)

    # ========== 模型 ==========
    model = CropDiseaseClassifier(
        num_classes=cfg["data"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    # ========== 损失函数 & 优化器 ==========
    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg["train"].get("label_smoothing", 0.0)
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    # ========== 混合精度 ==========
    use_amp = cfg["train"].get("mixed_precision", True) and device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    print(f"混合精度训练(AMP): {'[OK] 开启' if scaler else '[OFF] 关闭'}")

    # ========== 训练循环 ==========
    save_dir = Path(cfg["checkpoint"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = cfg["train"]["epochs"]
    patience = cfg["train"].get("patience", 5)
    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0

    print(f"\n{'='*50}")
    print(f"开始训练 — Epochs: {epochs}  Batch: {cfg['train']['batch_size']}  "
          f"LR: {cfg['train']['learning_rate']}")
    print(f"训练样本: {len(train_loader.dataset)}  "
          f"验证样本: {len(val_loader.dataset)}  "
          f"类别数: {cfg['data']['num_classes']}")
    print(f"{'='*50}\n")

    for epoch in range(epochs):
        epoch_start = time.time()

        # 学习率调度
        current_lr = cosine_schedule_with_warmup(
            optimizer,
            warmup_epochs=cfg["train"].get("warmup_epochs", 2),
            total_epochs=epochs,
            epoch=epoch,
            base_lr=cfg["train"]["learning_rate"],
        )

        # 训练 & 验证
        train_loss, train_acc1, train_acc3 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, cfg
        )
        val_loss, val_acc1, val_acc3 = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        print(f"\n[Epoch] Epoch {epoch+1}/{epochs} | Time: {epoch_time:.1f}s | LR: {current_lr:.6f}")
        print(f"  Train — Loss: {train_loss:.4f}  Acc@1: {train_acc1:.2f}%  Acc@3: {train_acc3:.2f}%")
        print(f"  Val   — Loss: {val_loss:.4f}  Acc@1: {val_acc1:.2f}%  Acc@3: {val_acc3:.2f}%")

        # 保存最佳模型
        if val_acc1 > best_val_acc:
            best_val_acc = val_acc1
            best_epoch = epoch + 1
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc1,
                "config": cfg,
            }, cfg["checkpoint"]["best_model"])
            print(f"  [*] 最佳模型已保存 (Acc@1: {val_acc1:.2f}%)")
        else:
            no_improve += 1
            print(f"  未提升 ({no_improve}/{patience})")

        # Early Stop
        if no_improve >= patience:
            print(f"\n[STOP] Early Stop — 验证集 {patience} 轮未提升, 停止训练")
            break

    # 保存最终模型
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc1,
        "config": cfg,
    }, cfg["checkpoint"]["final_model"])

    print(f"\n{'='*50}")
    print(f"[OK] 训练完成! 最佳 Acc@1: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"模型保存至: {save_dir.resolve()}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
