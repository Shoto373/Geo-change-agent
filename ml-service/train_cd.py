"""
Скрипт обучения модели Bitemporal Image Transformer (BIT) для Change Detection.
Оптимизирован под NVIDIA RTX 5060 (8 ГБ VRAM) с использованием Mixed Precision (AMP FP16).
"""

import os
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from models.bit_model import BitemporalImageTransformer
from dataset.oscd_dataset import BitemporalChangeDataset
from dataset.losses import ChangeDetectionCombinedLoss

def compute_metrics(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    """Вычисляет Precision, Recall, F1 и IoU для детекции изменений."""
    probs = torch.sigmoid(preds)
    pred_mask = (probs >= threshold).float()
    target_mask = (targets >= 0.5).float()

    tp = (pred_mask * target_mask).sum().item()
    fp = (pred_mask * (1.0 - target_mask)).sum().item()
    fn = ((1.0 - pred_mask) * target_mask).sum().item()

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)

    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}

def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0

    for step, (t1, t2, targets) in enumerate(loader):
        t1 = t1.to(device, non_blocking=True)
        t2 = t2.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Смешанная точность (FP16) под RTX 5060
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(t1, t2)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}

    for t1, t2, targets in loader:
        t1 = t1.to(device, non_blocking=True)
        t2 = t2.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(t1, t2)
            loss = criterion(logits, targets)

        total_loss += loss.item()
        batch_m = compute_metrics(logits, targets)
        for k in all_metrics:
            all_metrics[k] += batch_m[k]

    n = len(loader)
    avg_loss = total_loss / n
    avg_metrics = {k: v / n for k, v in all_metrics.items()}
    return avg_loss, avg_metrics

def main():
    parser = argparse.ArgumentParser(description="Train BIT Change Detection Model")
    parser.add_argument("--epochs", type=int, default=15, help="Количество эпох")
    parser.add_argument("--batch-size", type=int, default=8, help="Размер батча (8 оптимален для 8GB VRAM)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning Rate")
    parser.add_argument("--in-channels", type=int, default=4, help="Каналы (4 = RGB + NIR)")
    parser.add_argument("--samples", type=int, default=120, help="Количество тайлов для тренировки")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Папка для весов")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n========================================================")
    print(f"  Обучение Bitemporal Image Transformer (BIT)")
    print(f"  Устройство:   {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Каналов:      {args.in_channels} (Sentinel-2: B, G, R, NIR)")
    print(f"  Батч:         {args.batch_size} (Память: ~3.5 ГБ VRAM)")
    print(f"========================================================\n")

    # 1. Датасет
    full_dataset = BitemporalChangeDataset(
        num_synthetic_samples=args.samples,
        in_channels=args.in_channels,
        img_size=256,
        is_train=True
    )
    val_size = max(int(len(full_dataset) * 0.2), 10)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=(device.type == "cuda"))

    # 2. Модель
    model = BitemporalImageTransformer(
        in_channels=args.in_channels,
        base_channels=64,
        num_tokens=16,
        transformer_depth=2
    ).to(device)

    total_params = sum(p.num_grad for p in model.parameters() if p.requires_grad)
    print(f"Параметров модели: {total_params / 1e6:.2f} млн\n")

    # 3. Оптимизатор, Scaler и Loss
    criterion = ChangeDetectionCombinedLoss(bce_weight=0.4, pos_weight=3.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    os.makedirs(args.save_dir, exist_ok=True)
    best_f1 = 0.0

    # 4. Обучающий цикл
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_m = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - ep_start
        print(f"Эпоха [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val F1: {val_m['f1']:.4f} | IoU: {val_m['iou']:.4f} | "
              f"Precision: {val_m['precision']:.4f} | Recall: {val_m['recall']:.4f}")

        # Сохранение лучшего чекпоинта
        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            save_path = os.path.join(args.save_dir, "best_bit_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "f1": best_f1,
                "iou": val_m["iou"],
                "in_channels": args.in_channels
            }, save_path)
            print(f"  --> Новый рекорд F1 ({best_f1:.4f}) сохранен в {save_path}")

    total_time = time.time() - start_time
    print(f"\n Обучение завершено за {total_time:.1f} сек. Лучший Val F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()
