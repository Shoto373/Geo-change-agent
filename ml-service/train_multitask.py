"""
Обучение Multi-Task модели: Change Detection (Маска) + Change Captioning (Текст).
Архитектура: Pretrained ResNet-18 + BIT Joint Self-Attention + UNet Skip Decoder + Beam Search.
Датасет: LEVIR-MCI (6 815 пар train, 1 008 пар val).
Метрики: IoU, F1 (для маски) и BLEU-1..4, ROUGE-L (для текста).
Оптимизировано под NVIDIA RTX 5060 (8 ГБ VRAM).
"""

import os
import sys
import io
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Принудительная UTF-8 кодировка для консоли Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from models.multitask_model import MultiTaskChangeModel
from dataset.levir_mci_dataset import LevirMCIDataset
from dataset.losses import ChangeDetectionCombinedLoss
from dataset.metrics import compute_iou_f1, compute_bleu, compute_rouge_l

def train_one_epoch(model, loader, mask_loss_fn, text_loss_fn, optimizer, scaler, device, epoch, lambda_text=1.0):
    model.train()
    total_loss = 0.0
    total_mask_loss = 0.0
    total_text_loss = 0.0
    num_batches = len(loader)

    for step, (t1, t2, masks, captions) in enumerate(loader, 1):
        t1 = t1.to(device, non_blocking=True)
        t2 = t2.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)

        inp_captions = captions[:, :-1]
        target_captions = captions[:, 1:]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            mask_logits, caption_logits = model(t1, t2, inp_captions)

            # 1. Потери маски (BCE + Dice)
            l_mask = mask_loss_fn(mask_logits, masks)

            # 2. Потери текста (Cross Entropy)
            b, t, v = caption_logits.shape
            l_text = text_loss_fn(caption_logits.reshape(b * t, v), target_captions.reshape(-1))

            loss = l_mask + lambda_text * l_text

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_mask_loss += l_mask.item()
        total_text_loss += l_text.item()

        # Прогресс каждые 150 батчей
        if step % 150 == 0 or step == num_batches:
            print(f"  [Эпоха {epoch:02d} | Батч {step:03d}/{num_batches:03d}] "
                  f"Loss: {loss.item():.4f} (Mask: {l_mask.item():.3f}, Text: {l_text.item():.3f})", flush=True)

    return total_loss / num_batches, total_mask_loss / num_batches, total_text_loss / num_batches

@torch.no_grad()
def evaluate_multitask(model, loader, mask_loss_fn, text_loss_fn, vocab, device, max_eval_samples=60):
    model.eval()
    total_loss = 0.0
    all_preds_masks = []
    all_gt_masks = []
    hypotheses = []
    references = []

    for step, (t1, t2, masks, captions) in enumerate(loader):
        t1 = t1.to(device, non_blocking=True)
        t2 = t2.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)

        inp_captions = captions[:, :-1]
        target_captions = captions[:, 1:]

        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            mask_logits, caption_logits = model(t1, t2, inp_captions)
            l_mask = mask_loss_fn(mask_logits, masks)
            b, t, v = caption_logits.shape
            l_text = text_loss_fn(caption_logits.reshape(b * t, v), target_captions.reshape(-1))
            loss = l_mask + l_text

        total_loss += loss.item()
        all_preds_masks.append(torch.sigmoid(mask_logits).cpu())
        all_gt_masks.append(masks.cpu())

        # Генерация текста через Beam Search для первых max_eval_samples
        if len(hypotheses) < max_eval_samples:
            _, gen_tokens = model.predict(t1, t2, beam_size=3, max_caption_len=25)
            for i in range(len(t1)):
                if len(hypotheses) >= max_eval_samples:
                    break
                hyp_str = vocab.decode(gen_tokens[i].cpu().tolist())
                ref_str = vocab.decode(captions[i].cpu().tolist())
                hypotheses.append(hyp_str)
                references.append(ref_str)

    cat_preds = torch.cat(all_preds_masks, dim=0)
    cat_gt = torch.cat(all_gt_masks, dim=0)
    det_metrics = compute_iou_f1(cat_preds, cat_gt)

    bleu = compute_bleu(hypotheses, references)
    rouge_l = compute_rouge_l(hypotheses, references)

    if hypotheses:
        print(f"\n  [Пример валидации]", flush=True)
        print(f"  Эталон:     {references[0]}", flush=True)
        print(f"  BeamSearch: {hypotheses[0]}", flush=True)

    return total_loss / len(loader), det_metrics, bleu, rouge_l

def main():
    parser = argparse.ArgumentParser(description="Multi-Task Change Intelligence Training")
    parser.add_argument("--epochs", type=int, default=20, help="Количество эпох")
    parser.add_argument("--batch-size", type=int, default=8, help="Размер батча")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default="./data/levir_mci/LEVIR-MCI-dataset")
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", action="store_true", help="Продолжить обучение с сохраненного чекпоинта")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=================================================================", flush=True)
    print(f"  ОБУЧЕНИЕ MULTI-TASK CHANGE INTELLIGENCE ({args.epochs} ЭПОХ)", flush=True)
    print(f"  Устройство:   {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f"  Бэкбон:       Pretrained ResNet-18 (ImageNet)", flush=True)
    print(f"  Фьюжн:        BIT Joint Self-Attention + Pixel Cross-Attention", flush=True)
    print(f"  Декодеры:     U-Net Mask Decoder + Beam Search Transformer", flush=True)
    print(f"=================================================================\n", flush=True)

    # 1. Загрузка реального датасета LEVIR-MCI
    print("[1/4] Загрузка датасета LEVIR-MCI...", flush=True)
    train_dataset = LevirMCIDataset(root_dir=args.data_dir, split="train", img_size=256)
    val_dataset = LevirMCIDataset(root_dir=args.data_dir, split="val", vocab=train_dataset.vocab, img_size=256)
    vocab = train_dataset.vocab

    print(f"  Обучающих пар:   {len(train_dataset)}", flush=True)
    print(f"  Валидационных:   {len(val_dataset)}", flush=True)
    print(f"  Размер словаря:  {len(vocab)} слов", flush=True)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        pin_memory=(device.type == "cuda"),
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        pin_memory=(device.type == "cuda"),
        num_workers=2
    )

    # 2. Инициализация MultiTask модели
    print("\n[2/4] Инициализация модели...", flush=True)
    model = MultiTaskChangeModel(
        backbone_name="resnet18",
        in_channels=3,
        embed_dim=256,
        num_tokens=16,
        fusion_depth=4,
        vocab_size=len(vocab),
        caption_layers=3,
        pretrained_backbone=True
    ).to(device)

    start_epoch = 1
    best_iou = 0.0
    checkpoint_path = os.path.join(args.save_dir, "best_multitask_model.pth")

    # Восстановление с чекпоинта при необходимости
    if args.resume and os.path.exists(checkpoint_path):
        print(f"  -> Загрузка чекпоинта: {checkpoint_path}", flush=True)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 1) + 1
        best_iou = ckpt.get("iou", 0.0)
        print(f"  -> Восстановлено с эпохи {start_epoch}, текущий рекорд IoU: {best_iou:.4f}", flush=True)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Всего параметров: {total_params / 1e6:.2f} млн", flush=True)

    # 3. Функция потерь и оптимизатор
    mask_loss_fn = ChangeDetectionCombinedLoss(bce_weight=0.5, pos_weight=2.5)
    text_loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.word2idx[vocab.PAD_TOKEN])

    if start_epoch <= 2:
        print("\n[3/4] Фазирование: Warm-up голов (ResNet заморожен)", flush=True)
        model.freeze_backbone()
        lr_current = args.lr
    else:
        print("\n[3/4] Фазирование: Сквозной Fine-Tuning всей сети", flush=True)
        model.unfreeze_backbone()
        lr_current = args.lr * 0.5

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=lr_current, 
        weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler('cuda' if device.type == "cuda" else 'cpu')

    os.makedirs(args.save_dir, exist_ok=True)

    # 4. Основной тренировочный цикл
    print(f"\n[4/4] Запуск цикла с эпохи {start_epoch} до {args.epochs}...", flush=True)
    start_total_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        if epoch == 3:
            print("\n  >>> [Разморозка бэкбона ResNet-18 для сквозного обучения] <<<", flush=True)
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.5, weight_decay=1e-4)

        ep_start = time.time()
        train_loss, l_mask, l_text = train_one_epoch(
            model, train_loader, mask_loss_fn, text_loss_fn, optimizer, scaler, device, epoch
        )
        val_loss, det_m, bleu, rouge = evaluate_multitask(
            model, val_loader, mask_loss_fn, text_loss_fn, vocab, device
        )

        dt = time.time() - ep_start
        print(f"\nИТОГИ ЭПОХИ [{epoch:02d}/{args.epochs:02d}] ({dt:.1f}s) | "
              f"Loss: {train_loss:.4f} (Mask: {l_mask:.3f}, Text: {l_text:.3f}) | "
              f"IoU: {det_m['iou']:.4f} | F1: {det_m['f1']:.4f} | "
              f"BLEU-4: {bleu['bleu4']} | ROUGE-L: {rouge:.1f}%\n" + "-" * 75, flush=True)

        if det_m["iou"] > best_iou:
            best_iou = det_m["iou"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "vocab": vocab.word2idx,
                "iou": best_iou,
                "bleu4": bleu["bleu4"]
            }, checkpoint_path)
            print(f"  [РЕКОРД] Чекпоинт сохранен: {checkpoint_path}\n", flush=True)

    total_time = time.time() - start_total_time
    print(f"\n Обучение завершено за {total_time / 60:.1f} мин! Лучший IoU: {best_iou:.4f}", flush=True)

if __name__ == "__main__":
    main()
