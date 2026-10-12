import os
import sys
import time
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dataset.levir_mci_dataset import LevirMCIDataset
from models.multitask_model import MultiTaskChangeModel
from dataset.metrics import compute_iou_f1, compute_bleu, compute_rouge_l
from utils.translator import translate_change_caption

class ChangeDetectionCombinedLoss(nn.Module):
    """
    Комбинированный лосс для детекции изменений:
    0.5 * Weighted BCE (штраф за пропуск мелких объектов) + 0.5 * Dice Loss (стабильность площадей).
    """
    def __init__(self, bce_weight: float = 0.5, pos_weight: float = 2.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.pos_weight = torch.tensor([pos_weight])
        self.dice_weight = 1.0 - bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_w = self.pos_weight.to(logits.device)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_w
        )
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(-2, -1))
        union = probs.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
        dice = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = dice.mean()
        return self.bce_weight * bce + self.dice_weight * dice_loss

def apply_pair_augmentations(t1: torch.Tensor, t2: torch.Tensor, masks: torch.Tensor):
    """
    Синхронные геометрические и радиометрические аугментации для батча снимков.
    """
    b = t1.size(0)
    for i in range(b):
        # 1. Случайное отражение по горизонтали
        if random.random() > 0.5:
            t1[i] = torch.flip(t1[i], dims=[-1])
            t2[i] = torch.flip(t2[i], dims=[-1])
            masks[i] = torch.flip(masks[i], dims=[-1])

        # 2. Случайное отражение по вертикали
        if random.random() > 0.5:
            t1[i] = torch.flip(t1[i], dims=[-2])
            t2[i] = torch.flip(t2[i], dims=[-2])
            masks[i] = torch.flip(masks[i], dims=[-2])

        # 3. Случайный поворот на 90, 180 или 270 градусов
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            t1[i] = torch.rot90(t1[i], k=k, dims=[-2, -1])
            t2[i] = torch.rot90(t2[i], k=k, dims=[-2, -1])
            masks[i] = torch.rot90(masks[i], k=k, dims=[-2, -1])

        # 4. Радиометрическая вариация (сезонные изменения освещенности)
        if random.random() > 0.5:
            brightness_t1 = 1.0 + (random.random() - 0.5) * 0.15
            brightness_t2 = 1.0 + (random.random() - 0.5) * 0.15
            t1[i] = torch.clamp(t1[i] * brightness_t1, 0.0, 1.0)
            t2[i] = torch.clamp(t2[i] * brightness_t2, 0.0, 1.0)

    return t1, t2, masks

def train_one_epoch(model, loader, mask_loss_fn, text_loss_fn, optimizer, scaler, device, epoch, lambda_text=2.0):
    model.train()
    total_loss = 0.0
    total_mask_loss = 0.0
    total_text_loss = 0.0
    num_batches = len(loader)

    for step, (t1, t2, masks, captions) in enumerate(loader):
        t1 = t1.to(device, non_blocking=True)
        t2 = t2.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)

        # Аугментация
        t1, t2, masks = apply_pair_augmentations(t1, t2, masks)

        inp_captions = captions[:, :-1]
        target_captions = captions[:, 1:]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            mask_logits, caption_logits = model(t1, t2, inp_captions)
            l_mask = mask_loss_fn(mask_logits, masks)
            
            b, t, v = caption_logits.shape
            l_text = text_loss_fn(caption_logits.reshape(b * t, v), target_captions.reshape(-1))

            loss = l_mask + lambda_text * l_text

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_mask_loss += l_mask.item()
        total_text_loss += l_text.item()

        if (step + 1) % 150 == 0 or (step + 1) == num_batches:
            print(f"  [Эп {epoch:03d} | Батч {step+1:04d}/{num_batches:04d}] "
                  f"Total Loss: {total_loss / (step + 1):.4f} | "
                  f"Mask: {total_mask_loss / (step + 1):.4f} | "
                  f"Text: {total_text_loss / (step + 1):.4f}", flush=True)

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
            loss = l_mask + 2.0 * l_text

        total_loss += loss.item()
        all_preds_masks.append(torch.sigmoid(mask_logits).cpu())
        all_gt_masks.append(masks.cpu())

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
        print(f"  Эталон EN:  {references[0]}", flush=True)
        print(f"  Модель EN:  {hypotheses[0]}", flush=True)
        print(f"  Модель RU:  {translate_change_caption(hypotheses[0])}", flush=True)

    return total_loss / len(loader), det_metrics, bleu, rouge_l

def main():
    parser = argparse.ArgumentParser(description="Deep 6-Hour Multi-Task Training")
    parser.add_argument("--max-hours", type=float, default=6.0, help="Лимит времени в часах")
    parser.add_argument("--epochs", type=int, default=350, help="Максимальное количество эпох")
    parser.add_argument("--batch-size", type=int, default=8, help="Размер батча")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default="./data/levir_mci/LEVIR-MCI-dataset")
    parser.add_argument("--base-ckpt", type=str, default="./checkpoints/best_multitask_finetuned.pth")
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=================================================================", flush=True)
    print(f"  ГЛУБОКОЕ ОБУЧЕНИЕ MULTI-TASK CHANGE INTELLIGENCE (ДО {args.max_hours:.1f} ЧАСОВ)", flush=True)
    print(f"  Устройство:    {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f"  Инициализация: {args.base_ckpt}", flush=True)
    print(f"=================================================================\n", flush=True)

    # 1. Датасет
    print("[1/4] Инициализация сбалансированного датасета с аугментациями...", flush=True)
    train_dataset = LevirMCIDataset(root_dir=args.data_dir, split="train", img_size=256, balance_changes=True)
    val_dataset = LevirMCIDataset(root_dir=args.data_dir, split="val", vocab=train_dataset.vocab, img_size=256)
    vocab = train_dataset.vocab

    print(f"  Обучающих пар:    {len(train_dataset)}", flush=True)
    print(f"  Валидационных пар: {len(val_dataset)}", flush=True)
    print(f"  Размер словаря:   {len(vocab)} слов", flush=True)

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

    # 2. Модель
    print("\n[2/4] Загрузка весов и настройка дифференцированных LR...", flush=True)
    model = MultiTaskChangeModel(
        backbone_name="resnet18",
        in_channels=3,
        embed_dim=256,
        num_tokens=16,
        fusion_depth=4,
        vocab_size=len(vocab),
        caption_layers=3,
        pretrained_backbone=False
    ).to(device)

    best_ckpt_path = os.path.join(args.save_dir, "best_multitask_deep.pth")
    latest_ckpt_path = os.path.join(args.save_dir, "latest_multitask_checkpoint.pth")

    if os.path.exists(args.base_ckpt):
        print(f"  -> Инициализация весов из: {args.base_ckpt}", flush=True)
        ckpt = torch.load(args.base_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print("  -> Веса успешно загружены!", flush=True)
    elif os.path.exists(os.path.join(args.save_dir, "best_multitask_model.pth")):
        fallback = os.path.join(args.save_dir, "best_multitask_model.pth")
        print(f"  -> Загрузка из базового чекпоинта: {fallback}", flush=True)
        ckpt = torch.load(fallback, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    # Дифференцированные оптимизаторы: бэкбон учится медленнее (0.3x LR), декодеры быстрее
    backbone_params = list(model.backbone.parameters())
    head_params = (
        list(model.joint_fusion.parameters()) + 
        list(model.mask_decoder.parameters()) + 
        list(model.caption_decoder.parameters())
    )

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.3},
        {"params": head_params, "lr": args.lr}
    ], weight_decay=1e-4)

    # Расписание обучения: Cosine Annealing на весь 6-часовой бюджет
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    mask_loss_fn = ChangeDetectionCombinedLoss(bce_weight=0.5, pos_weight=2.0)
    text_loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.word2idx[vocab.PAD_TOKEN])
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu")

    best_combined_score = 0.0
    start_total_time = time.time()
    max_seconds = args.max_hours * 3600.0

    print(f"\n[3/4] Старт 6-часового цикла глубокого обучения (лимит: {max_seconds/3600:.1f} ч)...", flush=True)

    for epoch in range(1, args.epochs + 1):
        elapsed_sec = time.time() - start_total_time
        if elapsed_sec >= max_seconds:
            print(f"\n  [ВРЕМЯ ИСТЕКЛО] Достигнут лимит {args.max_hours:.1f} часов ({elapsed_sec/3600:.2f} ч).", flush=True)
            break

        ep_start = time.time()
        tr_loss, l_m, l_t = train_one_epoch(
            model, train_loader, mask_loss_fn, text_loss_fn, optimizer, scaler, device, epoch, lambda_text=2.0
        )
        scheduler.step()

        # Валидация каждые 2 эпохи
        do_eval = (epoch % 2 == 0) or (epoch == 1) or (epoch == args.epochs)
        if do_eval:
            val_loss, det_m, bleu, rouge = evaluate_multitask(
                model, val_loader, mask_loss_fn, text_loss_fn, vocab, device
            )
            combined_score = det_m["iou"] + (bleu["bleu4"] / 100.0)
        else:
            det_m = {"iou": 0.0, "f1": 0.0}
            bleu = {"bleu4": 0.0}
            rouge = 0.0
            combined_score = 0.0

        dt = time.time() - ep_start
        cur_elapsed = time.time() - start_total_time
        rem_sec = max(0.0, max_seconds - cur_elapsed)
        cur_lr = scheduler.get_last_lr()[1]

        if do_eval:
            print(f"\nИТОГИ ЭПОХИ [{epoch:03d}/{args.epochs:03d}] ({dt:.1f}s, прошло {cur_elapsed/3600:.2f}ч, осталось ~{rem_sec/3600:.2f}ч) | "
                  f"LR: {cur_lr:.2e} | Loss: {tr_loss:.4f} | "
                  f"IoU: {det_m['iou']:.4f} | F1: {det_m['f1']:.4f} | "
                  f"BLEU-4: {bleu['bleu4']} | ROUGE-L: {rouge:.1f}%\n" + "=" * 80, flush=True)

            if combined_score > best_combined_score or epoch == 1:
                best_combined_score = combined_score
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab.word2idx,
                    "iou": det_m["iou"],
                    "f1": det_m["f1"],
                    "bleu4": bleu["bleu4"],
                    "hours_elapsed": cur_elapsed / 3600.0
                }, best_ckpt_path)
                print(f"  [РЕКОРД] Лучший чекпоинт сохранен: {best_ckpt_path}\n", flush=True)
        else:
            print(f"  [Эпоха {epoch:03d}] {dt:.1f}s | Loss: {tr_loss:.4f} (Mask: {l_m:.3f}, Text: {l_t:.3f}) | Прошло {cur_elapsed/3600:.2f}ч, осталось ~{rem_sec/3600:.2f}ч", flush=True)

        # Резервный чекпоинт каждые 5 эпох
        if epoch % 5 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "vocab": vocab.word2idx,
                "hours_elapsed": cur_elapsed / 3600.0
            }, latest_ckpt_path)

    total_time = time.time() - start_total_time
    print(f"\n Обучение завершено! Общее время: {total_time / 3600:.2f} часов. Лучший чекпоинт: {best_ckpt_path}", flush=True)

if __name__ == "__main__":
    main()
