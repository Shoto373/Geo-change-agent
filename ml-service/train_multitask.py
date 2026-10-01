"""
Обучение Multi-Task модели: Change Detection (Маска) + Change Captioning (Текст).
Архитектура: Pretrained ResNet-18 + BIT Joint Self-Attention + UNet Skip Decoder + Beam Search.
Метрики: IoU, F1 (для маски) и BLEU-1..4, ROUGE-L (для текста).
Оптимизировано под NVIDIA RTX 5060 (8 ГБ VRAM).
"""

import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from models.multitask_model import MultiTaskChangeModel
from dataset.levir_mci_dataset import LevirMCIDataset
from dataset.losses import ChangeDetectionCombinedLoss
from dataset.metrics import compute_iou_f1, compute_bleu, compute_rouge_l

def train_one_epoch(model, loader, mask_loss_fn, text_loss_fn, optimizer, scaler, device, lambda_text=1.0):
    model.train()
    total_loss = 0.0
    total_mask_loss = 0.0
    total_text_loss = 0.0

    for step, (t1, t2, masks, captions) in enumerate(loader):
        t1 = t1.to(device, non_blocking=True)
        t2 = t2.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)

        inp_captions = captions[:, :-1]
        target_captions = captions[:, 1:]

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            mask_logits, caption_logits = model(t1, t2, inp_captions)

            # 1. Потери маски (BCE + Dice)
            l_mask = mask_loss_fn(mask_logits, masks)

            # 2. Потери текста (Cross Entropy по словарю)
            b, t, v = caption_logits.shape
            l_text = text_loss_fn(caption_logits.reshape(b * t, v), target_captions.reshape(-1))

            loss = l_mask + lambda_text * l_text

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_mask_loss += l_mask.item()
        total_text_loss += l_text.item()

    n = len(loader)
    return total_loss / n, total_mask_loss / n, total_text_loss / n

@torch.no_grad()
def evaluate_multitask(model, loader, mask_loss_fn, text_loss_fn, vocab, device, max_samples=40):
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

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            mask_logits, caption_logits = model(t1, t2, inp_captions)
            l_mask = mask_loss_fn(mask_logits, masks)
            b, t, v = caption_logits.shape
            l_text = text_loss_fn(caption_logits.reshape(b * t, v), target_captions.reshape(-1))
            loss = l_mask + l_text

        total_loss += loss.item()

        # Маски
        pred_mask = torch.sigmoid(mask_logits)
        all_preds_masks.append(pred_mask.cpu())
        all_gt_masks.append(masks.cpu())

        # Текст через Beam Search для первых max_samples сэмплов
        if len(hypotheses) < max_samples:
            _, gen_tokens = model.predict(t1, t2, beam_size=3, max_caption_len=25)
            for i in range(len(t1)):
                if len(hypotheses) >= max_samples:
                    break
                hyp_str = vocab.decode(gen_tokens[i].cpu().tolist())
                ref_str = vocab.decode(captions[i].cpu().tolist())
                hypotheses.append(hyp_str)
                references.append(ref_str)

    # 1. Метрики маски
    cat_preds = torch.cat(all_preds_masks, dim=0)
    cat_gt = torch.cat(all_gt_masks, dim=0)
    det_metrics = compute_iou_f1(cat_preds, cat_gt)

    # 2. Метрики текста
    bleu = compute_bleu(hypotheses, references)
    rouge_l = compute_rouge_l(hypotheses, references)

    # Демонстрационный сэмпл
    if hypotheses:
        print(f"\n  [Пример валидации]")
        print(f"  Эталон:     {references[0]}")
        print(f"  BeamSearch: {hypotheses[0]}")

    return total_loss / len(loader), det_metrics, bleu, rouge_l

def main():
    parser = argparse.ArgumentParser(description="Multi-Task Change Intelligence Training")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n========================================================")
    print(f"  Обучение Multi-Task: Change Detection + Captioning")
    print(f"  Бэкбон:       Pretrained ResNet-18 (ImageNet)")
    print(f"  Фьюжн:        BIT Joint Self-Attention + Pixel Cross-Attn")
    print(f"  Декодеры:     U-Net Mask Decoder + Beam Search Transformer")
    print(f"  Устройство:   {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"========================================================\n")

    # 1. Датасет
    dataset = LevirMCIDataset(img_size=256, num_synthetic=120)
    vocab = dataset.vocab
    print(f"Размер словаря: {len(vocab)} слов")

    val_size = max(int(len(dataset) * 0.2), 10)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # 2. Модель
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

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Параметров Multi-Task модели: {total_params / 1e6:.2f} млн\n")

    mask_loss_fn = ChangeDetectionCombinedLoss(bce_weight=0.5, pos_weight=2.5)
    text_loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.word2idx[vocab.PAD_TOKEN])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    os.makedirs(args.save_dir, exist_ok=True)
    best_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss, l_mask, l_text = train_one_epoch(
            model, train_loader, mask_loss_fn, text_loss_fn, optimizer, scaler, device
        )
        val_loss, det_m, bleu, rouge = evaluate_multitask(
            model, val_loader, mask_loss_fn, text_loss_fn, vocab, device
        )

        dt = time.time() - t0
        print(f"Эпоха [{epoch:02d}/{args.epochs:02d}] ({dt:.1f}s) | "
              f"Loss: {loss:.4f} (Mask: {l_mask:.3f}, Text: {l_text:.3f}) | "
              f"IoU: {det_m['iou']:.4f} | F1: {det_m['f1']:.4f} | "
              f"BLEU-4: {bleu['bleu4']} | ROUGE-L: {rouge:.1f}%")

        if det_m["iou"] > best_iou:
            best_iou = det_m["iou"]
            save_path = os.path.join(args.save_dir, "best_multitask_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "vocab": vocab.word2idx,
                "iou": best_iou,
                "bleu4": bleu["bleu4"]
            }, save_path)
            print(f"  --> Чекпоинт сохранен: {save_path}")

    print(f"\n Обучение завершено! Лучший IoU: {best_iou:.4f}")

if __name__ == "__main__":
    main()
