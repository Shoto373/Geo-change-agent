import os
import sys
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from models.multitask_model import MultiTaskChangeModel
from dataset.levir_mci_dataset import ChangeCaptionVocab
from utils.translator import translate_change_caption

def generate_visual_report():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "checkpoints/best_multitask_deep.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/best_multitask_finetuned.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/best_multitask_model.pth"
        
    print(f"Loading checkpoint: {ckpt_path} on {device}...")
    ckpt = torch.load(ckpt_path, map_location=device)

    vocab = ChangeCaptionVocab()
    vocab.word2idx = ckpt["vocab"]
    vocab.idx2word = {i: w for w, i in vocab.word2idx.items()}

    model = MultiTaskChangeModel(vocab_size=len(vocab), pretrained_backbone=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with open("data/levir_mci/LEVIR-MCI-dataset/LevirCCcaptions.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    ann_dict = {x["filename"]: [s["raw"] for s in x["sentences"]] for x in meta["images"]}

    samples = [
        ("test_000004.png", "Вырубка растительности, кольцевая дорога и виллы"),
        ("test_000019.png", "Строительство коттеджного массива"),
        ("test_000017.png", "Новая дорога снизу и рассредоточенные дома"),
        ("test_000001.png", "Контрольная пара без изменений (Zero Change)")
    ]

    fig, axes = plt.subplots(len(samples), 5, figsize=(23, 4.6 * len(samples)))
    plt.subplots_adjust(wspace=0.12, hspace=0.28)

    print("\n" + "=" * 105)
    print(f"{'Sample':<16} | {'GT Mask':<10} | {'Pred Mask':<10} | {'IoU':<8} | {'Russian Caption'}")
    print("=" * 105)

    for row_idx, (sample, title_desc) in enumerate(samples):
        p_a = f"data/levir_mci/LEVIR-MCI-dataset/images/test/A/{sample}"
        p_b = f"data/levir_mci/LEVIR-MCI-dataset/images/test/B/{sample}"
        p_mask = f"data/levir_mci/LEVIR-MCI-dataset/images/test/label/{sample}"

        img_a = Image.open(p_a).convert("RGB").resize((256, 256), Image.BILINEAR)
        img_b = Image.open(p_b).convert("RGB").resize((256, 256), Image.BILINEAR)
        
        # Порог > 50 для честного эталона (включая дорогу 128 и дома 255)
        if os.path.exists(p_mask):
            raw_lbl = np.array(Image.open(p_mask).convert("L").resize((256, 256), Image.NEAREST))
            gt_mask = raw_lbl > 50
            gt_has_road = ((raw_lbl >= 100) & (raw_lbl <= 150)).sum() > 0
        else:
            gt_mask = np.zeros((256, 256), dtype=bool)
            gt_has_road = False

        t1 = torch.from_numpy(np.array(img_a, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        t2 = torch.from_numpy(np.array(img_b, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_mask_tensor, gen_tokens = model.predict(t1, t2, beam_size=3, max_caption_len=30)

        pred_mask = pred_mask_tensor.squeeze().cpu().numpy() > 0.5
        if pred_mask.sum() == 0:
            caption_en = "there is no difference ."
        else:
            caption_en = vocab.decode(gen_tokens.squeeze().cpu().tolist())
        caption_ru = translate_change_caption(caption_en)
        gt_caption_en = ann_dict.get(sample, ["N/A"])[0].strip()
        gt_caption_ru = translate_change_caption(gt_caption_en)

        # Вычисление честного IoU
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        sample_iou = (intersection / union) if union > 0 else (1.0 if pred_mask.sum() == 0 else 0.0)

        print(f"{sample:<16} | {gt_mask.sum():<10} | {pred_mask.sum():<10} | {sample_iou*100:>5.1f}% | {caption_ru}")
        print(f"  -> EN Pred: \"{caption_en}\"")
        print(f"  -> GT EN:   \"{gt_caption_en}\"")
        print("-" * 105)

        # 1. T1
        axes[row_idx, 0].imshow(img_a)
        axes[row_idx, 0].set_title(f"Снимок ДО (T1)\n{title_desc[:35]}", fontsize=10, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # 2. T2
        axes[row_idx, 1].imshow(img_b)
        axes[row_idx, 1].set_title(f"Снимок ПОСЛЕ (T2)\nGT: \"{gt_caption_ru[:35]}...\"", fontsize=9)
        axes[row_idx, 1].axis("off")

        # 3. Ground Truth Mask (Road 128 + Houses 255)
        axes[row_idx, 2].imshow(gt_mask, cmap="gray")
        road_note = " (с учетом дороги)" if gt_has_road else ""
        axes[row_idx, 2].set_title(f"Эталонная маска (GT)\n{gt_mask.sum()} px{road_note}", fontsize=9)
        axes[row_idx, 2].axis("off")

        # 4. Neural Network Prediction Mask
        axes[row_idx, 3].imshow(pred_mask, cmap="gray")
        axes[row_idx, 3].set_title(f"Предсказанная маска сети\n{pred_mask.sum()} px (IoU: {sample_iou*100:.1f}%)", fontsize=9, fontweight="bold", color="darkgreen" if sample_iou > 0.6 else "black")
        axes[row_idx, 3].axis("off")

        # 5. Overlay + Russian Caption
        axes[row_idx, 4].imshow(img_b)
        if pred_mask.sum() > 0:
            overlay = np.ma.masked_where(~pred_mask, pred_mask)
            axes[row_idx, 4].imshow(overlay, cmap="autumn", alpha=0.65, interpolation="none")
        
        # Переносим длинный русский текст на строки
        words = caption_ru.split()
        wrapped_ru = ""
        line = ""
        for w in words:
            if len(line + " " + w) > 32:
                wrapped_ru += line + "\n"
                line = w
            else:
                line = (line + " " + w).strip()
        wrapped_ru += line

        axes[row_idx, 4].set_title(f"Отчет на русском:\n\"{wrapped_ru}\"", fontsize=9, fontweight="bold", color="darkred")
        axes[row_idx, 4].axis("off")

    out_file = "test_evaluation_deep_6h.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nFinal Visual Report saved to: {out_file}")

if __name__ == "__main__":
    generate_visual_report()
