"""
Инференс обученной Multi-Task модели (Маска + Текст).
Принимает два снимка, прогоняет через обученный чекпоинт и выводит:
  1. Текстовое описание изменений (Beam Search).
  2. Визуализацию: T1, T2, предсказанную маску и наложение.
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from models.multitask_model import MultiTaskChangeModel
from dataset.levir_mci_dataset import ChangeCaptionVocab
from utils.translator import translate_change_caption

def run_inference(img_a_path: str, img_b_path: str, checkpoint_path: str = "./checkpoints/best_multitask_finetuned.pth", output_path: str = "inference_result.png"):
    if not os.path.exists(checkpoint_path) and os.path.exists("./checkpoints/best_multitask_model.pth"):
        checkpoint_path = "./checkpoints/best_multitask_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Загрузка модели из {checkpoint_path} на {device}...")

    ckpt = torch.load(checkpoint_path, map_location=device)
    word2idx = ckpt["vocab"]
    
    vocab = ChangeCaptionVocab()
    vocab.word2idx = word2idx
    vocab.idx2word = {idx: word for word, idx in word2idx.items()}

    # Инициализация модели
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

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Загрузка и подготовка пары снимков
    img_a = Image.open(img_a_path).convert("RGB").resize((256, 256), Image.BILINEAR)
    img_b = Image.open(img_b_path).convert("RGB").resize((256, 256), Image.BILINEAR)

    t1_np = np.array(img_a, dtype=np.float32) / 255.0
    t2_np = np.array(img_b, dtype=np.float32) / 255.0

    t1_tensor = torch.from_numpy(t1_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
    t2_tensor = torch.from_numpy(t2_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # Предсказание
    with torch.no_grad():
        pred_mask, gen_tokens = model.predict(t1_tensor, t2_tensor, mask_threshold=0.5, beam_size=3, max_caption_len=30)

    pred_mask_np = pred_mask.squeeze().cpu().numpy()
    num_changed_pixels = int(pred_mask_np.sum())
    change_percentage = (num_changed_pixels / (256 * 256)) * 100

    if num_changed_pixels == 0:
        generated_en = "there is no difference ."
    else:
        generated_en = vocab.decode(gen_tokens.squeeze().cpu().tolist())

    generated_ru = translate_change_caption(generated_en)

    print("\n" + "=" * 65)
    print("  РЕЗУЛЬТАТ ИНФЕРЕНСА MULTI-TASK МОДЕЛИ")
    print("=" * 65)
    print(f"  Текст EN:              \"{generated_en}\"")
    print(f"  Текст RU (Перевод):    \"{generated_ru}\"")
    print(f"  Изменившихся пикселей: {num_changed_pixels} / {256*256} ({change_percentage:.1f}%)")
    print("=" * 65)

    # Визуализация
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(t1_np)
    axes[0].set_title("Снимок ДО (T1)", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(t2_np)
    axes[1].set_title("Снимок ПОСЛЕ (T2)", fontsize=13)
    axes[1].axis("off")

    axes[2].imshow(pred_mask_np, cmap="gray")
    axes[2].set_title(f"Маска изменений ({num_changed_pixels} px, {change_percentage:.1f}%)", fontsize=13)
    axes[2].axis("off")

    axes[3].imshow(t2_np)
    if num_changed_pixels > 0:
        overlay = np.ma.masked_where(pred_mask_np == 0, pred_mask_np)
        axes[3].imshow(overlay, cmap="autumn", alpha=0.6, interpolation="none")
    axes[3].set_title(f"\"{generated_ru}\"", fontsize=11, fontweight="bold", color="darkred")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Диагностический снимок сохранен: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-a", type=str, default="data/levir_mci/LEVIR-MCI-dataset/images/test/A/test_000004.png")
    parser.add_argument("--img-b", type=str, default="data/levir_mci/LEVIR-MCI-dataset/images/test/B/test_000004.png")
    parser.add_argument("--ckpt", type=str, default="./checkpoints/best_multitask_finetuned.pth")
    parser.add_argument("--out", type=str, default="inference_result.png")
    args = parser.parse_args()

    run_inference(args.img_a, args.img_b, args.ckpt, args.out)
