import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from detector import analyze_earth_changes, compute_spectral_indices
from models.multitask_model import MultiTaskChangeModel
from dataset.levir_mci_dataset import ChangeCaptionVocab
from utils.translator import translate_change_caption
import torch

def generate_disaster_showcase():
    print("=== ГЕНЕРАЦИЯ ДИАГНОСТИЧЕСКОГО ОТЧЕТА ПО ЧС И СТРОИТЕЛЬСТВУ ===")
    size = 256
    y, x = np.ogrid[:size, :size]

    fig, axes = plt.subplots(4, 4, figsize=(18, 17))
    plt.subplots_adjust(wspace=0.10, hspace=0.28)

    # -------------------------------------------------------------
    # 1. СЦЕНАРИЙ: НАВОДНЕНИЕ / ПАВОДОК (FLOOD)
    # -------------------------------------------------------------
    np.random.seed(101)
    flood_t1 = {
        "B02": np.random.normal(300, 20, (size, size)),
        "B03": np.random.normal(550, 30, (size, size)),
        "B04": np.random.normal(350, 25, (size, size)),
        "B08": np.random.normal(3200, 100, (size, size)),
        "B12": np.random.normal(800, 40, (size, size))
    }
    flood_t2 = {k: v.copy() for k, v in flood_t1.items()}
    # Зона разлива реки
    flood_mask_gt = ((x - 128)**2 / (75**2) + (y - 128)**2 / (45**2)) <= 1
    flood_t2["B02"][flood_mask_gt] = 1300
    flood_t2["B03"][flood_mask_gt] = 1100
    flood_t2["B04"][flood_mask_gt] = 380
    flood_t2["B08"][flood_mask_gt] = 180  # Сильное поглощение водой
    flood_t2["B12"][flood_mask_gt] = 100

    mask_flood, stats_flood = analyze_earth_changes(flood_t1, flood_t2)

    # RGB для отображения
    t1_rgb = np.stack([flood_t1["B04"], flood_t1["B03"], flood_t1["B02"]], axis=-1) / 3000.0
    t2_rgb = np.stack([flood_t2["B04"], flood_t2["B03"], flood_t2["B02"]], axis=-1) / 3000.0

    axes[0, 0].imshow(np.clip(t1_rgb, 0, 1))
    axes[0, 0].set_title("1. Наводнение: Снимок ДО\nНормальное русло реки", fontsize=10, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(np.clip(t2_rgb, 0, 1))
    axes[0, 1].set_title("Снимок ПОСЛЕ\nРазлив паводковых вод", fontsize=10, fontweight="bold")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(mask_flood, cmap="Blues_r")
    axes[0, 2].set_title(f"Спектральная маска (ΔNDWI > 0.2)\nВодное зеркало: {stats_flood['total_area_hectares']} га", fontsize=10)
    axes[0, 2].axis("off")

    axes[0, 3].imshow(np.clip(t2_rgb, 0, 1))
    overlay = np.ma.masked_where(~mask_flood, mask_flood)
    axes[0, 3].imshow(overlay, cmap="cool", alpha=0.7)
    axes[0, 3].set_title(f"Класс: {stats_flood['category']} ({stats_flood['severity']})\n{stats_flood['description_ru'][:40]}...", fontsize=9, fontweight="bold", color="blue")
    axes[0, 3].axis("off")

    # -------------------------------------------------------------
    # 2. СЦЕНАРИЙ: ЛЕСНОЙ ПОЖАР / ГАРЬ (WILDFIRE)
    # -------------------------------------------------------------
    np.random.seed(202)
    fire_t1 = {
        "B02": np.random.normal(280, 20, (size, size)),
        "B03": np.random.normal(520, 30, (size, size)),
        "B04": np.random.normal(320, 25, (size, size)),
        "B08": np.random.normal(3400, 120, (size, size)),
        "B12": np.random.normal(750, 40, (size, size))
    }
    fire_t2 = {k: v.copy() for k, v in fire_t1.items()}
    fire_mask_gt = ((x - 140)**2 / (65**2) + (y - 120)**2 / (50**2)) <= 1
    fire_t2["B04"][fire_mask_gt] = 850   # Пепел/зола
    fire_t2["B08"][fire_mask_gt] = 950   # Обрушение NIR
    fire_t2["B12"][fire_mask_gt] = 2700  # Всплеск SWIR

    mask_fire, stats_fire = analyze_earth_changes(fire_t1, fire_t2)

    f_t1_rgb = np.stack([fire_t1["B04"], fire_t1["B03"], fire_t1["B02"]], axis=-1) / 3000.0
    f_t2_rgb = np.stack([fire_t2["B04"], fire_t2["B03"], fire_t2["B02"]], axis=-1) / 3000.0

    axes[1, 0].imshow(np.clip(f_t1_rgb, 0, 1))
    axes[1, 0].set_title("2. Пожар: Снимок ДО\nХвойный лесной массив", fontsize=10, fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(np.clip(f_t2_rgb, 0, 1))
    axes[1, 1].set_title("Снимок ПОСЛЕ\nТермопоражение / Зола", fontsize=10, fontweight="bold")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(mask_fire, cmap="copper")
    axes[1, 2].set_title(f"Спектральная маска (ΔNBR > 0.25)\nПлощадь гари: {stats_fire['total_area_hectares']} га", fontsize=10)
    axes[1, 2].axis("off")

    axes[1, 3].imshow(np.clip(f_t2_rgb, 0, 1))
    overlay = np.ma.masked_where(~mask_fire, mask_fire)
    axes[1, 3].imshow(overlay, cmap="hot", alpha=0.7)
    axes[1, 3].set_title(f"Класс: {stats_fire['category']} ({stats_fire['severity']})\n{stats_fire['description_ru'][:40]}...", fontsize=9, fontweight="bold", color="darkred")
    axes[1, 3].axis("off")

    # -------------------------------------------------------------
    # 3. СЦЕНАРИЙ: ОБЕЗЛЕСЕНИЕ / ВЫРУБКА (DEFORESTATION)
    # -------------------------------------------------------------
    np.random.seed(303)
    cut_t1 = {
        "B02": np.random.normal(260, 20, (size, size)),
        "B03": np.random.normal(530, 30, (size, size)),
        "B04": np.random.normal(310, 25, (size, size)),
        "B08": np.random.normal(3300, 110, (size, size)),
        "B12": np.random.normal(780, 40, (size, size))
    }
    cut_t2 = {k: v.copy() for k, v in cut_t1.items()}
    cut_mask_gt = (x > 40) & (x < 180) & (y > 60) & (y < 140)
    cut_t2["B04"][cut_mask_gt] = 1600  # Обнаженный сухой грунт
    cut_t2["B08"][cut_mask_gt] = 1500
    cut_t2["B12"][cut_mask_gt] = 1350

    mask_cut, stats_cut = analyze_earth_changes(cut_t1, cut_t2)

    c_t1_rgb = np.stack([cut_t1["B04"], cut_t1["B03"], cut_t1["B02"]], axis=-1) / 3000.0
    c_t2_rgb = np.stack([cut_t2["B04"], cut_t2["B03"], cut_t2["B02"]], axis=-1) / 3000.0

    axes[2, 0].imshow(np.clip(c_t1_rgb, 0, 1))
    axes[2, 0].set_title("3. Обезлесение: Снимок ДО\nДевственный лес (Тропики)", fontsize=10, fontweight="bold")
    axes[2, 0].axis("off")

    axes[2, 1].imshow(np.clip(c_t2_rgb, 0, 1))
    axes[2, 1].set_title("Снимок ПОСЛЕ\nУчасток сплошной вырубки", fontsize=10, fontweight="bold")
    axes[2, 1].axis("off")

    axes[2, 2].imshow(mask_cut, cmap="YlOrBr")
    axes[2, 2].set_title(f"Спектральная маска (ΔNDVI < -0.2)\nУтрачено: {stats_cut['total_area_hectares']} га", fontsize=10)
    axes[2, 2].axis("off")

    axes[2, 3].imshow(np.clip(c_t2_rgb, 0, 1))
    overlay = np.ma.masked_where(~mask_cut, mask_cut)
    axes[2, 3].imshow(overlay, cmap="autumn", alpha=0.7)
    axes[2, 3].set_title(f"Класс: {stats_cut['category']} ({stats_cut['severity']})\n{stats_cut['description_ru'][:40]}...", fontsize=9, fontweight="bold", color="brown")
    axes[2, 3].axis("off")

    # -------------------------------------------------------------
    # 4. СЦЕНАРИЙ: НЕЙРОСЕТЕВАЯ ДЕТЕКЦИЯ СТРОИТЕЛЬСТВА (BIT NEURAL)
    # -------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "checkpoints/best_multitask_deep.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/best_multitask_finetuned.pth"
    
    ckpt = torch.load(ckpt_path, map_location=device)
    vocab = ChangeCaptionVocab()
    vocab.word2idx = ckpt["vocab"]
    vocab.idx2word = {i: w for w, i in vocab.word2idx.items()}

    model = MultiTaskChangeModel(vocab_size=len(vocab), pretrained_backbone=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    p_a = "data/levir_mci/LEVIR-MCI-dataset/images/test/A/test_000004.png"
    p_b = "data/levir_mci/LEVIR-MCI-dataset/images/test/B/test_000004.png"
    img_a = Image.open(p_a).convert("RGB").resize((256, 256), Image.BILINEAR)
    img_b = Image.open(p_b).convert("RGB").resize((256, 256), Image.BILINEAR)

    t1_t = torch.from_numpy(np.array(img_a, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    t2_t = torch.from_numpy(np.array(img_b, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_mask_tensor, gen_tokens = model.predict(t1_t, t2_t, beam_size=3)

    pred_mask_neural = pred_mask_tensor.squeeze().cpu().numpy() > 0.5
    caption_en = vocab.decode(gen_tokens.squeeze().cpu().tolist())
    caption_ru = translate_change_caption(caption_en)

    axes[3, 0].imshow(img_a)
    axes[3, 0].set_title("4. Инфраструктура: Снимок ДО\nРастительность / Пустырь", fontsize=10, fontweight="bold")
    axes[3, 0].axis("off")

    axes[3, 1].imshow(img_b)
    axes[3, 1].set_title("Снимок ПОСЛЕ\nНовые коттеджи и дорога", fontsize=10, fontweight="bold")
    axes[3, 1].axis("off")

    axes[3, 2].imshow(pred_mask_neural, cmap="gray")
    axes[3, 2].set_title(f"Нейросетевая маска BIT (U-Net)\nПикселей изменений: {pred_mask_neural.sum()}", fontsize=10)
    axes[3, 2].axis("off")

    axes[3, 3].imshow(img_b)
    overlay = np.ma.masked_where(~pred_mask_neural, pred_mask_neural)
    axes[3, 3].imshow(overlay, cmap="autumn", alpha=0.65)
    axes[3, 3].set_title(f"Нейросеть BIT + Перевод:\n\"{caption_ru}\"", fontsize=9, fontweight="bold", color="darkgreen")
    axes[3, 3].axis("off")

    out_file = "disaster_intelligence_report.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Диагностический отчет сохранен в: {out_file}")

if __name__ == "__main__":
    generate_disaster_showcase()
