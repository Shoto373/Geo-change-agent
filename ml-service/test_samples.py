import os
import sys
import json
import torch
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

from models.multitask_model import MultiTaskChangeModel
from dataset.levir_mci_dataset import ChangeCaptionVocab

def test_inference_suite():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = "checkpoints/best_multitask_model.pth"
    ckpt = torch.load(ckpt_path, map_location=device)
    
    vocab = ChangeCaptionVocab()
    vocab.word2idx = ckpt["vocab"]
    vocab.idx2word = {i: w for w, i in vocab.word2idx.items()}

    model = MultiTaskChangeModel(vocab_size=len(vocab), pretrained_backbone=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load annotations to see Ground Truth
    with open("data/levir_mci/LEVIR-MCI-dataset/LevirCCcaptions.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    ann_dict = {x["filename"]: [s["raw"] for s in x["sentences"]] for x in meta["images"]}

    test_samples = [
        "test_000004.png",
        "test_000017.png",
        "test_000018.png",
        "test_000019.png",
        "test_000020.png",
        "test_000001.png"
    ]

    print("\n" + "=" * 90)
    print(f"{'Sample':<18} | {'Mask Pixels':<12} | {'Change %':<8} | {'Generated Caption':<35}")
    print("=" * 90)

    for sample in test_samples:
        p_a = f"data/levir_mci/LEVIR-MCI-dataset/images/test/A/{sample}"
        p_b = f"data/levir_mci/LEVIR-MCI-dataset/images/test/B/{sample}"
        if not os.path.exists(p_a):
            continue

        img_a = Image.open(p_a).convert("RGB").resize((256, 256), Image.BILINEAR)
        img_b = Image.open(p_b).convert("RGB").resize((256, 256), Image.BILINEAR)

        t1 = torch.from_numpy(np.array(img_a, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        t2 = torch.from_numpy(np.array(img_b, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_mask, gen_tokens = model.predict(t1, t2, beam_size=3, max_caption_len=30)

        mask_np = pred_mask.squeeze().cpu().numpy()
        n_pixels = int(mask_np.sum())
        pct = (n_pixels / (256 * 256)) * 100
        caption = vocab.decode(gen_tokens.squeeze().cpu().tolist())
        gt_cap = ann_dict.get(sample, ["N/A"])[0]

        print(f"{sample:<18} | {n_pixels:<12} | {pct:>6.1f}% | {caption}")
        print(f"   [Ground Truth]: {gt_cap.strip()}")
        print("-" * 90)

if __name__ == "__main__":
    test_inference_suite()
