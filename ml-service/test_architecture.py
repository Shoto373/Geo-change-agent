"""
Комплексный Unit-тест архитектуры MultiTaskChangeModel:
Проверяет формы тензоров (Tensor Shapes), поток градиентов (Gradient Flow)
и инференс Beam Search для Change Detection + Change Captioning.
"""

import sys
import torch

from models.multitask_model import MultiTaskChangeModel
from dataset.metrics import compute_bleu, compute_rouge_l, compute_iou_f1

def run_architectural_test():
    print("=" * 70)
    print("  ПРОВЕРКА АРХИТЕКТУРЫ: MultiTaskChangeModel (BIT + ResNet + Beam Search)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1] Устройство проверки: {device}")

    # Инициализация модели
    b, c, h, w = 2, 3, 256, 256
    vocab_size = 500
    caption_len = 15

    model = MultiTaskChangeModel(
        backbone_name="resnet18",
        in_channels=c,
        embed_dim=256,
        num_tokens=16,
        fusion_depth=2,
        vocab_size=vocab_size,
        caption_layers=2,
        pretrained_backbone=False # Для быстрого теста без скачивания весов
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[2] Параметры модели: всего {total_params / 1e6:.2f} млн (обучаемых: {trainable_params / 1e6:.2f} млн)")

    # 1. Проверка прямого прохода при обучении (Training Forward)
    print("\n[3] Тестирование Training Forward Pass...")
    t1 = torch.randn(b, c, h, w, device=device)
    t2 = torch.randn(b, c, h, w, device=device)
    caption_ids = torch.randint(4, vocab_size, (b, caption_len), device=device)

    mask_logits, caption_logits = model(t1, t2, caption_ids)

    print(f"  Вход T1:            {tuple(t1.shape)}")
    print(f"  Вход T2:            {tuple(t2.shape)}")
    print(f"  Выход Маска:        {tuple(mask_logits.shape)} (Ожидается: {(b, 1, h, w)})")
    print(f"  Выход Текст:        {tuple(caption_logits.shape)} (Ожидается: {(b, caption_len, vocab_size)})")

    assert mask_logits.shape == (b, 1, h, w), f"Ошибка размерности маски: {mask_logits.shape}"
    assert caption_logits.shape == (b, caption_len, vocab_size), f"Ошибка размерности текста: {caption_logits.shape}"
    print("  -> Все формы тензоров идеально совпали!")

    # 2. Проверка потока градиентов (Gradient Flow Check)
    print("\n[4] Тестирование обратного прохода (Backward Pass)...")
    target_mask = torch.randint(0, 2, (b, 1, h, w), device=device).float()
    target_caption = torch.randint(0, vocab_size, (b, caption_len), device=device)

    loss_mask = torch.nn.functional.binary_cross_entropy_with_logits(mask_logits, target_mask)
    loss_text = torch.nn.functional.cross_entropy(caption_logits.view(-1, vocab_size), target_caption.view(-1))
    total_loss = loss_mask + loss_text
    total_loss.backward()

    # Проверка градиентов по компонентам
    grad_checks = {
        "Backbone (conv1)": model.backbone.conv1.weight.grad is not None,
        "BIT Joint Fusion": model.joint_fusion.time_embed.grad is not None,
        "Mask Decoder": model.mask_decoder.out_conv.weight.grad is not None,
        "Caption Decoder": model.caption_decoder.fc_out.weight.grad is not None,
    }
    for name, has_grad in grad_checks.items():
        status = "OK" if has_grad else "FAIL (нет градиента)"
        print(f"  Градиент {name:20s}: {status}")
        assert has_grad, f"Разрыв графа градиентов в {name}!"

    # 3. Проверка инференса с Beam Search
    print("\n[5] Тестирование Inference с Beam Search (beam_size=3)...")
    pred_mask, gen_tokens = model.predict(t1, t2, mask_threshold=0.5, beam_size=3, max_caption_len=20)
    print(f"  Сгенерированная маска: {tuple(pred_mask.shape)}, значения в [{pred_mask.min().item()}, {pred_mask.max().item()}]")
    print(f"  Сгенерированные токены: {tuple(gen_tokens.shape)}, сэмпл: {gen_tokens[0].cpu().tolist()[:10]}...")

    # 4. Проверка метрик (BLEU, ROUGE-L, IoU)
    print("\n[6] Тестирование академических метрик (BLEU, ROUGE, IoU)...")
    hyps = ["a new road and two houses were built in the forest", "the trees were removed"]
    refs = ["a new road and three houses were built in the green forest", "trees were cleared"]
    bleu_scores = compute_bleu(hyps, refs)
    rouge_score = compute_rouge_l(hyps, refs)
    iou_scores = compute_iou_f1(pred_mask, target_mask)

    print(f"  BLEU-1..4: {bleu_scores}")
    print(f"  ROUGE-L:   {rouge_score:.2f}%")
    print(f"  Mask IoU:  {iou_scores['iou']:.4f} | F1: {iou_scores['f1']:.4f}")

    print("\n" + "=" * 70)
    print("  РЕЗУЛЬТАТ: Все математические проверки и градиентные потоки пройдены успешно!")
    print("=" * 70)

if __name__ == "__main__":
    run_architectural_test()
