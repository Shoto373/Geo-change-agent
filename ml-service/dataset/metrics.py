import math
from collections import Counter
from typing import List, Dict, Tuple

def compute_iou_f1(pred_mask, gt_mask, threshold: float = 0.5):
    """Вычисляет IoU, F1, Precision, Recall для маски изменений."""
    pred = (pred_mask >= threshold).float()
    gt = (gt_mask >= 0.5).float()

    tp = (pred * gt).sum().item()
    fp = (pred * (1.0 - gt)).sum().item()
    fn = ((1.0 - pred) * gt).sum().item()

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)

    return {"iou": iou, "f1": f1, "precision": precision, "recall": recall}

def compute_ngram_matches(candidate: List[str], reference: List[str], n: int) -> Tuple[int, int]:
    """Подсчитывает совпадение n-грамм между кандидатом и эталоном."""
    if len(candidate) < n or len(reference) < n:
        return 0, max(len(candidate) - n + 1, 0)

    cand_ngrams = Counter([tuple(candidate[i:i+n]) for i in range(len(candidate) - n + 1)])
    ref_ngrams = Counter([tuple(reference[i:i+n]) for i in range(len(reference) - n + 1)])

    overlap = sum(min(count, ref_ngrams.get(ng, 0)) for ng, count in cand_ngrams.items())
    total = max(len(candidate) - n + 1, 0)
    return overlap, total

def compute_bleu(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    """
    Вычисляет классический BLEU (1, 2, 3, 4) с Brevity Penalty по корпусу.
    """
    p_n = [0.0, 0.0, 0.0, 0.0]
    total_cand_len = 0
    total_ref_len = 0

    for n in range(1, 5):
        num_matches = 0
        total_possible = 0
        for hyp_str, ref_str in zip(hypotheses, references):
            hyp = hyp_str.strip().lower().split()
            ref = ref_str.strip().lower().split()

            if n == 1:
                total_cand_len += len(hyp)
                total_ref_len += len(ref)

            matches, poss = compute_ngram_matches(hyp, ref, n)
            num_matches += matches
            total_possible += poss

        p_n[n - 1] = (num_matches + 1e-7) / (total_possible + 1e-7)

    # Brevity Penalty (штраф за слишком короткие предложения)
    if total_cand_len > total_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - float(total_ref_len + 1e-7) / float(total_cand_len + 1e-7))

    bleu1 = bp * p_n[0]
    bleu2 = bp * math.exp(0.5 * (math.log(p_n[0]) + math.log(p_n[1])))
    bleu3 = bp * math.exp((1/3) * sum(math.log(p_n[i]) for i in range(3)))
    bleu4 = bp * math.exp(0.25 * sum(math.log(p_n[i]) for i in range(4)))

    return {
        "bleu1": round(bleu1 * 100, 2),
        "bleu2": round(bleu2 * 100, 2),
        "bleu3": round(bleu3 * 100, 2),
        "bleu4": round(bleu4 * 100, 2)
    }

def compute_rouge_l(hypotheses: List[str], references: List[str]) -> float:
    """Вычисляет ROUGE-L на основе Longest Common Subsequence (LCS)."""
    scores = []
    for hyp_str, ref_str in zip(hypotheses, references):
        hyp = hyp_str.strip().lower().split()
        ref = ref_str.strip().lower().split()

        m, n = len(ref), len(hyp)
        if m == 0 or n == 0:
            scores.append(0.0)
            continue

        # Таблица динамического программирования для LCS
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if ref[i] == hyp[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
        lcs = dp[m][n]

        r_lcs = lcs / float(m)
        p_lcs = lcs / float(n)
        beta = 1.2
        f_lcs = ((1 + beta**2) * r_lcs * p_lcs) / (beta**2 * p_lcs + r_lcs + 1e-7)
        scores.append(f_lcs)

    return round(sum(scores) / len(scores) * 100, 2) if scores else 0.0
