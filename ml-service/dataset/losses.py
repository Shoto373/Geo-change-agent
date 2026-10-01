import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Dice Loss для бинарной сегментации редких изменений (борьба с дисбалансом классов).
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice

class ChangeDetectionCombinedLoss(nn.Module):
    """
    Комбинированная функция потерь: BCEWithLogits + Dice Loss.
    Идеальна для задач Change Detection, где изменения занимают малую долю площади.
    """
    def __init__(self, bce_weight: float = 0.5, pos_weight: float = 2.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.pos_weight = torch.tensor([pos_weight])
        self.dice_loss = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_w = self.pos_weight.to(logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_w)
        dice = self.dice_loss(logits, targets)
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice
