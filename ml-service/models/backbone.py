import logging
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights
from typing import Tuple, Optional

logger = logging.getLogger("backbone")

class PretrainedSiameseBackbone(nn.Module):
    """
    Сиамский энкодер на базе предобученного ResNet (ImageNet).
    
    Особенности:
      1. Загружает официальные веса ImageNet (ResNet18 или ResNet34).
      2. При работе с 4 каналами (RGB + NIR) инициализирует 4-й канал 
         средним значением весов RGB фильтров (стандартный трюк в Remote Sensing).
      3. Возвращает многомасштабные признаки (skip-connections) для масочного декодера:
         - c1: H/4,  W/4  (64 канала)
         - c2: H/8,  W/8  (128 каналов)
         - c3: H/16, W/16 (256 каналов)
         - c4: H/32, W/32 (512 каналов)
      4. Позволяет замораживать бэкбон на первые эпохи (warm-up).
    """
    def __init__(
        self, 
        model_name: str = "resnet18", 
        in_channels: int = 3, 
        pretrained: bool = True,
        out_dim: int = 256
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim

        # 1. Загрузка ResNet с предобученными весами
        if model_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base_model = resnet18(weights=weights)
        elif model_name == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            base_model = resnet34(weights=weights)
        else:
            raise ValueError(f"Неподдерживаемый бэкбон: {model_name}. Используйте resnet18 или resnet34.")

        # 2. Адаптация первого слоя conv1 под нестандартное число каналов (например, 4 для Sentinel-2)
        if in_channels != 3:
            old_conv = base_model.conv1
            new_conv = nn.Conv2d(
                in_channels, 
                old_conv.out_channels, 
                kernel_size=old_conv.kernel_size, 
                stride=old_conv.stride, 
                padding=old_conv.padding, 
                bias=old_conv.bias is not None
            )
            with torch.no_grad():
                # Копируем веса для первых 3 каналов (RGB)
                new_conv.weight[:, :3, :, :] = old_conv.weight
                # 4-й канал (NIR) инициализируем средним по RGB
                if in_channels > 3:
                    mean_weights = old_conv.weight.mean(dim=1, keepdim=True)
                    for c in range(3, in_channels):
                        new_conv.weight[:, c:c+1, :, :] = mean_weights
            base_model.conv1 = new_conv

        # 3. Декомпозиция слоев ResNet
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1 # H/4,  64 каналов
        self.layer2 = base_model.layer2 # H/8,  128 каналов
        self.layer3 = base_model.layer3 # H/16, 256 каналов
        self.layer4 = base_model.layer4 # H/32, 512 каналов

        # Проекционный слой в единую размерность эмбеддинга
        self.proj = nn.Sequential(
            nn.Conv2d(512, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def freeze(self):
        """Замораживает веса бэкбона (для фазы разминки голов)."""
        for param in self.parameters():
            param.requires_grad = False
        # Проекционный слой оставляем обучаемым
        for param in self.proj.parameters():
            param.requires_grad = True

    def unfreeze(self):
        """Размораживает все веса для сквозного fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Извлекает глубокие признаки и многомасштабные skip-connections.
        :return: (top_feat: BxCx(H/32)x(W/32), skips: (c1, c2, c3))
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c0 = x
        x = self.maxpool(x)

        c1 = self.layer1(x)  # H/4,  64
        c2 = self.layer2(c1) # H/8,  128
        c3 = self.layer3(c2) # H/16, 256
        c4 = self.layer4(c3) # H/32, 512

        top_feat = self.proj(c4) # H/32, out_dim
        return top_feat, (c1, c2, c3)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor):
        """
        Сиамский прогон снимков T1 и T2 через разделяемые веса (Shared Weights).
        """
        f1, skips_1 = self.extract_features(t1)
        f2, skips_2 = self.extract_features(t2)
        return (f1, f2), (skips_1, skips_2)
