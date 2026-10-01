import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from models.backbone import SiameseEncoder
from models.transformer import (
    SpatialTokenExtractor, 
    CrossTemporalTransformerBlock, 
    TokenToSpatialDecoder
)

class BitemporalImageTransformer(nn.Module):
    """
    Полная архитектура BIT (Bitemporal Image Transformer) для детекции изменений.
    Поддерживает мультиспектральные снимки Sentinel-2 (например, 4 канала: B, G, R, NIR).
    """
    def __init__(
        self, 
        in_channels: int = 4, 
        base_channels: int = 64, 
        num_tokens: int = 16,
        transformer_depth: int = 2
    ):
        super().__init__()
        # 1. Сиамский бэкбон
        self.encoder = SiameseEncoder(in_channels=in_channels, base_channels=base_channels)
        feat_dim = self.encoder.out_channels # base_channels * 4

        # 2. Семантический токенизатор
        self.tokenizer = SpatialTokenExtractor(in_channels=feat_dim, num_tokens=num_tokens)

        # 3. Кросс-временной трансформер (стек блоков)
        self.transformer_layers = nn.ModuleList([
            CrossTemporalTransformerBlock(dim=feat_dim, num_heads=4, mlp_ratio=2.0)
            for _ in range(transformer_depth)
        ])

        # 4. Проекция токенов обратно в пространство
        self.decoder_spatial = TokenToSpatialDecoder(token_dim=feat_dim, feat_channels=feat_dim)

        # 5. Классификационная голова (Upsample до исходного разрешения H, W)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(feat_dim, feat_dim // 2, kernel_size=4, stride=2, padding=1, bias=False), # H/8
            nn.BatchNorm2d(feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat_dim // 2, feat_dim // 4, kernel_size=4, stride=2, padding=1, bias=False), # H/4
            nn.BatchNorm2d(feat_dim // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat_dim // 4, feat_dim // 8, kernel_size=4, stride=2, padding=1, bias=False), # H/2
            nn.BatchNorm2d(feat_dim // 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat_dim // 8, 32, kernel_size=4, stride=2, padding=1, bias=False), # H
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1) # 1 канал логитов изменений
        )

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """
        :param t1: Тензор первой даты (B, C, H, W)
        :param t2: Тензор второй даты (B, C, H, W)
        :return: Логиты изменений (B, 1, H, W)
        """
        # Шаг 1: Извлечение признаков сиамским энкодером
        f1, f2 = self.encoder(t1, t2) # (B, feat_dim, H/16, W/16)

        # Шаг 2: Извлечение семантических токенов
        tokens_t1 = self.tokenizer(f1) # (B, L, feat_dim)
        tokens_t2 = self.tokenizer(f2) # (B, L, feat_dim)

        # Шаг 3: Кросс-временное внимание между T1 и T2
        for layer in self.transformer_layers:
            tokens_t1, tokens_t2 = layer(tokens_t1, tokens_t2)

        # Шаг 4: Разность контекстных токенов
        token_diff = torch.abs(tokens_t2 - tokens_t1)

        # Шаг 5: Пространственная разность фичей + инъекция токенов
        feat_diff = torch.abs(f2 - f1)
        refined_feat = self.decoder_spatial(token_diff, feat_diff)

        # Шаг 6: Восстановление исходного разрешения и классификация пикселей
        logits = self.upsample(refined_feat)
        return logits

    @torch.no_grad()
    def predict_mask(self, t1: torch.Tensor, t2: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Инференс: выдает бинарную маску изменений (0 или 1)."""
        self.eval()
        logits = self.forward(t1, t2)
        probs = torch.sigmoid(logits)
        return (probs >= threshold).float()
