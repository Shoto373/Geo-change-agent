import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SpatialTokenizer(nn.Module):
    """
    Семантический токенизатор BIT (Chen et al., IEEE TGRS):
    Проецирует пространственную карту признаков X (B, C, H, W) в L семантических концепт-токенов (B, L, C).
    """
    def __init__(self, in_channels: int, num_tokens: int = 16):
        super().__init__()
        self.num_tokens = num_tokens
        # Свертка 1x1 генерирует L карт вероятностей принадлежности пикселей к концептам
        self.conv_attn = nn.Conv2d(in_channels, num_tokens, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # (B, L, H*W)
        spatial_weights = self.conv_attn(x).view(b, self.num_tokens, h * w)
        spatial_weights = F.softmax(spatial_weights, dim=-1)

        # (B, C, H*W)
        feats = x.view(b, c, h * w)

        # Вычисляем токены: W * X^T -> (B, L, C)
        tokens = torch.bmm(spatial_weights, feats.transpose(1, 2))
        return tokens

class PixelToTokenCrossProjector(nn.Module):
    """
    Обратная проекция токенов в пиксельное пространство (Pixel-Level Cross Attention):
    Каждый пиксель карты признаков (Query) сопоставляется с обогащенными токенами изменений (Key, Value).
    Это обеспечивает точечную пространственную локализацию без усреднения и размытия.
    """
    def __init__(self, in_channels: int, num_heads: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        self.q_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.v_proj = nn.Linear(in_channels, in_channels)
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, spatial_feat: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """
        :param spatial_feat: Пространственная карта признаков (B, C, H, W)
        :param tokens: Обогащенные токены изменений (B, L, C)
        :return: Восстановленная карта признаков изменений (B, C, H, W)
        """
        b, c, h, w = spatial_feat.shape
        hw = h * w
        l = tokens.size(1)

        # Query: из пространственных пикселей (B, num_heads, HW, head_dim)
        q = self.q_proj(spatial_feat).view(b, self.num_heads, self.head_dim, hw).permute(0, 1, 3, 2)

        # Key, Value: из токенов (B, num_heads, L, head_dim)
        k = self.k_proj(tokens).view(b, l, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(tokens).view(b, l, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Внимание: (B, num_heads, HW, L)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)

        # Контекст: (B, num_heads, HW, head_dim) -> (B, C, H, W)
        ctx = torch.matmul(attn, v).permute(0, 1, 3, 2).contiguous().view(b, c, h, w)
        out = self.out_proj(ctx)

        # Residual connection
        return spatial_feat + out

class BITJointFusion(nn.Module):
    """
    Канонический BIT Joint Self-Attention Fusion:
    1. Токенизация F1 и F2 в L семантических токенов.
    2. Добавление временных маркеров (Time Embeddings).
    3. Joint Self-Attention: все 2*L токенов взаимодействуют в едином трансформере.
    4. Обратная Cross-Attention проекция в пространство пикселей.
    5. Вычисление честной разности изменений.
    """
    def __init__(
        self, 
        embed_dim: int = 256, 
        num_tokens: int = 16, 
        depth: int = 4, 
        num_heads: int = 4
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.tokenizer = SpatialTokenizer(in_channels=embed_dim, num_tokens=num_tokens)

        # Временные метки для T1 и T2
        self.time_embed = nn.Parameter(torch.randn(1, 2, embed_dim) * 0.02)

        # Transformer Encoder для Joint Attention над 2*L токенами
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Обратные проекторы токенов в пиксели для T1 и T2
        self.projector1 = PixelToTokenCrossProjector(in_channels=embed_dim, num_heads=num_heads)
        self.projector2 = PixelToTokenCrossProjector(in_channels=embed_dim, num_heads=num_heads)

    def forward(
        self, 
        f1: torch.Tensor, 
        f2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param f1: Пространственные признаки снимка ДО (B, C, H, W)
        :param f2: Пространственные признаки снимка ПОСЛЕ (B, C, H, W)
        :return: (spatial_change_features, joint_tokens_for_captioning)
        """
        # 1. Токенизация
        t1_tok = self.tokenizer(f1) # (B, L, C)
        t2_tok = self.tokenizer(f2) # (B, L, C)

        # 2. Добавление Time Embedding
        t1_tok = t1_tok + self.time_embed[:, 0:1, :]
        t2_tok = t2_tok + self.time_embed[:, 1:2, :]

        # 3. Объединение в общий пул (B, 2*L, C)
        joint_pool = torch.cat([t1_tok, t2_tok], dim=1)

        # 4. Joint Self-Attention: все токены обоих снимков взаимодействуют одновременно
        joint_context = self.transformer(joint_pool)

        # Разделяем обогащенные токены
        ctx_t1 = joint_context[:, :self.num_tokens, :]
        ctx_t2 = joint_context[:, self.num_tokens:, :]

        # 5. Обратная проекция в пиксели через Cross-Attention
        refined_f1 = self.projector1(f1, ctx_t1)
        refined_f2 = self.projector2(f2, ctx_t2)

        # 6. Итоговая разностная карта изменений (абсолютная разность в пиксельном пространстве)
        spatial_diff = torch.abs(refined_f2 - refined_f1)

        # Возвращаем:
        # - spatial_diff: для масочного декодера (Segmentation)
        # - joint_context: для текстового трансформера (Captioning)
        return spatial_diff, joint_context
