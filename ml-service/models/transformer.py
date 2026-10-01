import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTokenExtractor(nn.Module):
    """
    Преобразует карту признаков (B, C, H, W) в компактный набор пространственно-семантических токенов (B, L, C).
    """
    def __init__(self, in_channels: int, num_tokens: int = 16):
        super().__init__()
        self.num_tokens = num_tokens
        # Свертка генерирует веса внимания для каждого токена
        self.conv_attn = nn.Conv2d(in_channels, num_tokens, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # (B, L, H*W)
        spatial_weights = self.conv_attn(x).view(b, self.num_tokens, h * w)
        spatial_weights = F.softmax(spatial_weights, dim=-1)

        # (B, C, H*W)
        features = x.view(b, c, h * w)

        # Вычисляем токены как взвешенную сумму фичей: (B, L, C)
        tokens = torch.bmm(spatial_weights, features.transpose(1, 2))
        return tokens

class CrossTemporalTransformerBlock(nn.Module):
    """
    Моделирует контекстное взаимодействие между токенами времени T1 и T2 через Multi-Head Attention.
    """
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, tokens_t1: torch.Tensor, tokens_t2: torch.Tensor):
        """
        Cross-attention: токены T1 обогащаются информацией из T2, и наоборот.
        """
        # T1 queries, T2 keys/values
        t1_norm = self.norm1(tokens_t1)
        t2_norm = self.norm1(tokens_t2)

        attn_out_1, _ = self.cross_attn(query=t1_norm, key=t2_norm, value=t2_norm)
        t1 = tokens_t1 + attn_out_1
        t1 = t1 + self.mlp(self.norm2(t1))

        # T2 queries, T1 keys/values
        attn_out_2, _ = self.cross_attn(query=t2_norm, key=t1_norm, value=t1_norm)
        t2 = tokens_t2 + attn_out_2
        t2 = t2 + self.mlp(self.norm2(t2))

        return t1, t2

class TokenToSpatialDecoder(nn.Module):
    """
    Проецирует обогащенные токены изменений обратно в пространственную карту признаков.
    """
    def __init__(self, token_dim: int, feat_channels: int):
        super().__init__()
        self.token_proj = nn.Linear(token_dim, feat_channels)
        self.refine_conv = nn.Sequential(
            nn.Conv2d(feat_channels * 2, feat_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, token_diff: torch.Tensor, spatial_feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = spatial_feat.shape
        # token_diff: (B, L, D) -> усредняем по токенам или проецируем
        token_ctx = token_diff.mean(dim=1) # (B, D)
        token_ctx = self.token_proj(token_ctx).unsqueeze(-1).unsqueeze(-1) # (B, C, 1, 1)
        token_ctx = token_ctx.expand(-1, -1, h, w)

        combined = torch.cat([spatial_feat, token_ctx], dim=1)
        return self.refine_conv(combined)
