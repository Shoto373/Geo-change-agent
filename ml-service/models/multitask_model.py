import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from models.backbone import PretrainedSiameseBackbone
from models.joint_fusion import BITJointFusion
from models.caption_decoder import CaptionDecoder

class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UNetMaskDecoder(nn.Module):
    """
    Полноценный U-Net декодер с многомасштабными skip-connections.
    Объединяет глубокие признаки изменений от BIT с текстурными деталями высокого разрешения (H/4, H/8, H/16).
    Обеспечивает четкие геометрические границы объектов (здания, вырубки, дороги).
    """
    def __init__(self, in_channels: int = 256):
        super().__init__()
        # Вход: H/32 (256 каналов)
        self.up4 = nn.ConvTranspose2d(in_channels, 128, kernel_size=2, stride=2) # -> H/16
        self.conv4 = ConvBlock(128 + 256, 128) # + skip c3 (256 каналов)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # -> H/8
        self.conv3 = ConvBlock(64 + 128, 64) # + skip c2 (128 каналов)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2) # -> H/4
        self.conv2 = ConvBlock(32 + 64, 32) # + skip c1 (64 каналов)

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=4) # -> H
        self.out_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, diff_feat: torch.Tensor, skips_1: Tuple, skips_2: Tuple) -> torch.Tensor:
        # skips: (c1: 64, c2: 128, c3: 256)
        c1_1, c2_1, c3_1 = skips_1
        c1_2, c2_2, c3_2 = skips_2

        # Разностные skip-connections (симметрично между T1 и T2)
        diff_c3 = torch.abs(c3_2 - c3_1)
        diff_c2 = torch.abs(c2_2 - c2_1)
        diff_c1 = torch.abs(c1_2 - c1_1)

        # H/32 -> H/16
        x = self.up4(diff_feat)
        x = torch.cat([x, diff_c3], dim=1)
        x = self.conv4(x)

        # H/16 -> H/8
        x = self.up3(x)
        x = torch.cat([x, diff_c2], dim=1)
        x = self.conv3(x)

        # H/8 -> H/4
        x = self.up2(x)
        x = torch.cat([x, diff_c1], dim=1)
        x = self.conv2(x)

        # H/4 -> H (исходное разрешение)
        x = self.up1(x)
        logits = self.out_conv(x)
        return logits

class MultiTaskChangeModel(nn.Module):
    """
    Академически строгая Multi-Task архитектура:
      1. Сиамский ImageNet-предобученный бэкбон (ResNet-18).
      2. BIT Joint Self-Attention Fusion с пиксельным Cross-Attention.
      3. U-Net Mask Decoder с разностными skip-connections (Change Detection).
      4. Autoregressive Transformer Decoder с Beam Search (Change Captioning).
    """
    def __init__(
        self,
        backbone_name: str = "resnet18",
        in_channels: int = 3,
        embed_dim: int = 256,
        num_tokens: int = 16,
        fusion_depth: int = 4,
        vocab_size: int = 1500,
        caption_layers: int = 3,
        pretrained_backbone: bool = True
    ):
        super().__init__()
        # 1. Сиамский энкодер
        self.backbone = PretrainedSiameseBackbone(
            model_name=backbone_name,
            in_channels=in_channels,
            pretrained=pretrained_backbone,
            out_dim=embed_dim
        )

        # 2. BIT Joint Self-Attention Fusion
        self.joint_fusion = BITJointFusion(
            embed_dim=embed_dim,
            num_tokens=num_tokens,
            depth=fusion_depth,
            num_heads=4
        )

        # 3. Ветка А: Маска изменений (U-Net с разностными скипами)
        self.mask_decoder = UNetMaskDecoder(in_channels=embed_dim)

        # 4. Ветка Б: Текстовый декодер описания
        self.caption_decoder = CaptionDecoder(
            vocab_size=vocab_size,
            d_model=embed_dim,
            num_layers=caption_layers
        )

    def freeze_backbone(self):
        self.backbone.freeze()

    def unfreeze_backbone(self):
        self.backbone.unfreeze()

    def forward(
        self, 
        t1: torch.Tensor, 
        t2: torch.Tensor, 
        caption_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Прямой проход при обучении.
        """
        # Шаг 1: Сиамское извлечение признаков и многомасштабных скипов
        (f1, f2), (skips_1, skips_2) = self.backbone(t1, t2)

        # Шаг 2: BIT Joint Self-Attention + Пиксельный Cross-Attention
        spatial_diff, joint_context = self.joint_fusion(f1, f2)

        # Шаг 3: Генерация маски (U-Net)
        mask_logits = self.mask_decoder(spatial_diff, skips_1, skips_2)

        # Шаг 4: Генерация логитов текста (если переданы таргет-слова)
        caption_logits = None
        if caption_ids is not None:
            caption_logits = self.caption_decoder(joint_context, caption_ids)

        return mask_logits, caption_logits

    @torch.no_grad()
    def predict(
        self, 
        t1: torch.Tensor, 
        t2: torch.Tensor, 
        mask_threshold: float = 0.5,
        beam_size: int = 3,
        max_caption_len: int = 35
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Инференс: возвращает бинарную маску (ГДЕ) и сгенерированные через Beam Search слова (ЧТО).
        """
        self.eval()
        (f1, f2), (skips_1, skips_2) = self.backbone(t1, t2)
        spatial_diff, joint_context = self.joint_fusion(f1, f2)

        # 1. Маска
        mask_logits = self.mask_decoder(spatial_diff, skips_1, skips_2)
        pred_mask = (torch.sigmoid(mask_logits) >= mask_threshold).float()

        # 2. Текст через Beam Search
        generated_token_ids = self.caption_decoder.beam_search_decode(
            joint_context, 
            beam_size=beam_size, 
            max_len=max_caption_len
        )

        return pred_mask, generated_token_ids
