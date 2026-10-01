import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, List

class BitemporalChangeDataset(Dataset):
    """
    Универсальный PyTorch Dataset для задач Change Detection (OSCD, PRODES или синтетические пары).
    Гарантирует синхронные аугментации для пары T1, T2 и эталонной маски.
    """
    def __init__(
        self,
        samples: Optional[List[dict]] = None,
        num_synthetic_samples: int = 200,
        img_size: int = 256,
        in_channels: int = 4,
        is_train: bool = True
    ):
        self.samples = samples or []
        self.num_synthetic_samples = num_synthetic_samples
        self.img_size = img_size
        self.in_channels = in_channels
        self.is_train = is_train

        # Если реальный датасет еще не скачан, создаем синтетические пары лесных вырубок для отладки
        self.use_synthetic = len(self.samples) == 0

    def __len__(self) -> int:
        if self.use_synthetic:
            return self.num_synthetic_samples
        return len(self.samples)

    def _generate_synthetic_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Генерирует битемпоральную пару снимков (C, H, W) и маску (1, H, W)."""
        np.random.seed(idx if not self.is_train else None)
        h = w = self.img_size

        # Фоновый лес на T1: высокий NIR (канал 3), умеренный Red (канал 2)
        # Каналы: 0: Blue, 1: Green, 2: Red, 3: NIR
        t1 = np.zeros((self.in_channels, h, w), dtype=np.float32)
        t1[0] = np.random.normal(0.04, 0.01, (h, w)) # Blue
        t1[1] = np.random.normal(0.08, 0.015, (h, w)) # Green
        t1[2] = np.random.normal(0.05, 0.01, (h, w)) # Red
        t1[3] = np.random.normal(0.40, 0.04, (h, w)) # NIR

        t2 = t1.copy()
        mask = np.zeros((1, h, w), dtype=np.float32)

        # С вероятностью 70% на снимке есть вырубка леса
        if np.random.rand() > 0.3:
            num_cuts = np.random.randint(1, 4)
            for _ in range(num_cuts):
                cx, cy = np.random.randint(40, w - 40), np.random.randint(40, h - 40)
                rx, ry = np.random.randint(15, 45), np.random.randint(15, 35)
                y, x = np.ogrid[:h, :w]
                cut_region = ((x - cx)**2 / rx**2 + (y - cy)**2 / ry**2) <= 1
                
                # Обнаженная почва на T2: падение NIR, рост Red
                t2[2, cut_region] = np.random.normal(0.22, 0.02, np.sum(cut_region))
                t2[3, cut_region] = np.random.normal(0.15, 0.02, np.sum(cut_region))
                mask[0, cut_region] = 1.0

        # Сезонный шум освещения на всем снимке T2
        t2 += np.random.normal(0.0, 0.01, t2.shape)

        return np.clip(t1, 0, 1), np.clip(t2, 0, 1), mask

    def _apply_augmentations(
        self, 
        t1: np.ndarray, 
        t2: np.ndarray, 
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Синхронные пространственные аугментации для пары растров и маски."""
        if random.random() > 0.5:
            t1 = np.flip(t1, axis=-1).copy()
            t2 = np.flip(t2, axis=-1).copy()
            mask = np.flip(mask, axis=-1).copy()

        if random.random() > 0.5:
            t1 = np.flip(t1, axis=-2).copy()
            t2 = np.flip(t2, axis=-2).copy()
            mask = np.flip(mask, axis=-2).copy()

        # Случайный поворот на 90, 180, 270 градусов
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            t1 = np.rot90(t1, k, axes=(-2, -1)).copy()
            t2 = np.rot90(t2, k, axes=(-2, -1)).copy()
            mask = np.rot90(mask, k, axes=(-2, -1)).copy()

        return t1, t2, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_synthetic:
            t1, t2, mask = self._generate_synthetic_pair(idx)
        else:
            # Загрузка из диска для OSCD / реальных тайлов
            item = self.samples[idx]
            t1 = np.load(item["t1_path"]).astype(np.float32)
            t2 = np.load(item["t2_path"]).astype(np.float32)
            mask = np.load(item["mask_path"]).astype(np.float32)

        if self.is_train:
            t1, t2, mask = self._apply_augmentations(t1, t2, mask)

        return (
            torch.from_numpy(t1).float(),
            torch.from_numpy(t2).float(),
            torch.from_numpy(mask).float()
        )
