import os
import json
import random
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional

class ChangeCaptionVocab:
    """
    Словарь для токенизации и декодирования предложений Change Captioning.
    """
    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.word2idx = {
            self.PAD_TOKEN: 0,
            self.BOS_TOKEN: 1,
            self.EOS_TOKEN: 2,
            self.UNK_TOKEN: 3
        }
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def build_vocab(self, sentences: List[str]):
        counter = Counter()
        for s in sentences:
            tokens = s.lower().strip().replace(".", " .").replace(",", " ,").split()
            counter.update(tokens)

        idx = len(self.word2idx)
        for word, count in counter.items():
            if count >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def encode(self, sentence: str, max_len: int = 40) -> List[int]:
        tokens = sentence.lower().strip().replace(".", " .").replace(",", " ,").split()
        ids = [self.word2idx[self.BOS_TOKEN]]
        for t in tokens:
            ids.append(self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]))
        ids.append(self.word2idx[self.EOS_TOKEN])

        # Padding
        if len(ids) < max_len:
            ids.extend([self.word2idx[self.PAD_TOKEN]] * (max_len - len(ids)))
        else:
            ids = ids[:max_len - 1] + [self.word2idx[self.EOS_TOKEN]]
        return ids

    def decode(self, token_ids: List[int]) -> str:
        words = []
        for tid in token_ids:
            if tid == self.word2idx[self.BOS_TOKEN] or tid == self.word2idx[self.PAD_TOKEN]:
                continue
            if tid == self.word2idx[self.EOS_TOKEN]:
                break
            words.append(self.idx2word.get(tid, self.UNK_TOKEN))
        return " ".join(words).replace(" .", ".").replace(" ,", ",")

    def __len__(self):
        return len(self.word2idx)

class LevirMCIDataset(Dataset):
    """
    Датасет для LEVIR-MCI (Multi-task Change Interpretation):
    Возвращает кортеж: (Снимок T1, Снимок T2, Маска, Токены описания).
    """
    DEFAULT_VOCAB_PHRASES = [
        "some trees were removed and small buildings were constructed .",
        "the dense forest canopy was cleared for agricultural land .",
        "a new road was built through the green field .",
        "several large warehouses appeared in the vacant ground .",
        "residential houses were built where there was bare soil .",
        "no significant changes occurred in this rural area ."
    ]

    def __init__(
        self,
        data_dir: Optional[str] = None,
        split: str = "train",
        vocab: Optional[ChangeCaptionVocab] = None,
        img_size: int = 256,
        num_synthetic: int = 150
    ):
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.num_synthetic = num_synthetic

        # Инициализация словаря
        if vocab is None:
            self.vocab = ChangeCaptionVocab()
            self.vocab.build_vocab(self.DEFAULT_VOCAB_PHRASES)
        else:
            self.vocab = vocab

        self.samples = []
        if data_dir and os.path.exists(os.path.join(data_dir, f"{split}.json")):
            with open(os.path.join(data_dir, f"{split}.json"), "r", encoding="utf-8") as f:
                self.samples = json.load(f)

        self.use_synthetic = len(self.samples) == 0

    def __len__(self) -> int:
        if self.use_synthetic:
            return self.num_synthetic
        return len(self.samples)

    def _generate_synthetic(self, idx: int):
        h = w = self.img_size
        t1 = np.random.normal(0.4, 0.05, (3, h, w)).astype(np.float32)
        t2 = t1.copy()
        mask = np.zeros((1, h, w), dtype=np.float32)

        has_change = (idx % 4 != 0)
        if has_change:
            y, x = np.ogrid[:h, :w]
            region = ((x - 128)**2 + (y - 128)**2) <= (45**2)
            t2[:, region] = np.random.normal(0.75, 0.04, (3, np.sum(region)))
            mask[0, region] = 1.0
            sentence = random.choice(self.DEFAULT_VOCAB_PHRASES[:-1])
        else:
            sentence = self.DEFAULT_VOCAB_PHRASES[-1]

        token_ids = self.vocab.encode(sentence, max_len=30)
        return np.clip(t1, 0, 1), np.clip(t2, 0, 1), mask, np.array(token_ids, dtype=np.int64)

    def __getitem__(self, idx: int):
        if self.use_synthetic:
            t1, t2, mask, token_ids = self._generate_synthetic(idx)
        else:
            item = self.samples[idx]
            # Загрузка реальных изображений из диска
            t1 = np.load(os.path.join(self.data_dir, "A", item["filename"])).astype(np.float32)
            t2 = np.load(os.path.join(self.data_dir, "B", item["filename"])).astype(np.float32)
            mask = np.load(os.path.join(self.data_dir, "label", item["filename"])).astype(np.float32)
            sentence = random.choice(item["sentences"])
            token_ids = np.array(self.vocab.encode(sentence, max_len=30), dtype=np.int64)

        return (
            torch.from_numpy(t1).float(),
            torch.from_numpy(t2).float(),
            torch.from_numpy(mask).float(),
            torch.from_numpy(token_ids).long()
        )
