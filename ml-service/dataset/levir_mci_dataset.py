import os
import json
import random
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Dict, Tuple, Optional

class ChangeCaptionVocab:
    """
    Словарь токенов для Change Captioning.
    Автоматически обучается на всех предложениях обучающей выборки LEVIR-MCI.
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
    Официальный загрузчик датасета LEVIR-MCI (Beihang University).
    Загружает реальные тройки: (Снимок T1, Снимок T2, Маска, Текстовое описание).
    """
    def __init__(
        self,
        root_dir: str = "./data/levir_mci/LEVIR-MCI-dataset",
        split: str = "train",
        vocab: Optional[ChangeCaptionVocab] = None,
        img_size: int = 256,
        max_caption_len: int = 35,
        balance_changes: bool = False
    ):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.max_caption_len = max_caption_len

        json_path = os.path.join(root_dir, "LevirCCcaptions.json")
        self.samples = []
        all_train_sentences = []

        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Извлекаем список изображений
            items = data["images"] if isinstance(data, dict) and "images" in data else data

            for item in items:
                # Фильтруем по сплиту: 'train', 'val', 'test'
                item_split = item.get("split", item.get("filepath", "train"))
                raw_sentences = [s["raw"] if isinstance(s, dict) else s for s in item.get("sentences", [])]

                if item_split == "train":
                    all_train_sentences.extend(raw_sentences)

                if item_split == self.split:
                    self.samples.append({
                        "filename": item["filename"],
                        "split": item_split,
                        "sentences": raw_sentences,
                        "changeflag": item.get("changeflag", 1)
                    })

        # Если включен баланс для обучения: устраняем 50% перекос неизменных шаблонов
        if self.split == "train" and balance_changes:
            changed_samples = [s for s in self.samples if s.get("changeflag", 1) == 1]
            unchanged_samples = [s for s in self.samples if s.get("changeflag", 1) == 0]
            rng = random.Random(42)
            # Берем около 700 неизменных пар вместо 3400, чтобы избежать доминирования шаблона
            sampled_unchanged = rng.sample(unchanged_samples, min(len(unchanged_samples), max(len(changed_samples) // 4, 1)))
            self.samples = changed_samples + sampled_unchanged
            rng.shuffle(self.samples)

        # Построение словаря
        if vocab is None:
            self.vocab = ChangeCaptionVocab(min_freq=2)
            if all_train_sentences:
                self.vocab.build_vocab(all_train_sentences)
            else:
                self.vocab.build_vocab(["trees were removed and buildings were built ."])
        else:
            self.vocab = vocab

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        split_dir = item["split"]
        fname = item["filename"]

        path_a = os.path.join(self.root_dir, "images", split_dir, "A", fname)
        path_b = os.path.join(self.root_dir, "images", split_dir, "B", fname)
        path_label = os.path.join(self.root_dir, "images", split_dir, "label", fname)

        # Загрузка и нормализация изображений
        img_a = Image.open(path_a).convert("RGB")
        img_b = Image.open(path_b).convert("RGB")
        img_label = Image.open(path_label).convert("L")

        if self.img_size != 256:
            img_a = img_a.resize((self.img_size, self.img_size), Image.BILINEAR)
            img_b = img_b.resize((self.img_size, self.img_size), Image.BILINEAR)
            img_label = img_label.resize((self.img_size, self.img_size), Image.NEAREST)

        t1 = np.array(img_a, dtype=np.float32).transpose(2, 0, 1) / 255.0
        t2 = np.array(img_b, dtype=np.float32).transpose(2, 0, 1) / 255.0
        # Порог > 50 для захвата и дорог (128), и построек (255)
        mask = (np.array(img_label, dtype=np.float32) > 50).astype(np.float32)[np.newaxis, :, :]

        # При обучении берем случайное из 5 предложений; при валидации - первое фиксированное
        if self.split == "train":
            sentence = random.choice(item["sentences"])
            # Синхронные аугментации для пары растров и маски
            if random.random() > 0.5:
                t1 = np.flip(t1, axis=-1).copy()
                t2 = np.flip(t2, axis=-1).copy()
                mask = np.flip(mask, axis=-1).copy()
            if random.random() > 0.5:
                t1 = np.flip(t1, axis=-2).copy()
                t2 = np.flip(t2, axis=-2).copy()
                mask = np.flip(mask, axis=-2).copy()
        else:
            sentence = item["sentences"][0] if item["sentences"] else "no change ."

        token_ids = self.vocab.encode(sentence, max_len=self.max_caption_len)

        return (
            torch.from_numpy(t1).float(),
            torch.from_numpy(t2).float(),
            torch.from_numpy(mask).float(),
            torch.tensor(token_ids, dtype=torch.long)
        )
