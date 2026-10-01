import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

class CaptionDecoder(nn.Module):
    """
    Текстовый декодер для Change Captioning с поддержкой Beam Search.
    
    Особенности:
      1. Causal Masking для Teacher Forcing при обучении.
      2. Полноценный Beam Search с шириной луча (Beam Width) и штрафом за длину (Length Penalty).
      3. Cross-Attention на визуальные токены BIT Joint Fusion.
    """
    def __init__(
        self, 
        vocab_size: int = 1500, 
        d_model: int = 256, 
        nhead: int = 4, 
        num_layers: int = 3,
        pad_idx: int = 0,
        bos_idx: int = 1,
        eos_idx: int = 2,
        max_len: int = 40
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.max_len = max_len

        self.word_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 2, 
            dropout=0.1, 
            batch_first=True,
            activation="gelu"
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)
        return mask

    def forward(
        self, 
        visual_memory: torch.Tensor, 
        tgt_caption_ids: torch.Tensor
    ) -> torch.Tensor:
        """Прямой проход при обучении (Teacher Forcing)."""
        device = visual_memory.device
        t = tgt_caption_ids.size(1)

        tgt_emb = self.word_embed(tgt_caption_ids) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        causal_mask = self._generate_causal_mask(t, device)
        pad_mask = (tgt_caption_ids == self.pad_idx)

        dec_out = self.transformer_decoder(
            tgt=tgt_emb,
            memory=visual_memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=pad_mask
        )
        logits = self.fc_out(dec_out)
        return logits

    @torch.no_grad()
    def beam_search_decode(
        self, 
        visual_memory: torch.Tensor, 
        beam_size: int = 3, 
        max_len: Optional[int] = None,
        length_penalty: float = 0.7
    ) -> torch.Tensor:
        """
        Полноценный Beam Search для генерации связного текста без повторов.
        :param visual_memory: Токены из BIT (B, 2*L, D)
        :param beam_size: Ширина луча (обычно 3 или 5)
        :param length_penalty: Степень нормализации длины предложения (alpha)
        :return: Лучшая последовательность токенов (B, seq_len)
        """
        self.eval()
        device = visual_memory.device
        b = visual_memory.size(0)
        max_l = max_len or self.max_len

        all_results = []

        # Обрабатываем каждый элемент батча
        for i in range(b):
            mem = visual_memory[i:i+1] # (1, 2*L, D)

            # Кандидат: (log_prob, [token_ids], is_finished)
            candidates = [(0.0, [self.bos_idx], False)]

            for step in range(max_l - 1):
                new_candidates = []
                all_finished = True

                for score, seq, is_done in candidates:
                    if is_done:
                        new_candidates.append((score, seq, True))
                        continue

                    all_finished = False
                    curr_input = torch.tensor([seq], dtype=torch.long, device=device)
                    t = curr_input.size(1)

                    tgt_emb = self.word_embed(curr_input) * math.sqrt(self.d_model)
                    tgt_emb = self.pos_encoder(tgt_emb)
                    causal_mask = self._generate_causal_mask(t, device)

                    dec_out = self.transformer_decoder(
                        tgt=tgt_emb,
                        memory=mem,
                        tgt_mask=causal_mask
                    )
                    logits = self.fc_out(dec_out[:, -1, :]) # (1, vocab_size)
                    log_probs = F.log_softmax(logits, dim=-1).squeeze(0)

                    # Берем top-K лучших следующих слов
                    topk_log_probs, topk_tokens = torch.topk(log_probs, k=beam_size)

                    for k in range(beam_size):
                        next_token = topk_tokens[k].item()
                        next_score = score + topk_log_probs[k].item()
                        next_seq = seq + [next_token]
                        next_is_done = (next_token == self.eos_idx)
                        new_candidates.append((next_score, next_seq, next_is_done))

                if all_finished:
                    break

                # Сортировка с учетом штрафа за длину (length penalty)
                # Нормализованный скор: score / (len^alpha)
                def sort_key(item):
                    s, sq, _ = item
                    penalty = (len(sq) ** length_penalty)
                    return s / penalty

                new_candidates.sort(key=sort_key, reverse=True)
                candidates = new_candidates[:beam_size]

            # Выбираем лучший результат для данного сэмпла
            best_seq = candidates[0][1]
            all_results.append(best_seq)

        # Приводим к единому тензору с дополнением нулями (padding)
        max_res_len = max(len(s) for s in all_results)
        padded = torch.full((b, max_res_len), self.pad_idx, dtype=torch.long, device=device)
        for i, seq in enumerate(all_results):
            padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)

        return padded
