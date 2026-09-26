'''Frozen encoder wrapper around Hugging Face AutoModel.'''
from __future__ import annotations

import torch
from transformers import AutoModel, AutoTokenizer


def mean_pool(last_hidden_state: torch.Tensor,
              attention_mask: torch.Tensor) -> torch.Tensor:
    '''Mean-pool token embeddings using the attention mask.'''
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    return summed / mask.sum(dim=1).clamp(min=1e-9)


class FrozenEncoder:
    '''Wraps a pretrained encoder, freezes it, and returns pooled features.'''

    def __init__(self, model_name: str, device: str | None = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # CPU runs benefit from fp32 (bf16 is slow on CPU and breaks int8).
        dtype = torch.float32 if str(self.device).startswith('cpu') else None
        self.model = AutoModel.from_pretrained(
            model_name, **({'dtype': dtype} if dtype is not None else {}),
        ).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.hidden_dim = self.model.config.hidden_size

    def encode(self, texts: list[str], batch_size: int = 32,
               max_length: int = 512) -> torch.Tensor:
        '''Return [N, hidden_dim] pooled features for the input texts.'''
        features: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                enc = self.tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=max_length, return_tensors='pt',
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = self.model(**enc)
                pooled = mean_pool(out.last_hidden_state, enc['attention_mask'])
                features.append(pooled.cpu())
        return torch.cat(features, dim=0)
