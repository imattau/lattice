'''Generative baseline for Phase 0.

The SAME model that serves as our frozen encoder is instead asked to
classify by scoring the likelihood of each candidate intent, under a
few-shot chat prompt that primes the snake_case output format. Raw
zero-shot cloze on BANKING77 is invalid (shared label prefixes give a
base model no way to disambiguate); a few-shot format-primed prompt is
the fair generative comparison and directly motivates typed readouts.
'''
from __future__ import annotations
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from lattice.calibration import expected_calibration_error
from lattice.phases.phase0 import _split_indices, load_banking77


def build_prefix_ids(tok, examples: list[tuple[str, str]], query: str,
                     system: str) -> list[int]:
    messages = [{'role': 'system', 'content': system}]
    for text, label in examples:
        messages.append({'role': 'user', 'content': text})
        messages.append({'role': 'assistant', 'content': label})
    messages.append({'role': 'user', 'content': query})
    out = tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=False,
    )
    # transformers 5.x may return a tokenizers.Encoding or a list.
    for attr in ('ids', 'input_ids'):
        if hasattr(out, attr):
            out = getattr(out, attr)
    return list(out)


@torch.no_grad()
def label_logprobs(model: nn.Module, tok, queries, label_texts, prefix_fn, device,
                   label_chunk: int = 16):
    label_ids = [tok(l, add_special_tokens=False)['input_ids'] for l in label_texts]
    max_label = max(len(x) for x in label_ids)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    scores = torch.zeros(len(queries), len(label_texts))
    for i, q in enumerate(queries):
        head_ids = prefix_fn(q)
        L = len(head_ids)
        n = len(label_ids)
        for c0 in range(0, n, label_chunk):
            idxs = list(range(c0, min(c0 + label_chunk, n)))
            m = len(idxs)
            seq = torch.full((m, L + max_label), pad_id, dtype=torch.long)
            attn = torch.zeros((m, L + max_label), dtype=torch.long)
            seq[:, :L] = torch.tensor(head_ids)
            attn[:, :L] = 1
            valid = torch.zeros((m, max_label), dtype=torch.bool)
            for jj, j in enumerate(idxs):
                ids = label_ids[j]
                seq[jj, L:L + len(ids)] = torch.tensor(ids)
                attn[jj, L:L + len(ids)] = 1
                valid[jj, :len(ids)] = True
            seq, attn = seq.to(device), attn.to(device)
            logits = model(input_ids=seq, attention_mask=attn).logits
            tgt = seq[:, L:L + max_label]
            lp = torch.log_softmax(
                logits[:, L - 1:L - 1 + max_label].float(), dim=-1)
            tok_lp = lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
            v = valid.to(device).float()
            mean_lp = (tok_lp * v).sum(1) / v.sum(1).clamp(min=1)
            scores[i, idxs] = mean_lp.float().cpu()
            del logits
    return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen/Qwen3-0.6B')
    p.add_argument('--data', default='data/banking77.csv')
    p.add_argument('--device', default=None)
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--shots', type=int, default=3)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    texts, labels, label_names = load_banking77(args.data)
    train_idx, _, test_idx = _split_indices(len(texts), 0.70, 0.15, args.seed)
    test_idx = test_idx[:args.n]

    system = ('Classify the customer support query into exactly one intent '
              'label written in snake_case, from the known set of banking '
              'intents. Respond with only the label.')
    examples = [(texts[train_idx[k]], label_names[labels[train_idx[k]]])
                for k in range(args.shots)]

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to(device)  # type: ignore[arg-type]
    model.eval()

    def prefix_fn(q):
        return build_prefix_ids(tok, examples, q, system)

    queries = [texts[i] for i in test_idx]
    test_labels = torch.tensor([labels[i] for i in test_idx], dtype=torch.long)

    t0 = time.time()
    scores = label_logprobs(model, tok, queries, label_names, prefix_fn, device)
    lat_ms = (time.time() - t0) / max(1, len(queries)) * 1000.0

    probs = F.softmax(scores, dim=-1)
    acc = (probs.argmax(dim=-1) == test_labels).float().mean().item()
    ece = expected_calibration_error(probs, test_labels)
    print(f'Generative baseline ({args.model}, {args.shots}-shot cloze, '
          f'N={len(queries)}):')
    print(f'  accuracy    = {acc:.3f}')
    print(f'  ECE (raw)   = {ece:.3f}')
    print(f'  latency     = {lat_ms:.1f} ms/example')


if __name__ == '__main__':
    main()
