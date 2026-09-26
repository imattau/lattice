'''Generative baseline for Phase 0 (constrained decoding).

The right generative comparison is the SAME KIND of local model asked to
choose an intent by *generating* it, under a constraint that forces the
output to be one of the 77 valid labels. Naive likelihood scoring of raw
snake_case labels is invalid here because BANKING77 labels share long
prefixes (card_arrival vs card_arrival_estimate); a model is never shown
the vocabulary and cannot disambiguate by early tokens.

Constrained decoding fixes this: we restrict the next-token set to the
prefix trie of the label set, and the probability of each label is the
joint likelihood of its token path plus the stop token:

    P(label | prompt) ~ exp(sum_t logP(tok_t | prompt, tok_<t))

Softmax over these 77 constrained leaf probabilities is exactly the
posterior induced by constrained decoding, so we get both a hard decision
and calibrated probabilities. We report accuracy, ECE, Brier, latency,
cost, and prefix-collision rate so the label-collision issue is visible.
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


def build_prefix_ids(tok, examples, query, system):
    messages = [{'role': 'system', 'content': system}]
    for text, label in examples:
        messages.append({'role': 'user', 'content': text})
        messages.append({'role': 'assistant', 'content': label})
    messages.append({'role': 'user', 'content': query})
    out = tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=False,
    )
    for attr in ('ids', 'input_ids'):
        if hasattr(out, attr):
            out = getattr(out, attr)
    return list(out)


@torch.no_grad()
def constrained_leaf_logprobs(model, tok, queries, label_texts, prefix_fn,
                              device, label_chunk=24):
    '''Leaf logprob = sum(logP(label tokens)) + logP(EOS after label).

    Returns [len(queries), len(labels)]. This is the constrained-decoding
    joint probability of each valid label (trie leaves only, so shared
    prefixes are handled correctly by construction).
    '''
    label_ids = [tok(l, add_special_tokens=False)['input_ids'] for l in label_texts]
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    max_label = max(len(x) for x in label_ids)
    width = max_label + 1  # label tokens + terminating EOS
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id
    scores = torch.zeros(len(queries), len(label_texts))
    for i, q in enumerate(queries):
        head_ids = prefix_fn(q)
        L = len(head_ids)
        n = len(label_ids)
        for c0 in range(0, n, label_chunk):
            idxs = list(range(c0, min(c0 + label_chunk, n)))
            m = len(idxs)
            seq = torch.full((m, L + width), pad_id, dtype=torch.long)
            attn = torch.zeros((m, L + width), dtype=torch.long)
            seq[:, :L] = torch.tensor(head_ids)
            attn[:, :L] = 1
            valid = torch.zeros((m, width), dtype=torch.bool)
            for jj, j in enumerate(idxs):
                ids = label_ids[j]
                seq[jj, L:L + len(ids)] = torch.tensor(ids)
                seq[jj, L + len(ids)] = eos_id          # terminate
                valid[jj, :len(ids) + 1] = True
            attn[:, L:L + max_label + 1] = 1
            seq, attn = seq.to(device), attn.to(device)
            logits = model(input_ids=seq, attention_mask=attn).logits
            tgt = seq[:, L:L + width]
            lp = torch.log_softmax(
                logits[:, L - 1:L - 1 + width].float(), dim=-1)
            tok_lp = lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
            v = valid.to(device).float()
            joint = (tok_lp * v).sum(1)                 # sum, not mean
            scores[i, idxs] = joint.float().cpu()
            del logits
    return scores


def prefix_collision_rate(pred, true_labels, label_texts, label_ids_map):
    '''Fraction of errors whose predicted label shares >=1 leading token with
    the gold label. Shows the label-collision issue is real (per the plan),
    even though constrained decoding prevents it from being the failure mode.'''
    pred = pred.tolist()
    true = true_labels.tolist()
    first = {l: label_ids_map[l][0] for l in label_texts}
    wrong = [(p, t) for p, t in zip(pred, true) if p != t]
    if not wrong:
        return 0.0
    share = sum(1 for p, t in wrong
                if first[label_texts[p]] == first[label_texts[t]])
    return share / len(wrong)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen/Qwen3-0.6B')
    p.add_argument('--data', default='data/banking77.csv')
    p.add_argument('--device', default=None)
    p.add_argument('--n', type=int, default=200)
    p.add_argument('--shots', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    texts, labels, label_names = load_banking77(args.data)
    train_idx, _, test_idx = _split_indices(len(texts), 0.70, 0.15, args.seed)
    test_idx = test_idx[:args.n]

    system = ('Classify the customer support query into exactly one intent '
              'from the known set of banking intent labels in snake_case. '
              'Respond with only the label.')
    examples = [(texts[train_idx[k]], label_names[labels[train_idx[k]]])
                for k in range(args.shots)]

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if str(device).startswith('cuda')
        else torch.float32)
    model.to(device)  # type: ignore[arg-type]
    model.eval()

    label_ids_map = {l: tok(l, add_special_tokens=False)['input_ids']
                     for l in label_names}

    def prefix_fn(q):
        return build_prefix_ids(tok, examples, q, system)

    queries = [texts[i] for i in test_idx]
    test_labels = torch.tensor([labels[i] for i in test_idx], dtype=torch.long)

    t0 = time.time()
    scores = constrained_leaf_logprobs(
        model, tok, queries, label_names, prefix_fn, device)
    total = time.time() - t0
    lat_ms = total / max(1, len(queries)) * 1000.0

    probs = F.softmax(scores, dim=-1)
    pred = probs.argmax(dim=-1)
    acc = (pred == test_labels).float().mean().item()
    ece = expected_calibration_error(probs, test_labels)
    brier = ((probs - F.one_hot(test_labels, len(label_names))) ** 2
             ).sum(-1).mean().item()
    coll = prefix_collision_rate(pred, test_labels, label_names, label_ids_map)

    print(f'Generative baseline — constrained decoding ({args.model}, '
          f'{args.shots}-shot, N={len(queries)}):')
    print(f'  accuracy (exact)      = {acc:.3f}')
    print(f'  ECE (raw)             = {ece:.3f}')
    print(f'  Brier                 = {brier:.3f}')
    print(f'  prefix-collision rate = {coll:.3f} of errors')
    print(f'  latency               = {lat_ms:.1f} ms/example')
    print(f'  cost                  = ~{scores.shape[1]} leaf scorings/example')


if __name__ == '__main__':
    main()
