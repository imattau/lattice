'''Phase 1.5 toy corpus: a small, diverse, fully offline text corpus for the
JEPA-vs-AR falsification experiment (PLAN.md "Phase 1.5: Toy-Scale
Falsification").

The plan calls for "code (e.g. The Stack subset), natural language (e.g. C4
subset), structured text (JSON, tables)" at 100M-1B tokens. This environment
has no budget in a single interactive session to download and train on
internet-scale corpora at that volume (see PHASE1_5_RESULTS.md for the full
scope-reduction rationale — this mirrors how the Phase 1 corpus generation
deviated from "frontier-LLM generation" and documented it rather than
overclaiming). Instead:

  - **natural language**: real text already produced by this project
    (BANKING77 customer-service queries, Phase 1 triage document bodies) —
    not a substitute corpus, actual previously-generated text on disk.
  - **code** and **structured text**: small deterministic template
    generators (same pattern as lattice/synthetic_triage.py), since no
    code/JSON corpus is available offline.

This is a toy-of-the-toy: enough source diversity to ask "does the JEPA
objective produce representations comparable to AR at matched capacity on
*some* small diverse corpus" — not a claim about internet-scale text or
about the specific 10-50M-parameter / 100M-1B-token regime the plan
specifies.
'''
from __future__ import annotations
import csv
import json
import random
from pathlib import Path


def load_banking77_texts(path: str | Path = 'data/banking77.csv',
                         offset: int = 0, limit: int = 4000) -> list[str]:
    texts = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < offset:
                continue
            texts.append(row['text'])
            if len(texts) >= limit:
                break
    return texts


def load_banking77_labeled(path: str | Path = 'data/banking77.csv',
                           offset: int = 0, limit: int = 3000
                           ) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < offset:
                continue
            texts.append(row['text'])
            labels.append(row['category'])
            if len(texts) >= limit:
                break
    return texts, labels


def load_triage_texts(path: str | Path = 'data/triage/train.jsonl',
                      limit: int = 4000) -> list[str]:
    texts = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            texts.append(json.loads(line)['text'])
    return texts


def load_triage_labeled(path: str | Path = 'data/triage/val.jsonl',
                        limit: int = 3000) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            doc = json.loads(line)
            texts.append(doc['text'])
            labels.append(doc['department'])
    return texts, labels


_VAR_NAMES = ['x', 'y', 'total', 'count', 'result', 'value', 'data', 'item',
             'acc', 'n', 'buffer', 'index', 'key', 'output']
_OPS = ['+', '-', '*', '//', '%']
_FUNC_NAMES = ['compute', 'process', 'update', 'normalize', 'filter_items',
              'transform', 'validate', 'merge', 'reduce', 'aggregate']


def gen_code_snippet(rng: random.Random) -> str:
    fname = rng.choice(_FUNC_NAMES)
    a, b = rng.sample(_VAR_NAMES, 2)
    op = rng.choice(_OPS)
    lines = [f'def {fname}({a}, {b}):', f'    result = {a} {op} {b}']
    if rng.random() < 0.5:
        lines += ['    if result > 0:', '        return result', '    return 0']
    else:
        lines.append('    return result')
    return '\n'.join(lines)


_JSON_KEYS = ['id', 'name', 'status', 'count', 'active', 'tags', 'score']
_JSON_NAMES = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
_JSON_STATUS = ['ok', 'pending', 'error', 'retrying']
_JSON_TAGS = ['a', 'b', 'c', 'd', 'e']


def gen_json_record(rng: random.Random) -> str:
    keys = rng.sample(_JSON_KEYS, k=rng.randint(3, len(_JSON_KEYS)))
    obj = {}
    for k in keys:
        if k == 'id':
            obj[k] = rng.randint(1, 99999)
        elif k == 'name':
            obj[k] = rng.choice(_JSON_NAMES)
        elif k == 'status':
            obj[k] = rng.choice(_JSON_STATUS)
        elif k == 'count':
            obj[k] = rng.randint(0, 500)
        elif k == 'active':
            obj[k] = rng.choice([True, False])
        elif k == 'tags':
            obj[k] = rng.sample(_JSON_TAGS, k=rng.randint(1, 3))
        elif k == 'score':
            obj[k] = round(rng.uniform(0, 1), 3)
    return json.dumps(obj)


def build_pretrain_corpus(
    seed: int = 0,
    n_code: int = 4000,
    n_json: int = 4000,
    n_banking: int = 4000,
    n_triage: int = 4000,
) -> list[str]:
    '''Deterministic, offline, four-source toy corpus for JEPA/AR pretraining.'''
    rng = random.Random(seed)
    code = [gen_code_snippet(rng) for _ in range(n_code)]
    struct = [gen_json_record(rng) for _ in range(n_json)]
    banking = load_banking77_texts(offset=0, limit=n_banking)
    triage = load_triage_texts(limit=n_triage)
    corpus = code + struct + banking + triage
    rng.shuffle(corpus)
    return corpus


class CharTokenizer:
    '''Character-level tokenizer: no external dependency, no download, and
    (unlike a subword tokenizer) its vocabulary doesn't dominate parameter
    count at this model scale, so nearly all trained capacity sits in the
    transformer body rather than an embedding table — closer to the plan's
    intent of comparing objectives at matched *transformer* capacity.
    '''
    PAD = 0
    MASK = 1
    UNK = 2
    _RESERVED = 3

    def __init__(self, chars: list[str]):
        self.chars = chars
        self.stoi = {c: i + self._RESERVED for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars) + self._RESERVED

    @classmethod
    def fit(cls, texts: list[str]) -> 'CharTokenizer':
        chars = sorted(set(''.join(texts)))
        return cls(chars)

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.stoi.get(c, self.UNK) for c in text[:max_len]]
        ids += [self.PAD] * (max_len - len(ids))
        return ids

    def encode_batch(self, texts: list[str], max_len: int):
        import torch
        return torch.tensor([self.encode(t, max_len) for t in texts],
                            dtype=torch.long)
