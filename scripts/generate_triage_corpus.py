'''Generate the Phase 1 synthetic triage corpus. Plan §Phase 1 / Data.

Writes train/val/test JSONL splits plus a small, hard-case-heavy red-team
set (spec: "50-100 adversarial inputs" for the Phase 1 red-teaming eval).
Also prints label-distribution summary stats so corpus balance can be
checked without loading the files elsewhere.
'''
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path

from lattice.synthetic_triage import (
    DEPARTMENT_NAMES, HARD_CASES, document_to_dict, generate_corpus,
)


def _write_jsonl(path: Path, docs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for doc in docs:
            f.write(json.dumps(document_to_dict(doc)) + '\n')


def _summarize(name: str, docs) -> None:
    n = len(docs)
    dept = Counter(d.department for d in docs)
    urg = Counter(d.urgency for d in docs)
    hard = Counter(d.hard_case for d in docs)
    esc_rate = sum(d.escalate for d in docs) / n
    safety_rate = sum(d.safety_flag for d in docs) / n
    print(f'\n{name}: {n} documents')
    print('  department: ' + ', '.join(
        f'{DEPARTMENT_NAMES[k]}={v}' for k, v in sorted(dept.items())))
    print('  urgency (0=low,1=med,2=high): '
          + ', '.join(f'{k}={v}' for k, v in sorted(urg.items())))
    print(f'  escalation rate = {esc_rate:.3f}')
    print(f'  safety_flag rate = {safety_rate:.3f}')
    print('  hard cases: ' + ', '.join(
        f'{k or "none"}={v}' for k, v in sorted(hard.items(), key=str)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=30000,
                    help='total corpus size before splitting (plan: 20K-50K)')
    p.add_argument('--out', default='data/triage')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--hard_case_rate', type=float, default=0.2)
    p.add_argument('--redteam_n', type=int, default=100)
    p.add_argument('--train_frac', type=float, default=0.70)
    p.add_argument('--val_frac', type=float, default=0.15)
    args = p.parse_args()

    docs = generate_corpus(args.n, seed=args.seed,
                           hard_case_rate=args.hard_case_rate)
    n_train = int(args.n * args.train_frac)
    n_val = int(args.n * args.val_frac)
    train, val, test = (docs[:n_train], docs[n_train:n_train + n_val],
                        docs[n_train + n_val:])

    # A held-out "gold" validation slice, stratified-by-construction (same
    # generator, disjoint seed) — a stand-in for the plan's 500-1,000
    # human-annotated set until real human annotation happens. NOT
    # human-verified; see PHASE1_RESULTS.md.
    gold = generate_corpus(1000, seed=args.seed + 10_000, hard_case_rate=0.3)

    # Red-team set: adversarial-only, heavier hard-case concentration.
    redteam = generate_corpus(args.redteam_n, seed=args.seed + 20_000,
                              hard_case_rate=1.0)

    out = Path(args.out)
    _write_jsonl(out / 'train.jsonl', train)
    _write_jsonl(out / 'val.jsonl', val)
    _write_jsonl(out / 'test.jsonl', test)
    _write_jsonl(out / 'gold_validation.jsonl', gold)
    _write_jsonl(out / 'redteam.jsonl', redteam)

    print(f'Wrote corpus to {out}/  (seed={args.seed}, '
          f'hard_case_rate={args.hard_case_rate})')
    for name, split in [('train', train), ('val', val), ('test', test),
                        ('gold_validation', gold), ('redteam', redteam)]:
        _summarize(name, split)


if __name__ == '__main__':
    main()
