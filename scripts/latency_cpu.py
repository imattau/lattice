'''CPU end-to-end latency gate for Phase 0/1 (< 50 ms/decision).

The Phase 0 readout number (0.04 ms) was the head alone, on GPU. This
measures the number that actually matters for deployment: frozen
encoder forward + typed head on CPU, per decision. Also evaluates
dynamic int8 quantization of the encoder, since CPU cannot use the
4-bit GPU path.
'''
from __future__ import annotations
import argparse
import time

import numpy as np
import torch

from lattice.encoders import FrozenEncoder
from lattice.interfaces import DecisionType
from lattice.phases.phase0 import _split_indices, encode_cached, train_readout
from lattice.readout_heads import ReadoutHead


def _percentiles(times_ms):
    a = np.asarray(times_ms)
    return a.mean(), np.percentile(a, 50), np.percentile(a, 95)


@torch.inference_mode()
def measure(encoder: FrozenEncoder, head: ReadoutHead, texts, n, max_len):
    times = []
    for t in texts[:n]:
        t0 = time.perf_counter()
        feats = encoder.encode([t], batch_size=1, max_length=max_len)
        _ = head(feats)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--encoders', nargs='+',
                   default=['distilbert-base-uncased', 'Qwen/Qwen3-0.6B'])
    p.add_argument('--data', default='data/banking77.csv')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--n', type=int, default=150)
    p.add_argument('--max_len', type=int, default=64)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--quantize', action='store_true',
                   help='also test dynamic int8 quantization of the encoder')
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    from lattice.phases.phase0 import load_banking77
    texts, labels, label_names = load_banking77(args.data)
    train_idx, _, test_idx = _split_indices(len(texts), 0.70, 0.15, args.seed)
    labels_t = torch.tensor(labels, dtype=torch.long)
    test_texts = [texts[i] for i in test_idx]

    print(f'threads={args.threads}  n={args.n}  max_len={args.max_len}')
    hdr = f'{"encoder":<30}{"mean_ms":>9}{"p50":>8}{"p95":>8}{"<50ms?":>8}'
    print(hdr); print('-' * len(hdr))

    for enc in args.encoders:
        # Train the typed head on cached (fp32) features, once.
        feats = encode_cached(enc, texts, device='cpu')
        head = train_readout(feats[train_idx], labels_t[train_idx],
                             len(label_names), epochs=args.epochs, lr=3e-3)
        head.eval()
        encoder = FrozenEncoder(enc, device='cpu')
        _ = measure(encoder, head, test_texts, 10, args.max_len)  # warmup
        m, p50, p95 = _percentiles(measure(encoder, head, test_texts,
                                           args.n, args.max_len))
        ok = 'yes' if p95 < 50 else 'NO'
        print(f'{enc:<30}{m:>9.1f}{p50:>8.1f}{p95:>8.1f}{ok:>8}')

        if args.quantize:
            q = torch.ao.quantization.quantize_dynamic(
                encoder.model, {torch.nn.Linear}, dtype=torch.qint8)
            encoder.model = q
            _ = measure(encoder, head, test_texts, 10, args.max_len)
            m2, p50_2, p95_2 = _percentiles(measure(encoder, head, test_texts,
                                                    args.n, args.max_len))
            ok2 = 'yes' if p95_2 < 50 else 'NO'
            print(f'{enc+" [int8]":<30}{m2:>9.1f}{p50_2:>8.1f}{p95_2:>8.1f}{ok2:>8}')


if __name__ == '__main__':
    main()
