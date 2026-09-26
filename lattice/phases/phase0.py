'''Phase 0: Calibrated typed readout heads + constraint layer on BANKING77.

This is the smallest real-world slice of the Lattice:
  - Freeze a pretrained encoder.
  - Train a typed readout head with the RLCD surrogate loss.
  - Fit a post-hoc temperature on a calibration split.
  - Evaluate accuracy, ECE, and Brier on a held-out test split.
  - Route decisions through the constraint layer + controller.
'''
from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset

from lattice.calibration import (
    expected_calibration_error, fit_temperature, rlcd_surrogate,
)
from lattice.constraint_layer import ConstraintLayer, Rule
from lattice.controller import Controller, ControllerConfig
from lattice.encoders import FrozenEncoder
from lattice.interfaces import DecisionType, Stakes
from lattice.readout_heads import ReadoutHead


@dataclass
class Phase0Result:
    accuracy: float
    ece_raw: float
    ece_cal: float
    brier: float
    latency_ms: float
    temperature: float
    output_dir: str

    def __str__(self) -> str:
        return (
            f'Phase0Result(accuracy={self.accuracy:.4f}, '
            f'ece_raw={self.ece_raw:.4f}, ece_cal={self.ece_cal:.4f}, '
            f'brier={self.brier:.4f}, latency_ms={self.latency_ms:.2f}, '
            f'temperature={self.temperature:.4f})'
        )


def _split_indices(n: int, train_frac: float, cal_frac: float,
                   seed: int) -> tuple[list[int], list[int], list[int]]:
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()
    n_train = int(n * train_frac)
    n_cal = int(n * cal_frac)
    return perm[:n_train], perm[n_train:n_train + n_cal], perm[n_train + n_cal:]


def load_banking77(data_path: str = 'data/banking77.csv'):
    '''Return (texts, label_ids, label_names) from the local BANKING77 CSV.'''
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f'BANKING77 CSV not found at {path}. Download it from: '
            'https://raw.githubusercontent.com/PolyAI-LDN/task-specific-'
            'datasets/master/banking_data/train.csv'
        )
    ds = load_dataset('csv', data_files={'train': str(path)})
    texts = ds['train']['text']
    categories = ds['train']['category']
    label_names = sorted(set(categories))
    label2id = {name: i for i, name in enumerate(label_names)}
    labels = [label2id[c] for c in categories]
    return texts, labels, label_names


def _slug(name: str) -> str:
    import re
    return re.sub(r'[^0-9a-zA-Z]+', '_', name.strip('/')).lower().strip('_')


def encode_cached(encoder_name: str, texts: list[str],
                  cache_dir: str = 'data/feats', device: str | None = None,
                  batch_size: int = 32) -> torch.Tensor:
    '''Encode texts with a FrozenEncoder, caching the [N, D] tensor to disk.'''
    cache = Path(cache_dir) / (_slug(encoder_name) + '.pt')
    if cache.exists():
        return torch.load(cache, weights_only=True)
    encoder = FrozenEncoder(encoder_name, device=device)
    feats = encoder.encode(texts, batch_size=batch_size)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feats, cache)
    return feats


def prepare_data(encoder: FrozenEncoder,
                 data_path: str = 'data/banking77.csv',
                 train_frac: float = 0.70,
                 cal_frac: float = 0.15, seed: int = 0):
    '''Load BANKING77 from a local CSV, encode utterances, and split train/cal/test.'''
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f'Download BANKING77 CSV to {path}. '
            'See scripts/run_phase0.py --help for the source URL.'
        )
    ds = load_dataset('csv', data_files={'train': str(path)})
    texts = ds['train']['text']
    categories = ds['train']['category']
    label_names = sorted(set(categories))
    label2id = {name: i for i, name in enumerate(label_names)}
    labels = [label2id[c] for c in categories]
    num_classes = len(label_names)

    train_idx, cal_idx, test_idx = _split_indices(
        len(texts), train_frac, cal_frac, seed,
    )
    all_features = encoder.encode(texts)

    def subset(idxs: list[int]):
        return (
            all_features[idxs],
            torch.tensor([labels[i] for i in idxs], dtype=torch.long),
        )

    return {
        'train': subset(train_idx),
        'cal': subset(cal_idx),
        'test': subset(test_idx),
        'num_classes': num_classes,
        'label_names': label_names,
    }


def _build_policy() -> list[Rule]:
    '''Hard constraints for the Phase 0 routing demo.'''
    def low_confidence_high_stakes(action):
        if action.stakes != Stakes.HIGH:
            return False
        return float(action.confidence.mean()) < 0.3 and action.action_type != 'escalate'

    return [
        Rule(
            'escalate_low_conf_high_stakes',
            low_confidence_high_stakes,
            'Escalate high-stakes decisions when confidence is below 0.3.',
        ),
    ]


def train_readout(features: torch.Tensor, labels: torch.Tensor,
                  num_classes: int, epochs: int = 30, lr: float = 1e-3,
                  beta: float = 1.0) -> ReadoutHead:
    '''Train a ReadoutHead with the RLCD surrogate loss.'''
    head = ReadoutHead(
        hidden_dim=features.size(-1),
        num_options=num_classes,
        decision_type=DecisionType.ROUTING,
    )
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    for _ in range(epochs):
        out = head(features)
        loss = rlcd_surrogate(out.distribution, labels, beta=beta)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return head


def _evaluate(head: ReadoutHead, features: torch.Tensor,
              labels: torch.Tensor, num_classes: int,
              temperature: float | None = None) -> dict:
    head.eval()
    with torch.no_grad():
        out = head(features)
        probs = out.distribution
        logits = head.logit_head(features)
        if temperature is None:
            cal = probs
        else:
            cal = torch.softmax(logits / temperature, dim=-1)
        acc = (probs.argmax(dim=-1) == labels).float().mean().item()
        ece = expected_calibration_error(probs, labels)
        ece_cal = expected_calibration_error(cal, labels)
        brier = ((probs - F.one_hot(labels, num_classes).float()) ** 2
                 ).sum(dim=-1).mean().item()
    return {
        'accuracy': acc,
        'ece_raw': ece,
        'ece_cal': ece_cal,
        'brier': brier,
    }


def _measure_latency(head: ReadoutHead, features: torch.Tensor,
                     n: int = 200) -> float:
    head.eval()
    device = next(head.parameters()).device
    sample = features[:1].to(device)
    # Warmup
    with torch.no_grad():
        _ = head(sample)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n):
            _ = head(sample)
    return (time.time() - t0) / n * 1000.0


def _demo_decisions(head: ReadoutHead, features: torch.Tensor,
                    labels: torch.Tensor, label_names: list[str],
                    controller: Controller, device: str,
                    n: int = 20) -> list[dict]:
    head.eval()
    decisions = []
    with torch.no_grad():
        for i in range(min(n, len(features))):
            feat = features[i:i + 1].to(device)
            readout = head(feat)
            true_name = label_names[int(labels[i].item())]
            # Treat a few intent categories as high-stakes for demonstration.
            high_stakes = {'lost_or_stolen_card', 'pin_blocked', 'cancel_transfer'}
            stakes = Stakes.HIGH if true_name in high_stakes else Stakes.LOW
            proposal, result = controller.decide(
                {DecisionType.ROUTING: readout}, stakes=stakes,
            )
            decisions.append({
                'true_label': true_name,
                'predicted_label': label_names[int(readout.distribution.argmax(dim=-1).item())],
                'confidence': float(readout.confidence.mean()),
                'proposed_action': proposal.action_type,
                'verdict': result.verdict.value,
            })
    return decisions


def run_phase0(
    encoder_name: str = 'distilbert-base-uncased',
    data_path: str = 'data/banking77.csv',
    output_dir: str = './runs/phase0',
    device: str | None = None,
    seed: int = 0,
    readout_epochs: int = 30,
    readout_lr: float = 1e-3,
    rlcd_beta: float = 1.0,
) -> Phase0Result:
    '''End-to-end Phase 0 pipeline.'''
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    encoder = FrozenEncoder(encoder_name, device=device)
    data = prepare_data(encoder, data_path=data_path, seed=seed)
    num_classes = data['num_classes']
    label_names = data['label_names']
    train_features, train_labels = data['train']
    cal_features, cal_labels = data['cal']
    test_features, test_labels = data['test']

    # Train readout head.
    head = train_readout(
        train_features, train_labels, num_classes,
        epochs=readout_epochs, lr=readout_lr, beta=rlcd_beta,
    )
    head.to(device)

    # Fit post-hoc temperature on calibration split.
    with torch.no_grad():
        cal_logits = head.logit_head(cal_features.to(device))
    temperature = fit_temperature(cal_logits, cal_labels.to(device))

    # Evaluate on test split.
    metrics = _evaluate(
        head, test_features.to(device), test_labels.to(device),
        num_classes, temperature=temperature,
    )
    latency_ms = _measure_latency(head, test_features)

    # Demonstrate controller + constraint layer.
    constraints = ConstraintLayer(_build_policy())
    controller = Controller(ControllerConfig(), constraints)
    decisions = _demo_decisions(
        head, test_features, test_labels, label_names,
        controller, device,
    )

    # Save artifacts.
    torch.save({
        'head': head.state_dict(),
        'temperature': temperature,
        'label_names': label_names,
    }, out_path / 'readout.pt')
    with open(out_path / 'metrics.json', 'w') as f:
        json.dump({**metrics, 'latency_ms': latency_ms,
                   'temperature': temperature}, f, indent=2)
    with open(out_path / 'decisions.json', 'w') as f:
        json.dump(decisions, f, indent=2)

    return Phase0Result(
        accuracy=metrics['accuracy'],
        ece_raw=metrics['ece_raw'],
        ece_cal=metrics['ece_cal'],
        brier=metrics['brier'],
        latency_ms=latency_ms,
        temperature=temperature,
        output_dir=str(out_path),
    )
