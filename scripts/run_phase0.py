'''Run Phase 0 pipeline.'''
from __future__ import annotations
import argparse

from lattice.phases.phase0 import run_phase0


def main():
    p = argparse.ArgumentParser(description='Run Lattice Phase 0 on BANKING77')
    p.add_argument('--encoder', default='distilbert-base-uncased')
    p.add_argument('--data', default='data/banking77.csv')
    p.add_argument('--output_dir', default='./runs/phase0')
    p.add_argument('--device', default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--beta', type=float, default=1.0)
    args = p.parse_args()

    result = run_phase0(
        encoder_name=args.encoder,
        data_path=args.data,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        readout_epochs=args.epochs,
        readout_lr=args.lr,
        rlcd_beta=args.beta,
    )
    print(result)


if __name__ == '__main__':
    main()
