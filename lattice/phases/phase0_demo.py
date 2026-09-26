'''Phase 0 demonstration. Spec §10, Phase 0.

Runs the full decision → constraint → action path on synthetic data.
This is the smallest end-to-end slice of the Lattice.
'''
from __future__ import annotations
import torch
import torch.nn.functional as F

from lattice.interfaces import DecisionType, Stakes
from lattice.readout_heads import ReadoutBank
from lattice.constraint_layer import ConstraintLayer, default_policy
from lattice.controller import Controller, ControllerConfig
from lattice.calibration import expected_calibration_error, fit_temperature


def main() -> None:
    torch.manual_seed(0)

    # Synthetic shared representation (stand-in for a frozen encoder)
    hidden_dim = 128
    num_samples = 1024
    pooled = torch.randn(num_samples, hidden_dim)

    # Ground-truth labels for a routing decision
    routing_labels = (pooled[:, 0] > 0).long()  # binary for demo

    # Readout bank
    bank = ReadoutBank(
        hidden_dim=hidden_dim,
        specs={DecisionType.ROUTING: 2, DecisionType.URGENCY: 3},
        rationale_dim=16,
    )

    # Train the routing head on the synthetic task
    opt = torch.optim.AdamW(bank.parameters(), lr=1e-3)
    for step in range(500):
        out = bank(pooled)[DecisionType.ROUTING]
        loss = F.nll_loss(torch.log(out.distribution.clamp(min=1e-8)),
                          routing_labels)
        opt.zero_grad(); loss.backward(); opt.step()

    # Calibrate post-hoc
    with torch.no_grad():
        out = bank(pooled)[DecisionType.ROUTING]
        logits = torch.log(out.distribution.clamp(min=1e-8))
    T = fit_temperature(logits, routing_labels)
    print(f'Fitted temperature: {T:.3f}')

    with torch.no_grad():
        calibrated = torch.softmax(logits / T, dim=-1)
        ece_raw = expected_calibration_error(out.distribution, routing_labels)
        ece_cal = expected_calibration_error(calibrated, routing_labels)
    print(f'ECE (raw):        {ece_raw:.4f}')
    print(f'ECE (calibrated): {ece_cal:.4f}')

    # Constraint layer
    constraints = ConstraintLayer(default_policy())
    controller = Controller(ControllerConfig(), constraints)

    # End-to-end decision on one sample
    sample = pooled[:1]
    with torch.no_grad():
        readouts = bank(sample)
    proposal, result = controller.decide(readouts, stakes=Stakes.HIGH)
    print(f'\nProposed action: {proposal.action_type}')
    print(f'Constraint verdict: {result.verdict.value}')
    print(f'Policy hash: {result.policy_hash[:16]}...')
    for entry in result.trace:
        print(f"  [{entry['outcome']}] {entry['rule_id']}")


if __name__ == '__main__':
    main()
