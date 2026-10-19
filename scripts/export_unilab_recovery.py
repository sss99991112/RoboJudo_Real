"""Run with UniLab's uv environment; preserve raw and old exported weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn


class ExplicitExpertStudent(nn.Module):
    def __init__(self, student):
        super().__init__()
        if len(student.experts) != 3:
            raise ValueError("recovery export requires three experts")
        self.walk = student.experts[0]
        self.stand = student.experts[1]
        self.recovery = student.experts[2]

    def forward(self, obs, expert_index):
        actions = torch.stack([self.walk(obs), self.stand(obs), self.recovery(obs)], dim=1)
        indices = expert_index[:, None, None].expand(-1, 1, 29)
        return torch.gather(actions, 1, indices).squeeze(1)


def export_recovery(checkpoint, output):
    from unilab.algos.torch.distill import load_distillation_student_policy
    from unilab.algos.torch.distill.recovery_integration import (
        RECOVERY_ROLES,
        RECOVERY_ROUTING_CONTRACT,
        validate_recovery_student,
    )

    from robojudo.policy.recovery_route import ROLES, ROUTING_CONTRACT, STUDENT_SHA256

    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if output.exists() or output == checkpoint:
        raise FileExistsError("export must create a new artifact, never overwrite weights")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != STUDENT_SHA256:
        raise ValueError("checkpoint SHA does not match the frozen migration student")
    loaded = load_distillation_student_policy(checkpoint, device="cpu")
    validate_recovery_student(loaded.policy, loaded.distill_runtime_cfg)
    if RECOVERY_ROLES != ROLES or RECOVERY_ROUTING_CONTRACT != ROUTING_CONTRACT:
        raise ValueError("source and target recovery contracts differ")
    module = ExplicitExpertStudent(loaded.policy).eval()
    with torch.inference_mode(), torch.random.fork_rng():
        torch.manual_seed(19)
        example = torch.zeros(1, 99)
        traced = torch.jit.trace(module, (example, torch.zeros(1, dtype=torch.int64)), check_trace=False)
        for batch in (1, 17):
            obs = torch.randn(batch, 99)
            for role in range(3):
                indices = torch.full((batch,), role, dtype=torch.int64)
                torch.testing.assert_close(traced(obs, indices), loaded.policy.experts[role](obs), rtol=0, atol=1e-6)
            indices = torch.arange(batch) % 3
            torch.testing.assert_close(traced(obs, indices), module(obs, indices), rtol=0, atol=1e-6)
    metadata = {
        "obs_dim": 99,
        "action_dim": 29,
        "roles": ROLES,
        "routing": ROUTING_CONTRACT,
        "source_sha256": digest,
        "inputs": ["obs", "expert_index"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output), _extra_files={"contract.json": json.dumps(metadata, sort_keys=True)})
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(export_recovery(args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()
