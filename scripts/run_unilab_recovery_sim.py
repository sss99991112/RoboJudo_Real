"""Bounded RoboJudo three-expert simulation; no UniLab import or SDK path."""

import argparse
import json
import time

import numpy as np
from box import Box

from robojudo.config.g1.g1_recovery_cfg import G1UniLabRecoveryEnvCfg, g1_unilab_recovery_sim
from robojudo.pipeline.unilab_recovery_sim_pipeline import UniLabRecoverySimPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--supine", action="store_true")
    parser.add_argument("--walk", action="store_true", help="Request keyboard-equivalent 0.4 m/s forward")
    args = parser.parse_args()
    if not 1 <= args.steps <= 1500:
        parser.error("--steps must be 1..1500 (maximum 30 simulated seconds)")
    cfg = g1_unilab_recovery_sim(env=G1UniLabRecoveryEnvCfg(headless=args.headless, reset_supine=args.supine), ctrl=[])
    pipeline = UniLabRecoverySimPipeline(cfg)
    counts = [0, 0, 0]
    try:
        for frame in range(args.steps):
            events = [{"type": "keyboard", "name": "w", "pressed": True}] if args.walk and frame == 0 else []
            controls = Box({"KeyboardCtrl": {"keyboard_event": events}, "COMMANDS": []})
            pipeline.ctrl_manager.get_ctrl_data = lambda _data, controls=controls: controls
            if pipeline.env.viewer is not None and not pipeline.env.viewer.is_alive:
                break
            started = time.perf_counter()
            pipeline.step()
            counts[pipeline.policy._selected_expert] += 1
            if not args.headless:
                time.sleep(max(0, pipeline.dt - (time.perf_counter() - started)))
        data = pipeline.env.get_data()
        print(
            json.dumps(
                {
                    "steps": sum(counts),
                    "route_counts": counts,
                    "final_height_m": float(data.recovery_height),
                    "final_tilt_deg": float(np.rad2deg(np.arccos(np.clip(-data.policy_gravity[2], -1, 1)))),
                    "settle_remaining": pipeline.policy.route.settle_remaining,
                    "hardware": False,
                },
                indent=2,
            )
        )
    finally:
        pipeline.env.shutdown()


if __name__ == "__main__":
    main()
