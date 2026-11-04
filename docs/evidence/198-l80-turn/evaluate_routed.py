"""Bounded non-training candidate: r1 straight, t3 only for |yaw cmd| > .05.

The loader adapter keeps turn_eval's physics, commands and instrumentation intact.
No model files are written. A passing turn probe triggers the existing switch gate.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'sim/rl'))
import stage_gates as sg
import turn_eval
from evaluate import load
from gate_judge import judge

OUT = Path(__file__).resolve().parent/'eval/decision_b'
STRAIGHT = ROOT/'sim/rl/runs/l80_r1_6M'
TURN = ROOT/'sim/rl/runs/l80_t3'
WALK = ROOT/'sim/rl/runs/l80_w2'


class RoutedPolicy:
    def __init__(self, env):
        self.env = env
        self.policies = {}
        self.switches = 0
        self.using = None
        self.counts = dict(straight=0, turn=0)
        for name, directory in (('straight', STRAIGHT), ('turn', TURN)):
            model, vn = load(str(directory/'model.zip'), env)
            scale = sg.policy_scales(json.loads((directory/'args.json').read_text()))
            scale = np.array([scale[sg.joint_type(env.active[i])] for i in env.policy_idx])
            self.policies[name] = model, vn, scale

    def reset(self):
        return self.policies['straight'][1].reset()

    def normalize_obs(self, raw):
        return raw  # each policy's own VecNormalize is applied in predict

    def predict(self, obs, deterministic=True):
        name = 'turn' if abs(float(self.env.command[2])) > .05 else 'straight'
        self.switches += self.using is not None and self.using != name
        self.using = name
        self.counts[name] += 1
        model, vn, scale = self.policies[name]
        raw = sg.observation_for_policy(obs[0], self.env, scale)
        action = model.predict(vn.normalize_obs(raw[None]), deterministic=deterministic)[0][0]
        return sg.action_in_env(action, scale, self.env.scale[self.env.policy_idx])[None], None

    def close(self):
        for _, vn, _ in self.policies.values():
            vn.close()


def main():
    torch.set_num_threads(1)
    adapters = []

    def routed_load(model_path, env):
        if Path(model_path).parent == TURN:
            adapter = RoutedPolicy(env)
            adapters.append(adapter)
            return adapter, adapter
        return load(model_path, env)

    rows = []
    with (OUT/'routed_turn.jsonl').open('x') as out, patch.object(turn_eval, 'load', routed_load):
        for speed in (2., 4.):
            for sign in (1., -1.):
                row = dict(gate='turn', run='r1_straight_t3_turn', quick=False,
                           **turn_eval.turn_condition(TURN, speed, sign*(1. if speed == 2. else .5), range(7300, 7304)))
                row['policy'] = 'r1_straight_t3_turn'
                for episode, adapter in zip(row['episodes'], adapters[-4:]):
                    episode['policy_switches'] = adapter.switches
                    episode['policy_steps'] = adapter.counts
                rows.append(row)
                out.write(json.dumps(row, allow_nan=False)+'\n'); out.flush()
                print(json.dumps(dict(cmd=speed, sign=sign, verdict=judge(row))), flush=True)
    passed = all(judge(r)[0] is True for r in rows)
    if passed:
        with (OUT/'routed_switch.jsonl').open('x') as out, patch.object(sg, 'load', routed_load):
            for row in sg.switch(WALK, run_dir=TURN):
                out.write(json.dumps(dict(gate='switch', run='r1_straight_t3_turn', **row), allow_nan=False)+'\n')
    (OUT/'routed_decision.json').write_text(json.dumps(dict(turn_passed=passed,
        switch_executed=passed, decision='continue full verification' if passed else 'reject; no threshold search'), indent=2)+'\n')


if __name__ == '__main__':
    main()
