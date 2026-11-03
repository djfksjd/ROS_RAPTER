"""Design-freeze plan F0 (docs/DESIGN_FREEZE_PLAN.ko.md): the experiment contract. Hashes of the model, the R-03 config,
the realism layer and the training/eval code; per policy axis the joint stop range and the range a policy can actually
command (q0 +- scale, clipped) with the stand policy r3_s1ho settings; the R-03 policies and whether they are still
valid on this model. Regenerate after any change; a hash change means earlier results are not directly comparable.
Usage: .venv-sim/bin/python modeling/freeze_contract.py [--out docs/design/freeze_contract.json]"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'sim'/'rl'))
import stage_gates as sg  # noqa: E402
from run_env import joint_type  # noqa: E402

FILES = ['sim/raptor_r03.xml', 'sim/rl/rev3_b.json', 'sim/rl/realism_robstride_rev3.json', 'sim/rl/run_env.py',
         'sim/rl/realism.py', 'sim/rl/train_run.py', 'sim/rl/stage_gates.py']
REF = 'r3_s1ho'  # stage 1a pass (evidence 188)


def sha(p):
    return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT/'docs/design/freeze_contract.json'))
    o = ap.parse_args()
    a = json.loads((ROOT/f'sim/rl/runs/{REF}/args.json').read_text())
    e = sg.RunEnv(**sg.env_kwargs(a), randomize=False, seed=0)
    e.reset(seed=0)
    axes = {}
    for i in e.policy_idx:
        n = e.active[i]
        if not n.startswith('left') and not n.startswith('tail'):
            continue
        lo, hi, q0, s = float(e.lo[i]), float(e.hi[i]), float(e.q0[i]), float(e.scale[i])
        axes[n.replace('_joint', '')] = dict(stop=[round(lo, 3), round(hi, 3)], q0=round(q0, 3), scale=s,
                                             commandable=[round(max(lo, q0-s), 3), round(min(hi, q0+s), 3)])
    runs = sorted(p.name for p in (ROOT/'sim/rl/runs').glob('r3_*'))
    out = dict(
        date='2026-11-03', model_rev=3, mass_kg=round(float(e.model.body_subtreemass[0]), 2),
        hashes={f: sha(f) for f in FILES}, reference_policy=REF, scale_override=a.get('scale_override'),
        nominal=a.get('nominal'), policy_axes_left_and_tail=axes,
        note=('commandable < stop means a policy with these settings can never reach the joint stop '
              '(evidence 192: the knee was capped at 2.0 of 2.26 in every get-up test before r3_gxa/gxb)'),
        r3_policies=runs, valid_on_this_contract='all r3_* runs, until a hash above changes; rev2_* runs are records '
                                                  '(model_rev 2, invalid on R-03)')
    Path(o.out).write_text(json.dumps(out, indent=1, ensure_ascii=False)+'\n')
    print(json.dumps(axes, indent=1))


if __name__ == '__main__':
    main()
