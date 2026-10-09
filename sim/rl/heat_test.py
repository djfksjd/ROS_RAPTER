"""60 s constant-command runs with the motor heat measure, env rebuilt exactly from the run's args.json (train_run.env_kwargs).

Per command: start from the standing reset pose with cold motors (randomize off), hold the command for 60 s with the
deterministic policy. Reported over the last 40 s: mean forward speed, steady heat per motor type = mean of
(motor torque / continuous torque)^2 (continuous = 40 % of the class peak; 1 = rating; independent of the 3 s thermal
time constant), falls. Speed = forward velocity in the body heading frame (the heading drifts over 60 s). Also the best 3 s mean speed (burst). Verdicts:
  sustainable: no fall, every motor's steady heat <= 1.0 (1 ms torque; *_20hz: after a 20 Hz low-pass, lower bound);  tracked: last-40 s speed >= 90 % of the command;
  40 km/h reached: sustainable AND last-40 s speed >= 11.1 m/s (judged separately from the 90 % rule, review 2026-10-01).
T1 virtual-actuator simulation with assumed motor curves and a heat proxy, not motor temperature or hardware.
Usage: .venv-sim/bin/python sim/rl/heat_test.py sim/rl/runs/<run> [--cmds 3 4 5 6 8 11.1] [--seconds 60] [--seed 7]
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from evaluate import load  # noqa: E402
from run_env import RunEnv, joint_type  # noqa: E402
from train_run import env_kwargs  # noqa: E402


def run(run_dir, cmd, seconds=60., seed=7, model_file='model.zip', init_seed=None):
    a = json.loads((Path(run_dir)/'args.json').read_text())
    kw = {**env_kwargs(a), 'episode_s': seconds+5., 'level': 0.}
    env = RunEnv(**kw, randomize=False, seed=seed)
    env.init_seed = init_seed   # seeded start perturbation (None = the nominal start)
    model, venv = load(str(Path(run_dir)/model_file), env)
    venv.reset()
    env.command = np.array([cmd, 0., 0.]); env.resample_steps = 0
    obs = venv.normalize_obs(env._obs()[None])
    xs, heats, lps, fell, t = [], [], [], False, 0.
    for _ in range(int(seconds/.02)):
        act, _ = model.predict(obs, deterministic=True)
        raw, _, term, trunc, _ = env.step(act[0])
        obs = venv.normalize_obs(raw[None])
        t += .02
        xs.append(float(env.body_velocity(env.base)[1][0])); heats.append(env.heat_inst.copy()); lps.append(env.heat_lp_inst.copy())  # forward speed, body heading
        if term:
            fell = True
            break
    xs, heats, lps = np.array(xs), np.array(heats), np.array(lps)
    n40 = int(40/.02)
    tail = slice(max(0, len(xs)-n40), len(xs))
    v40 = float(np.mean(xs[tail])) if len(xs) else 0.
    w = int(3/.02)
    burst = float(np.convolve(xs, np.ones(w)/w, 'valid').max()) if len(xs) >= w else 0.
    types, types_lp = {}, {}
    for i, n in enumerate(env.active):
        if env.motor_cont[i] < 1e8:
            types.setdefault(joint_type(n), []).append(float(np.mean(heats[tail, i])))
            types_lp.setdefault(joint_type(n), []).append(float(np.mean(lps[tail, i])))
    steady = {k: round(max(v), 2) for k, v in types.items()}          # rounded for display only
    steady_lp = {k: round(max(v), 2) for k, v in types_lp.items()}
    hmax = max(max(v) for v in types.values())                       # verdicts on unrounded values (UNI_AI review)
    hmax_lp = max(max(v) for v in types_lp.values())
    # Preserve precision across the JSON boundary: downstream comparisons must not
    # turn a failing heat value (e.g. 1.0004) or a sub-40 km/h speed into a pass.
    return dict(cmd=cmd, init_seed=init_seed, fell=fell, t_s=round(t, 1), speed_last40_mps=v40, burst3s_mps=burst,
                steady_heat=steady, max_heat=hmax, steady_heat_20hz=steady_lp, max_heat_20hz=hmax_lp,
                mass_kg=round(float(env.model.body_mass.sum()), 2),
                sustainable=(not fell) and hmax <= 1.0, sustainable_20hz=(not fell) and hmax_lp <= 1.0,
                tracked=(not fell) and v40 >= .9*cmd,
                reached_40kmh=(not fell) and hmax <= 1.0 and v40 >= 100/9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_dir')
    p.add_argument('--cmds', type=float, nargs='+', default=[3., 4., 5., 6., 8., 11.1])
    p.add_argument('--seconds', type=float, default=60.)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--model', default='model.zip')
    p.add_argument('--init-seeds', type=int, nargs='+', help='distinct seeded starts per command (default: one nominal start)')
    a = p.parse_args()
    rows = [run(a.run_dir, c, a.seconds, a.seed, a.model, s) for c in a.cmds for s in (a.init_seeds or [None])]
    for r in rows:
        print(json.dumps(r), flush=True)
    ok = [r for r in rows if r['sustainable'] and r['tracked']]
    best = max(ok, key=lambda r: r['speed_last40_mps']) if ok else None
    sus = max((r['speed_last40_mps'] for r in rows if r['sustainable']), default=0.)
    sus_lp = max((r['speed_last40_mps'] for r in rows if r['sustainable_20hz']), default=0.)
    print(json.dumps({'run': a.run_dir, 'highest_sustainable_speed_mps': sus, 'highest_sustainable_speed_20hz_mps': sus_lp,
                      'highest_tracked_sustainable_cmd': best['cmd'] if best else None,
                      'reached_40kmh': any(r['reached_40kmh'] for r in rows), 'falls': sum(r['fell'] for r in rows)}))


if __name__ == '__main__':
    main()
