"""Finite decision-B evaluations: two full policy sets, shared w2 regressions.

No training and no runs/ writes. All episodes use the existing full-gate routines.
Run from the repository root with .venv-sim/bin/python; outputs refuse overwrite.
"""
import concurrent.futures
import hashlib
import json
import multiprocessing
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'sim/rl'))
import torch
import stage_gates as sg
from turn_rules import TURN_CRITERIA

OUT = Path(__file__).resolve().parent/'eval/decision_b'
WALK = ROOT/'sim/rl/runs/l80_w2'
RUNS = ('l80_t3', 'l80_r1_6M')


def evaluate(task):
    torch.set_num_threads(1)
    run, kind, value = task
    d = ROOT/'sim/rl/runs'/run
    if kind in ('stand', 'walk', 'switch'):
        rows = getattr(sg, kind)(WALK, run_dir=d) if kind == 'switch' else getattr(sg, kind)(WALK)
        return [dict(gate=kind, run=WALK.name, run_policy=run, **r) for r in rows]
    if kind == 'turn':
        speed, sign = value
        policy = WALK if speed == 1. else d
        return [dict(gate='turn', run=policy.name, quick=False,
                     **sg.turn_condition(policy, speed, sign*TURN_CRITERIA[speed]['yaw'], range(7300, 7304)))]
    if kind in ('run_regression', 'speed'):
        episodes = []
        for seed in range(6700, 6704):
            fell, v, der, hot = sg.run_rollout(d, seed, value, 20.)
            episodes.append(dict(seed=seed, t_fall=fell, v_mean=float(v[500:].mean()) if fell is None else None,
                                 min_derate=der, t_winding_max_c=hot))
        return [dict(gate='turn' if kind == 'run_regression' else 'run', run=run, cond=kind, cmd=value,
                     n=4, falls=sum(e['t_fall'] is not None for e in episodes), episodes=episodes)]
    if kind == 'heat':
        cmd, mount = value
        fell, v, der, hot = sg.run_rollout(d, 6800, cmd, 600., mount=mount)
        return [dict(gate='run', run=run, cond=kind, cmd=cmd, mount=mount, n=1, falls=int(fell is not None),
                     t_fall=fell, min_derate=der, t_winding_max_c=hot, v_mean=round(float(v[500:].mean()), 3))]
    assert kind == 'lateral'
    episodes = []
    for seed in range(6900, 6908):
        fell, _, _, _ = sg.run_rollout(d, seed, 4., 15., kick=(0., .5 if value == 'left' else -.5))
        episodes.append(dict(seed=seed, t_fall=fell))
    return [dict(gate='run', run=run, cond=kind, dir=value, cmd=4., n=8,
                 falls=sum(e['t_fall'] is not None for e in episodes), episodes=episodes)]


def hashes():
    paths = [ROOT/'sim/rl/runs'/r/f for r in (*RUNS, WALK.name) for f in ('args.json', 'model.zip', 'vecnorm.pkl')]
    paths += [ROOT/'sim/rl'/f for f in ('stage_gates.py', 'gate_judge.py', 'turn_eval.py', 'turn_rules.py',
                                      'run_env.py', 'train_run.py', 'rev3_l80.json', 'realism_robstride_l80.json')]
    paths.append(Path(__file__))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    OUT.mkdir(exist_ok=False)
    before = hashes()
    (OUT/'inputs.json').write_text(json.dumps(dict(before=before), indent=2)+'\n')
    tasks = [(r, 'turn', (v, s)) for r in RUNS for v in TURN_CRITERIA for s in (1., -1.)]
    tasks += [(r, 'switch', None) for r in RUNS]
    tasks += [(r, 'run_regression', v) for r in RUNS for v in (2., 4., 6.)]
    tasks += [(WALK.name, gate, None) for gate in ('stand', 'walk')]
    tasks += [(r, 'heat', (v, m)) for r in RUNS for v in (4., 5., 6.) for m in ('plate', 'worst')]
    tasks += [(r, 'speed', 8.) for r in RUNS]
    tasks += [(r, 'lateral', side) for r in RUNS for side in ('left', 'right')]
    files = {(r, part): (OUT/f'{r}_{part}.jsonl').open('x') for r in RUNS for part in ('full', 'extra')}
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn')) as pool:
            pending = {pool.submit(evaluate, t): t for t in tasks}
            for future in concurrent.futures.as_completed(pending):
                task = pending[future]
                rows = future.result()
                targets = RUNS if task[0] == WALK.name else (task[0],)
                part = 'extra' if task[1] in ('heat', 'speed', 'lateral') else 'full'
                for r in targets:
                    for row in rows:
                        files[r, part].write(json.dumps(row, allow_nan=False)+'\n')
                    files[r, part].flush()
                print(json.dumps(dict(task=task, rows=len(rows), falls=sum(r['falls'] for r in rows))), flush=True)
    finally:
        for f in files.values():
            f.close()
        after = hashes()
        (OUT/'inputs.json').write_text(json.dumps(dict(before=before, after=after, unchanged=before == after), indent=2)+'\n')
    assert before == after


if __name__ == '__main__':
    main()
