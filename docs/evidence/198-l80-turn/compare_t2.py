"""24 fixed diagnostic episodes or offline summary; never train or write runs/."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from diagnose_turn import EVIDENCE, ROOT, rollout
from audit_check import audit

RUNS = ('l80_t1_3M', 'l80_t2_3M')
CONDITIONS = ((2., 1.), (2., -1.), (4., .5), (4., -.5), (6., .5), (6., -.5))
SEEDS = (7200, 7201)
OUT = EVIDENCE/'eval/t2_comparison.jsonl'


def probe():
    if OUT.exists():
        raise FileExistsError(f'refusing to overwrite {OUT}')
    traces = OUT.with_suffix('')
    traces.mkdir(exist_ok=False)
    files = [ROOT/'sim/rl/runs'/r/n for r in RUNS for n in ('model.zip', 'vecnorm.pkl', 'args.json')]
    files += [ROOT/'sim/rl'/n for n in ('rev3_l80.json', 'realism_robstride_l80.json',
              'run_env.py', 'train_run.py', 'evaluate.py', 'realism.py')]
    files += [EVIDENCE/'diagnose_turn.py', Path(__file__)]
    def digest():
        return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    before = digest()
    torch.set_num_threads(1)
    try:
        with OUT.open('x') as f:
            for run in RUNS:
                for v, w in CONDITIONS:
                    for seed in SEEDS:
                        row = rollout(run, v, w, seed, traces)
                        vm, wm = row['steady']['vx'], row['steady']['wz']
                        row['radius_m'] = abs(vm/wm) if abs(wm) > 1e-6 else None
                        f.write(json.dumps(row, allow_nan=False)+'\n'); f.flush()
                        print(run, v, w, seed, row['complete'], vm, wm, flush=True)
    finally:
        after = digest()
        OUT.with_suffix('.manifest.json').write_text(json.dumps(dict(before=before, after=after,
            unchanged=before == after, seeds=SEEDS, conditions=CONDITIONS, runs=RUNS,
            randomize=False, training=False, episode_budget=24), indent=2)+'\n')
    assert before == after


def summarize():
    assert json.loads(OUT.with_suffix('.manifest.json').read_text())['unchanged']
    rows = [json.loads(s) for s in OUT.read_text().splitlines()]
    expected = {(r, v, w, s) for r in RUNS for v, w in CONDITIONS for s in SEEDS}
    assert len(rows) == 24 and {(r['run'], r['speed'], r['yaw'], r['seed']) for r in rows} == expected
    assert all(r['complete'] and r['joint_samples'] == 10000 for r in rows)
    groups = []
    for run in RUNS:
        for v, w in CONDITIONS:
            g = [r for r in rows if (r['run'], r['speed'], r['yaw']) == (run, v, w)]
            local_slopes = []
            for r in g:
                with np.load(OUT.with_suffix('')/r['trace']) as trace:
                    data = trace['data']
                    error = w-data[data[:, 0] > 8.+1e-6, 6]
                    # Analytic reward sensitivity at recorded states, not a policy gradient.
                    local_slopes.extend((40*np.abs(error)*np.exp(-error**2/.2)).tolist())
            joints = {}
            for kind in ('hip_yaw', 'hip_roll'):
                jj = [j for r in g for n, j in r['joints'].items() if kind in n]
                actions = []
                for r in g:
                    with np.load(OUT.with_suffix('')/r['trace']) as trace:
                        columns = trace['columns'].tolist()
                        indices = [i for i, c in enumerate(columns) if c.startswith('action:') and kind in c]
                        actions.extend(np.abs(trace['data'][250:, indices]).ravel().tolist())
                joints[kind] = dict(angle_min=min(j['angle_min'] for j in jj), angle_max=max(j['angle_max'] for j in jj),
                    max_stop_share=max(j['near_stop_share'] for j in jj), max_torque_share=max(j['torque_at_limit_share'] for j in jj),
                    max_peak_nm=max(j['torque_peak_nm'] for j in jj), max_rms_nm=max(j['torque_rms_nm'] for j in jj),
                    action_clip_mean=float(np.mean([j['action_at_clip_share'] for j in jj])),
                    action_exact_clip_mean=float(np.mean(np.asarray(actions) >= 1.)),
                    action_abs_mean=float(np.mean(actions)), action_abs_p95=float(np.quantile(actions, .95)),
                    action_clip_max=max(j['action_at_clip_share'] for j in jj))
            groups.append(dict(run=run, speed=v, yaw=w, falls=sum(r['t_fall'] is not None for r in g),
                steady={k: float(np.mean([r['steady'][k] for r in g])) for k in g[0]['steady']},
                radius_mean=float(np.mean([r['radius_m'] for r in g])),
                radius_per_seed=[r['radius_m'] for r in g],
                track_yaw_local_slope_mean=float(np.mean(local_slopes)),
                contact_ge98_mean=float(np.mean([r['contact']['tangent_utilization_ge98_share'] for r in g])),
                rewards={k: float(np.mean([r['weighted_reward_per_second'][k] for r in g])) for k in g[0]['weighted_reward_per_second']},
                joints=joints))
    differences = []
    checks = []
    from gate_judge import judge
    for run in ('l80_t1', 'l80_t2'):
        for block in (1, 2, 3):
            refs = [json.loads(s) for s in (EVIDENCE/f'eval/{run}/turn_{block}M.jsonl').read_text().splitlines()]
            for ref in refs:
                if ref['cmd'] < 2:
                    continue
                for e in ref['episodes']:
                    passed, why = judge({**ref, 'n': 1, 'falls': int(e['t_fall'] is not None), 'episodes': [e]})
                    checks.append(dict(run=run, block=block, cmd=ref['cmd'], yaw_cmd=ref['yaw_cmd'],
                                       passed=passed, reason=why, **e))
                    if block == 3:
                        found = next(r for r in rows if (r['run'], r['speed'], r['yaw'], r['seed']) ==
                                     (run+'_3M', ref['cmd'], ref['yaw_cmd'], e['seed']))
                        differences.append(max(abs(e['v_mean']-found['steady']['vx']),
                                               abs(e['yaw_rate_mean']-found['steady']['wz'])))
    assert len(differences) == 8 and max(differences) < 1e-12
    result = dict(episodes=24, falls=sum(r['t_fall'] is not None for r in rows),
        instrumentation=dict(reference_episodes=8, max_velocity_difference=max(differences)),
        step_caveat='Equal additional steps per run; t2 resumes t1 3M, so this is not equal cumulative training or a controlled causal ablation.',
        audits={run: audit(run) for run in ('l80_t1', 'l80_t2')}, groups=groups, saved_check_episodes=checks)
    (EVIDENCE/'eval/t2_comparison_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    x = np.arange(6)
    for run, offset, color, label in zip(RUNS, (-.12, .12), ('#666666', '#2364a0'), ('t1 +3M', 't2 +3M (resumes t1)')):
        g = [r for r in groups if r['run'] == run]
        for i, (v, w) in enumerate(CONDITIONS):
            raw = [np.sign(w)*r['steady']['wz'] for r in rows if (r['run'], r['speed'], r['yaw']) == (run, v, w)]
            ax.scatter([i+offset]*2, raw, color=color, s=18, alpha=.55)
        ax.plot(x+offset, [np.sign(r['yaw'])*r['steady']['wz'] for r in g], 'o', color=color, label=label)
    ax.scatter(x, [.65, .65, .325, .325, .325, .325], marker='_', s=450, color='black', label='Yaw lower bound only')
    ax.set_xticks(x, ['2 / +1', '2 / -1', '4 / +0.5', '4 / -0.5', '6 / +0.5', '6 / -0.5'])
    ax.set(xlabel='Forward / yaw command (m/s / rad/s)', ylabel='Signed-direction body yaw rate (rad/s)', ylim=(0, 1.1))
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='.9'); ax.legend(frameon=False, fontsize=9)
    ax.set_title('Same +2,998,272 steps per run; randomize off, seeds 7200/7201\n8–15 s means; dots show both seeds; other gate criteria still apply', fontsize=10)
    fig.tight_layout(); fig.savefig(EVIDENCE/'media/t1_vs_t2.png', dpi=160); plt.close(fig)
    print(json.dumps(dict(episodes=24, instrumentation=result['instrumentation'], audits=result['audits']), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['probe', 'summary'])
    a = p.parse_args()
    probe() if a.mode == 'probe' else summarize()
