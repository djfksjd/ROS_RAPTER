"""Aggregate diagnosis.jsonl and draw one plain chart, without running a policy."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

EVIDENCE = Path(__file__).resolve().parent


def main():
    rows = [json.loads(s) for s in (EVIDENCE/'eval/diagnosis.jsonl').read_text().splitlines()]
    assert len(rows) == 48 and len({(r['run'], r['speed'], r['yaw'], r['seed']) for r in rows}) == 48
    assert all(r['complete'] and r['joint_samples'] == 10000 for r in rows)
    groups = []
    for run in ('l80_r1_6M', 'l80_t1'):
        for v in (2., 4., 6.):
            for w in (.5, -.5, 1., 1.5):
                g = [r for r in rows if (r['run'], r['speed'], r['yaw']) == (run, v, w)]
                cf = [c for r in g for c in r['counterfactual']]
                joints = {}
                for kind in ('hip_yaw', 'hip_roll'):
                    jj = [j for r in g for n, j in r['joints'].items() if kind in n]
                    joints[kind] = dict(angle_min=min(j['angle_min'] for j in jj), angle_max=max(j['angle_max'] for j in jj),
                        max_stop_share=max(j['near_stop_share'] for j in jj), max_torque_share=max(j['torque_at_limit_share'] for j in jj),
                        max_peak_nm=max(j['torque_peak_nm'] for j in jj), max_rms_nm=max(j['torque_rms_nm'] for j in jj),
                        action_clip_mean=float(np.mean([j['action_at_clip_share'] for j in jj])),
                        action_clip_max=max(j['action_at_clip_share'] for j in jj))
                groups.append(dict(run=run, speed=v, yaw=w, n=len(g),
                    steady={k: float(np.mean([r['steady'][k] for r in g])) for k in g[0]['steady']},
                    rewards={k: float(np.mean([r['weighted_reward_per_second'][k] for r in g]))
                             for k in g[0]['weighted_reward_per_second']}, joints=joints,
                    cf_l2_mean=float(np.mean([c['l2'] for c in cf])), cf_l2_min=min(c['l2'] for c in cf),
                    cf_target_abs_mean=np.mean([np.abs(c['hip_delta_target_rad']) for c in cf], axis=0).tolist()))
    physics = []
    # Necessary average-force screen: level constant-speed circular motion, no slip.
    # Standard w=heading rate; separately map the actual gate's body gyro-z to lean
    # for an ideal banked turn (pitch=0): w_body=w_heading*cos(lean).
    for v in (2., 4., 6.):
        for w in (.5, 1., 1.5):
            a = v*w; ratio = a/9.81
            bank_body = math.asin(ratio)
            physics.append(dict(speed=v, yaw=w, v_times_w=a, mu_required_heading=ratio,
                lean_heading_deg=math.degrees(math.atan(ratio)),
                lean_body_z_deg=math.degrees(bank_body), mu_required_body_z=math.tan(bank_body),
                body_z_below_tilt_limit=bank_body < .8,
                available_a_mu08=9.81*.8, available_a_mu12=9.81*1.2))
    ideal_reward = [dict(yaw=w, straight_track_yaw=4*math.exp(-w*w/.2),
        straight_yaw_err=-w, perfect_track_yaw=4., perfect_track_lin=5., perfect_progress=3.,
        straight_track_yaw_slope=40*w*math.exp(-w*w/.2)) for w in (.25, .5, 1., 1.5)]
    # Cross-check instrumentation/counterfactual queries against existing unguided trajectories.
    reference = [json.loads(s) for s in (EVIDENCE/'eval/turn_baseline.jsonl').read_text().splitlines()]
    reference += [json.loads(s) for s in (EVIDENCE/'eval/l80_t1/turn_3M.jsonl').read_text().splitlines()]
    errors = []
    for r in rows:
        candidates = [e for ref in reference if ref['cmd'] == r['speed'] and ref['yaw_cmd'] == r['yaw']
                      and ref['policy'] == ('l80_t1_3M' if r['run'] == 'l80_t1' else r['run'])
                      for e in ref['episodes'] if e['seed'] == r['seed']]
        if candidates:
            ref = candidates[0]
            errors.append(max(abs(ref['v_mean']-r['steady']['vx']), abs(ref['yaw_rate_mean']-r['steady']['wz'])))
    assert len(errors) == 28 and max(errors) < 1e-12
    result = dict(episodes=len(rows), falls=sum(r['t_fall'] is not None for r in rows),
                  instrumentation=dict(reference_episodes=len(errors), max_velocity_difference=max(errors)),
                  normalization={run: next(r['normalization'] for r in rows if r['run'] == run)
                                 for run in ('l80_r1_6M', 'l80_t1')},
                  physics=physics, ideal_reward=ideal_reward, groups=groups)
    (EVIDENCE/'eval/diagnosis_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, v in zip(axes, (2., 4., 6.)):
        ax.plot([0, 1.5], [0, 1.5], '--', color='.6', lw=1, label='Command')
        for run, color in [('l80_r1_6M', '#333333'), ('l80_t1', '#1f77b4')]:
            g = [r for r in groups if r['run'] == run and r['speed'] == v and r['yaw'] > 0]
            ax.plot([r['yaw'] for r in g], [r['steady']['wz'] for r in g], 'o-', color=color, ms=4, label=run)
            raw = [r for r in rows if r['run'] == run and r['speed'] == v and r['yaw'] > 0]
            ax.scatter([r['yaw'] for r in raw], [r['steady']['wz'] for r in raw], color=color, s=10, alpha=.5)
        ax.set(title=f'{v:g} m/s command', xlabel='Yaw command (rad/s)', xlim=(0, 1.6), ylim=(0, 1.65))
        ax.set_xticks([0, .5, 1, 1.5]); ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='.9', lw=.5)
    axes[0].set_ylabel('Measured body yaw rate (rad/s)')
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('No training: randomize off, seeds 7200/7201, mean over 8–15 s', fontsize=10)
    fig.tight_layout()
    (EVIDENCE/'media').mkdir(exist_ok=True)
    fig.savefig(EVIDENCE/'media/turn_diagnosis.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
