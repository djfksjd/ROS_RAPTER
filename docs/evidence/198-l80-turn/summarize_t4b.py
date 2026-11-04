"""Audit t4b's fixed budget and compare three policy sets from measured JSONL."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from check_training_b import MAX_STEPS, metrics, stop_reason

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'sim/rl'))
from gate_judge import judge, sustained_speed, turn_episode_ok, turn_suite_status
from turn_rules import TURN_CRITERIA

E = Path(__file__).resolve().parent
OUT = E/'eval/t4b'


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row):
    return tuple(row.get(k) for k in ('gate', 'cond', 'cmd', 'yaw_cmd', 'dir', 'dv', 'mount'))


def audit():
    run = ROOT/'sim/rl/runs/l80_t4b'
    history = json.loads((run/'turn_checks_b.json').read_text())
    assert len(history) == 2
    blocks, conditions = [], []
    for entry in history:
        block = entry['block']
        rows = read(E/f'eval/l80_t4b/development_{block}M.jsonl')
        assert len(rows) == 11 and sum(r['n'] for r in rows) == 22
        assert metrics(rows) == {k: entry[k] for k in metrics(rows)}
        conditions.append(sorted((r['cond'], r['cmd'], r.get('yaw_cmd', 0), e['seed']) for r in rows for e in r['episodes']))
        with (run.with_name(f'l80_t4b_{block}M')/'progress.csv').open() as f:
            lengths = [float(r['rollout/ep_len_mean']) for r in csv.DictReader(f) if r.get('rollout/ep_len_mean')]
        assert entry['added_steps'] == 999_424*block
        assert stop_reason(entry['added_steps'], lengths) == entry['decision']
        blocks.append(dict(**entry, first10_length=sum(lengths[:10])/10,
                           last10_length=sum(lengths[-10:])/10, updates=len(lengths)))
    assert conditions[0] == conditions[1]
    initial = json.loads((run/'initial-policy.json').read_text())['steps']
    args = json.loads((run/'args.json').read_text())
    previous = json.loads((run.with_name('l80_t3')/'args.json').read_text())
    differences = {k for k in args.keys() | previous.keys() if args.get(k) != previous.get(k)}
    assert differences == {'name', 'init', 'steps', 'weights', 'checkpoint_check'}
    old_w, new_w = json.loads(previous['weights']), json.loads(args['weights'])
    assert {k for k in old_w if old_w[k] != new_w[k]} == {'track_lin'}
    assert (old_w['track_lin'], new_w['track_lin']) == (5, 8)
    with zipfile.ZipFile(run/'model.zip') as z:
        saved = json.loads(z.read('data'))
    with (run/'progress.csv').open() as f:
        progress = list(csv.DictReader(f))
    final = int(progress[-1]['time/total_timesteps'])
    assert final == saved['num_timesteps'] and final-initial == MAX_STEPS
    rollout = saved['n_steps']*saved['n_envs']
    assert int(args['steps']) == 2_000_000 and int(args['steps'])//rollout*rollout == MAX_STEPS
    log = [json.loads(l) for l in (ROOT/'sim/rl/runs_l80_t4b.log').read_text().splitlines() if l.startswith('{')]
    assert [r for r in log if 'block' in r] == history
    assert [r['steps']-initial for r in log if 'steps' in r] == [999_424, MAX_STEPS]
    same = {name: digest(run/name) == digest(run.with_name('l80_t4b_2M')/name) for name in ('model.zip', 'vecnorm.pkl')}
    assert all(same.values())
    stored_full = read(E/'eval/l80_t4b/full_2M.jsonl')
    eligible, reason = turn_suite_status(stored_full)
    stored_adoption = json.loads((E/'eval/l80_t4b/adoption_2M.json').read_text())
    assert stored_adoption == dict(eligible=eligible, reason=reason)
    result = dict(blocks=blocks, same_development_conditions=True, initial_steps=initial, final_steps=final,
                  declared_budget=2_000_000, consumed=MAX_STEPS, unused_usable_budget=0,
                  rollout_steps=rollout, current_equals_2M=same, log_and_progress_match=True,
                  only_reward_change={'track_lin': [5, 8]}, stored_adoption=stored_adoption,
                  termination_valid=True, extension_allowed=False)
    (OUT/'check_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def summarize():
    manifest = json.loads((OUT/'inputs.json').read_text())
    assert manifest['unchanged'] and manifest['before'] == manifest['after']
    assert all(digest(ROOT/p) == h for p, h in manifest['after'].items())
    expected = {key(r) for r in read(E/'eval/l80_t4b/full_2M.jsonl')}
    old_inputs = json.loads((E/'eval/decision_b/inputs.json').read_text())['after']
    assert all(digest(ROOT/p) == h for p, h in old_inputs.items())
    result, chart_data = {}, []
    for run in ('l80_r1_6M', 'l80_t3', 'l80_t4b'):
        directory = OUT if run == 'l80_t4b' else E/'eval/decision_b'
        full, extra = (read(directory/f'{run}_{part}.jsonl') for part in ('full', 'extra'))
        assert len(full) == 36 and sum(r['n'] for r in full) == 240
        assert len(extra) == 9 and sum(r['n'] for r in extra) == 26
        assert len({key(r) for r in full+extra}) == 45 and {key(r) for r in full} == expected
        assert {(r['cmd'], r['mount']) for r in extra if r['cond'] == 'heat'} == {
            (v, m) for v in (4., 5., 6.) for m in ('plate', 'worst')}
        turns, straight, verdicts = [], [], []
        for r in full+extra:
            if 'episodes' in r:
                assert len(r['episodes']) == r['n']
                assert r['falls'] == sum(e['t_fall'] is not None for e in r['episodes'])
            verdicts.append(dict(selector=key(r), passed=judge(r)[0], reason=judge(r)[1]))
            if r['cond'] == 'moving_turn':
                assert {e['seed'] for e in r['episodes']} == set(range(7300, 7304)) and not r['quick']
                assert all(e['load_samples'] == 10000 and len(e['joints']) == 2 for e in r['episodes'])
                item = dict(cmd=r['cmd'], yaw_cmd=r['yaw_cmd'], passed=sum(turn_episode_ok(r, e) for e in r['episodes']),
                            n=r['n'], falls=r['falls'], judged=r['cmd'] != 6.)
                for k in ('v_mean', 'yaw_rate_mean', 'radius_m', 't_90_s'):
                    values = [e[k] for e in r['episodes'] if e[k] is not None]
                    item[k] = [min(values), max(values)] if values else None
                turns.append(item)
            if r['cond'] == 'run_regression':
                values = [e['v_mean'] for e in r['episodes'] if e['v_mean'] is not None]
                straight.append(dict(cmd=r['cmd'], passed=judge(r)[0], falls=r['falls'], n=r['n'],
                    v_min=min(values), v_max=max(values), v_mean=sum(values)/len(values)))
        suites = {}
        for gate in ('stand', 'walk', 'switch'):
            rr = [r for r in full if r['gate'] == gate]
            jj = [r for r in rr if judge(r)[0] is not None]
            suites[gate] = dict(passing_conditions=sum(judge(r)[0] is True for r in jj),
                judged_conditions=len(jj), falls=sum(r['falls'] for r in rr), episodes=sum(r['n'] for r in rr))
        exit_codes = {}
        for part in ('full', 'extra'):
            with (OUT/f'{run}_{part}.judge.txt').open('w') as f:
                proc = subprocess.run([sys.executable, str(ROOT/'sim/rl/gate_judge.py'), str(directory/f'{run}_{part}.jsonl')],
                                      stdout=f, check=False)
            exit_codes[part] = proc.returncode
        result[run] = dict(full_suite=dict(zip(('passed', 'reason'), turn_suite_status(full))), turns=turns,
            judged_turn_passed=sum(r['passed'] for r in turns if r['judged']), judged_turn_total=24,
            straight=straight, regressions=suites, heat=[r for r in extra if r['cond'] == 'heat'],
            sustained_command=sustained_speed(extra), speed8=next(r for r in extra if r['cond'] == 'speed'),
            lateral=[r for r in extra if r['cond'] == 'lateral'], verdicts=verdicts, judge_exit_codes=exit_codes)
        chart_data.append(dict(run=run, turns=[r for r in turns if r['judged']], straight=straight))
    recorded = {key(r): r for r in read(E/'eval/l80_t4b/full_2M.jsonl')}
    for row in read(OUT/'l80_t4b_full.jsonl'):
        assert row['episodes'] == recorded[key(row)]['episodes'], key(row)
    result['verification'] = dict(new_rollouts=266, input_hashes_unchanged=len(manifest['after']),
                                  previous_comparison_inputs_unchanged=True, automatic_full_240_episodes_identical=True)
    (OUT/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (OUT/'chart_data.json').write_text(json.dumps(chart_data, indent=2)+'\n')
    chart(chart_data)
    return result


def chart(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    conditions = [(v, sign*TURN_CRITERIA[v]['yaw']) for v in (1., 2., 4.) for sign in (1., -1.)]
    colors = ('#777777', '#397fa6', '#bb7335')
    for j, (item, color) in enumerate(zip(data, colors)):
        offset = (j-1)*.23
        passed = [next(r['passed'] for r in item['turns'] if (r['cmd'], r['yaw_cmd']) == c) for c in conditions]
        axes[0].bar(np.arange(6)+offset, passed, width=.22, color=color, label=item['run'].replace('l80_', ''))
        rr = sorted(item['straight'], key=lambda r: r['cmd'])
        means = [r['v_mean']/r['cmd'] for r in rr]
        lower = [(r['v_mean']-r['v_min'])/r['cmd'] for r in rr]
        upper = [(r['v_max']-r['v_mean'])/r['cmd'] for r in rr]
        axes[1].errorbar(np.arange(3)+offset, means, yerr=[lower, upper], fmt='o', capsize=4, color=color)
    axes[0].set_xticks(range(6), [f'{v:g} / {w:+g}' for v, w in conditions])
    axes[0].set_ylabel('Passing episodes / 4'); axes[0].set_ylim(0, 4.7); axes[0].set_yticks(range(5))
    axes[0].set_xlabel('Turn: forward / yaw command (m/s / rad/s)')
    axes[0].legend(frameon=False, ncol=3, loc='upper right')
    axes[1].axhspan(.9, 1.1, color='.93', label='Allowed range')
    axes[1].axhline(1., color='.4', lw=.8, ls='--')
    axes[1].axhline(.9, color='.6', lw=.8); axes[1].axhline(1.1, color='.6', lw=.8)
    axes[1].set_xticks(range(3), ['2', '4', '6']); axes[1].set_ylim(.86, 1.26)
    axes[1].set_xlabel('Straight command (m/s)'); axes[1].set_ylabel('Actual speed / command')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('Policy sets: fixed w2 + candidate run policy\nSame reused gate seeds; 6 m/s turns deferred. Straight points: mean with min/max.', fontsize=11)
    fig.tight_layout(); fig.savefig(E/'media/candidate_sets_t4b.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(audit(), indent=2))
    if not args.audit_only:
        result = summarize()
        print(json.dumps({run: {k: value[k] for k in ('full_suite', 'judged_turn_passed', 'sustained_command')}
                          for run, value in result.items() if run != 'verification'}, indent=2))
