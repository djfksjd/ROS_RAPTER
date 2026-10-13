"""Summarise the model_rev 2 evidence into rev2_results.json (read by finalize.py and verify.py)."""
import collections, json
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; rd = lambda n: [json.loads(l) for l in (H/n).read_text().splitlines()]


def table(rows):
    g = collections.defaultdict(list)
    for r in rows:
        g[f"{r['label']}@{r['cmd']:g}"].append(r)
    out = {}
    for k, v in sorted(g.items()):
        ok = [r for r in v if not r['fell']]; T = sum(r['t'] for r in v)
        out[k] = dict(n=len(v), falls=sum(r['fell'] for r in v), fall_t_min=min([r['t'] for r in v if r['fell']], default=None), fall_t_max=max([r['t'] for r in v if r['fell']], default=None),
                      speed_min=min([r['speed_tail'] for r in ok], default=None), heat_max=max([r['heat_max'] for r in ok], default=None), heat_knee_median=float(np.median([r['heat_knee'] for r in v])),
                      inject_W=sum(r['energy_injected_J'] for r in v)/T, engagements=sum(r['engagements'] for r in v), engage_nonzero=sum(r['engage_nonzero'] for r in v),
                      latched_stance_frac=float(np.mean([r['latched_stance_frac'] for r in v])), self_contact_step_frac=float(np.mean([r['self_contact_step_frac'] for r in v])),
                      pad_clearance_start_mm=float(np.mean([r['pad_clearance_start_m'] for r in v])*1000))
    return out


opt, spr, st = rd('options.jsonl'), rd('springs.jsonl'), rd('stand.jsonl')
sid = json.loads((H/'spring_id.json').read_text()); drop = json.loads((H/'drop_rig.json').read_text())
S = table(spr); none = {c: S[f'none@{c}'] for c in ('2', '4', '7')}
same_as_none = {lab: all(all(S[f'{lab}@{c}'][k] == none[c][k] for k in ('falls', 'speed_min', 'heat_max')) for c in ('2', '4', '7')) for lab in sorted({r['label'] for r in spr})}
res = dict(regression=json.loads((H/'regression.json').read_text()), options=table(opt), springs=S, springs_identical_to_none=same_as_none, stand=table(st),
           rollouts=dict(options=len(opt), springs=len(spr), stand=len(st), spring_replay_recordings=3, drop_rig=len(drop), regression=4),
           spring_replay={c: dict(gait=sid[c]['gait'], candidates=sid[c]['candidates']) for c in ('2', '4', '7')},
           drop_rig={f"{r['label']}@h{r['h']:g}": {k: r[k] for k in ('k', 'q_td', 'h', 'knee_flex', 'motor_peak_ratio', 'motor_sat_frac', 'heat_integral_s', 'motor_neg_J', 'spring_E_peak_J', 'drop_energy_J', 'grf_peak_per_leg_BW', 'restitution', 'rig_tilt_max', 'stance_s', 'lifted_off')} for r in drop if r['touched'] and '_q' not in r['label']},
           static_knee_capacity=json.loads((H/'static_knee_capacity.json').read_text()))
(H/'rev2_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(json.dumps(res['regression'])); print(same_as_none); print(res['rollouts'])
