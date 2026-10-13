"""Aggregate the gate rollouts into gate_results.json (numbers quoted by the report)."""
import json, sys
from pathlib import Path
import numpy as np
G = Path(sys.argv[1]); L = lambda n: [json.loads(l) for l in (G/n).read_text().splitlines()]
main, sc, combo = L('gate_rollouts.jsonl'), L('gate_rollouts_selfcol.jsonl'), L('gate_rollouts_combo.jsonl')
rows = [r for r in main if r['mode'] == 'policy' and r['kind'] not in ('base', 'selfcol')]+sc+combo   # base/selfcol re-run after the collision-mask fix
zero = [r for r in main if r['mode'] == 'zero_action']
base_old = {(r['cmd'], r['init_seed']): r for r in main if r['kind'] == 'base' and r['mode'] == 'policy'}
out = {'conditions': {}, 'zero_action': {}}
for kind in dict.fromkeys(r['kind'] for r in rows):
    for cmd in (4., 7.):
        rs = [r for r in rows if r['kind'] == kind and r['cmd'] == cmd]; ok = [r for r in rs if not r['fell']]; T = sum(r['t'] for r in rs)
        c = dict(n=len(rs), falls=len(rs)-len(ok), fall_times=sorted(round(r['t'], 2) for r in rs if r['fell']), speed_min=min((r['speed_tail'] for r in ok), default=None),
                 speed_mean=float(np.mean([r['speed_tail'] for r in ok])) if ok else None, heat_max=max((r['heat_max'] for r in ok), default=None), heat_pass=sum(r['heat_max'] <= 1. for r in ok),
                 knee_min=min(r['knee_min'] for r in rs), knee_max=max(r['knee_max'] for r in rs), knee_frac_outside_band=float(np.mean([r['knee_frac_below_band']+r['knee_frac_above_band'] for r in rs])),
                 engagements_per_s=sum(r['engagements'] for r in rs)/T, engage_nonzero_frac=sum(r['engage_nonzero'] for r in rs)/max(1, sum(r['engagements'] for r in rs)),
                 engage_defl_mean=float(np.mean([r['engage_abs_defl_mean'] for r in rs])), engage_defl_max=max(r['engage_abs_defl_max'] for r in rs),
                 inject_W=sum(r['energy_injected_J'] for r in rs)/T, release_W=sum(r['energy_released_J'] for r in rs)/T, motor_pos_W=sum(r['motor_positive_work_J'] for r in rs)/T,
                 start_gap_mm=[round(1000*min(r['start_gap_m'] for r in rs), 2), round(1000*max(r['start_gap_m'] for r in rs), 2)])
        c['engagements'] = sum(r['engagements'] for r in rs); c['engagements_zero_deflection'] = c['engagements']-sum(r['engage_nonzero'] for r in rs)
        c['inject_over_motor'] = c['inject_W']/c['motor_pos_W']; c['inject_share_of_positive_work'] = c['inject_W']/(c['inject_W']+c['motor_pos_W'])
        if kind == 'base':
            ov, steps = {}, sum(int(round(r['t']/.02)) for r in rs)
            for r in rs:
                for k, v in r['overlaps'].items():
                    ov[k] = [ov.get(k, [0, 0.])[0]+v[0], max(ov.get(k, [0, 0.])[1], v[1])]
            c.update(overlap_step_frac=float(np.mean([r['overlap_step_frac'] for r in rs])), rollouts_with_overlap=sum(bool(r['overlaps']) for r in rs),
                     overlap_max_penetration_mm=1000*max((v[1] for v in ov.values()), default=0.), overlap_pairs=len(ov), hip_roll_in_max=max(r['hip_roll_in_max'] for r in rs),
                     identical_to_first_run=all(abs(r['speed_tail']-base_old[(r['cmd'], r['init_seed'])]['speed_tail']) < 1e-12 and abs(r['heat_max']-base_old[(r['cmd'], r['init_seed'])]['heat_max']) < 1e-12 for r in rs))
        if 'hip_roll_in_max' in rs[0]:
            c['hip_roll_in_max'] = max(r['hip_roll_in_max'] for r in rs)
        out['conditions'][f'{kind}@{cmd:g}'] = c
for kind in dict.fromkeys(r['kind'] for r in zero):
    rs = [r for r in zero if r['kind'] == kind]
    out['zero_action'][kind] = dict(n=len(rs), fall_times=sorted(r['t'] for r in rs), knee_heat=[min(r['heat_max'] for r in rs), max(r['heat_max'] for r in rs)], heat_motor=sorted({r['heat_argmax'] for r in rs}),
                                    vz_max=max(r['vz_max'] for r in rs), initial_preload_J=rs[0]['initial_preload_J'], injected_J=[min(r['energy_injected_J'] for r in rs), max(r['energy_injected_J'] for r in rs)])
ok60 = [r['wall_s'] for r in rows if not r['fell']]
out['timing'] = dict(wall_s_per_60s_rollout=float(np.mean(ok60)), rollouts=len(rows)+len(zero)+sum(1 for r in main if r['mode'] == 'policy' and r['kind'] in ('base', 'selfcol')))
out['statistics'] = {str(n): 1-.05**(1/n) for n in (10, 20, 30, 59, 299)}
out['seeds'] = sorted({r['init_seed'] for r in rows}); out['probe_check'] = json.loads((G/'selfcol_probe_check.json').read_text())
JR = 81*1.002e-4
out['armature'] = dict(rotor_inertia_kgm2=1.002e-4, internal_ratio=9, module_output=JR, hip_pitch=JR*1.5**2, knee_at_N1p4=JR*1.4**2, tail=JR, hip_roll_assumed=JR*1.7**2)
(G/'gate_results.json').write_text(json.dumps(out, indent=1))
for k, c in out['conditions'].items():
    f = lambda x, n=3: '-' if x is None else f'{x:.{n}f}'
    print(f"{k:42s} falls {c['falls']}/{c['n']} vmin {f(c['speed_min'])} heat {f(c['heat_max'])} pass {c['heat_pass']} knee [{f(c['knee_min'])},{f(c['knee_max'])}] out {f(c['knee_frac_outside_band'],4)} inj {f(c['inject_W'],1)} motor {f(c['motor_pos_W'],1)} ratio {f(c['inject_over_motor'])} nonzero {f(c['engage_nonzero_frac'],2)}")
print(json.dumps(out['zero_action'])[:900]); print(out['timing'], out['statistics'])
print({k: {x: c[x] for x in ('overlap_step_frac', 'rollouts_with_overlap', 'overlap_max_penetration_mm', 'overlap_pairs', 'hip_roll_in_max', 'identical_to_first_run')} for k, c in out['conditions'].items() if k.startswith('base')})
