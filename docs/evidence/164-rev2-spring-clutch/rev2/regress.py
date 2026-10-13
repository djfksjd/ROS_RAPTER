import json, sys
from pathlib import Path
G = Path('output/pdf/RAPTOR_최종문서_20261012/evidence/gates'); sys.path.insert(0, str(G))
import gates
old = {(r['cmd'], r['init_seed']): r for r in map(json.loads, (G/'gate_rollouts_selfcol.jsonl').read_text().splitlines()) if r['kind'] == 'base'}
keys = ('fell', 't', 'speed_tail', 'heat_max', 'energy_injected_J', 'motor_positive_work_J', 'knee_min', 'knee_max', 'engagements', 'vz_max')
ok = True; rows = []
for cmd in (4., 7.):
    for seed in (95001, 95007):
        r = gates.rollout(('base', cmd, seed, 60., 'policy')); o = old[(cmd, seed)]
        same = all(r[k] == o[k] for k in keys); ok &= same
        rows.append(dict(cmd=cmd, seed=seed, identical=same, **{k: r[k] for k in keys}))
        print(cmd, seed, 'identical' if same else {k: (r[k], o[k]) for k in keys if r[k] != o[k]})
print('REGRESSION', 'PASS' if ok else 'FAIL')
(Path(__file__).resolve().parent/'regression.json').write_text(json.dumps(dict(passed=bool(ok), compared_keys=keys, rows=rows), indent=1))
