"""Print a compact table from a rev2_accept.py output file. Usage: quick_look.py <file.jsonl>"""
import json, sys, collections, statistics as st
rows = [json.loads(l) for l in open(sys.argv[1])]; g = collections.defaultdict(list)
for r in rows: g[(r['cmd'], r['label'])].append(r)
for k, v in sorted(g.items()):
    ok = [x for x in v if not x['fell']] or v; m = lambda q, G=ok: round(st.mean(x[q] for x in G), 3)
    ha = {j: round(st.mean(x['heat_all'][j] for x in ok), 2) for j in ok[0]['heat_all']} if 'heat_all' in ok[0] else {}
    hot = max(ha, key=ha.get) if ha else ''
    print(k, 'n', len(v), 'falls', sum(x['fell'] for x in v), 'spd', m('speed_tail'), 'knee', m('heat_knee'), 'max', m('heat_max'), hot, 'tailP', ha.get('tail_pitch'), 'hipR', max(ha.get('left_hip_roll', 0), ha.get('right_hip_roll', 0)),
          'on', m('latched_stance_frac', v), 'active', m('spring_active_stance_frac', v) if 'spring_active_stance_frac' in v[0] else None, 'peakJ', m('spring_peak_J') if 'spring_peak_J' in v[0] else None,
          'Einj', round(sum(x['energy_injected_J'] for x in v), 3), 'Erel/Wmot', round(sum(x['energy_released_J'] for x in v)/max(1e-9, sum(x['motor_positive_work_J'] for x in v)), 4),
          'lim', m('near_limit_frac_max', v) if 'near_limit_frac_max' in v[0] else None, collections.Counter(x.get('near_limit_joint') for x in v).most_common(1)[0][0], 'self', m('self_contact_step_frac', v))
