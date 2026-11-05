"""Pass / fail of a stage gate output (stage_gates.py JSON lines) against the rules of docs/STAGED_PLAN.ko.md.
stand (1a): hold 8 nominal: 0 falls, drift <= 0.5 m; hold_dr 24: falls <= 1; push 0.3 m/s each direction: falls <= 1
            (0.5 / 0.8 m/s recorded, not judged).
walk (2, fixed 2026-11-03 before any walking run): every nominal condition 0 falls; track: mean speed error <= max(0.1,
            0.15 cmd) per episode; turn: |mean yaw rate - cmd| <= 0.25 |cmd| and drift <= 0.5 m; stopstart: 0.8 m/s
            within 2 s of each start, |speed| < 0.15 m/s 2 s after each stop; track_dr: falls <= 1.
run (3, adjusted by the user 2026-11-04, 11.1 m/s deferred): speed 2/4/6 m/s: 0 falls and mean within +-10 %; 8 m/s:
            0 falls and mean >= 6.5 m/s; 10 / 11.1 m/s recorded; heat: the sustained speed (highest heat command on the
            vendor heat-sink mount with no derating and 0 falls) must be >= 4 m/s; lateral: falls <= 1 per side.
switch (3, policy set walk + run, 2026-11-04): commands rate-limited to 2 m/s^2: 0 falls nominal, <= 1 of 8 randomized;
            step commands recorded.
turn (4, fixed 2026-11-04 before training): turn_rules.py; every episode must satisfy speed, yaw, radius, signed
            heading and 90-degree deadline. Full adoption also requires complete held-out turns and earlier regressions.
            User decision B, 2026-11-04 AFTER seeing the t3 full gate: judge 1/2/4 m/s turns only;
            6 m/s turns are recorded, not judged. Earlier regressions, including straight 6 m/s, are unchanged.
jump (5a, fixed by supervisor 2026-11-04 BEFORE jump training): 0.10 s both-feet flight, +0.05 m over
            standing height, landing <=0.30 m from takeoff, then continuous 1 s standing. Nominal 8/8;
            DR >=22/24 successful and <=2 falls. Impact/current recorded only. Full retained-policy regressions.
            User decision 2026-11-05 AFTER j3: 5a-1 .20 m, then 5a-2 .30 m. Tagged rows use that height;
            untagged historical rows retain .05 m. Mixed levels cannot form an adoption suite.
            This judges 5a, not all stage 5; step/drop 0.15 m (5b) and forward jump are separate.
Usage: .venv-sim/bin/python sim/rl/gate_judge.py GATE.jsonl [GATE2.jsonl ...]"""
import json
import math
import sys
from turn_rules import TURN_CRITERIA


def turn_episode_ok(row, episode):
    rule = TURN_CRITERIA.get(row['cmd'])
    if rule is None or abs(row['yaw_cmd']) != rule['yaw']:
        return False
    v, w, radius, t90, angle = (episode.get(k) for k in
                              ('v_mean', 'yaw_rate_mean', 'radius_m', 't_90_s', 'heading_change_rad'))
    values = (v, w, radius, t90, angle)
    if any(x is None or not math.isfinite(x) for x in values):
        return False
    return (episode['t_fall'] is None and abs(v-row['cmd']) <= rule['v_tol']*row['cmd']
            and abs(w-row['yaw_cmd']) <= rule['yaw_tol']*rule['yaw']
            and 0 <= radius <= rule['radius'] and 0 < t90 <= rule['t90']
            and math.copysign(1., row['yaw_cmd'])*angle >= math.pi/2)


def jump_episode_ok(e, height_target_m=.05):
    fields = ('request_time_s', 'standing_height_m', 'flight_s', 'apex_rise_m', 'landing_error_m',
              'touchdown_s', 'standing_hold_s', 'success_time_s', 'elapsed_s', 'landing_impact_peak_n',
              'current_peak_a', 'current_over_limit_peak', 'torque_peak_nm', 'return_observed_s')
    if any(not isinstance(e.get(k), (int, float)) or not math.isfinite(e[k]) for k in fields):
        return False
    return (height_target_m in (.05, .20, .30) and e.get('height_target_m', .05) == height_target_m
            and e.get('t_fall') is None and e['request_time_s'] >= 2. and e['standing_height_m'] > 0
            and e['flight_s'] >= .10 and e['apex_rise_m'] >= height_target_m and 0 <= e['landing_error_m'] <= .30
            and e['standing_hold_s'] >= 1. and e['request_time_s'] < e['touchdown_s'] < e['success_time_s'] <= e['elapsed_s']
            and e['success_time_s'] <= 12.+1e-9 and e['elapsed_s'] <= 13.+1e-9
            and e['return_observed_s'] >= 1.-1e-9 and e.get('return_to') == e.get('handoff_from')
            and bool(e.get('return_to')) and e.get('sample_dt_s') == .001
            and e.get('load_samples') == round(e['elapsed_s']/.001) and bool(e.get('raw_trace'))
            and all(e[k] >= 0 for k in ('landing_impact_peak_n', 'current_peak_a', 'current_over_limit_peak', 'torque_peak_nm')))


def judge(row):
    c, e = row['cond'], row.get('episodes', [])
    if row['gate'] == 'jump':
        if c in ('in_place', 'in_place_dr'):
            height = row.get('height_target_m', .05)
            passed = sum(jump_episode_ok(x, height) for x in e)
            allowed = 0 if c == 'in_place' else 2
            valid = (height in (.05, .20, .30) and bool(e) and len(e) == row['n']
                     and all(x.get('height_target_m', .05) == height for x in e)
                     and row['falls'] == sum(x.get('t_fall') is not None for x in e))
            return valid and row['falls'] <= allowed and passed >= max(1, row['n']-allowed), f"jump height {height:g} m success {passed}/{row['n']}, falls {row['falls']}; impact/current recorded only"
        if c == 'run_regression':
            ok = bool(e) and len(e) == row['n'] and row['falls'] == 0 and all(
                x['t_fall'] is None and x['v_mean'] is not None and math.isfinite(x['v_mean'])
                and abs(x['v_mean']-row['cmd']) <= .1*row['cmd'] for x in e)
            return ok, f"falls {row['falls']}/{row['n']}, v {[x['v_mean'] for x in e]}"
    if row['gate'] == 'turn':
        if c == 'moving_turn':
            if row['cmd'] == 6.:
                return None, f"falls {row['falls']}/{row['n']} (6 m/s turn deferred by user 2026-11-04; recorded)"
            ok = bool(e) and len(e) == row['n'] and row['falls'] == 0 and all(turn_episode_ok(row, x) for x in e)
            return ok, f"falls {row['falls']}/{row['n']}, passing episodes {sum(turn_episode_ok(row, x) for x in e)}/{len(e)}"
        if c == 'run_regression':
            ok = bool(e) and len(e) == row['n'] and row['falls'] == 0 and all(
                x['t_fall'] is None and x['v_mean'] is not None and math.isfinite(x['v_mean'])
                and abs(x['v_mean']-row['cmd']) <= .1*row['cmd'] for x in e)
            return ok, f"falls {row['falls']}/{row['n']}, v {[x['v_mean'] for x in e]}"
    if row['gate'] == 'stand':
        if c == 'hold':
            return row['falls'] == 0 and row['drift_max_m'] <= .5, f"falls {row['falls']}, drift {row['drift_max_m']}"
        if c == 'hold_dr':
            return row['falls'] <= 1, f"falls {row['falls']}/{row['n']}"
        if c == 'push':
            if row['dv'] != .3:
                return None, f"falls {row['falls']} (recorded)"
            return row['falls'] <= 1, f"falls {row['falls']}"
    if row['gate'] == 'walk':
        if c == 'track':
            tol = max(.1, .15*row['cmd'])
            v = [x['v_mean'] for x in e]
            ok = row['falls'] == 0 and all(x is not None and abs(x-row['cmd']) <= tol for x in v)
            return ok, f"v {[None if x is None else round(x, 2) for x in v]}, tol {tol:.2f}"
        if c == 'turn':
            w = [x['yaw_rate_mean'] for x in e]
            ok = row['falls'] == 0 and all(x is not None and abs(x-row['cmd']) <= .25*abs(row['cmd']) for x in w) \
                and all(x['drift_m'] <= .5 for x in e)
            return ok, f"yaw {[None if x is None else round(x, 2) for x in w]}, drift max {max(x['drift_m'] for x in e)}"
        if c == 'stopstart':
            reach = [t for x in e for t in x['t_reach_0p8']]
            stop = [v for x in e for v in x['speed_2s_after_stop']]
            ok = row['falls'] == 0 and all(t is not None and t <= 2. for t in reach) and all(v is not None and abs(v) < .15 for v in stop)
            return ok, f"reach max {max((t for t in reach if t is not None), default=None)}, stop speeds {[None if v is None else round(v, 2) for v in stop]}"
        if c == 'track_dr':
            return row['falls'] <= 1, f"falls {row['falls']}/{row['n']}"
    if row['gate'] == 'run':  # stage 3 as adjusted by the user on 2026-11-04 (11.1 m/s deferred, evidence 197)
        if c == 'speed':
            v = [x['v_mean'] for x in e]
            vs = f"v {[None if x is None else round(x, 2) for x in v]}, derate min {min(x['min_derate'] for x in e)}"
            if row['cmd'] <= 6.:
                return row['falls'] == 0 and all(x is not None and abs(x-row['cmd']) <= .1*row['cmd'] for x in v), vs
            if row['cmd'] == 8.:
                return row['falls'] == 0 and all(x is not None and x >= 6.5 for x in v), vs+' (top speed: >= 6.5)'
            return None, vs+' (recorded)'
        if c == 'heat':
            why = f"{row['mount']}: falls {row['falls']}, derate min {row['min_derate']}, winding max {row['t_winding_max_c']} C, v {row['v_mean']}"
            return None, why+' (sustained speed below)'
        if c == 'lateral':
            return row['falls'] <= 1, f"falls {row['falls']}/{row['n']}"
    if row['gate'] == 'switch':  # stage 3 policy set (walk + run policy), 2026-11-04
        why = f"falls {row['falls']}/{row['n']}, switches {[x['switches'] for x in e]}"
        if row['cond'] == 'switch_ramp':
            return row['falls'] == 0, why
        if row['cond'] == 'switch_ramp_dr':
            return row['falls'] <= 1, why
        return None, why+' (step commands, recorded)'
    return None, 'not judged'


def turn_suite_status(rows):
    """Adoption requires a complete full suite, never an empty/partial/quick file."""
    required = []
    for speed, rule in TURN_CRITERIA.items():
        if speed == 6.:
            continue  # User decision B after t3 data, 2026-11-04; still measured by stage_gates.
        for sign in (1., -1.):
            required.append((dict(gate='turn', cond='moving_turn', cmd=speed, yaw_cmd=sign*rule['yaw']), 4))
    required += [(dict(gate='turn', cond='run_regression', cmd=v), 4) for v in (2., 4., 6.)]
    required += [(dict(gate='stand', cond='hold'), 8), (dict(gate='stand', cond='hold_dr'), 24)]
    required += [(dict(gate='stand', cond='push', dv=.3, dir=d), 8) for d in ('front', 'back', 'left', 'right')]
    required += [(dict(gate='walk', cond='track', cmd=v), 4) for v in (.3, .6, 1., 1.5)]
    required += [(dict(gate='walk', cond='turn', cmd=w), 4) for w in (.5, -.5)]
    required += [(dict(gate='walk', cond='stopstart'), 4), (dict(gate='walk', cond='track_dr'), 24),
                 (dict(gate='switch', cond='switch_ramp'), 4), (dict(gate='switch', cond='switch_ramp_dr'), 8)]
    for selector, n in required:
        matches = [r for r in rows if all(r.get(k) == v for k, v in selector.items())]
        if len(matches) != 1:
            return False, f'missing or duplicate condition: {selector}'
        r = matches[0]
        if r.get('quick') or r.get('n') != n or len(r.get('episodes', [])) != n:
            return False, f'incomplete or quick condition: {selector}'
        if r['falls'] != sum(e['t_fall'] is not None for e in r['episodes']):
            return False, f'inconsistent fall count: {selector}'
        if selector['gate'] == 'turn' and selector['cond'] == 'moving_turn':
            if {e['seed'] for e in r['episodes']} != set(range(7300, 7304)):
                return False, f'held-out seeds required: {selector}'
            for e in r['episodes']:
                joints = e.get('joints', {})
                if e.get('load_samples') != 10000 or set(joints) != {'left_hip_yaw_joint', 'right_hip_yaw_joint'}:
                    return False, f'incomplete hip yaw load measurement: {selector}'
                for j in joints.values():
                    if any(not isinstance(j.get(k), (int, float)) or not math.isfinite(j[k]) or j[k] < 0
                           for k in ('rms_nm', 'peak_nm', 'peak_rad_s', 'at_limit')) or j['at_limit'] > 1:
                        return False, f'invalid hip yaw load measurement: {selector}'
        if judge(r)[0] is not True:
            return False, f'failed condition: {selector}'
    return True, 'full turn + stand/walk/run 2/4/6/switch regression passed'


def jump_suite_status(rows, height_target_m=None):
    """Complete 5a + retained policy regressions only. No 5b/stage-5 adoption claim."""
    if any(r.get('quick') for r in rows):
        return False, 'quick rows are development only'
    heights = {r.get('height_target_m', .05) for r in rows if r.get('gate') == 'jump' and r.get('cond') in ('in_place', 'in_place_dr')}
    if len(heights) != 1 or not heights <= {.05, .20, .30} or (height_target_m is not None and heights != {height_target_m}):
        return False, 'missing/mixed/wrong jump height contract'
    required = [(dict(gate='jump', cond='in_place'), 8), (dict(gate='jump', cond='in_place_dr'), 24)]
    required += [(dict(gate='jump', cond='run_regression', cmd=v), 4) for v in (2., 4., 6.)]
    required += [(dict(gate='stand', cond='hold'), 8), (dict(gate='stand', cond='hold_dr'), 24)]
    required += [(dict(gate='stand', cond='push', dv=.3, dir=d), 8) for d in ('front', 'back', 'left', 'right')]
    required += [(dict(gate='walk', cond='track', cmd=v), 4) for v in (.3, .6, 1., 1.5)]
    required += [(dict(gate='walk', cond='turn', cmd=v), 4) for v in (.5, -.5)]
    required += [(dict(gate='walk', cond='stopstart'), 4), (dict(gate='walk', cond='track_dr'), 24),
                 (dict(gate='switch', cond='switch_ramp'), 4), (dict(gate='switch', cond='switch_ramp_dr'), 8)]
    for selector, n in required:
        matches = [r for r in rows if all(r.get(k) == v for k, v in selector.items())]
        if len(matches) != 1:
            return False, f'missing or duplicate condition: {selector}'
        r = matches[0]; episodes = r.get('episodes', [])
        if r.get('quick') or r.get('n') != n or len(episodes) != n:
            return False, f'incomplete or quick condition: {selector}'
        if r.get('falls') != sum(e.get('t_fall') is not None for e in episodes):
            return False, f'inconsistent falls: {selector}'
        if selector['gate'] == 'jump' and selector['cond'] in ('in_place', 'in_place_dr'):
            first = 9400 if selector['cond'] == 'in_place' else 9500
            if {e.get('seed') for e in episodes} != set(range(first, first+n)):
                return False, f'held-out seeds required: {selector}'
            if not r.get('walk') or any(e.get('handoff_from') != r['walk'] for e in episodes):
                return False, 'missing retained-walk handoff'
            # Failed DR episodes still need complete finite load/contact evidence.
            for e in episodes:
                if any(not isinstance(e.get(k), (int, float)) or not math.isfinite(e[k]) or e[k] < 0
                       for k in ('landing_impact_peak_n', 'current_peak_a', 'current_over_limit_peak', 'torque_peak_nm')):
                    return False, 'invalid load summary'
                if (not isinstance(e.get('elapsed_s'), (int, float)) or not math.isfinite(e['elapsed_s'])
                        or e.get('sample_dt_s') != .001 or e.get('load_samples') != round(e['elapsed_s']/.001)
                        or e.get('load_samples', 0) <= 0 or not e.get('raw_trace')):
                    return False, 'missing 1 ms measurement'
        if judge(r)[0] is not True:
            return False, f'failed condition: {selector}'
    return True, '5a + stand/walk/run 2/4/6/switch passed; 5b not evaluated, stage 5 not passed'


def sustained_speed(rows):
    ok = [r['cmd'] for r in rows if r['gate'] == 'run' and r['cond'] == 'heat' and r['mount'] == 'plate'
          and r['falls'] == 0 and r['min_derate'] >= 1.]
    return max(ok, default=0.)


def held_speed(rows):
    vs = 0.
    for r in sorted((r for r in rows if r['gate'] == 'run' and r['cond'] == 'speed'), key=lambda r: r['cmd']):
        if not judge(r)[0]:
            break
        vs = r['cmd']
    return vs


def main():
    allok, rows = True, []
    for p in sys.argv[1:]:
        for l in open(p):
            row = json.loads(l)
            rows.append(row)
            ok, why = judge(row)
            tag = {True: 'PASS', False: 'FAIL', None: '----'}[ok]
            allok &= ok is not False
            print(f"{tag} {row['gate']:5} {row['cond']:9} {str(row.get('cmd', row.get('dir', ''))):>5} {row.get('dv', ''):>4}  {why}")
    if any(r['gate'] == 'run' for r in rows):
        vh = sustained_speed(rows)
        allok &= vh >= 4.
        print(f'{"PASS" if vh >= 4. else "FAIL"} run   heat      sustained 10 min speed without derating {vh} m/s (needs >= 4)')
    if any(r['gate'] == 'jump' for r in rows):
        if any(r.get('quick') for r in rows):
            print('QUICK PASS (development only; not adoption)' if allok else 'QUICK NOT PASSED')
        else:
            ok, why = jump_suite_status(rows); allok &= ok
            print(f'{"PASS" if ok else "FAIL"} jump 5a suite: {why}')
        return 0 if allok else 1
    if any(r['gate'] == 'turn' for r in rows):
        if any(r.get('quick') for r in rows):
            print('QUICK PASS (development only; not adoption)' if allok else 'QUICK NOT PASSED')
        else:
            ok, why = turn_suite_status(rows)
            allok &= ok
            print(f'{"PASS" if ok else "FAIL"} turn suite: {why}')
            print('ALL PASS' if allok else 'NOT PASSED')
        return 0 if allok else 1
    print('ALL PASS' if allok else 'NOT PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
