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
Usage: .venv-sim/bin/python sim/rl/gate_judge.py GATE.jsonl [GATE2.jsonl ...]"""
import json
import sys


def judge(row):
    c, e = row['cond'], row.get('episodes', [])
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
    print('ALL PASS' if allok else 'NOT PASSED')


if __name__ == '__main__':
    main()
