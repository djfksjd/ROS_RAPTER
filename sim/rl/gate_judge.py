"""Pass / fail of a stage gate output (stage_gates.py JSON lines) against the rules of docs/STAGED_PLAN.ko.md.
stand (1a): hold 8 nominal: 0 falls, drift <= 0.5 m; hold_dr 24: falls <= 1; push 0.3 m/s each direction: falls <= 1
            (0.5 / 0.8 m/s recorded, not judged).
walk (2, fixed 2026-11-03 before any walking run): every nominal condition 0 falls; track: mean speed error <= max(0.1,
            0.15 cmd) per episode; turn: |mean yaw rate - cmd| <= 0.25 |cmd| and drift <= 0.5 m; stopstart: 0.8 m/s
            within 2 s of each start, |speed| < 0.15 m/s 2 s after each stop; track_dr: falls <= 1.
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
    return None, 'not judged'


def main():
    allok = True
    for p in sys.argv[1:]:
        for l in open(p):
            row = json.loads(l)
            ok, why = judge(row)
            tag = {True: 'PASS', False: 'FAIL', None: '----'}[ok]
            allok &= ok is not False
            print(f"{tag} {row['gate']:5} {row['cond']:9} {str(row.get('cmd', row.get('dir', ''))):>5} {row.get('dv', ''):>4}  {why}")
    print('ALL PASS' if allok else 'NOT PASSED')


if __name__ == '__main__':
    main()
