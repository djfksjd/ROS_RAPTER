"""Equal-condition structure comparison (evidence 86 §18; protocol reviewed by UNI_AI gpt-6-sol 2026-10-01).

For each run directory (env rebuilt from its args.json):
  A) heat_test at commands CMDS x distinct seeded starts (3 by default): 60 s, cold motors, last-40 s body-frame speed,
     steady heat per motor at 1 ms (upper bound) and after a 20 Hz low-pass (lower bound, reported only).
  B) sustained speed S = at the highest command where EVERY start passes (no fall and max 1 ms heat <= 1.0), the
     slowest of those starts' last-40 s speeds (conservative). S_20hz the same with the 20 Hz heat (sensitivity only,
     never a tie-breaker). Burst = best 3 s mean speed of any run. "40 km/h reached" only if S >= 11.111 m/s.
  C) robustness and turns via evaluate_run.py --from-args (distinct seeded starts): yaw impulses 0.5/1.0/1.5 N·m·s and
     pitch impulses 1/2 N·m·s at 4 m/s, turns 0.5/1.0/1.5 rad/s at 4 m/s, 5 episodes each (exploratory).
Decision (applied by the reader, printed as a hint): pass/fail at 1 ms heat first, then S; a gap <= 0.5 m/s between
the top two is UNDECIDED -> train more seeds for both. Results come from one training seed each: label them so.
T1 virtual-actuator simulation with assumed motor curves and a heat proxy; not hardware performance.
Usage: .venv-sim/bin/python sim/rl/compare_structures.py RUN_DIR [RUN_DIR ...] --out sim/rl/runs/screen_logs/compare
       [--starts 101 102 103] [--cmds ...] [--skip-robust]
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CMDS = [3., 4., 5., 6., 7., 8., 9., 9.5, 10., 10.5, 11.2]


def heat_grid(run_dir, cmds, starts, case_dir=None):
    from heat_test import run
    rows = []
    path = Path(case_dir)/f'{Path(run_dir).name}.jsonl' if case_dir else None
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
    for c in cmds:
        for s in starts:
            row = run(run_dir, c, 60., 7, 'model.zip', s)
            rows.append(row)
            if path:
                with path.open('a') as f:
                    f.write(json.dumps(row)+'\n')
            print(json.dumps({'run': Path(run_dir).name, **row}), flush=True)
    return rows


def summarize(rows):
    by = {}
    for r in rows:
        by.setdefault(r['cmd'], []).append(r)
    def sustained(key):
        ok = [c for c, rs in by.items() if all((not r['fell']) and r[key] for r in rs)]
        if not ok:
            return 0., None
        c = max(ok, key=lambda c: min(r['speed_last40_mps'] for r in by[c]))
        return min(r['speed_last40_mps'] for r in by[c]), c
    S, c1 = sustained('sustainable')
    S_lp, c2 = sustained('sustainable_20hz')
    per_cmd = {c: dict(falls=sum(r['fell'] for r in rs), n=len(rs), speed_min=min(r['speed_last40_mps'] for r in rs),
                       speed_max=max(r['speed_last40_mps'] for r in rs), heat_max=max(r['max_heat'] for r in rs),
                       heat20_max=max(r['max_heat_20hz'] for r in rs),
                       hottest=max(rs[0]['steady_heat'], key=rs[0]['steady_heat'].get)) for c, rs in sorted(by.items())}
    return dict(S=S, S_cmd=c1, S_20hz=S_lp, S_20hz_cmd=c2, burst=max(r['burst3s_mps'] for r in rows),
                reached_40kmh=S >= 100/9, mass_kg=rows[0]['mass_kg'], per_cmd=per_cmd)


def robust(run_dir, episodes=5):
    model = str(Path(run_dir)/'model.zip')
    py = sys.executable
    out = {}
    for name, extra in (('yaw', ['--impulses', '0.5', '1.0', '1.5', '--impulse-axis', 'z']),
                        ('pitch', ['--impulses', '1', '2', '--impulse-axis', 'y']),
                        ('turn', ['--turns', '0.5', '1.0', '1.5', '--turn-cmd', '4'])):
        cmd = [py, str(HERE/'evaluate_run.py'), model, '--from-args', '--commands', '4', '--episodes', str(episodes),
               '--seconds', '10', '--impulse-cmd', '4', *extra]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode:
            raise RuntimeError(f'{run_dir}: {name} evaluation failed ({res.returncode}): {res.stderr[-2000:]}')
        rows = [json.loads(l) for l in res.stdout.splitlines() if l.startswith('{')]
        key = 'turn_cmd_rad_s' if name == 'turn' else 'impulse_nms'
        expected = {'yaw': [0.5, 1.0, 1.5], 'pitch': [1., 2.], 'turn': [0.5, 1., 1.5]}[name]
        cases = [r for r in rows if key in r]
        if (len(rows) != 1+len(expected) or sorted(r[key] for r in cases) != expected
                or any(r.get('episodes') != episodes for r in rows)):
            raise RuntimeError(f'{run_dir}: {name} evaluation returned incomplete or duplicate cases')
        out[name] = rows
        print(json.dumps({'run': Path(run_dir).name, 'robust': name, 'rows': rows}), flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+')
    p.add_argument('--out', required=True)
    p.add_argument('--starts', type=int, nargs='+', default=[101, 102, 103])
    p.add_argument('--cmds', type=float, nargs='+', default=CMDS)
    p.add_argument('--skip-robust', action='store_true')
    a = p.parse_args()
    case_dir = Path(a.out + '.cases')
    if case_dir.exists():
        p.error(f'case directory already exists: {case_dir}; choose a new --out to preserve previous results')
    case_dir.mkdir(parents=True)
    with ProcessPoolExecutor(max_workers=len(a.runs)) as ex:
        grids = list(ex.map(heat_grid, a.runs, [a.cmds]*len(a.runs), [a.starts]*len(a.runs), [case_dir]*len(a.runs)))
    res = {}
    for r, rows in zip(a.runs, grids):
        res[Path(r).name] = dict(summary=summarize(rows), heat_rows=rows)
    if not a.skip_robust:
        with ProcessPoolExecutor(max_workers=len(a.runs)) as ex:
            for r, rb in zip(a.runs, ex.map(robust, a.runs)):
                res[Path(r).name]['robust'] = rb
    Path(a.out + '.json').write_text(json.dumps(res, indent=1))
    ranked = sorted(res.items(), key=lambda kv: -kv[1]['summary']['S'])
    lines = ['| run | mass kg | S (1 ms heat) m/s @cmd | S_20hz m/s | burst 3 s m/s | 40 km/h | yaw falls (0.5/1/1.5) | pitch falls (1/2) | turn angle ratio (0.5/1/1.5) |',
             '|---|---|---|---|---|---|---|---|---|']
    for name, v in ranked:
        s = v['summary']; rb = v.get('robust', {})
        yf = '/'.join(str(r['falls']) for r in rb.get('yaw', []) if 'impulse_nms' in r) or '-'
        pf = '/'.join(str(r['falls']) for r in rb.get('pitch', []) if 'impulse_nms' in r) or '-'
        tr = '/'.join((f"{r['turn_rad']/r['turn_rad_cmd']:.2f}" if r.get('turn_rad') is not None else f"fell {r['falls']}")
                      for r in rb.get('turn', []) if 'turn_rad_cmd' in r) or '-'
        lines.append(f"| {name} | {s['mass_kg']} | {s['S']:.2f} @{s['S_cmd']} | {s['S_20hz']:.2f} | {s['burst']:.2f} | {'yes' if s['reached_40kmh'] else 'no'} | {yf} | {pf} | {tr} |")
    if len(ranked) > 1:
        gap = ranked[0][1]['summary']['S'] - ranked[1][1]['summary']['S']
        lines.append('')
        lines.append(f"Top-two gap {gap:.2f} m/s -> {'UNDECIDED: train seeds 1 and 2 for both' if gap <= .5 else 'leader by > 0.5 m/s (single training seed)'}")
    Path(a.out + '.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
