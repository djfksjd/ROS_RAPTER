#!/usr/bin/env python3
"""Roll-fall diagnosis: the same time series from a Gazebo probe log and a MuJoCo run (evidence 72).

Gazebo inputs: rock_gz_probe.py --diag JSON and a `gz topic -e -t /world/raptor_world/dynamic_pose/info
--json-output` stream (link poses are model-relative; world = model pose * link pose). MuJoCo: rock_probe.run
with a read-only trace hook. Series (100 Hz, probe time t = sim time since the probe started):
roll, roll rate, pad-centre y per foot, whole-body COM y and extrapolated COM (y + vy/omega0), normal and
lateral contact force per foot, pad horizontal speed while loaded. Diagnosis only; no control change.

Usage: python roll_diagnosis.py OUT.json --gazebo DIAG.json POSES.jsonl [...] --mujoco-kv 20 30 100
"""
import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'src/raptor_control/scripts'))
from lateral_feasibility import origin  # noqa: E402

URDF = HERE/'raptor_digitigrade.urdf'
G = 9.81


def quat_matrix(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def urdf_bodies():
    root = ET.parse(URDF).getroot()
    masses = {l.get('name'): (float(l.find('inertial/mass').get('value')), origin(l.find('inertial/origin'))[:3, 3])
              for l in root.findall('link') if l.find('inertial') is not None}
    foot = next(l for l in root.findall('link') if l.get('name') == 'left_foot_link')
    pad = origin(foot.findall('collision')[0].find('origin'))[:3, 3]
    base_offset = origin(next(j for j in root.findall('joint') if j.get('name') == 'base_root_joint').find('origin'))
    return masses, pad, base_offset


def summarize(series, weight, window, fall_side_t=None):
    """Window statistics and verdict inputs from equal-length arrays in `series`."""
    t = np.asarray(series['t'])
    m = (t >= window[0]) & (t <= window[1])
    out = {'window_s': window, 'samples': int(m.sum())}
    if not m.any():
        return out
    roll = np.asarray(series['roll'])[m]
    out['roll_final'] = float(roll[-1])
    out['roll_min_max'] = [float(roll.min()), float(roll.max())]
    out['roll_rate_abs_max'] = float(np.abs(np.asarray(series['roll_rate'])[m]).max())
    for side in ('left', 'right'):
        fz = np.asarray(series[f'fz_{side}'])[m]
        fy = np.asarray(series[f'fy_{side}'])[m]
        slip = np.asarray(series[f'slip_{side}'])[m]
        loaded = fz > .2*weight
        out[side] = {'pad_y_min_max': [float(np.min(np.asarray(series[f'pad_y_{side}'])[m])),
                                       float(np.max(np.asarray(series[f'pad_y_{side}'])[m]))],
                     'loaded_fraction': float(loaded.mean()),
                     'lateral_to_normal_p90': float(np.percentile(np.abs(fy[loaded])/fz[loaded], 90)) if loaded.any() else None,
                     'slip_speed_loaded_p90': float(np.nanpercentile(slip[loaded], 90)) if loaded.any() else None,
                     'slip_speed_loaded_max': float(np.nanmax(slip[loaded])) if loaded.any() else None}
    com = np.asarray(series['com_y'])[m]
    xcom = np.asarray(series['xcom_y'])[m]
    out['com_y_min_max'] = [float(com.min()), float(com.max())]
    out['xcom_y_min_max'] = [float(xcom.min()), float(xcom.max())]
    return out


def per_cycle_peaks(series, frequency, start):
    t, roll = np.asarray(series['t']), np.asarray(series['roll'])
    peaks = []
    for k in range(int((t[-1]-start)*frequency)+1):
        m = (t >= start+k/frequency) & (t < start+(k+1)/frequency)
        if m.any():
            peaks.append([round(float(roll[m].min()), 4), round(float(roll[m].max()), 4)])
    return peaks


def gazebo_series(diag_path, poses_path):
    masses, pad, base_offset = urdf_bodies()
    log = json.loads(Path(diag_path).read_text())
    start = log['start_sim_time']
    text, decoder, i, poses = Path(poses_path).read_text(), json.JSONDecoder(), 0, []
    while i < len(text):
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            break
        msg, i = decoder.raw_decode(text, i)
        stamp = msg['header']['stamp']
        ts = float(stamp.get('sec', 0))+float(stamp.get('nsec', 0))*1e-9-start
        entries = {p['name']: p for p in msg['pose']}
        if 'raptor' not in entries:
            continue

        def transform(p):
            pos = p.get('position', {})
            ori = p.get('orientation', {})
            T = np.eye(4)
            T[:3, :3] = quat_matrix([ori.get('w', 1.), ori.get('x', 0.), ori.get('y', 0.), ori.get('z', 0.)])
            T[:3, 3] = [pos.get('x', 0.), pos.get('y', 0.), pos.get('z', 0.)]
            return T
        model = transform(entries['raptor'])
        world = {n: model@transform(p) for n, p in entries.items() if n != 'raptor'}
        # base_link is merged into base_root by the URDF->SDF conversion (fixed joint base_root_joint).
        world.setdefault('base_link', world['base_root']@base_offset)
        total = sum(m for n, (m, _) in masses.items() if n in world)
        com = sum(m*(world[n][:3, :3]@c+world[n][:3, 3]) for n, (m, c) in masses.items() if n in world)/total
        pads = {s: world[f'{s}_foot_link'][:3, :3]@pad+world[f'{s}_foot_link'][:3, 3] for s in ('left', 'right')}
        soles = {s: math.atan2(world[f'{s}_foot_link'][2, 1], world[f'{s}_foot_link'][2, 2]) for s in ('left', 'right')}
        poses.append((ts, com, pads, world['base_link'][:3, 3].copy(), soles))
    poses.sort(key=lambda r: r[0])
    pt = np.array([p[0] for p in poses])
    diag = log['diag']
    grid = np.arange(max(pt[0], diag[0]['t']), min(pt[-1], diag[-1]['t']), .01)

    def interp(values, times):
        return np.interp(grid, times, values)
    dt_ = np.array([r['t'] for r in diag])
    series = {'t': grid, 'roll': interp([r['roll'] for r in diag], dt_),
              'roll_rate': interp([r['roll_rate'] for r in diag], dt_)}
    for s in ('left', 'right'):
        series[f'fz_{s}'] = interp([r['contact'][s]['f1'][2] for r in diag], dt_)
        series[f'fy_{s}'] = interp([r['contact'][s]['f1'][1] for r in diag], dt_)
        y = interp([p[2][s][1] for p in poses], pt)
        x = interp([p[2][s][0] for p in poses], pt)
        series[f'pad_y_{s}'] = y
        series[f'slip_{s}'] = np.r_[np.nan, np.hypot(np.diff(x), np.diff(y))/.01]
        series[f'sole_roll_{s}'] = interp([p[4][s] for p in poses], pt)  # foot link roll in the world frame
    series['body_y'] = interp([p[3][1] for p in poses], pt)  # base_link origin (body centre), not the COM
    com_y = interp([p[1][1] for p in poses], pt)
    com_z = interp([p[1][2] for p in poses], pt)
    vy = np.r_[0., np.diff(com_y)/.01]
    series['com_y'], series['xcom_y'] = com_y, com_y+vy/np.sqrt(G/np.maximum(com_z, .1))
    return series, log


def mujoco_series(kv, cycles, frequency=2.5, stride=.08, amplitude=.08, window_width=.35, mirror=False,
                  model='raptor_digitigrade.xml', delay=0., abduction=0., pll=None, damping=None, **gait):
    sys.path.insert(0, str(HERE))
    import mujoco
    from rock_probe import foot_forces, run
    _, pad, _ = urdf_bodies()
    rows, force = [], np.zeros(6)

    def trace(t, model, data):
        if round(t*1000) % 10:
            return
        w, x, y, z = data.qpos[3:7]
        roll = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
        root = model.body('base_root').id
        row = {'t': t, 'roll': roll, 'roll_rate': float(data.qvel[3]), 'com': data.subtree_com[root].copy(),
               'body_y': float(data.xpos[model.body('base_link').id][1])}
        for s in ('left', 'right'):
            b = model.body(f'{s}_foot_link').id
            row[f'pad_{s}'] = data.xmat[b].reshape(3, 3)@pad+data.xpos[b]
            m3 = data.xmat[b].reshape(3, 3)
            row[f'sole_{s}'] = math.atan2(m3[2, 1], m3[2, 2])
            fz = fy = 0.
            for i in range(data.ncon):
                c = data.contact[i]
                geom = c.geom1 if model.geom_bodyid[c.geom1] else c.geom2
                if not model.body(model.geom_bodyid[geom]).name.startswith(s+'_'):
                    continue
                mujoco.mj_contactForce(model, data, i, force)
                f = c.frame.reshape(3, 3).T@force[:3]  # contact frame -> world (force on geom2)
                sign = 1. if c.geom2 == geom else -1.
                fz += sign*f[2]
                fy += sign*f[1]
            row[f'fz_{s}'], row[f'fy_{s}'] = fz, fy
        rows.append(row)
    centers = (('left', .78), ('right', .28)) if mirror else (('left', .28), ('right', .78))
    result = run(-amplitude if mirror else amplitude, frequency, cycles=cycles, stride=stride,
                 pitch_feedback=(.5, .05, .15), servo_kv=kv, jtc=(.02, .02, delay), model_path=str(HERE/model),
                 crouch_hip=-.10, crouch_knee=.5, step_hook=trace, lift_width=window_width, lift_centers=centers,
                 abduction=abduction, pll=pll, damping=damping, **gait)
    t = np.array([r['t'] for r in rows])
    series = {'t': t, 'roll': np.array([r['roll'] for r in rows]), 'roll_rate': np.array([r['roll_rate'] for r in rows]),
              'body_y': np.array([r['body_y'] for r in rows])}
    for s in ('left', 'right'):
        series[f'fz_{s}'] = np.array([r[f'fz_{s}'] for r in rows])
        series[f'fy_{s}'] = np.array([r[f'fy_{s}'] for r in rows])
        p = np.array([r[f'pad_{s}'] for r in rows])
        series[f'pad_y_{s}'] = p[:, 1]
        series[f'slip_{s}'] = np.r_[np.nan, np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))/.01]
        series[f'sole_roll_{s}'] = np.array([r[f'sole_{s}'] for r in rows])
    com = np.array([r['com'] for r in rows])
    vy = np.r_[0., np.diff(com[:, 1])/.01]
    series['com_y'], series['xcom_y'] = com[:, 1], com[:, 1]+vy/np.sqrt(G/np.maximum(com[:, 2], .1))
    return series, result


def band_amplitude(t, y, lo, hi, f0, f1):
    """Hann-windowed FFT amplitude (sinusoid peak units) over [f0, f1) Hz within t in [lo, hi].

    Band RSS divided by sqrt(1.5) (Hann equivalent noise bandwidth, 1.5 bins) so a pure sine returns its amplitude.
    Before 2026-09-29 (evidence 72-74 tables) the division was missing: those band amplitudes are 1.225x too high.
    """
    t, y = np.asarray(t), np.asarray(y)
    m = (t >= lo) & (t <= hi)
    y = y[m]-y[m].mean()
    n = len(y)
    if n < 20:
        return None, None
    f = np.fft.rfftfreq(n, .01)
    a = np.abs(np.fft.rfft(y*np.hanning(n)))*4/n  # Hann coherent gain 0.5
    band = (f >= f0) & (f < f1)
    if not band.any():  # window too short to resolve this band
        return None, None
    return float(np.sqrt(np.sum(a[band]**2)/1.5)), float(f[band][np.argmax(a[band])])


def step_growth(t, y, lo, hi, frequency):
    """Per-step growth of the lateral oscillation: peak |y - moving mean| in each step (half cycle), then
    (A[k+3]/A[k])**(1/3) over consecutive 3-step spans. Returns (median, max, per-step amplitudes)."""
    t, y = np.asarray(t), np.asarray(y)
    m = (t >= lo) & (t <= hi)
    t, y = t[m], y[m]
    if len(t) < 50:
        return None, None, []
    k = max(1, int(round(1/frequency/.01)))  # one gait cycle of samples
    if len(y) < 2*k:  # shorter than two cycles: no growth estimate
        return None, None, []
    mean = np.convolve(y, np.ones(k)/k, mode='same')
    dev = np.abs(y-mean)
    step = .5/frequency
    amps = [float(dev[(t >= a) & (t < a+step)].max()) for a in np.arange(t[0], t[-1]-step, step)]
    ratios = [(amps[i+3]/amps[i])**(1/3) for i in range(len(amps)-3) if amps[i] > 1e-4]
    return (float(np.median(ratios)) if ratios else None, float(np.max(ratios)) if ratios else None, amps)


def table_metrics(series, lo, hi, frequency, weight=17.3*G):
    """Metrics requested for evidence 73 tables over [lo, hi] s."""
    t = np.asarray(series['t'])
    m = (t >= lo) & (t <= hi)
    low, low_f = band_amplitude(t, series['com_y'], lo, hi, .3, 1.6)
    gait, _ = band_amplitude(t, series['com_y'], lo, hi, 2., 3.)
    body_gait, _ = band_amplitude(t, series['body_y'], lo, hi, 2., 3.)
    roll_gait, _ = band_amplitude(t, series['roll'], lo, hi, 2., 3.)
    ratios = []
    for s in ('left', 'right'):
        fz, fy = np.asarray(series[f'fz_{s}'])[m], np.asarray(series[f'fy_{s}'])[m]
        loaded = fz > .2*weight
        ratios.extend(list(np.abs(fy[loaded])/fz[loaded]))
    fl, fr = np.asarray(series['fz_left'])[m], np.asarray(series['fz_right'])[m]
    growth_med, growth_max, _ = step_growth(t, series['com_y'], lo, hi, frequency)
    roll_low, roll_low_f = band_amplitude(t, series['roll'], lo, hi, .3, 1.6)
    # period-3 subharmonic and fundamental separately (the 0.3-1.6 Hz band contains the fundamental when f <= 1.6)
    sub3, _ = band_amplitude(t, series['com_y'], lo, hi, frequency/3-.05, frequency/3+.05)
    fund, _ = band_amplitude(t, series['com_y'], lo, hi, frequency-.05, frequency+.05)
    single = single_support_median(t[m], np.asarray(series['fz_left'])[m], np.asarray(series['fz_right'])[m], weight)
    return {'window_s': [round(lo, 2), round(hi, 2)], 'com_y_0.3_1.6Hz_m': low, 'com_y_low_peak_hz': low_f,
            'com_y_2_3Hz_m': gait, 'body_y_2_3Hz_m': body_gait, 'roll_2_3Hz_rad': roll_gait,
            'loaded_fy_fz_p90': float(np.percentile(ratios, 90)) if ratios else None,
            'load_share_left': float(fl.sum()/max(fl.sum()+fr.sum(), 1e-9)),
            'growth_per_step_median': growth_med, 'growth_per_step_max': growth_max,
            'roll_0.3_1.6Hz_rad': roll_low, 'roll_low_peak_hz': roll_low_f, 'single_support_median_s': single,
            'com_y_f_over_3_m': sub3, 'com_y_f_m': fund, 'landing_sole_roll_p90_rad': landing_sole(series, m, weight)}


def landing_sole(series, m, weight):
    """p90 of |foot roll| at touchdowns (normal force crossing 20 % of the weight upward) inside mask m."""
    values = []
    for s in ('left', 'right'):
        if f'sole_roll_{s}' not in series:
            return None
        fz, roll = np.asarray(series[f'fz_{s}'])[m], np.asarray(series[f'sole_roll_{s}'])[m]
        up = np.nonzero((fz[1:] >= .2*weight) & (fz[:-1] < .2*weight))[0]+1
        values += list(np.abs(roll[up]))
    return float(np.percentile(values, 90)) if values else None


def single_support_median(t, fl, fr, weight):
    """Median duration of runs where only the left or only the right foot carries >= 20 % of the weight
    (a run ends when the support state changes; runs shorter than 30 ms are ignored)."""
    if len(t) < 2:
        return None
    left, right = fl >= .2*weight, fr >= .2*weight
    state = np.where(left & ~right, 1, np.where(right & ~left, 2, 0))
    runs, start = [], 0
    for i in range(1, len(state)+1):
        if i == len(state) or state[i] != state[start]:
            if state[start]:
                runs.append(t[i-1]-t[start]+(t[1]-t[0]))
            start = i
    runs = [r for r in runs if r >= .03]
    return float(np.median(runs)) if runs else None


KEYS = ('roll', 'roll_rate', 'com_y', 'xcom_y', 'pad_y_left', 'pad_y_right', 'fz_left', 'fz_right',
        'fy_left', 'fy_right', 'slip_left', 'slip_right')


def window(series, w):
    t = np.asarray(series['t'])
    m = (t >= w[0]) & (t <= w[1])
    return {'t': [round(float(v), 3) for v in t[m]]} | {
        k: [None if not np.isfinite(v) else round(float(v), 4) for v in np.asarray(series[k])[m]] for k in KEYS}


def plot(path, panels, title):
    """Stacked line plots with Pillow (no matplotlib in .venv-sim). panels: [(label, [(name, t, y, rgb)], (lo, hi))]."""
    from PIL import Image, ImageDraw
    width, height, left = 1400, 230, 90
    image = Image.new('RGB', (width, 40+height*len(panels)), 'white')
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), title, fill='black')
    for k, (label, lines, (lo, hi)) in enumerate(panels):
        top = 40+k*height
        draw.rectangle([left, top+10, width-20, top+height-25], outline='gray')
        draw.text((10, top+12), label, fill='black')
        draw.text((10, top+height-40), f'{lo:g}..{hi:g}', fill='gray')
        t0 = min(min(t) for _, t, _, _ in lines if len(t))
        t1 = max(max(t) for _, t, _, _ in lines if len(t))
        x = lambda v: left+(v-t0)/max(t1-t0, 1e-9)*(width-20-left)  # noqa: E731
        yv = lambda v: top+height-25-(min(max(v, lo), hi)-lo)/(hi-lo)*(height-35)  # noqa: E731
        if lo < 0 < hi:
            draw.line([(left, yv(0)), (width-20, yv(0))], fill=(220, 220, 220))
        for n, (name, t, y, rgb) in enumerate(lines):
            pts = [(x(a), yv(b)) for a, b in zip(t, y) if b is not None and np.isfinite(b)]
            if len(pts) > 1:
                draw.line(pts, fill=rgb, width=2)
            draw.text((left+10+n*260, top+height-22), name, fill=rgb)
        draw.text((width-150, top+height-22), f't {t0:.2f}..{t1:.2f} s', fill='gray')
    image.save(path)


def compare_plot(path, a, b, w, title):
    A, B = window(a[1], w), window(b[1], w)
    red, blue, green, orange = (200, 40, 40), (40, 70, 200), (30, 140, 60), (230, 140, 20)

    def pair(key, lo, hi, label):
        return (label, [(f'{a[0]} {key}', A['t'], A[key], red), (f'{b[0]} {key}', B['t'], B[key], blue)], (lo, hi))
    plot(path, [pair('roll', -.3, .3, 'body roll [rad] (+ = tips right)'),
                pair('roll_rate', -2.5, 2.5, 'roll rate [rad/s]'),
                (f'{a[0]}: COM y (red), XCOM y (orange), pads (green)',
                 [('COM', A['t'], A['com_y'], red), ('XCOM', A['t'], A['xcom_y'], orange),
                  ('pad L', A['t'], A['pad_y_left'], green), ('pad R', A['t'], A['pad_y_right'], green)], (-.3, .3)),
                (f'{b[0]}: COM y (blue), XCOM y (orange), pads (green)',
                 [('COM', B['t'], B['com_y'], blue), ('XCOM', B['t'], B['xcom_y'], orange),
                  ('pad L', B['t'], B['pad_y_left'], green), ('pad R', B['t'], B['pad_y_right'], green)], (-.3, .3)),
                pair('fz_left', 0, 400, 'left foot normal force [N]'),
                pair('fz_right', 0, 400, 'right foot normal force [N]')], title)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out')
    parser.add_argument('--gazebo', nargs=2, action='append', metavar=('DIAG', 'POSES'), default=[])
    parser.add_argument('--mujoco-kv', type=float, nargs='*', default=[20., 30., 100.])
    parser.add_argument('--cycles', type=int, default=40)
    parser.add_argument('--series-out', help='JSON with the 5.0-6.5 s and pre-fall windows of every run')
    parser.add_argument('--plot-dir', help='write side-by-side PNGs (first Gazebo run vs each MuJoCo kv)')
    args = parser.parse_args()
    all_series = []
    weight = 17.3*G
    report = {'gazebo': [], 'mujoco': []}
    for diag, poses in args.gazebo:
        series, log = gazebo_series(diag, poses)
        fall = float(series['t'][-1])
        all_series.append((f'gazebo {Path(diag).stem}', series, [fall-1.5, fall]))
        report['gazebo'].append({'log': Path(diag).name, 'stop_reason': log['stop_reason'], 'end_t': fall,
                                 'pre_fall': summarize(series, weight, [fall-1.5, fall]),
                                 'window_5_6p5': summarize(series, weight, [5., 6.5]),
                                 'steady_2_to_3p5': summarize(series, weight, [2., 3.5]),
                                 'cycle_roll_peaks': per_cycle_peaks(series, 2.5, 2.)})
    for kv in args.mujoco_kv:
        series, result = mujoco_series(kv, args.cycles)
        end = float(series['t'][-1])
        fall_at = result['stopped_by_guard_at_s']
        all_series.append((f'mujoco kv{kv:g}', series, [fall_at-1.5, fall_at] if fall_at else [end-1.5, end]))
        report['mujoco'].append({'kv': kv, 'fall_at': result['stopped_by_guard_at_s'], 'end_t': end,
                                 'pre_end_or_fall': summarize(series, weight, [end-1.5, end]),
                                 'window_5_6p5': summarize(series, weight, [5., 6.5]),
                                 'steady_2_to_3p5': summarize(series, weight, [2., 3.5]),
                                 'cycle_roll_peaks': per_cycle_peaks(series, 2.5, 2.)[:20]})
    Path(args.out).write_text(json.dumps(report, indent=1))
    if args.series_out:
        Path(args.series_out).write_text(json.dumps({name: {'window_5_6p5': window(s, [5., 6.5]),
                                                              'pre_fall_or_end': window(s, w)}
                                                       for name, s, w in all_series}, separators=(',', ':')))
    if args.plot_dir:
        gz = next((x for x in all_series if x[0].startswith('gazebo')), None)
        for other in (x for x in all_series if x[0].startswith('mujoco')):
            if gz is None:
                break
            name = other[0].replace(' ', '_')
            compare_plot(Path(args.plot_dir)/f'{name}_5-6.5s.png', (gz[0], gz[1]), (other[0], other[1]), [5., 6.5],
                         f'{gz[0]} vs {other[0]}, 5.0-6.5 s (same gait: stride 0.08, 2.5 Hz, fixed tail)')
            compare_plot(Path(args.plot_dir)/f'{name}_gazebo-prefall.png', (gz[0], gz[1]), (other[0], other[1]),
                         gz[2], f'{gz[0]} last 1.5 s before the STOP vs {other[0]} same window')
    print('wrote', args.out)


if __name__ == '__main__':
    main()
