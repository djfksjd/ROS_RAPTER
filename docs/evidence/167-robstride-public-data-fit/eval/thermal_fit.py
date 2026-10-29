"""Fit the realism 2-node thermal model (realism.Thermal2RC, loss 1.5 i^2 r_line (1+a(T-25)), i = tau/kt_out) to the
RobStride user-manual overload tables (RobStride/Product_Information@384cc56: RS03User Manual261013.pdf p.11,
RS06User Manual260713.pdf p.10): time from 25 C to the 145 C winding limit at constant load, rotating at 100 rpm on the
vendor aluminium heat sink. Rotating table = fit target, stall table (single-phase heat x1.414 per the manual) = check.
Prints current-model vs fitted times and writes thermal_fit.json."""
import json, sys
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, 'sim/rl')
import realism
OUT = 'docs/evidence/167-robstride-public-data-fit/eval/thermal_fit.json'
TABLE = {  # load N·m: seconds (None = 'rated', continuous)
    'RS03': dict(kt_out=2.36, rotate={60: 13, 50: 34, 40: 67, 30: 393, 20: None}, stall={60: 1, 50: 4, 40: 9, 30: 25, 20: 190, 13: None}),
    'RS06': dict(kt_out=1.09, rotate={36: 5, 34: 6, 32: 8, 28: 11, 23: 21, 18: 70, 15: 280, 13: 1360, 11: None},
                 stall={36: 1, 30: 1, 25: 5, 20: 11, 17: 15, 11: 200, 8: None}),
}
cfg = json.load(open('sim/rl/realism_robstride.json'))
ov = {k: {f: v['value'] for f, v in d.items()} for k, d in cfg['motor']['spec_override'].items()}
spec = realism.load_motor_specs('docs/design/motor_candidates.json', ov, set(TABLE))
A, T0, TMAX, DT, HORIZON = .00393, 25., 145., .01, 4000.

def t_to_limit(p, tau, kt_out, r_line, heat=1.):
    """Seconds until the winding reaches TMAX at constant load (inf if it settles below)."""
    th = realism.Thermal2RC(*p); th.reset(T0)
    i = tau/kt_out
    n = int(HORIZON/DT)
    for k in range(n):
        th.step(heat*1.5*i*i*r_line*(1+A*(th.t_w-T0)), DT, T0)
        if th.t_w >= TMAX:
            return (k+1)*DT
    return np.inf

def steady(p, tau, kt_out, r_line, heat=1.):
    # fixed point of T = T0 + P(T) (r_wh + r_ha)
    i, R = tau/kt_out, p[0]+p[1]
    q = heat*1.5*i*i*r_line
    return T0+q*R/(1-q*R*A) if q*R*A < 1 else np.inf

def report(name, p, tab):
    r = spec[name]['r_phase']
    rows = []
    for kind, heat in (('rotate', 1.), ('stall', 1.414)):
        for tau, t in tab[kind].items():
            rows.append(dict(kind=kind, load=tau, manual_s=t, model_s=round(t_to_limit(p, tau, tab['kt_out'], r, heat), 1) if t else None,
                             steady_c=round(steady(p, tau, tab['kt_out'], r, heat), 1)))
    return rows

def loss(lp, name, tab):
    p = np.exp(lp); r = spec[name]['r_phase']; e = 0.
    for tau, t in tab['rotate'].items():
        if t is None:  # 'rated': settles at or just below the limit
            e += 4*(min(steady(p, tau, tab['kt_out'], r)-TMAX, 0)/20)**2 + 4*max(steady(p, tau, tab['kt_out'], r)-TMAX, 0)**2
        else:
            m = t_to_limit(p, tau, tab['kt_out'], r)
            e += (np.log(m/t) if np.isfinite(m) else 5.)**2
    return e

res = {}
for name, tab in TABLE.items():
    s = spec[name]; p0 = np.array([s['r_th_wh'], s['r_th_ha'], s['c_th_w'], s['c_th_h']])
    cur = report(name, p0, tab)
    fit = minimize(loss, np.log(p0), args=(name, tab), method='Nelder-Mead', options=dict(maxiter=600, xatol=1e-3, fatol=1e-4))
    pf = np.exp(fit.x)
    res[name] = dict(current=dict(r_th_wh=p0[0], r_th_ha=p0[1], c_th_w=p0[2], c_th_h=p0[3], rows=cur, loss=loss(np.log(p0), name, tab)),
                     fitted=dict(r_th_wh=round(pf[0], 4), r_th_ha=round(pf[1], 4), c_th_w=round(pf[2], 2), c_th_h=round(pf[3], 2),
                                 rows=report(name, pf, tab), loss=float(fit.fun)))
    print(name, 'current loss', round(res[name]['current']['loss'], 3), 'fitted loss', round(fit.fun, 3), {k: v for k, v in res[name]['fitted'].items() if k.startswith(('r_', 'c_'))})
    for a, b in zip(cur, res[name]['fitted']['rows']):
        print(f"  {a['kind']:6s} {a['load']:>3} N·m  manual {str(a['manual_s']):>5} s | current {str(a['model_s']):>6} s ss {a['steady_c']:>6} | fitted {str(b['model_s']):>6} s ss {b['steady_c']:>6}")
json.dump(res, open(OUT, 'w'), indent=1, default=float)
