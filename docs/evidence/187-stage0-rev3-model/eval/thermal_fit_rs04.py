"""Fit the realism 2-node thermal model to the RobStride RS04 overload table (RobStride/Product_Information@384cc56
RS04User Manual260713.pdf p.11 items 13-14; values in docs/design/motor_candidates.json 'overload_table'): seconds from 25 C
to the 145 C winding limit at constant load on a 345x345 mm aluminium heat sink, rotating at 50 rpm (fit target) and
stalled (single-phase heat x1.414, check). Same method as evidence 167 (thermal_fit.py for RS03/RS06): loss 1.5 i^2 r_line
(1 + a (T - 25)), i = tau/kt_out, Nelder-Mead on log-time error, 'rated' = settles at or just below 145 C. The start point
is the RS03 fit scaled by the RS04/RS03 rated copper loss. Writes thermal_fit_rs04.json next to this file."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, 'sim/rl')
import realism
OUT = Path(__file__).with_name('thermal_fit_rs04.json')
mod = json.load(open('docs/design/motor_candidates.json'))['modules']['RS04']
tab = mod['overload_table']
KT, R = mod['kt_out'], mod['r_line']
ROT = {float(k): v for k, v in tab['rotate_50rpm'].items()}
STALL = {float(k): v for k, v in tab['stall'].items()}
A, T0, TMAX, DT, HORIZON = .00393, 25., 145., .01, 4000.


def t_to_limit(p, tau, heat=1.):
    th = realism.Thermal2RC(*p); th.reset(T0)
    i = tau/KT
    for k in range(int(HORIZON/DT)):
        th.step(heat*1.5*i*i*R*(1+A*(th.t_w-T0)), DT, T0)
        if th.t_w >= TMAX:
            return (k+1)*DT
    return np.inf


def steady(p, tau, heat=1.):
    i, Rt = tau/KT, p[0]+p[1]
    q = heat*1.5*i*i*R
    return T0+q*Rt/(1-q*Rt*A) if q*Rt*A < 1 else np.inf


def loss(lp):
    p, e = np.exp(lp), 0.
    for tau, t in ROT.items():
        if t is None:
            ss = steady(p, tau)
            e += 4*(min(ss-TMAX, 0)/20)**2 + 4*max(ss-TMAX, 0)**2
        else:
            m = t_to_limit(p, tau)
            e += (np.log(m/t) if np.isfinite(m) else 5.)**2
    return e


rs03 = json.load(open('sim/rl/realism_robstride_v2.json'))['motor']['spec_override']['RS03']
p_rs03 = np.array([rs03[k]['value'] for k in ('r_th_wh', 'r_th_ha', 'c_th_w', 'c_th_h')])
# rated copper loss ratio RS04/RS03 at the rated torques (40 vs 20 N·m): scale resistances down, capacities up
q04, q03 = 1.5*(40/KT)**2*R, 1.5*(20/2.36)**2*.39
p0 = p_rs03*np.array([q03/q04, q03/q04, q04/q03, q04/q03])
fit = minimize(loss, np.log(p0), method='Nelder-Mead', options=dict(maxiter=1500, xatol=1e-4, fatol=1e-5))
pf = np.exp(fit.x)
rows = [dict(kind=k, load=tau, manual_s=t, model_s=round(t_to_limit(pf, tau, h), 1) if t else None, steady_c=round(steady(pf, tau, h), 1))
        for k, h, T in (('rotate', 1., ROT), ('stall', 1.414, STALL)) for tau, t in T.items()]
res = dict(fitted=dict(r_th_wh=round(pf[0], 4), r_th_ha=round(pf[1], 4), c_th_w=round(pf[2], 2), c_th_h=round(pf[3], 2)),
           start=dict(zip(('r_th_wh', 'r_th_ha', 'c_th_w', 'c_th_h'), np.round(p0, 4).tolist())), loss=float(fit.fun), rows=rows)
print(json.dumps(res['fitted']), 'loss', round(fit.fun, 3))
for r in rows:
    print(f"  {r['kind']:6s} {r['load']:>5} N·m  manual {str(r['manual_s']):>5} s | fitted {str(r['model_s']):>6} s  steady {r['steady_c']} C")
OUT.write_text(json.dumps(res, indent=1, default=float))
