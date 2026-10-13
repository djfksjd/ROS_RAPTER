"""Inverse-dynamics replay on the gait of the policy trained on model_rev 2 (rev2_a8_h2): what would a knee spring that is
already engaged at touchdown do for THIS motion? The policy never engages the latch, so the recorded knee torque is all motor.
Scenario 'stance': unilateral spring (k, q0) active whenever the foot is loaded (the engagement and the work to deflect it
in late swing are ignored -> upper bound of the benefit). Scenario 'stance+pre': also active during the 100 ms before each
touchdown, where the motor has to hold the knee flexed against the spring in the air. The motion is held fixed.
Run from the project root: .venv-sim/bin/python spring_id_rev2.py <out.json>"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
out_path = Path(sys.argv[1]); sys.argv[1] = str(HERE)
import spring_id as S
S.RUN = Path.cwd()/'sim/rl/runs/rev2_a8_h2'
from run_env import MOTOR_CLASSES, LINK_ETA, motor_available


def heat(sub, ctl, kr, k, q0, pre, skip=250):
    res = []
    for j in range(2):
        q, qd, req = sub[:, j], sub[:, 2+j], sub[:, 4+j]+sub[:, 6+j]
        ld = ctl[:, 2+j].astype(bool); on = ld.copy()
        if pre:
            for i in np.flatnonzero(ld[1:] & ~ld[:-1])+1:
                on[max(0, i-pre):i] = True
        on = np.repeat(on, 20)
        tm = req-np.where(on, -k*np.clip(q-q0, 0., None), 0.); N = np.interp(q, *kr)
        lim = np.where(tm*qd >= 0, N*LINK_ETA*motor_available('B', N*qd), N*LINK_ETA*MOTOR_CLASSES['B'][0]); s = slice(skip*20, None)
        res.append((float(np.mean((tm[s]/(N[s]*LINK_ETA)/S.CONT)**2)), float(np.mean(np.abs(tm[s]) > lim[s]+1e-9)), float(np.abs(tm[s]).max())))
    return [max(r[i] for r in res) for i in range(3)]


res = {}; K = np.arange(0, 601, 20); Q0 = np.round(np.arange(.70, 1.301, .02), 3)
for cmd in (2., 4., 7.):
    sub, ctl, kr = S.record(cmd, 96001)
    st = np.repeat(ctl[:, 2:4].astype(bool), 20, axis=0); req = sub[:, 4:6]+sub[:, 6:8]
    row = dict(none=heat(sub, ctl, kr, 0, 1., 0), req_peak=float(np.abs(req).max()), req_stance_rms=float(np.sqrt(np.mean(req[st]**2))), stance_knee_mean=float(sub[:, :2][st].mean()))
    for name, pre in (('stance', 0), ('stance+pre', 5)):
        g = np.array([[heat(sub, ctl, kr, k, q0, pre)[0] for q0 in Q0] for k in K]); i, j = np.unravel_index(g.argmin(), g.shape)
        row[name] = dict(k170_q0956=heat(sub, ctl, kr, 170*11.2912/11.36, .956, pre), k340_q0956=heat(sub, ctl, kr, 340*11.2912/11.36, .956, pre),
                         best=dict(k=float(K[i]), q0=float(Q0[j]), res=heat(sub, ctl, kr, float(K[i]), float(Q0[j]), pre)), grid=g.tolist())
    res[f'{cmd:g}'] = row
    print(cmd, {k: (v if not isinstance(v, dict) else {a: b for a, b in v.items() if a != 'grid'}) for k, v in row.items()})
res['K'] = K.tolist(); res['Q0'] = Q0.tolist(); res['columns'] = ['heat', 'frac_over_envelope', 'peak_Nm']
out_path.write_text(json.dumps(res))
