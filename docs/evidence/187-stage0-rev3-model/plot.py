"""Evidence 187 figures. motor_curves.png: module output torque vs speed for RS03 / RS04 / RS06: the vendor 48 V T-N curve
(digitised, the rev 3 realism cap) against the DC model of realism.py alone (48 V, kt_out, r_line, i_peak), and the RS04
thermal fit against its manual overload table. rev3_model.png: R-03 render at the nominal pose and with both hip yaw
joints at +-0.3 rad. Run from repo root."""
import json, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
E = 'docs/evidence/187-stage0-rev3-model'
M = json.load(open('docs/design/motor_candidates.json'))['modules']
fit = json.load(open(f'{E}/eval/thermal_fit_rs04.json'))
fig, ax = plt.subplots(1, 4, figsize=(17, 4))
for a, m in zip(ax, ('RS03', 'RS04', 'RS06')):
    d = M[m]; c = d['tn_curve']
    w = np.linspace(0, d['w0']*1.1, 300)
    dc = np.minimum(np.clip((48-d['kt_out']*w)/d['r_line'], 0, d['i_peak'])*d['kt_out'], d['tau_peak'])
    a.plot(w*30/np.pi, dc, 'k--', label='DC model alone (realism v2)')
    a.plot(c['rpm'], c['nm'], 'o-', color='C3', ms=3, label='vendor 48 V T-N curve (rev 3 cap)')
    a.axhline(d['tau_cont'], color='C2', lw=1, label=f"rated {d['tau_cont']:g} N·m")
    a.set_xlabel('module output speed (rpm)'); a.set_ylabel('torque (N·m)'); a.set_title(m); a.grid(alpha=.3); a.legend(fontsize=7)
rows = [r for r in fit['rows'] if r['manual_s']]
for kind, col in (('rotate', 'C0'), ('stall', 'C1')):
    R = [r for r in rows if r['kind'] == kind]
    ax[3].plot([r['load'] for r in R], [r['manual_s'] for r in R], 'o', color=col, label=f'manual, {kind}')
    ax[3].plot([r['load'] for r in R], [r['model_s'] for r in R], '-', color=col, label=f'fitted model, {kind}')
ax[3].set_yscale('log'); ax[3].set_xlabel('constant load (N·m)'); ax[3].set_ylabel('seconds from 25 to 145 °C')
ax[3].set_title('RS04 overload table vs fitted thermal model\n(rotating = fit target, stall = check)', fontsize=9); ax[3].legend(fontsize=7); ax[3].grid(alpha=.3)
fig.tight_layout(); fig.savefig(f'{E}/media/motor_curves.png', dpi=100)

sys.path.insert(0, 'sim/rl')
import mujoco
from run_env import RunEnv
imgs = []
for yaw, az, el in ((0., 140., -15), (0., 90., -10), (0., 180., -60), (.3, 180., -60)):
    e = RunEnv(arch8=True, real_mass=True, model_rev=3, randomize=False, seed=1, render_mode='rgb_array')
    e.reset(seed=1)
    d = e.data
    d.qpos[e.q_adr] = e.q0
    for s in ('left', 'right'):
        d.qpos[e.q_adr[e.active.index(f'{s}_hip_yaw_joint')]] = yaw if s == 'left' else -yaw
    d.qvel[:] = 0
    mujoco.mj_forward(e.model, d)
    r = mujoco.Renderer(e.model, 480, 480)
    cam = mujoco.MjvCamera(); cam.lookat[:] = d.xpos[e.base]; cam.distance, cam.azimuth, cam.elevation = (1.9 if el > -30 else 1.3), az, el
    r.update_scene(d, cam); imgs.append(r.render())
fig, ax = plt.subplots(1, 4, figsize=(17, 4.8))
for a, im, t in zip(ax, imgs, ('nominal pose', 'side view', 'from above, hip yaw 0', 'from above, hip yaw +-0.3 rad (toes out)')):
    a.imshow(im); a.set_title(t); a.axis('off')
fig.suptitle('R-03 (model rev 3): R-02 + hip yaw per leg, 15.1 kg, 10 actions; MuJoCo render, not hardware', fontsize=10)
fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(f'{E}/media/rev3_model.png', dpi=90)
