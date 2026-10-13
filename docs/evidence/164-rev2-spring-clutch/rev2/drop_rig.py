"""Drop-and-rebound test on a virtual rig, no policy and no training (model_rev 2).

The torso orientation and lateral position are held by a rig (PD wrench on the torso each 1 ms physics step); height and
fore-aft position are free. Both legs are set to the same pose: hip at the nominal angle (level torso and flat toes fix
it), knee at the touchdown angle q_td, ankle through the 1:1 coupling. The robot is released with the toe pads h above
the ground and the knee servo keeps the target q_td. Spring variants start latched (the late-swing state of the latch
state machine) at q_td = q0, i.e. with zero deflection. Equal conditions for every variant: same pose, same servo, same h.
Measured over the first stance: knee flexion, knee motor load, heat integral, spring energy; then the rebound apex."""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT/'sim/rl')); sys.path.insert(0, str(ROOT/'modeling'))
RUN = ROOT/'sim/rl/runs/scr_a8_h2'


def run(task):
    label, k, q_td, h, frames = task
    import mujoco, run_env
    from run_env import RunEnv, MOTOR_CLASSES, LINK_ETA
    from train_run import env_kwargs
    kw = env_kwargs(json.loads((RUN/'args.json').read_text()))
    kw['springs_override'] = {'knee_pitch': [k, q_td, 'uni+', 'latch']} if k else None
    if not k:
        kw['springs'] = 'none'
    env = RunEnv(**{**kw, 'level': 0., 'episode_s': 30., 'model_rev': 2}, randomize=False, seed=7, render_mode='rgb_array' if frames else None, model_path=str(Path(__file__).with_name('raptor_r02_rig.xml')))
    env.reset(); m, d = env.model, env.data; env.command[:] = 0; env.resample_steps = 0
    for s in ('left', 'right'):
        kq, aq = env.q_adr[env.ix[f'{s}_knee_pitch_joint']], env.q_adr[env.ix[f'{s}_ankle_pitch_joint']]
        c = d.qpos[kq]+d.qpos[aq]; d.qpos[kq] = q_td; d.qpos[aq] = c-q_td
    d.qvel[:] = 0; mujoco.mj_forward(m, d); d.qpos[2] += h-env._pad_clearance()
    rq = [m.jnt_qposadr[m.joint(n).id] for n in ('rig_x', 'rig_z')]; d.qpos[rq] = [d.qpos[0], d.qpos[2]]; mujoco.mj_forward(m, d)
    if k:
        for s in ('left', 'right'):
            env.spring_was_on[(s, 'knee_pitch')] = True; env.latch_armed[s] = True; m.tendon_stiffness[m.tendon(f'spring_{s}_knee_pitch').id] = env.spring_engage[(s, 'knee_pitch')][1]
    env.loaded = {'left': False, 'right': False}; env.stance_seen = {}
    target = d.qpos[env.q_adr].copy(); d.ctrl[env.act] = target; env.ctrl_prev = target.copy()
    action = np.zeros(len(env.policy_idx)); pol = list(env.policy_idx)
    for s in ('left', 'right'):
        action[pol.index(env.ix[f'{s}_knee_pitch_joint'])] = q_td-env.q0[env.ix[f'{s}_knee_pitch_joint']]
    ki = env.ix['left_knee_pitch_joint']; kq, kv, ka = env.q_adr[ki], env.v_adr[ki], env.act[ki]
    base = env.base; quat0 = d.xquat[base].copy(); y0 = d.xpos[base][1]; rec = []
    step0 = mujoco.mj_step; neg = np.zeros(4); dq = np.zeros(4); rv = np.zeros(3)
    kspr = env.spring_engage[('left', 'knee_pitch')][1] if k else 0.
    tid = m.tendon('spring_left_knee_pitch').id if k else None

    def mj_step(m_, d_):
        step0(m_, d_)
        mujoco.mju_negQuat(neg, quat0); mujoco.mju_mulQuat(dq, d_.xquat[base], neg); mujoco.mju_quat2Vel(rv, dq, 1.)
        on = bool(m_.tendon_stiffness[tid] > 0) if k else False
        rec.append((d_.time, env._pad_clearance(), d_.qpos[kq], d_.qvel[kv], d_.actuator_force[ka], d_.qfrc_passive[kv], on, d_.xpos[base][2], np.linalg.norm(rv)))
    run_env.mujoco.mj_step = mj_step; grf = []; imgs = []
    if frames:
        from PIL import Image
        r = mujoco.Renderer(m, 360, 480); cam = mujoco.MjvCamera(); cam.distance, cam.azimuth, cam.elevation = 2.4, 90, -6
    try:
        for i in range(100):
            if frames and i % 4 == 0 and i < 48:
                cam.lookat[:] = [d.xpos[base][0], 0, .45]; r.update_scene(d, cam); imgs.append(r.render().copy())
            _, _, _, _, info = env.step(action); grf.append((info['grf']['left'], info['loaded']['left']))
    finally:
        run_env.mujoco.mj_step = step0
    if frames:
        W = np.concatenate([np.concatenate(imgs[j:j+4], 1) for j in (0, 4, 8)], 0); Image.fromarray(W).save(frames)
    a = np.array(rec); t, clr, q, qd, tau, tsp, on, bz, tilt = a.T; grf = np.array(grf)
    N = np.interp(q, *env.knee_ratio); ms = tau/(N*LINK_ETA); cont, peak = .4*MOTOR_CLASSES['B'][0], MOTOR_CLASSES['B'][0]
    ld = np.repeat(grf[:, 1].astype(bool), 20); td = int(np.argmax(ld)) if ld.any() else None
    row = dict(label=label, k=k, q_td=q_td, h=h, touched=td is not None, rig_tilt_max=float(tilt.max()))
    if td is not None:
        lo = td+int(np.argmax(~ld[td:])) if (~ld[td:]).any() else len(ld); st = slice(max(0, td-20), lo)
        after = slice(lo, min(len(t), lo+600))
        row.update(stance_s=float((lo-td)*.001), lifted_off=bool(lo < len(ld)), knee_max=float(q[st].max()), knee_flex=float(q[st].max()-q_td),
                   motor_peak_ratio=float(np.abs(ms[st]).max()/peak), motor_sat_frac=float(np.mean(np.abs(ms[st]) > .98*peak)),
                   heat_integral_s=float(np.sum((ms[st]/cont)**2)*.001), heat_mean_stance=float(np.mean((ms[st]/cont)**2)),
                   motor_neg_J=float(2*np.sum(np.clip(-tau[st]*qd[st], 0, None))*.001), motor_pos_J=float(2*np.sum(np.clip(tau[st]*qd[st], 0, None))*.001),
                   spring_peak_Nm=float(np.abs(tsp[st]).max()), spring_E_peak_J=float(2*.5*kspr*np.clip(q[st]-q_td, 0, None).max()**2) if k else 0.,
                   grf_peak_per_leg_BW=float(grf[:, 0].max()), base_drop_m=float(bz[0]-bz[st].min()),
                   rebound_apex_m=float(clr[after].max()) if lo < len(ld) else 0., restitution=float(clr[after].max()/h) if lo < len(ld) else 0.,
                   drop_energy_J=float(env.mass*9.81*h))
    env.close(); return row


if __name__ == '__main__':
    out = Path(sys.argv[1]); T = []
    for q in (.956, 1.1, 1.3):
        for h in (.02, .05, .10, .20):
            T += [(f'none@{q}', 0, q, h, None)]+[(f'k{k}@{q}', k, q, h, None) for k in (85, 170, 255, 340)]
    T += [(lab, k, q, .10, str(out/f'drop_{lab}.png')) for lab, k, q in (('none_q0.956', 0, .956), ('k170_q0.956', 170, .956), ('k170_q1.3', 170, 1.3))]
    with Pool(8) as p:
        R = p.map(run, T, chunksize=1)
    (out/'drop_rig.json').write_text(json.dumps(R, indent=0))
    for r in R[:60]:
        if r['touched']:
            print(f"{r['label']:12s} h{r['h']*100:4.0f}cm flex {r['knee_flex']:.3f} kmax {r['knee_max']:.3f} peak {r['motor_peak_ratio']:.2f} sat {r['motor_sat_frac']:.2f} heatI {r['heat_integral_s']:.3f}s negJ {r['motor_neg_J']:5.1f} posJ {r['motor_pos_J']:5.1f} sprE {r['spring_E_peak_J']:5.1f}/{r['drop_energy_J']:5.1f} grf {r['grf_peak_per_leg_BW']:.2f} stance {r['stance_s']:.2f}s lift {int(r['lifted_off'])} rest {r['restitution']:.2f} tilt {r['rig_tilt_max']:.3f}")
        else:
            print(r['label'], r['h'], 'no touchdown')
