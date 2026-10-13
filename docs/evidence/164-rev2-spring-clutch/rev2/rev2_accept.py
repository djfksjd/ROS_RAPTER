"""Acceptance and diagnosis of a policy trained on model_rev 2 (sim/rl/train_run.py --model-rev 2).
Seed list fixed before the first result existed: 96001..96299 (not used in any earlier diagnostic; those used 95001..95010).
Each task = (label, env overrides, knee spring | 'baseline' (as trained) | None (no spring), command, seed, seconds, mode).
mode 'policy' = deterministic policy, start perturbed by init_seed, no randomisation, no disturbances;
mode 'dr' = deterministic policy with the training randomisation (mass, friction on foot pads, start state) at the given level.
Run from the project root: .venv-sim/bin/python rev2_accept.py <run_dir> <out.jsonl> <suite> <cmds comma> [n_seeds] [level]"""
import json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path.cwd(); sys.path.insert(0, str(ROOT/'sim/rl')); sys.path.insert(0, str(ROOT/'modeling'))
RUN = Path(sys.argv[1]).resolve()
LEVEL = float(sys.argv[6]) if len(sys.argv) > 6 else 0.


def make(over, spring, seconds, init_seed):
    over = dict(over)
    from evaluate import load
    from run_env import RunEnv
    from train_run import env_kwargs
    a = json.loads((RUN/'args.json').read_text()); kw = env_kwargs(a)
    if spring != 'baseline':
        kw['springs_override'] = None if spring is None else {'knee_pitch': list(spring)}
        if spring is None:
            kw['springs'] = 'none'
    dr = over.pop('_dr', False) if '_dr' in over else False
    nop = over.pop('_no_param_dr', False)
    env = RunEnv(**{**kw, 'episode_s': seconds+5., 'level': LEVEL if dr else 0., **over}, randomize=dr, seed=init_seed if dr else 7); env.init_seed = init_seed
    if nop:  # keep the nominal mass, friction and torque limits (the other randomisation stays on)
        def nominal():
            env.model.body_mass[:] = env.nominal_mass; env.model.geom_friction[:] = env.nominal_friction; env.tau_max = env.tau_nom.copy()
        env.reset(seed=init_seed); env._randomize = nominal
    model, venv = load(str(RUN/'model.zip'), env); venv.reset(); venv.reset()
    return env, model, venv


def rollout(task):
    label, over, spring, cmd, init_seed, seconds, mode = task
    t0 = time.time(); env, model, venv = make({**over, '_dr': mode == 'dr'}, spring, seconds, init_seed); m, d = env.model, env.data
    env.command = np.array([cmd, 0., 0.]); env.resample_steps = 0
    obs = venv.normalize_obs(env._obs()[None])
    ki = [env.ix[f'{s}_knee_pitch_joint'] for s in ('left', 'right')]; kq = env.q_adr[ki]
    names = [m.tendon(i).name for i in range(m.ntendon)]
    tend = [names.index(f'spring_{s}_knee_pitch') for s in ('left', 'right')] if 'spring_left_knee_pitch' in names else []
    defl = lambda j, q: (lambda lo, hi: q-hi if q > hi else q-lo if q < lo else 0.)(*m.tendon_lengthspring[tend[j]])
    e_in = e_out = w_pos = t = 0.; n_eng = n_eng_nz = 0; latched_steps = 0; stance_steps = 0; latched_stance = 0
    xs, heats, knee, pairs, self_steps, fell = [], [], [], {}, 0, False
    u_peak = [0., 0.]; u_peaks = []; active_stance = 0; was_loaded = [True, True]; near_lim = np.zeros(len(env.active)); tail_i = None
    zero = np.zeros(len(env.policy_idx)); z0 = float(d.xpos[env.base][2]); pad0 = env._pad_clearance(); first_contact = None
    for step in range(int(seconds/.02)):
        k_pre = m.tendon_stiffness[tend].copy() if tend else []; q_pre = d.qpos[kq].copy()
        action = model.predict(obs, deterministic=True)[0][0] if mode in ('policy', 'dr') else zero
        raw, _, term, _, info = env.step(action); obs = venv.normalize_obs(raw[None]); t += .02
        for j in range(len(tend)):
            kp = m.tendon_stiffness[tend[j]]
            if k_pre[j] == 0 and kp > 0:
                u = .5*kp*defl(j, q_pre[j])**2; e_in += u; n_eng += 1; n_eng_nz += u > 1e-6
            elif k_pre[j] > 0 and kp == 0:
                e_out += .5*k_pre[j]*defl(j, q_pre[j])**2
            side = ('left', 'right')[j]; latched_steps += kp > 0; stance_steps += info['loaded'][side]; latched_stance += (kp > 0) and info['loaded'][side]
            u_now = .5*kp*defl(j, d.qpos[kq[j]])**2 if kp > 0 else 0.  # energy held by the spring at the end of this control step
            if info['loaded'][side]:
                u_peak[j] = max(u_peak[j], u_now); active_stance += u_now > 1e-3
            elif was_loaded[j]:
                if step > 250: u_peaks.append(u_peak[j])
                u_peak[j] = 0.
            was_loaded[j] = info['loaded'][side]
        near_lim += (d.qpos[env.q_adr] < env.lo+.03) | (d.qpos[env.q_adr] > env.hi-.03)
        w_pos += info['power']*.02
        if first_contact is None and any(info['loaded'].values()):
            first_contact = t
        xs.append(float(info['v_body'][0])); heats.append(env.heat_inst.copy()); knee.append(d.qpos[kq].copy())
        if env.self_collision:
            hit = False
            for i in range(d.ncon):
                c = d.contact[i]
                if c.geom1 in env.robot_geoms and c.geom2 in env.robot_geoms:
                    hit = True; key = '|'.join(sorted(m.body(m.geom_bodyid[g]).name.replace('_link', '') for g in (c.geom1, c.geom2))); pairs[key] = pairs.get(key, 0)+1
            self_steps += hit
        if term:
            fell = True; break
    xs, heats, knee = np.array(xs), np.array(heats), np.array(knee); n = len(xs)
    if getattr(env, 'ratchet', None):  # the ratchet sets its rest angle inside the physics step: use the environment's own counters
        e_in, e_out = env.spring_inject_J, env.spring_release_J
    tail = slice(max(0, n-int(40/.02)), n) if seconds >= 50 else slice(n//2, n)
    drive = [i for i in range(len(env.active)) if env.motor_cont[i] < 1e8]; hm = heats[tail][:, drive].mean(0)
    row = dict(label=label, spring=spring, cmd=cmd, init_seed=init_seed, seconds=seconds, mode=mode, fell=fell, t=round(t, 2), speed_tail=float(xs[tail].mean()),
               heat_max=float(hm.max()), heat_argmax=env.active[drive[int(hm.argmax())]], heat_knee=float(max(hm[[drive.index(i) for i in ki]])),
               knee_min=float(knee.min()), knee_max=float(knee.max()), engagements=n_eng, engage_nonzero=int(n_eng_nz), energy_injected_J=float(e_in), energy_released_J=float(e_out),
               motor_positive_work_J=float(w_pos), latched_frac=float(latched_steps/max(1, 2*n)), latched_stance_frac=float(latched_stance/max(1, stance_steps)),
               spring_peak_J=float(np.mean(u_peaks)) if u_peaks else 0., spring_active_stance_frac=float(active_stance/max(1, stance_steps)),
               heat_all={env.active[i].replace('_joint', ''): float(hm[k]) for k, i in enumerate(drive)}, near_limit_frac_max=float(near_lim.max()/max(1, n)), near_limit_joint=env.active[int(near_lim.argmax())].replace('_joint', ''),
               self_contact_step_frac=self_steps/max(1, n), self_pairs=pairs, pad_clearance_start_m=pad0, first_contact_s=first_contact,
               base_z_start=z0, base_z_end=float(d.xpos[env.base][2]), wall_s=round(time.time()-t0, 1))
    env.close(); return row


def suite(name, seeds, cmds):
    T = []
    if name == 'accept':       # as trained, fixed seeds, no randomisation
        T += [('accept', {}, 'baseline', c, s, 60., 'policy') for c in cmds for s in seeds]
    if name == 'dr':           # as trained, training randomisation at LEVEL
        T += [('dr', {}, 'baseline', c, s, 60., 'dr') for c in cmds for s in seeds]
    if name == 'ablate':       # same policy, knee spring removed or changed at evaluation time
        for lab, sp in [('as_trained', 'baseline'), ('no_spring', None), ('k85', (85, .956, 'uni+', 'latch')), ('k340', (340, .956, 'uni+', 'latch'))]:
            T += [(lab, {}, sp, c, s, 60., 'policy') for c in cmds for s in seeds]
    if name == 'clutch':       # same policy, knee spring switched to a clutch that locks its rest angle at the touchdown knee angle
        for lab, sp in [('as_trained', 'baseline'), ('td170', (170, .956, 'uni+', 'touchdown')), ('td340', (340, .956, 'uni+', 'touchdown')), ('td680', (680, .956, 'uni+', 'touchdown'))]:
            T += [(lab, {}, sp, c, s, 60., 'policy') for c in cmds for s in seeds]
    if name == 'clutch_ablate':  # policy trained with the touchdown clutch: spring removed or stiffness changed at evaluation time
        for lab, sp in [('as_trained', 'baseline'), ('no_spring', None), ('td170', (170, .956, 'uni+', 'touchdown')), ('td680', (680, .956, 'uni+', 'touchdown'))]:
            T += [(lab, {}, sp, c, s, 60., 'policy') for c in cmds for s in seeds]
    if name == 'tailgear':     # same policy, tail pitch motor behind a reduction (class B kept; hip pitch already uses 1.5:1)
        for lab, g in [('as_trained', None), ('tail_1p5', 1.5), ('tail_2p0', 2.)]:
            T += [(lab, {} if g is None else dict(arch8_override={'hip_pitch': ['B', 1.5], 'tail_pitch': ['B', g]}), 'baseline', c, s, 60., 'policy') for c in cmds for s in seeds]
    if name == 'dr_parts':     # training randomisation with one component removed at a time
        for lab, ov in [('no_yaw_impulse', dict(yaw_impulse=None)), ('no_push', dict(disturb_items=())), ('no_param', dict(_no_param_dr=True)),
                        ('init_only', dict(yaw_impulse=None, disturb_items=(), _no_param_dr=True))]:
            T += [(lab, ov, 'baseline', c, s, 60., 'dr') for c in cmds for s in seeds]
    if name == 'ratchet':      # same policy, knee spring switched to the ratchet clutch at several stiffnesses
        for lab, sp in [('as_trained', 'baseline'), ('rt340', (340, .956, 'uni+', 'ratchet')), ('rt500', (500, .956, 'uni+', 'ratchet')), ('rt680', (680, .956, 'uni+', 'ratchet'))]:
            T += [(lab, {}, sp, c, s, 60., 'policy') for c in cmds for s in seeds]
    if name == 'spring_off':   # same policy with and without the knee spring
        for lab, sp in [('as_trained', 'baseline'), ('no_spring', None)]:
            T += [(lab, {}, sp, c, s, 60., 'policy') for c in cmds for s in seeds]
    if name == 'rev1':         # same policy on the old model revision (old latch with touchdown fallback etc.)
        T += [('on_rev1', dict(model_rev=1), 'baseline', c, s, 60., 'policy') for c in cmds for s in seeds]
    return T


if __name__ == '__main__':
    out, name = Path(sys.argv[2]), sys.argv[3]; cmds = [float(x) for x in sys.argv[4].split(',')]; n = int(sys.argv[5]) if len(sys.argv) > 5 else 299
    tasks = suite(name, [96001+i for i in range(n)], cmds); t0 = time.time()
    with Pool(8) as pool, open(out, 'w') as f:
        for i, r in enumerate(pool.imap_unordered(rollout, tasks, chunksize=1)):
            f.write(json.dumps(r)+'\n'); f.flush()
    print('done', len(tasks), round(time.time()-t0), 's')
