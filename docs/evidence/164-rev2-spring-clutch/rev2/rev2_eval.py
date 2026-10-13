"""Diagnostics of the model_rev 2 options of sim/rl/run_env.py with the frozen policy scr_a8_h2 and with no policy at all.
No training. Each task = (label, env overrides, knee spring or None, command, init seed, seconds, mode).
mode 'policy' = deterministic frozen policy, 'zero' = zero action (PD hold of the nominal pose).
Run from the project root: .venv-sim/bin/python rev2_eval.py <out.jsonl> <suite> [n_seeds]"""
import json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path.cwd(); sys.path.insert(0, str(ROOT/'sim/rl')); sys.path.insert(0, str(ROOT/'modeling'))
RUN = ROOT/'sim/rl/runs/scr_a8_h2'
REV2 = dict(model_rev=2)
OPTS = dict(latch=dict(latch_fallback=False), friction=dict(friction_dr='foot'), selfcol=dict(self_collision=True), knee=dict(knee_band=(.7159, 2.2454)),
            start=dict(start_contact=True), armature=dict(armature=True))


def make(over, spring, seconds, init_seed):
    from evaluate import load
    from run_env import RunEnv
    from train_run import env_kwargs
    a = json.loads((RUN/'args.json').read_text()); kw = env_kwargs(a)
    if spring != 'baseline':
        kw['springs_override'] = None if spring is None else {'knee_pitch': list(spring)}
        if spring is None:
            kw['springs'] = 'none'
    env = RunEnv(**{**kw, 'episode_s': seconds+5., 'level': 0., **over}, randomize=False, seed=7); env.init_seed = init_seed
    model, venv = load(str(RUN/'model.zip'), env); venv.reset(); venv.reset()
    return env, model, venv


def rollout(task):
    label, over, spring, cmd, init_seed, seconds, mode = task
    t0 = time.time(); env, model, venv = make(over, spring, seconds, init_seed); m, d = env.model, env.data
    env.command = np.array([cmd, 0., 0.]); env.resample_steps = 0
    obs = venv.normalize_obs(env._obs()[None])
    ki = [env.ix[f'{s}_knee_pitch_joint'] for s in ('left', 'right')]; kq = env.q_adr[ki]
    names = [m.tendon(i).name for i in range(m.ntendon)]
    tend = [names.index(f'spring_{s}_knee_pitch') for s in ('left', 'right')] if 'spring_left_knee_pitch' in names else []
    defl = lambda j, q: (lambda lo, hi: q-hi if q > hi else q-lo if q < lo else 0.)(*m.tendon_lengthspring[tend[j]])
    e_in = e_out = w_pos = t = 0.; n_eng = n_eng_nz = 0; latched_steps = 0; stance_steps = 0; latched_stance = 0
    xs, heats, knee, pairs, self_steps, fell = [], [], [], {}, 0, False
    zero = np.zeros(len(env.policy_idx)); z0 = float(d.xpos[env.base][2]); pad0 = env._pad_clearance(); first_contact = None
    for step in range(int(seconds/.02)):
        k_pre = m.tendon_stiffness[tend].copy() if tend else []; q_pre = d.qpos[kq].copy()
        action = model.predict(obs, deterministic=True)[0][0] if mode == 'policy' else zero
        raw, _, term, _, info = env.step(action); obs = venv.normalize_obs(raw[None]); t += .02
        for j in range(len(tend)):
            kp = m.tendon_stiffness[tend[j]]
            if k_pre[j] == 0 and kp > 0:
                u = .5*kp*defl(j, q_pre[j])**2; e_in += u; n_eng += 1; n_eng_nz += u > 1e-6
            elif k_pre[j] > 0 and kp == 0:
                e_out += .5*k_pre[j]*defl(j, q_pre[j])**2
            side = ('left', 'right')[j]; latched_steps += kp > 0; stance_steps += info['loaded'][side]; latched_stance += (kp > 0) and info['loaded'][side]
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
    tail = slice(max(0, n-int(40/.02)), n) if seconds >= 50 else slice(n//2, n)
    drive = [i for i in range(len(env.active)) if env.motor_cont[i] < 1e8]; hm = heats[tail][:, drive].mean(0)
    row = dict(label=label, spring=spring, cmd=cmd, init_seed=init_seed, seconds=seconds, mode=mode, fell=fell, t=round(t, 2), speed_tail=float(xs[tail].mean()),
               heat_max=float(hm.max()), heat_argmax=env.active[drive[int(hm.argmax())]], heat_knee=float(max(hm[[drive.index(i) for i in ki]])),
               knee_min=float(knee.min()), knee_max=float(knee.max()), engagements=n_eng, engage_nonzero=int(n_eng_nz), energy_injected_J=float(e_in), energy_released_J=float(e_out),
               motor_positive_work_J=float(w_pos), latched_frac=float(latched_steps/max(1, 2*n)), latched_stance_frac=float(latched_stance/max(1, stance_steps)),
               self_contact_step_frac=self_steps/max(1, n), self_pairs=pairs, pad_clearance_start_m=pad0, first_contact_s=first_contact,
               base_z_start=z0, base_z_end=float(d.xpos[env.base][2]), wall_s=round(time.time()-t0, 1))
    env.close(); return row


def suite(name, seeds):
    T = []
    S170 = (170, .956, 'uni+', 'latch')
    if name == 'options':        # each option alone and all together, frozen policy, baseline spring definition
        for lab, ov in [('rev1', {}), *OPTS.items(), ('rev2_all', REV2), ('rev2_keep_fallback', {**REV2, 'latch_fallback': True})]:
            T += [(lab, ov, 'baseline', c, s, 60., 'policy') for c in (4., 7.) for s in seeds]
    if name == 'springs':        # rev 2, frozen policy, physical latch, spring grid
        grid = [('none', None)]+[(f'k{k}_q{q0}', (k, q0, 'uni+', 'latch')) for k in (85, 170, 255) for q0 in (.956, 1.1, 1.2, 1.3, 1.4, 1.5)]
        for lab, sp in grid:
            T += [(lab, REV2, sp, c, s, 60., 'policy') for c in (2., 4., 7.) for s in seeds]
    if name == 'stand':          # no policy: zero-action hold, 30 s
        for lab, ov, sp in [('rev1', {}, 'baseline'), ('rev2_k170_q0.956', REV2, S170), ('rev2_none', REV2, None), ('rev2_k170_q1.3', REV2, (170, 1.3, 'uni+', 'latch'))]:
            T += [(lab, ov, sp, 0., s, 30., 'zero') for s in seeds]
    return T


if __name__ == '__main__':
    out, name = Path(sys.argv[1]), sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    tasks = suite(name, [95001+i for i in range(n)]); t0 = time.time()
    with Pool(8) as pool, open(out, 'w') as f:
        for i, r in enumerate(pool.imap_unordered(rollout, tasks, chunksize=1)):
            f.write(json.dumps(r)+'\n'); f.flush()
    print('done', len(tasks), round(time.time()-t0), 's')
