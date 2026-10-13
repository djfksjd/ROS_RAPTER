"""Gate checks G2-G8 on the frozen baseline policy scr_a8_h2 (no training, repository code not modified).

Every condition is the unmodified RunEnv plus one change applied from this script. Deterministic policy, seeded starts.
Conditions:
  base        unchanged (also instrumented: knee-angle use, latch energy, self-overlap detection, start gap)
  mu_lo/mu_hi foot-geom friction scaled by 0.625 / 1.5625 (contact friction 0.5 / 1.25 at the 0.8 pads)
  selfcol     robot self-collision enabled by contype/conaffinity bit groups (left leg, right leg, tail, torso)
  knee30      knee target range clipped to the four-bar band where both linkage angles are >= 30 deg
  latch_nofb  latch without the touchdown fallback: it engages only in swing at q <= q0 (zero deflection); released at reset
  latch_clutch touchdown fallback engages at the current knee angle (zero deflection); swing latch at q0 unchanged
  nogap       start with the toes on the ground instead of 13 mm above it
  armature    reflected rotor inertia on the 8 driven joints (AK10-9 V3.0 rotor 1002 g cm^2, 9:1, external ratios)
Run from the project root:  .venv-sim/bin/python gates.py <out_dir> [n_seeds] [seconds]
"""
import copy, inspect, json, sys, textwrap, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT/'sim/rl')); sys.path.insert(0, str(ROOT/'modeling'))
RUN = ROOT/'sim/rl/runs/scr_a8_h2'
KNEE_BAND = (.7159, 2.2454)
JR_MODULE = 81*1.002e-4          # kg m^2 at the module output: 9^2 x 1002 g cm^2
ARMATURE = {'hip_pitch': JR_MODULE*1.5**2, 'knee_pitch': JR_MODULE*1.4**2, 'hip_roll': JR_MODULE*1.7**2, 'tail_yaw': JR_MODULE, 'tail_pitch': JR_MODULE}
CONDITIONS = ('base', 'mu_lo', 'mu_hi', 'selfcol', 'knee30', 'latch_nofb', 'latch_clutch', 'nogap', 'armature')


def variant_class(kind):
    import run_env
    if kind not in ('latch_nofb', 'latch_clutch'):
        return run_env.RunEnv
    src = textwrap.dedent(inspect.getsource(run_env.RunEnv.step))

    def rep(a, b):
        nonlocal src
        assert src.count(a) == 1, (kind, src.count(a), a[:50]); src = src.replace(a, b)
    a_q0 = "q0 = m.tendon_lengthspring[t][0] if m.tendon_lengthspring[t][0] > -1e5 else m.tendon_lengthspring[t][1]"
    I = ' '*12
    a_fb = I+"if self.loaded[s_]:\n"+I+"    on = True                                   # stays engaged through stance (fallback: engage at touchdown)\n"
    a_sw = I+"    if not was and qj <= q0:                   # extension past the rest angle in swing: latch\n"+I+"        on = True\n"
    rep(a_q0, "q0 = self.latch_q0")
    if kind == 'latch_nofb':
        rep(a_fb, I+"if self.loaded[s_]:\n"+I+"    on = was\n")
    else:
        rep(a_fb, I+"if self.loaded[s_]:\n"+I+"    if not was:\n"+I+"        m.tendon_lengthspring[t] = [qj, qj]\n"+I+"    on = True\n")
        rep(a_sw, I+"    if not was and qj <= q0:\n"+I+"        m.tendon_lengthspring[t] = [q0, q0]\n"+I+"        on = True\n")
    ns = dict(run_env.__dict__); exec(src, ns)
    return type('RunEnv_'+kind, (run_env.RunEnv,), {'step': ns['step'], 'latch_q0': .956})


def sync_body_masks(m):
    """MuJoCo's broad phase filters on per-body masks compiled from the geoms; refresh them after editing geom masks."""
    m.body_contype[:] = 0; m.body_conaffinity[:] = 0
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]; m.body_contype[b] |= m.geom_contype[g]; m.body_conaffinity[b] |= m.geom_conaffinity[g]


def make(kind, seconds, init_seed):
    import mujoco
    from evaluate import load
    from run_env import joint_type
    from train_run import env_kwargs
    a = json.loads((RUN/'args.json').read_text())
    env = variant_class(kind)(**{**env_kwargs(a), 'episode_s': seconds+5., 'level': 0.}, randomize=False, seed=7)
    env.init_seed = init_seed
    model, venv = load(str(RUN/'model.zip'), env)
    venv.reset()                                  # builds the MuJoCo model
    m = env.model
    foot = [g for s in env.foot_geoms for g in env.foot_geoms[s] if m.geom_priority[g] == 1]
    K = set(kind.split('+'))
    if K & {'mu_lo', 'mu_hi'}:
        env.nominal_friction[foot, 0] *= .625 if 'mu_lo' in K else 1.5625
    if 'selfcol' in K:
        for g in range(1, m.ngeom):
            nm = m.body(m.geom_bodyid[g]).name
            if nm == 'world':
                continue
            side = 2 if nm.startswith('left') else 4 if nm.startswith('right') else 0
            if nm == 'base_link':
                m.geom_contype[g], m.geom_conaffinity[g] = 1 | 16, 0
            elif nm.startswith('tail'):
                m.geom_contype[g], m.geom_conaffinity[g] = 1 | 8, 0
            elif 'thigh' in nm:
                m.geom_contype[g], m.geom_conaffinity[g] = 1 | side, 6-side
            else:
                m.geom_contype[g], m.geom_conaffinity[g] = 1 | side, (6-side) | 8 | 16
        sync_body_masks(m)
    if 'knee30' in K:
        for i, n in enumerate(env.active):
            if 'knee' in n:
                env.lo[i], env.hi[i] = KNEE_BAND
        m.actuator_ctrlrange[env.act] = np.c_[env.lo, env.hi]
    if 'armature' in K:
        for i, n in enumerate(env.active):
            if joint_type(n) in ARMATURE:
                m.dof_armature[env.v_adr[i]] = ARMATURE[joint_type(n)]
    venv.reset()                                  # same seeded start for every condition (second reset in all of them)
    gap = None
    d = env.data
    low = lambda: min(d.geom_xpos[g][2]-(m.geom_size[g][0] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE else abs(d.geom_xmat[g].reshape(3, 3)[2])@m.geom_size[g]) for g in foot)
    gap = float(low())
    if 'nogap' in K:
        d.qpos[2] -= gap-1e-4; mujoco.mj_forward(m, d); d.ctrl[env.act] = d.qpos[env.q_adr]
    return env, model, venv, gap


def overlap_probe(env):
    """Second model with every robot geom collidable; kinematics + collision only, never stepped."""
    import mujoco
    m2 = copy.copy(env.model); m2.geom_conaffinity[1:] = 1; sync_body_masks(m2); d2 = mujoco.MjData(m2)
    foot_group = lambda nm: nm.split('_')[0] if ('foot' in nm or 'toe' in nm) else None

    def probe():
        d2.qpos[:] = env.data.qpos; mujoco.mj_fwdPosition(m2, d2); out = {}
        for i in range(d2.ncon):
            c = d2.contact[i]; b1, b2 = (m2.body(m2.geom_bodyid[g]).name for g in (c.geom1, c.geom2))
            same_side = b1.split('_')[0] == b2.split('_')[0] and b1.split('_')[0] in ('left', 'right')
            built_in = (foot_group(b1) and foot_group(b1) == foot_group(b2)) or (same_side and {'metatarsus'} & {x for b in (b1, b2) for x in ('metatarsus',) if x in b} and ('toe' in b1 or 'toe' in b2)) \
                or ('base_link' in (b1, b2) and ('thigh' in b1 or 'thigh' in b2))
            if 'world' in (b1, b2) or built_in:
                continue        # ground contacts; overlaps that exist by construction of the box shapes (toes of one foot, metatarsus end / toe roots, thigh top / torso side)
            short = lambda n: n.replace('_link', '').replace('_proximal', '_p').replace('_distal', '_d')
            key = '|'.join(sorted((short(b1), short(b2)))); out[key] = max(out.get(key, 0.), float(-c.dist))
        return out
    return probe


def rollout(task):
    kind, cmd, init_seed, seconds, mode = task
    import mujoco
    t0 = time.time(); env, model, venv, gap = make(kind, seconds, init_seed); m, d = env.model, env.data
    env.command = np.array([cmd, 0., 0.]); env.resample_steps = 0
    obs = venv.normalize_obs(env._obs()[None])
    ki = [env.ix[f'{s}_knee_pitch_joint'] for s in ('left', 'right')]; kq = env.q_adr[ki]
    tend = [m.tendon(f'spring_{s}_knee_pitch').id for s in ('left', 'right')]
    probe = overlap_probe(env) if (kind == 'base' and mode == 'policy') else None
    xs, heats, knee, kforce, ovl = [], [], [], [], {}
    hroll = []; hq = env.q_adr[[env.ix[f'{s}_hip_roll_joint'] for s in ('left', 'right')]]
    eng, rel, e_in, e_out, w_pos, ovl_steps, first_contact, fell, t, vz_max = [], [], 0., 0., 0., 0, None, False, 0., 0.
    zero = np.zeros(len(env.policy_idx))
    if kind in ('latch_nofb', 'latch_clutch'):
        m.tendon_stiffness[tend] = 0.            # the variant decides on its first step; step() rewrites the stiffness anyway
    preload = float(sum(.5*m.tendon_stiffness[tj]*(d.qpos[kq[j]]-m.tendon_lengthspring[tj, 0])**2 for j, tj in enumerate(tend)))
    for step in range(int(seconds/.02)):
        k_pre = m.tendon_stiffness[tend].copy(); r_pre = m.tendon_lengthspring[tend, 0].copy(); q_pre = d.qpos[kq].copy()
        if mode == 'policy':
            act, _ = model.predict(obs, deterministic=True); action = act[0]
        else:
            action = zero
        raw, _, term, trunc, info = env.step(action)
        obs = venv.normalize_obs(raw[None]); t += .02
        k_post = m.tendon_stiffness[tend]; r_post = m.tendon_lengthspring[tend, 0]
        for j in range(2):
            if k_pre[j] == 0 and k_post[j] > 0:
                u = .5*k_post[j]*(q_pre[j]-r_post[j])**2; e_in += u; eng.append((t, float(q_pre[j]-r_post[j]), float(u)))
            elif k_pre[j] > 0 and k_post[j] == 0:
                u = .5*k_pre[j]*(q_pre[j]-r_pre[j])**2; e_out += u; rel.append(float(u))
        w_pos += info['power']*.02
        if first_contact is None and any(info['loaded'].values()):
            first_contact = t
        xs.append(float(env.body_velocity(env.base)[1][0])); heats.append(env.heat_inst.copy()); knee.append(d.qpos[kq].copy()); kforce.append(np.abs(d.actuator_force[env.act[ki]]).copy())
        vz_max = max(vz_max, float(d.qvel[2])); hroll.append(d.qpos[hq].copy())
        if probe is not None:
            o = probe(); ovl_steps += bool(o)
            for k_, v_ in o.items():
                ovl[k_] = [ovl.get(k_, [0, 0.])[0]+1, max(ovl.get(k_, [0, 0.])[1], v_)]
        if term:
            fell = True; break
    xs, heats, knee, kforce = np.array(xs), np.array(heats), np.array(knee), np.array(kforce); n = len(xs); hr = np.array(hroll)
    tail = slice(max(0, n-int(40/.02)), n) if seconds >= 50 else slice(n//2, n)
    drive = [i for i in range(len(env.active)) if env.motor_cont[i] < 1e8]
    hm = heats[tail][:, drive].mean(0)
    # the reset engages both springs on the first control step (loaded=True); count that engagement explicitly
    k0 = float(m.tendon_stiffness[tend[0]]) or float(list(env.spring_engage.values())[0][1])
    row = dict(kind=kind, mode=mode, cmd=cmd, init_seed=init_seed, seconds=seconds, fell=fell, t=round(t, 2), start_gap_m=gap, first_contact_s=first_contact,
               speed_tail=float(xs[tail].mean()) if n else 0., heat_max=float(hm.max()), heat_argmax=env.active[drive[int(hm.argmax())]], vz_max=vz_max,
               hip_roll_abs_max=float(np.abs(hr).max()), hip_roll_in_max=float(max(-hr[:, 0].min(), hr[:, 1].max())),
               knee_min=float(knee.min()), knee_max=float(knee.max()), knee_p01=float(np.quantile(knee, .01)), knee_p99=float(np.quantile(knee, .99)),
               knee_frac_below_band=float((knee < KNEE_BAND[0]).mean()), knee_frac_above_band=float((knee > KNEE_BAND[1]).mean()),
               knee_frac_above_2p30=float((knee > 2.30).mean()), knee_frac_above_closure=float((knee > 2.3481).mean()),
               knee_force_max=float(kforce.max()), knee_force_max_above_band=float(kforce[knee > KNEE_BAND[1]].max()) if (knee > KNEE_BAND[1]).any() else 0.,
               engagements=len(eng), engage_abs_defl_mean=float(np.mean([abs(e[1]) for e in eng])) if eng else 0., engage_abs_defl_max=float(max([abs(e[1]) for e in eng], default=0.)),
               engage_nonzero=int(sum(abs(e[1]) > .01 for e in eng)), energy_injected_J=float(e_in), energy_released_J=float(e_out), releases=len(rel), motor_positive_work_J=float(w_pos), initial_preload_J=preload,
               early_engagements=[e for e in eng if e[0] <= .5][:6], overlap_step_frac=(ovl_steps/n if probe is not None else None), overlaps=ovl if probe is not None else None, wall_s=round(time.time()-t0, 1))
    env.close(); return row


if __name__ == '__main__':
    OUT = Path(sys.argv[1]); nseed = int(sys.argv[2]) if len(sys.argv) > 2 else 10; seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 60.
    seeds = [95001+i for i in range(nseed)]
    only = sys.argv[4].split(',') if len(sys.argv) > 4 else CONDITIONS
    tasks = [(k, c, s, seconds, 'policy') for k in only for c in (4., 7.) for s in seeds]
    if len(sys.argv) <= 4:
        tasks += [(k, 0., s, 3., 'zero_action') for k in ('base', 'latch_nofb', 'latch_clutch', 'nogap') for s in seeds[:4]]
    t0 = time.time()
    with Pool(8) as pool, open(OUT/(sys.argv[5] if len(sys.argv) > 5 else 'gate_rollouts.jsonl'), 'w') as f:
        for i, r in enumerate(pool.imap_unordered(rollout, tasks, chunksize=1)):
            f.write(json.dumps(r)+'\n'); f.flush()
            if i % 20 == 0:
                print(i+1, '/', len(tasks), round(time.time()-t0), 's', flush=True)
    print('done', len(tasks), round(time.time()-t0), 's')
