#!/usr/bin/env python3
"""Lateral rocking (and optional stepping) from crouch in MuJoCo.

Both hip rolls move in parallel (left = -r, right = +r because the right axis is -x),
r = A*sin(phase) after a ramp; the phase is time-based or from `rhythm` (roll feedback).
Inside each foot's unloaded phase window: optional knee `lift` (hip/ankle -lift/2) and hip-pitch
`stride` (ankle cancels it); `pitch_feedback` adds a clipped ankle correction on base pitch.
Joint position targets only. A gravity-tilt guard (roll/pitch, not yaw; same as Gazebo TiltGuard) latches STOP.
Before 2026-09-28 evidence 70 the guard used the total rotation 2*acos|w|, which also counted body yaw. 'unloaded' = < 5% body weight.
Metrics are simulation diagnostics, not a walking certificate.
"""
import argparse
import json
import math
import sys
import numpy as np
import mujoco
from raptor_servo import GazeboLikeServo, JTCLikeServo
from stand_check import HERE, crouch, place_on_floor
from step_metrics import steps, summarize
sys.path.insert(0, str(HERE.parent/'src/raptor_control/scripts'))
from rock_law import GaitPhase, TailSync, tail_targets  # noqa: E402  shared with rock_gz_probe.py


def foot_forces(model, data, force):
    total = {'left': 0., 'right': 0.}
    for i in range(data.ncon):
        contact = data.contact[i]
        body = model.body(model.geom_bodyid[contact.geom1 or contact.geom2]).name
        side = body.split('_')[0]
        if side in total:
            mujoco.mj_contactForce(model, data, i, force)
            total[side] += force[0]
    return total


def lowest_z(model, data, geom):
    half, rot = model.geom_size[geom], data.geom_xmat[geom].reshape(3, 3)
    return float(data.geom_xpos[geom][2]-np.abs(rot[2]) @ half)


def contact_slip(model, data, side, force, velocity):
    """Normal-force-weighted horizontal speed of the foot material at its contact points."""
    total = weighted = 0.
    for i in range(data.ncon):
        contact = data.contact[i]
        geom = contact.geom1 or contact.geom2
        body = model.geom_bodyid[geom]
        if not model.body(body).name.startswith(side+'_'):
            continue
        mujoco.mj_contactForce(model, data, i, force)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
        point = velocity[3:]+np.cross(velocity[:3], contact.pos-data.xpos[body])
        total += force[0]
        weighted += force[0]*float(np.hypot(point[0], point[1]))
    return weighted/total if total > 0 else None


def yaw(quat):
    w, x, y, z = quat
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def roll_pitch(quat):
    w, x, y, z = quat
    return (math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)), math.asin(max(-1., min(1., 2*(w*y-z*x)))))


def run(amplitude, frequency, cycles=6, ramp=1., guard=.25, model_path=None, crouch_hip=-.15, log_every=0, crouch_knee=.4,
        lift=0., lift_width=.35, lift_centers=(('left', .28), ('right', .78)), stride=0., rhythm=None, friction=None, mass_scale=1., smooth_swing=False, pitch_feedback=None, servo_kv=None,
        jtc=None, tail=None, tail_sync=None, tail_mass_scale=1., step_hook=None):
    model = mujoco.MjModel.from_xml_path(model_path or str(HERE/'raptor.xml'))
    if friction is not None:
        model.geom_friction[:, 0] = friction
    model.body_mass[:] *= mass_scale
    if tail_mass_scale != 1.:  # 'no tail' comparison: shrink tail links (whole-body COM moves forward)
        for name in ('tail_yaw_link', 'tail_link'):
            model.body_mass[model.body(name).id] *= tail_mass_scale
            model.body_inertia[model.body(name).id] *= tail_mass_scale
    if servo_kv is not None:  # velocity-servo stiffness is an unvalidated approximation of the DART servo
        model.actuator_gainprm[:, 0], model.actuator_biasprm[:, 2] = servo_kv, -servo_kv
    data = mujoco.MjData(model)
    pose = crouch(crouch_hip, crouch_knee)
    for name, value in pose.items():
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    place_on_floor(model, data)
    # jtc=(sample, horizon): emulate the streamed JTC path of rock_gz_probe.py (evidence 68)
    servo = GazeboLikeServo(model) if jtc is None else JTCLikeServo(model, *jtc)
    servo.step(data)
    weight = float(model.body_mass.sum()*-model.opt.gravity[2])
    force = np.zeros(6)
    settle, dt = 1., model.opt.timestep
    duration = settle+ramp+cycles/frequency
    unloaded = {'left': 0, 'right': 0}
    longest, current = {'left': 0., 'right': 0.}, {'left': 0., 'right': 0.}
    both_off = 0
    both_run = longest_both = 0.
    worst = {'tilt': 0., 'roll': 0., 'pitch': 0., 'yaw': 0.}
    stopped_at, samples = None, []
    lifting = {'left': False, 'right': False}
    slip, slip_sum, loaded_time = {'left': 0., 'right': 0.}, {'left': 0., 'right': 0.}, {'left': 0., 'right': 0.}
    was_loaded, was_off = {'left': False, 'right': False}, {'left': False, 'right': False}
    touchdowns = {'left': [], 'right': []}
    velocity = np.zeros(6)
    trace = {side: {'force': [], 'slip': [], 'x': []} for side in ('left', 'right')}
    clearance = {'left': 0., 'right': 0.}
    gait, left_contact, last_f = GaitPhase(), False, {'left': 0., 'right': 0.}
    tail_law = TailSync(*tail_sync) if tail_sync else None
    legs = [model.actuator(n).id for n in (f'{s}_{j}_joint' for s in ('left', 'right')
                                           for j in ('hip_roll', 'hip_pitch', 'knee_pitch', 'ankle_pitch'))]
    tails = [model.actuator(n).id for n in ('tail_yaw_joint', 'tail_pitch_joint')]
    dofs = model.actuator_trnid[:, 0]
    work = {'legs': 0., 'tail': 0.}
    yaw_rates, yaws = [], []
    feet = {side: [g for g in range(model.ngeom) if model.body(model.geom_bodyid[g]).name.startswith(side+'_')
                   and ('foot' in model.body(model.geom_bodyid[g]).name or 'toe' in model.body(model.geom_bodyid[g]).name)]
            for side in ('left', 'right')}
    for k in range(int(duration/dt)):
        t = k*dt
        if stopped_at is None and t >= settle:
            phase = t-settle
            if rhythm is None:
                r = amplitude*min(1., phase/ramp)*math.sin(2*math.pi*frequency*phase)
                cycle = (phase*frequency) % 1
            else:  # closed loop: oscillator phase replaces time phase
                r = min(1., phase/ramp)*rhythm.step(roll_pitch(data.qpos[3:7])[0])
                cycle = (rhythm.theta/(2*math.pi)) % 1
            target = {'left_hip_roll_joint': -r, 'right_hip_roll_joint': r}
            for side, center in lift_centers:
                offset = (cycle-center+.5) % 1-.5
                bump = math.cos(math.pi*offset/lift_width)**2 if abs(offset) < lift_width/2 and phase >= ramp else 0.
                h = lift*bump
                # Stride: hip swings forward (negative pitch) across the unloaded window and returns
                # linearly during stance; ankle cancels it so the sole stays parallel in pitch.
                u = (offset+lift_width/2)/lift_width
                if 0 <= u <= 1:  # swing: cosine profile, zero hip velocity at liftoff and touchdown
                    swing = stride*.5*math.cos(math.pi*u) if smooth_swing else -stride*(u-.5)
                else:
                    swing = -stride*(.5-((offset-lift_width/2) % 1)/(1-lift_width))
                swing *= min(1., max(0., phase-ramp))
                target |= {f'{side}_hip_pitch_joint': pose[f'{side}_hip_pitch_joint']-h/2+swing,
                           f'{side}_knee_pitch_joint': pose[f'{side}_knee_pitch_joint']+h,
                           f'{side}_ankle_pitch_joint': pose[f'{side}_ankle_pitch_joint']-h/2-swing}
                lifting[side] = bump > .5
            # gait phase from left-foot contact (force hysteresis 20 % on / 5 % off), as a ROS node would from sensors
            left_contact = last_f['left'] >= .2*weight or (left_contact and last_f['left'] >= .05*weight)
            gait_phase = gait.update(t, left_contact)
            if tail_sync:  # gait-synchronised tail yaw (amp, phi0, k_fb); qvel[5] = body yaw rate
                target |= tail_law.update(t, gait_phase, data.qvel[5])
            if tail:  # tail balance (ky, kr, kp, kd); qvel[3:6] is the body-frame angular velocity
                body_roll, body_pitch = roll_pitch(data.qpos[3:7])
                target |= tail_targets(r, body_roll, data.qvel[3], body_pitch, data.qvel[4], tail)
            if pitch_feedback:  # ankle strategy on base pitch; position targets only, clipped
                kp, kd, limit = pitch_feedback
                correction = max(-limit, min(limit, kp*roll_pitch(data.qpos[3:7])[1]+kd*data.qvel[4]))
                for side in ('left', 'right'):
                    name = f'{side}_ankle_pitch_joint'
                    target[name] = target.get(name, pose[name])+correction
            servo.set_target(target)
        servo.step(data)
        if step_hook is not None:  # diagnostics hook (sim/roll_diagnosis.py); must not change the simulation
            step_hook(t+dt, model, data)
        roll, pitch = roll_pitch(data.qpos[3:7])
        w, x, y, z = data.qpos[3:7]
        tilt = math.acos(max(-1., min(1., 1-2*(x*x+y*y))))  # angle between body z and world z
        worst = {'tilt': max(worst['tilt'], tilt), 'roll': max(worst['roll'], abs(roll)),
                 'yaw': max(worst['yaw'], abs(yaw(data.qpos[3:7]))),
                 'pitch': max(worst['pitch'], abs(pitch))}
        if tilt > guard and stopped_at is None:
            stopped_at = t
            servo.stop()
        if t < settle+ramp or stopped_at is not None:
            continue
        f = last_f = foot_forces(model, data, force)
        off = {s: f[s] < .05*weight for s in f}
        power = np.abs(data.actuator_force*data.qvel[model.jnt_dofadr[dofs]])
        work['legs'] += float(power[legs].sum())*dt
        work['tail'] += float(power[tails].sum())*dt
        yaw_rates.append(float(data.qvel[5]))
        yaws.append(yaw(data.qpos[3:7]))
        for side in lifting:
            if lifting[side]:
                lowest = min(lowest_z(model, data, g) for g in feet[side])
                clearance[side] = max(clearance[side], lowest)
        both_off += off['left'] and off['right']
        both_run = both_run+dt if off['left'] and off['right'] else 0.
        longest_both = max(longest_both, both_run)
        for s in off:
            if off[s] and not off['left' if s == 'right' else 'right']:
                unloaded[s] += 1
                current[s] += dt
                longest[s] = max(longest[s], current[s])
            else:
                current[s] = 0.
        for side in ('left', 'right'):
            body = model.body(side+'_foot_link').id
            speed_now = contact_slip(model, data, side, force, velocity)
            trace[side]['force'].append(f[side])
            trace[side]['slip'].append(np.nan if speed_now is None else speed_now)
            trace[side]['x'].append(float(data.xpos[body][0]))
            loaded = f[side] >= .2*weight
            if loaded and was_loaded[side] and speed_now is not None:
                slip[side] = max(slip[side], speed_now)
                slip_sum[side] += speed_now*dt
                loaded_time[side] += dt
            if not off[side] and was_off[side]:
                touchdowns[side].append(float(data.xpos[body][0]))
            was_loaded[side], was_off[side] = loaded, off[side]
        if log_every and k % log_every == 0:
            samples.append({'t': round(t, 3), 'hip_roll_cmd': -servo.command[0], 'body_roll': roll,
                            'fz_left': f['left'], 'fz_right': f['right'], 'y': float(data.qpos[1])})
    active = (cycles/frequency)/dt
    yaw_trace = np.unwrap(np.array(yaws)) if yaws else np.zeros(1)
    trend = np.polyval(np.polyfit(np.arange(len(yaw_trace)), yaw_trace, 1), np.arange(len(yaw_trace))) \
        if len(yaw_trace) > 1 else yaw_trace
    return {'amplitude_rad': amplitude, 'frequency_hz': frequency,
            'stopped_by_guard_at_s': stopped_at, 'max_tilt_rad': worst['tilt'], 'max_abs_yaw_rad': worst['yaw'],
            'yaw_rate_rms': float(np.sqrt(np.mean(np.square(yaw_rates)))) if yaw_rates else None,
            'yaw_oscillation_std_rad': float(np.std(yaw_trace-trend)),
            'actuator_work_j': work, 'gait_period_s': gait.period, 'gait_duty': gait.duty,
            'max_body_roll_rad': worst['roll'], 'max_body_pitch_rad': worst['pitch'],
            'single_support_fraction': {s: unloaded[s]/active for s in unloaded},
            'longest_single_support_s': longest, 'both_feet_unloaded_fraction': both_off/active, 'longest_both_unloaded_s': longest_both,
            'friction': friction, 'mass_scale': mass_scale,
            'final_amplitude_rad': rhythm.amplitude if rhythm else amplitude,
            'lift_rad': lift, 'stride_rad': stride, 'final_yaw_rad': yaw(data.qpos[3:7]), 'max_swing_clearance_m': clearance,
            'contact_slip_max_m_s': slip,
            'contact_slip_mean_m_s': {k: slip_sum[k]/loaded_time[k] if loaded_time[k] else None for k in slip},
            'touchdown_step_m': {k: [b-a for a, b in zip(v, v[1:])] for k, v in touchdowns.items()},
            'clean_steps': summarize(*(steps(trace[k]['force'], trace[k]['slip'], trace[k]['x'], dt, weight)
                                       for k in ('left', 'right')), cycles),
            'final_xy': data.qpos[:2].tolist(), 'samples': samples}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--amplitudes', type=float, nargs='+', default=[.05, .1, .15, .2])
    parser.add_argument('--frequencies', type=float, nargs='+', default=[.5, 1., 1.5, 2.])
    parser.add_argument('--cycles', type=int, default=6)
    args = parser.parse_args()
    results = [run(a, f, args.cycles) for a in args.amplitudes for f in args.frequencies]
    print(json.dumps({'limitation': 'Open-loop MuJoCo rocking, no foot lift; not stepping or walking.',
                      'unloaded_threshold': '5% body weight', 'results': results}, indent=1))
