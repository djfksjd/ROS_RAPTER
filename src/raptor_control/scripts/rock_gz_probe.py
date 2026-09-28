#!/usr/bin/env python3
"""Bounded Gazebo lateral rocking / stride probe (simulation only, not a mission, not a walking certificate).

Streams position targets on sim time from rock_law.targets (the sim/rock_probe.py law with lift = 0):
hip rolls left = -r, right = +r, optional hip-pitch stride in each foot's unloaded window and optional
ankle pitch feedback on IMU pitch / pitch rate.
Each target is streamed every 20 ms with a 20 ms JTC horizon (50 ms halved hip-roll tracking, evidence 68).
Saturation watchdog (SIMULATION_METHODOLOGY.ko.md): STOP when a joint velocity reaches 90% of its
limit for 30 ms of sim time (single-sample touchdown spikes are counted, not stopped), a joint leaves its target by more than 0.15 rad, the IMU tilt exceeds 0.25 rad, joint/IMU data go
stale, or sim time stalls or runs backwards. STOP holds the measured positions (last finite target where a
measurement is unusable) and ends streaming; any exception also holds. A normal end returns to the crouch. Writes /raptor_ws/log/rock-gz-probe.json.
"""
import argparse
import json
import math
import time
from collections import deque
from pathlib import Path

import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from ros_gz_interfaces.msg import Contacts
from sensor_msgs.msg import Imu, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from motion_guard import TiltGuard
from motion_probe import JOINTS, pose
from rock_law import GaitPhase, TailSync, crouch_pose, step_roll, tail_targets, targets

# Velocity limits from raptor.urdf.xacro (rad/s); unchanged by leg_design.
VELOCITY_LIMIT = {'hip_roll': 2.0, 'hip_pitch': 2.5, 'knee_pitch': 3.0, 'ankle_pitch': 2.5,
                  'tail_yaw': 2.0, 'tail_pitch': 2.0}
TRACKING_LIMIT = .15
# Touchdown impacts give one-sample speed spikes with on-target positions (evidence 70); the DART runaway
# (evidence 61) stays at the limit for hundreds of ms. STOP on speed only when it persists this long.
SATURATION_HOLD = .03
STALE = .1  # s of sim time without joint/IMU updates
CLOCK_STALL = 2.  # s of wall time without a /clock advance after start
PERIOD = .02  # s of sim time between streamed targets


def limit_of(joint):
    return next(v for k, v in VELOCITY_LIMIT.items() if k in joint)


class RockProbe(Node):
    def __init__(self, args):
        super().__init__('raptor_rock_gz_probe')
        self.args = args
        self.crouch = crouch_pose(JOINTS, args.crouch_hip, args.crouch_knee)
        self.guard = TiltGuard(limit=.25, clock=lambda: self.sim_time or 0.)
        self.sim_time, self.joints, self.contacts, self.imu = None, {}, {}, None
        self.joints_at = self.imu_at = None
        self.clock_wall, self.clock_fault = time.monotonic(), None
        self.target, self.start, self.last_sent = dict(self.crouch), None, -math.inf
        self.stop_reason, self.rows = None, []
        self.gait, self.gait_phase = GaitPhase(), None
        self.tail_law = TailSync(*args.tail_sync) if args.tail_sync else None
        self.over_since, self.spikes, self.spike_keys = {}, [], set()
        self.raw = deque(maxlen=200)
        self.wrenches, self.diag = {}, []  # --diag: 100 Hz diagnostic log (no effect on control)  # last ~2 s of 100 Hz joint states, dumped around a STOP
        self.pub = self.create_publisher(JointTrajectory, '/raptor_joint_controller/joint_trajectory', 10)
        self.create_subscription(Clock, '/clock', self.on_clock, 10)
        self.create_subscription(JointState, '/joint_states', self.on_joints, 10)
        self.create_subscription(Imu, '/raptor/imu', self.on_imu, 10)
        for side in ('left', 'right'):
            self.create_subscription(Contacts, f'/raptor/{side}_foot/contact',
                                     lambda m, k=f'{side}_pad': self.on_contact(m, k), 10)
            for d in range(1, 4):
                for part in ('proximal', 'distal'):
                    self.create_subscription(Contacts, f'/raptor/{side}/toe_{d}_{part}/contact',
                                             lambda m, k=f'{side}_toe_{d}_{part}': self.on_contact(m, k), 10)

    def on_clock(self, msg):
        t = msg.clock.sec+msg.clock.nanosec*1e-9
        if self.sim_time is not None and t < self.sim_time:
            self.clock_fault = f'sim time went backwards {self.sim_time:.3f} -> {t:.3f}'
        if self.sim_time is None or t > self.sim_time:
            self.clock_wall = time.monotonic()
        self.sim_time = t

    def on_joints(self, msg):
        for i, name in enumerate(msg.name):
            if name in JOINTS and i < len(msg.position):
                self.joints[name] = (msg.position[i], msg.velocity[i] if i < len(msg.velocity) else float('nan'))
        self.joints_at = self.sim_time
        if self.args.diag and self.imu is not None and self.start is not None:
            fresh = {k: v for k, v in self.wrenches.items() if v[0] is not None and self.sim_time-v[0] < .03}
            self.diag.append({'t': round(self.sim_time-self.start, 4),
                              'q': {j: round(v[0], 5) for j, v in self.joints.items()},
                              'v': {j: round(v[1], 4) for j, v in self.joints.items()},
                              'roll': round(self.imu[0], 5), 'pitch': round(self.imu[1], 5),
                              'roll_rate': round(self.imu[4], 4), 'yaw_rate': round(self.imu[3], 4),
                              'cmd_hip_roll_right': round(self.target['right_hip_roll_joint'], 5),
                              'contact': {s: {'n': sum(n for k, (t, n) in self.contacts.items()
                                                       if k.startswith(s) and t is not None and self.sim_time-t < .03),
                                              'f1': [round(sum(v[1][i] for k, v in fresh.items() if k.startswith(s)), 2)
                                                     for i in range(3)],
                                              'f2': [round(sum(v[2][i] for k, v in fresh.items() if k.startswith(s)), 2)
                                                     for i in range(3)],
                                              'names': sorted({v[3] for k, v in fresh.items() if k.startswith(s)})}
                                          for s in ('left', 'right')}})
        self.raw.append({'t': self.sim_time, 'q': {j: round(v[0], 4) for j, v in self.joints.items()},
                         'v': {j: round(v[1], 4) for j, v in self.joints.items()},
                         'contact': {s: self.in_contact(s) for s in ('left', 'right')} if self.sim_time else None,
                         'target': {j: round(v, 4) for j, v in self.target.items()}})

    def on_imu(self, msg):
        q = msg.orientation
        self.imu_at = self.sim_time
        self.guard.observe(q.x, q.y, q.z, q.w, msg.orientation_covariance[0] != -1)
        self.imu = (math.atan2(2*(q.w*q.x+q.y*q.z), 1-2*(q.x*q.x+q.y*q.y)),
                    math.asin(max(-1., min(1., 2*(q.w*q.y-q.z*q.x)))), msg.angular_velocity.y, msg.angular_velocity.z,
                    msg.angular_velocity.x)

    def on_contact(self, msg, key):
        self.contacts[key] = (self.sim_time, len(msg.contacts))
        if self.args.diag:  # logging only: raw wrench sums, frame/sign convention not assumed
            f1 = [sum(getattr(w.body_1_wrench.force, a) for c in msg.contacts for w in c.wrenches) for a in 'xyz']
            f2 = [sum(getattr(w.body_2_wrench.force, a) for c in msg.contacts for w in c.wrenches) for a in 'xyz']
            names = msg.contacts[0].collision1.name if msg.contacts else ''
            self.wrenches[key] = (self.sim_time, f1, f2, names)

    def send(self, positions, horizon=.05):
        msg = JointTrajectory()
        msg.joint_names = JOINTS
        point = JointTrajectoryPoint()
        point.positions = [float(positions[j]) for j in JOINTS]
        point.time_from_start.nanosec = int(horizon*1e9)
        msg.points = [point]
        self.pub.publish(msg)

    def in_contact(self, side):
        return any(n > 0 and t is not None and self.sim_time-t < .1
                   for k, (t, n) in self.contacts.items() if k.startswith(side))

    def hold_positions(self):
        """Measured positions; the last commanded target where a measurement is missing or nonfinite."""
        return {j: self.joints[j][0] if j in self.joints and math.isfinite(self.joints[j][0]) else self.target[j]
                for j in JOINTS}

    def hold(self):
        positions = self.hold_positions()
        self.target = positions
        for _ in range(3):  # no controller acknowledgement on the topic interface; repeat the hold
            self.send(positions, .1)
            time.sleep(.05)

    def watchdog(self):
        if self.guard.failure:
            return self.guard.failure
        if self.clock_fault:
            return self.clock_fault
        if self.sim_time-(self.joints_at or -math.inf) > STALE or self.sim_time-(self.imu_at or -math.inf) > STALE:
            return 'stale joint states or IMU'
        for j, (q, v) in self.joints.items():
            if not math.isfinite(q) or not math.isfinite(v):
                return f'nonfinite state {j}'
            if abs(v) >= .9*limit_of(j):
                first = self.over_since.setdefault(j, self.joints_at)
                if (j, first) not in self.spike_keys:
                    self.spike_keys.add((j, first))
                    self.spikes.append([j, first, round(v, 3)])
                if self.joints_at-first >= SATURATION_HOLD:
                    return f'velocity saturation {j} {v:.3f} rad/s for {self.joints_at-first:.3f} s (limit {limit_of(j)})'
            else:
                self.over_since.pop(j, None)
            if abs(q-self.target[j]) > TRACKING_LIMIT:
                return f'tracking error {j} {q-self.target[j]:+.3f} rad'
        return None

    def step(self):
        t = self.sim_time
        if t is None or len(self.joints) < len(JOINTS) or self.joints_at is None or self.imu_at is None:
            return False
        if self.start is None:
            if t-self.joints_at > STALE/2 or t-self.imu_at > STALE/2:
                return False  # wait for fresh inputs (queues may lag after the blocking start pose query)
            error = max(abs(self.joints[j][0]-self.crouch[j]) for j in JOINTS)
            if error > .05:
                raise RuntimeError(f'not at the requested crouch (max error {error:.3f} rad); use crouched_start')
            self.start = t
        phase = t-self.start
        if self.stop_reason is None:
            reason = self.watchdog()
            if reason:
                self.stop_reason = f'{reason} at sim {phase:.3f} s'
                self.hold()
                return True
        if phase > self.args.settle+self.args.ramp+self.args.cycles/self.args.frequency:
            self.target = dict(self.crouch)
            self.send(self.target, .5)  # r = 0: back to the crouch
            return True
        if t-self.last_sent < PERIOD:
            return False
        self.last_sent = t
        a, g = phase-self.args.settle, self.args
        new = targets(self.crouch, a, g.amplitude, g.frequency, g.ramp, stride=g.stride, window=g.window,
                      pitch=self.imu[1], pitch_rate=self.imu[2], feedback=g.pitch_feedback, mirror=g.mirror,
                      abduction=g.abduction)
        self.gait_phase = self.gait.update(t, self.in_contact('left'))  # phase 0 at left touchdown (contact messages)
        if g.step_roll:  # experiment F: rate-limited hip-roll steps instead of the gait (no stride)
            r = step_roll(a, *g.step_roll)
            new = dict(self.crouch) | {'left_hip_roll_joint': -r, 'right_hip_roll_joint': r}
        if g.tail_sync and a >= 0:  # gait-synchronised tail yaw (amp, phi0, k_fb) with IMU yaw rate
            new |= self.tail_law.update(t, self.gait_phase, self.imu[3])
        if g.tail and a >= 0:  # tail balance: (ky, kr, kp, kd) on hip-roll command, IMU roll, pitch, pitch rate
            new |= tail_targets(new['right_hip_roll_joint'], self.imu[0], 0., self.imu[1], self.imu[2], g.tail)
        # Command-side saturation: never stream a target whose rate needs >= 90% of a joint's velocity limit.
        for j in JOINTS:
            rate = abs(new[j]-self.target[j])/max(g.horizon, PERIOD)
            if rate >= .9*limit_of(j):
                self.stop_reason = f'commanded speed {j} {rate:.3f} rad/s (limit {limit_of(j)}) at sim {phase:.3f} s'
                self.hold()
                return True
        self.target = new
        self.send(self.target, self.args.horizon)
        self.rows.append({'t': round(phase, 3), 'r_cmd': round(self.target['right_hip_roll_joint'], 5),
                          'roll': round(self.imu[0], 5),
                          'pitch': round(self.imu[1], 5), 'tilt': round(self.guard.tilt or 0., 5),
                          'contact': {s: self.in_contact(s) for s in ('left', 'right')},
                          'q': {j: round(self.joints[j][0], 4) for j in JOINTS},
                          'v': {j: round(self.joints[j][1], 4) for j in JOINTS},
                          'gait_phase': None if self.gait_phase is None else round(self.gait_phase, 3),
                          'yaw_rate': round(self.imu[3], 4), 'tail_yaw_cmd': round(self.target['tail_yaw_joint'], 4),
                          'hip_pitch_cmd': {s: round(self.target[f'{s}_hip_pitch_joint'], 4) for s in ('left', 'right')}})
        return False


def summary(rows, args):
    active = [r for r in rows if r['t'] >= args.settle+args.ramp]
    n = max(1, len(active))
    single = {s: sum(r['contact'][s] and not r['contact'][o] for r in active)/n for s, o in (('left', 'right'), ('right', 'left'))}
    return {'samples': len(rows), 'active_samples': len(active),
            'single_support_fraction': single, 'no_contact_fraction': sum(not any(r['contact'].values()) for r in active)/n,
            'max_body_roll_rad': max((abs(r['roll']) for r in active), default=None),
            'max_body_pitch_rad': max((abs(r['pitch']) for r in active), default=None),
            'max_tilt_rad': max((r['tilt'] for r in rows), default=None),
            'max_abs_velocity': {j: max((abs(r['v'][j]) for r in rows), default=None) for j in JOINTS},
            'yaw_rate_rms': math.sqrt(sum(r['yaw_rate']**2 for r in active)/n) if active else None,
            'phase_known_fraction': sum(r['gait_phase'] is not None for r in active)/n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--amplitude', type=float, default=.08)
    parser.add_argument('--frequency', type=float, default=2.)
    parser.add_argument('--cycles', type=int, default=20)
    parser.add_argument('--ramp', type=float, default=1.)
    parser.add_argument('--settle', type=float, default=1.)
    parser.add_argument('--crouch-hip', type=float, default=-.10)
    parser.add_argument('--crouch-knee', type=float, default=.5)
    parser.add_argument('--stride', type=float, default=0.)
    parser.add_argument('--pitch-feedback', type=float, nargs=3, metavar=('KP', 'KD', 'LIMIT'))
    parser.add_argument('--tail', type=float, nargs=4, metavar=('KY', 'KR', 'KP', 'KD'))
    parser.add_argument('--tail-sync', type=float, nargs=3, metavar=('AMP', 'PHI0', 'KFB'))
    parser.add_argument('--step-roll', type=float, nargs=2, metavar=('AMP', 'HOLD'),
                        help='experiment F: +AMP step at t=0 and -AMP at t=3 s (1.6 rad/s ramps), replaces the gait')
    parser.add_argument('--abduction', type=float, default=0., help='hip-roll outward offset on both legs (rad)')
    parser.add_argument('--window', type=float, default=.35, help='unloaded (swing) window, fraction of a cycle')
    parser.add_argument('--mirror', action='store_true', help='left/right mirrored gait (r -> -r, windows swapped)')
    parser.add_argument('--diag', action='store_true', help='log 100 Hz joint/IMU/contact wrench rows (no control effect)')
    parser.add_argument('--horizon', type=float, default=.02, help='JTC time_from_start of each streamed target (s)')
    parser.add_argument('--out', default='/raptor_ws/log/rock-gz-probe.json')
    args = parser.parse_args()
    if not (0 <= args.amplitude <= .12 and .5 <= args.frequency <= 3. and 1 <= args.cycles <= 200
            and .01 <= args.horizon <= .1 and .1 <= args.ramp <= 5 and 0 <= args.settle <= 5 and 0 <= args.stride <= .08
            and .2 <= args.window <= .4 and 0 <= args.abduction <= .06
            and (args.step_roll is None or (0 < args.step_roll[0] <= .1 and .1 <= args.step_roll[1] <= 1.))
            and (args.pitch_feedback is None or (0 <= args.pitch_feedback[0] <= 1 and 0 <= args.pitch_feedback[1] <= .2
                                                  and 0 <= args.pitch_feedback[2] <= .2))
            and (args.tail is None or all(abs(v) <= 3 for v in args.tail))
            and (args.tail_sync is None or (0 <= args.tail_sync[0] <= .2 and 0 <= args.tail_sync[1] < 1
                                            and abs(args.tail_sync[2]) <= .5))):
        parser.error('outside experimental bounds (A <= 0.12, 0.5 <= f <= 3, cycles <= 200, 0.1 <= ramp <= 5, '
                     'stride <= 0.08, feedback kp <= 1 kd <= 0.2 limit <= 0.2)')
    rclpy.init()
    node = RockProbe(args)
    error, wall, poses = None, time.monotonic(), {}
    try:
        poses['start'] = pose()['position']  # Gazebo world pose via gz transport (blocking, outside the loop)
        if 'raptor_mission_gate' in node.get_node_names():
            raise RuntimeError('stop the operator mission gate before a development probe')
        while not node.step():
            rclpy.spin_once(node, timeout_sec=.01)
            if node.stop_reason:
                stopped = node.sim_time
                while node.sim_time is not None and node.sim_time-stopped < .5 and time.monotonic()-wall < 60*20:
                    rclpy.spin_once(node, timeout_sec=.01)  # record 0.5 s of sim after the hold
                break
            if time.monotonic()-wall > 60*20:
                raise RuntimeError('wall-clock timeout')
            if node.start is not None and time.monotonic()-node.clock_wall > CLOCK_STALL:
                raise RuntimeError('sim time stalled')
            if node.start is None and time.monotonic()-wall > 30:
                raise RuntimeError('no clock/joint/IMU data')
    except Exception as exc:  # noqa: BLE001  any fault must end in a hold
        error = f'{type(exc).__name__}: {exc}'
    finally:
        if node.stop_reason is None and error and node.start is not None:
            node.hold()
        try:
            poses['end'] = pose()['position']
        except Exception as exc:  # noqa: BLE001  displacement is diagnostic only
            poses['end_error'] = str(exc)
        result = {'arguments': vars(args), 'stop_reason': node.stop_reason, 'error': error, 'base_pose': poses,
                  'wall_s': round(time.monotonic()-wall, 1), 'summary': summary(node.rows, args),
                  'raw_joint_states_tail': list(node.raw) if node.stop_reason else [],
                  'speed_spikes': node.spikes, 'diag': node.diag, 'start_sim_time': node.start, 'gait_period_s': node.gait.period, 'gait_duty': node.gait.duty,
                  'limitation': 'Gazebo DART open-loop rocking in place; contact = any foot/toe contact message; '
                                'not walking.', 'rows': node.rows}
        Path(args.out).write_text(json.dumps(result, separators=(',', ':')))
        print(json.dumps({k: result[k] for k in ('stop_reason', 'error', 'wall_s', 'base_pose', 'summary')}
                         | {'speed_spike_count': len(node.spikes)}, indent=1))
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
