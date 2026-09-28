#!/usr/bin/env python3
"""Bounded Gazebo lateral rocking probe (simulation only, not a mission, not a walking test).

Streams position targets on sim time: crouch pose + hip rolls left = -r, right = +r,
r = A*min(1, t/ramp)*sin(2*pi*f*t) (same open-loop law as sim/rock_probe.py, no stride).
Each target is streamed every 20 ms with a 20 ms JTC horizon (50 ms halved hip-roll tracking, evidence 68).
Saturation watchdog (SIMULATION_METHODOLOGY.ko.md): STOP when a joint velocity reaches 90% of its
limit, a joint leaves its target by more than 0.15 rad, the IMU tilt exceeds 0.25 rad, joint/IMU data go
stale, or sim time stalls or runs backwards. STOP holds the measured positions (last finite target where a
measurement is unusable) and ends streaming; any exception also holds. A normal end returns to the crouch. Writes /raptor_ws/log/rock-gz-probe.json.
"""
import argparse
import json
import math
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from ros_gz_interfaces.msg import Contacts
from sensor_msgs.msg import Imu, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from motion_guard import TiltGuard
from motion_probe import JOINTS

# Velocity limits from raptor.urdf.xacro (rad/s); unchanged by leg_design.
VELOCITY_LIMIT = {'hip_roll': 2.0, 'hip_pitch': 2.5, 'knee_pitch': 3.0, 'ankle_pitch': 2.5,
                  'tail_yaw': 2.0, 'tail_pitch': 2.0}
TRACKING_LIMIT = .15
STALE = .1  # s of sim time without joint/IMU updates
CLOCK_STALL = 2.  # s of wall time without a /clock advance after start
PERIOD = .02  # s of sim time between streamed targets


def limit_of(joint):
    return next(v for k, v in VELOCITY_LIMIT.items() if k in joint)


class RockProbe(Node):
    def __init__(self, args):
        super().__init__('raptor_rock_gz_probe')
        self.args = args
        self.crouch = dict.fromkeys(JOINTS, 0.)
        for side in ('left', 'right'):
            self.crouch |= {f'{side}_hip_pitch_joint': args.crouch_hip, f'{side}_knee_pitch_joint': args.crouch_knee,
                            f'{side}_ankle_pitch_joint': -args.crouch_hip-args.crouch_knee}
        self.guard = TiltGuard(limit=.25, clock=lambda: self.sim_time or 0.)
        self.sim_time, self.joints, self.contacts, self.imu = None, {}, {}, None
        self.joints_at = self.imu_at = None
        self.clock_wall, self.clock_fault = time.monotonic(), None
        self.target, self.start, self.last_sent = dict(self.crouch), None, -math.inf
        self.stop_reason, self.rows = None, []
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

    def on_imu(self, msg):
        q = msg.orientation
        self.imu_at = self.sim_time
        self.guard.observe(q.x, q.y, q.z, q.w, msg.orientation_covariance[0] != -1)
        self.imu = (math.atan2(2*(q.w*q.x+q.y*q.z), 1-2*(q.x*q.x+q.y*q.y)),
                    math.asin(max(-1., min(1., 2*(q.w*q.y-q.z*q.x)))))

    def on_contact(self, msg, key):
        self.contacts[key] = (self.sim_time, len(msg.contacts))

    def send(self, positions, horizon=.05):
        msg = JointTrajectory()
        msg.joint_names = JOINTS
        point = JointTrajectoryPoint()
        point.positions = [float(positions[j]) for j in JOINTS]
        point.time_from_start.nanosec = int(horizon*1e9)
        msg.points = [point]
        self.pub.publish(msg)

    def in_contact(self, side):
        return any(n > 0 and self.sim_time-t < .1 for k, (t, n) in self.contacts.items() if k.startswith(side))

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
                return f'velocity saturation {j} {v:.3f} rad/s (limit {limit_of(j)})'
            if abs(q-self.target[j]) > TRACKING_LIMIT:
                return f'tracking error {j} {q-self.target[j]:+.3f} rad'
        return None

    def step(self):
        t = self.sim_time
        if t is None or len(self.joints) < len(JOINTS) or self.joints_at is None or self.imu_at is None:
            return False
        if self.start is None:
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
        a = phase-self.args.settle
        r = 0. if a < 0 else self.args.amplitude*min(1., a/self.args.ramp)*math.sin(2*math.pi*self.args.frequency*a)
        self.target = dict(self.crouch) | {'left_hip_roll_joint': -r, 'right_hip_roll_joint': r}
        self.send(self.target, self.args.horizon)
        self.rows.append({'t': round(phase, 3), 'r_cmd': round(r, 5), 'roll': round(self.imu[0], 5),
                          'pitch': round(self.imu[1], 5), 'tilt': round(self.guard.tilt or 0., 5),
                          'contact': {s: self.in_contact(s) for s in ('left', 'right')},
                          'q': {j: round(self.joints[j][0], 4) for j in JOINTS},
                          'v': {j: round(self.joints[j][1], 4) for j in JOINTS}})
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
            'max_abs_velocity': {j: max((abs(r['v'][j]) for r in rows), default=None) for j in JOINTS}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--amplitude', type=float, default=.08)
    parser.add_argument('--frequency', type=float, default=2.)
    parser.add_argument('--cycles', type=int, default=20)
    parser.add_argument('--ramp', type=float, default=1.)
    parser.add_argument('--settle', type=float, default=1.)
    parser.add_argument('--crouch-hip', type=float, default=-.10)
    parser.add_argument('--crouch-knee', type=float, default=.5)
    parser.add_argument('--horizon', type=float, default=.02, help='JTC time_from_start of each streamed target (s)')
    parser.add_argument('--out', default='/raptor_ws/log/rock-gz-probe.json')
    args = parser.parse_args()
    if not (0 <= args.amplitude <= .12 and .5 <= args.frequency <= 2.5 and 1 <= args.cycles <= 120
            and .01 <= args.horizon <= .1 and .1 <= args.ramp <= 5 and 0 <= args.settle <= 5):
        parser.error('outside experimental bounds (A <= 0.12, 0.5 <= f <= 2.5, cycles <= 120, 0.1 <= ramp <= 5)')
    rclpy.init()
    node = RockProbe(args)
    error, wall = None, time.monotonic()
    try:
        if 'raptor_mission_gate' in node.get_node_names():
            raise RuntimeError('stop the operator mission gate before a development probe')
        while not node.step():
            rclpy.spin_once(node, timeout_sec=.01)
            if node.stop_reason:
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
        result = {'arguments': vars(args), 'stop_reason': node.stop_reason, 'error': error,
                  'wall_s': round(time.monotonic()-wall, 1), 'summary': summary(node.rows, args),
                  'limitation': 'Gazebo DART open-loop rocking in place; contact = any foot/toe contact message; '
                                'not walking.', 'rows': node.rows}
        Path(args.out).write_text(json.dumps(result, separators=(',', ':')))
        print(json.dumps({k: result[k] for k in ('stop_reason', 'error', 'wall_s', 'summary')}, indent=1))
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
