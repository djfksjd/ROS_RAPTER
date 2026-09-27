#!/usr/bin/env python3
"""Simulation mission gate. Unverified navigation/gaits are rejected explicitly."""
from collections import deque
import json
import math
import time
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from motion_probe import JOINTS


from mission_policy import Policy


class MissionGate(Node):
    def __init__(self):
        super().__init__('raptor_mission_gate', parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.policy = Policy()
        self.state = {}
        self.last_state = 0.
        self.seen = deque(maxlen=128)
        qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status = self.create_publisher(String, '/raptor/mission_status', qos)
        self.motion = self.create_publisher(JointTrajectory,
            '/raptor_joint_controller/joint_trajectory', 10)
        self.create_subscription(JointState, '/joint_states', self.observe, 10)
        self.create_subscription(String, '/raptor/mission_command', self.command, 10)
        self.pending = None
        self.create_timer(.1, self.monitor)

    def observe(self, msg):
        self.state = dict(zip(msg.name, msg.position))
        self.last_state = time.monotonic()

    def report(self, request_id, action, result):
        data = {'id': request_id, 'action': action, 'result': result,
                'stop_latched': self.policy.stopped}
        self.status.publish(String(data=json.dumps(data)))
        self.get_logger().info(json.dumps(data))

    def command(self, msg):
        request_id, action = 'invalid', 'invalid'
        try:
            if len(msg.data) > 1000:
                raise ValueError('Oversized payload')
            data = json.loads(msg.data)
            if not isinstance(data, dict) or set(data) != {'id','action','sent_at'}:
                raise ValueError('Invalid schema')
            request_id, action = data['id'], data['action']
            if not isinstance(request_id,str) or not 1 <= len(request_id) <= 64 or not isinstance(action,str):
                raise ValueError('Invalid identity/action')
            if request_id in self.seen:
                raise ValueError('Duplicate request')
            stamp = data['sent_at']
            if type(stamp) not in (float,int) or not math.isfinite(stamp) or abs(time.time()-stamp)>10:
                raise ValueError('Stale request')
            self.seen.append(request_id)
            outcome = self.policy.check(action)
            if outcome.startswith('rejected'):
                self.report(request_id,action,outcome); return
            if set(JOINTS)-self.state.keys() or time.monotonic()-self.last_state>1:
                raise ValueError('Joint feedback unavailable')
            positions = [self.state[j] for j in JOINTS]
            if not all(math.isfinite(v) for v in positions):
                raise ValueError('Invalid feedback')
            if self.pending:
                self.report(self.pending['id'], self.pending['action'], 'preempted')
            if outcome == 'stand':
                positions = [0.]*10
            point = JointTrajectoryPoint(positions=positions)
            point.time_from_start.sec = 2 if outcome == 'stand' else 0
            point.time_from_start.nanosec = 0 if outcome == 'stand' else 100000000
            self.motion.publish(JointTrajectory(joint_names=JOINTS,points=[point]))
            self.pending = {'id':request_id,'action':action,'positions':positions,
                            'deadline':time.monotonic()+12,'earliest':time.monotonic()+2.5}
            self.report(request_id,action,'accepted')
        except (ValueError,TypeError,KeyError):
            self.report(request_id,action,'rejected_invalid_or_stale_input')

    def monitor(self):
        p = self.pending
        if p is None or time.monotonic()<p['earliest']:
            return
        if time.monotonic()-self.last_state<1 and all(
            abs(self.state.get(j,float('inf'))-v)<.025 for j,v in zip(JOINTS,p['positions'])):
            self.report(p['id'],p['action'],'joint_target_reached')
            self.pending=None
        elif time.monotonic()>p['deadline']:
            self.report(p['id'],p['action'],'failed_tracking')
            self.pending=None


def main():
    rclpy.init(); node=MissionGate()
    try:rclpy.spin(node)
    finally:node.destroy_node();rclpy.shutdown()


if __name__ == '__main__':main()
