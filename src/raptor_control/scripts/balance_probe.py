#!/usr/bin/env python3
"""Bounded left-support IMU experiment; never enabled as an operator mission."""
import json
import argparse
import math
import time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.parameter import Parameter
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from rclpy.qos import QoSProfile, DurabilityPolicy
from ros_gz_interfaces.msg import Contacts
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from motion_probe import Probe, JOINTS, pose
from support_model import SupportModel


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fast-response',action='store_true',help='Failed bandwidth comparison, not a validated fix')
    args=parser.parse_args()
    max_delta=.030 if args.fast_response else .012
    horizon_ns=20000000 if args.fast_response else 80000000
    rclpy.init()
    node = Probe()
    node.set_parameters([Parameter('use_sim_time', value=True)])
    description = {}
    node.create_subscription(String, '/robot_description',
        lambda msg: description.update(xml=msg.data),
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    imu = {}; contacts = {}; rows = []; report = {'passed': False, 'fast_response':args.fast_response, 'samples': rows}
    def observe_imu(msg):
        imu.update(message=msg, received=time.monotonic())
    def observe_contact(msg, side):
        contacts[side] = {'received': time.monotonic(), 'force_z': sum(
            w.body_1_wrench.force.z for c in msg.contacts for w in c.wrenches)}
    node.create_subscription(Imu, '/raptor/imu', observe_imu, 10)
    for side in ['left', 'right']:
        node.create_subscription(Contacts, f'/raptor/{side}_foot/contact',
                                 lambda msg, s=side: observe_contact(msg, s), 10)
    pub = node.create_publisher(JointTrajectory, '/raptor_joint_controller/joint_trajectory', 10)
    control_started=False
    try:
        if not node.client.wait_for_server(timeout_sec=10):
            raise RuntimeError('Controller unavailable')
        node.settle(1)
        deadline=time.monotonic()+10
        while not description:
            if time.monotonic()>deadline:raise RuntimeError('No live robot_description')
            rclpy.spin_once(node,timeout_sec=.05)
        model=SupportModel(description['xml'])
        if any('_toe_' in j[0] or j[0]=='test_fixture_joint' for j in model.joints):
            raise RuntimeError('This balance approximation requires the free rigid-foot baseline')
        if 'raptor_mission_gate' in node.get_node_names():
            raise RuntimeError('Stop operator mission gate before a development probe')
        target = dict.fromkeys(JOINTS, 0.)
        for side in ['left', 'right']:
            target.update({f'{side}_hip_pitch_joint': -.15, f'{side}_knee_pitch_joint': .4,
                           f'{side}_ankle_pitch_joint': -.25})
        control_started=True
        node.move(target)
        target.update(left_hip_roll_joint=-.34, right_hip_roll_joint=.34)
        node.move(target)
        start = node.get_clock().now().nanoseconds/1e9
        wall_start = time.monotonic(); last = -1.; roll_target = .34
        while True:
            rclpy.spin_once(node, timeout_sec=.01)
            elapsed = node.get_clock().now().nanoseconds/1e9-start
            if time.monotonic()-wall_start > 90:
                raise RuntimeError('Simulation experiment wall-time limit')
            if elapsed > 8:
                break
            if elapsed-last < .02:
                continue
            last = elapsed
            if (time.monotonic()-imu.get('received', 0)>1 or
                    time.monotonic()-node.received>1):
                raise RuntimeError('Stale IMU or joint state')
            msg = imu['message']; q = msg.orientation
            roll = math.atan2(2*(q.w*q.x+q.y*q.z), 1-2*(q.x*q.x+q.y*q.y))
            pitch = math.asin(max(-1., min(1., 2*(q.w*q.y-q.z*q.x))))
            if not all(math.isfinite(v) for v in [roll, pitch, msg.angular_velocity.x]):
                raise RuntimeError('Nonfinite IMU')
            if abs(roll)>.25 or abs(pitch)>.25:
                raise RuntimeError('Body tilt exceeded 0.25 rad')
            x,y,z,w=q.x,q.y,q.z,q.w
            R=np.eye(4);R[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
            # Lift only after a short feedback settling period; no forward step.
            fraction=max(0.,min(1.,(elapsed-1)/2))
            target.update(right_knee_pitch_joint=.4+.55*fraction,
                          right_hip_pitch_joint=-.15-.25*fraction,
                          right_ankle_pitch_joint=-.25-.30*fraction)
            desired_margin=.03*roll+.03*msg.angular_velocity.x
            lo,hi=.1,.45
            for _ in range(12):
                mid=(lo+hi)/2
                trial=target|{'left_hip_roll_joint':-mid,'right_hip_roll_joint':mid}
                if model.margin(trial,R)>desired_margin:hi=mid
                else:lo=mid
            requested=(lo+hi)/2
            roll_target=max(roll_target-max_delta,min(roll_target+max_delta,requested))
            target.update(left_hip_roll_joint=-roll_target,right_hip_roll_joint=roll_target)
            point=JointTrajectoryPoint(positions=[target[j] for j in JOINTS])
            point.time_from_start.nanosec=horizon_ns
            pub.publish(JointTrajectory(joint_names=JOINTS,points=[point]))
            rows.append({'sim_seconds':elapsed,'roll':roll,'pitch':pitch,
                'roll_rate':msg.angular_velocity.x,'roll_target':roll_target,
                'requested_roll_target':requested,
                'actual_left_hip_roll':node.state['left_hip_roll_joint'],
                'margin':model.margin(node.state,R), 'desired_margin':desired_margin,
                'contacts':{s:{'recent':time.monotonic()-v['received']<.5,
                               'force_z':v['force_z']} for s,v in contacts.items()}})
        report['final_pose']=pose()
        support=rows[-100:]
        report['passed']=len(support)==100 and all(
            row['contacts'].get('left',{}).get('recent',False) and
            not row['contacts'].get('right',{}).get('recent',True) for row in support)
        if not report['passed']:
            raise RuntimeError('Sustained single-foot contact not established')
    except RuntimeError as exc:
        report['error']=str(exc)
    finally:
        # End the experiment with a position hold, even on failed criteria.
        if control_started and all(j in node.state and math.isfinite(node.state[j]) for j in JOINTS):
            point=JointTrajectoryPoint(positions=[node.state[j] for j in JOINTS])
            point.time_from_start.nanosec=100000000
            pub.publish(JointTrajectory(joint_names=JOINTS,points=[point]))
            rclpy.spin_once(node,timeout_sec=.1)
        Path('/raptor_ws/log/balance-probe.json').write_text(json.dumps(report,indent=2))
        node.destroy_node();rclpy.shutdown()
    print(json.dumps({k:v for k,v in report.items() if k!='samples'}))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
