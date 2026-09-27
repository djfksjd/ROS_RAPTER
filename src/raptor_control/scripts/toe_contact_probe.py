#!/usr/bin/env python3
"""Read-only, five simulated-second passive toe contact measurement."""
import json
import math
import time
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState, Imu
from ros_gz_interfaces.msg import Contacts
from motion_probe import JOINTS, pose


def main():
    rclpy.init();node=Node('toe_contact_probe',parameter_overrides=[Parameter('use_sim_time',value=True)])
    toes=[f'{s}_toe_{d}_{p}_joint' for s in ['left','right'] for d in range(1,4)
          for p in ['proximal','distal']]
    states={};imu={};contacts={};report={'samples':[],'passed_static_observation':False}
    def joints(msg):
        for name,value in zip(msg.name,msg.position):states[name]=(value,time.monotonic())
    def inertial(msg):imu.update(message=msg,received=time.monotonic())
    def contact(msg,key):
        contacts[key]={'raw_body1_force_z':sum(w.body_1_wrench.force.z for c in msg.contacts for w in c.wrenches),
                       'received':time.monotonic(), 'pairs':[(c.collision1.name,c.collision2.name) for c in msg.contacts]}
    node.create_subscription(JointState,'/joint_states',joints,10)
    node.create_subscription(Imu,'/raptor/imu',inertial,10)
    for side in ['left','right']:
        for digit in range(1,4):
            for part in ['proximal','distal']:
                key=f'{side}_toe_{digit}_{part}'
                node.create_subscription(Contacts,f'/raptor/{side}/toe_{digit}_{part}/contact',
                                         lambda msg,k=key:contact(msg,k),10)
    try:
        deadline=time.monotonic()+15
        while not (set(JOINTS+toes)<=states.keys() and imu and node.get_clock().now().nanoseconds>0):
            if time.monotonic()>deadline:raise RuntimeError('Incomplete active/passive/IMU feedback')
            rclpy.spin_once(node,timeout_sec=.05)
        start=node.get_clock().now().nanoseconds/1e9;last=-1.;deadline=time.monotonic()+45
        while True:
            rclpy.spin_once(node,timeout_sec=.02)
            elapsed=node.get_clock().now().nanoseconds/1e9-start
            if elapsed>=5:break
            if time.monotonic()>deadline:raise RuntimeError('Simulation clock stalled or too slow')
            if elapsed-last<.2:continue
            last=elapsed
            if any(time.monotonic()-states[j][1]>1 for j in JOINTS+toes) or time.monotonic()-imu['received']>1:
                raise RuntimeError('Stale sensor feedback')
            q=imu['message'].orientation
            roll=math.atan2(2*(q.w*q.x+q.y*q.z),1-2*(q.x*q.x+q.y*q.y))
            pitch=math.asin(max(-1.,min(1.,2*(q.w*q.y-q.z*q.x))))
            angles={j:states[j][0] for j in toes}
            row={'sim_seconds':elapsed,'roll':roll,'pitch':pitch,'toe_angles':angles,
                 'contacts':{k:{'recent':time.monotonic()-v['received']<.5,'raw_body1_force_z':v['raw_body1_force_z'], 'collision_pairs':v['pairs']}
                             for k,v in contacts.items()}}
            report['samples'].append(row)
            if not all(math.isfinite(v) and -.551<=v<=.651 for v in angles.values()):
                raise RuntimeError('Passive joint limit or finite-value check failed')
            if not math.isfinite(roll+pitch) or abs(roll)>.1 or abs(pitch)>.1:
                raise RuntimeError('Static attitude exceeded 0.1 rad')
        report['final_pose']=pose();report['passed_static_observation']=True
    except RuntimeError as exc:report['error']=str(exc)
    finally:
        Path('/raptor_ws/log/toe-contact-probe.json').write_text(json.dumps(report,indent=2))
        node.destroy_node();rclpy.shutdown()
    print(json.dumps({k:v for k,v in report.items() if k!='samples'}))
    if not report['passed_static_observation']:raise SystemExit(1)


if __name__=='__main__':main()
