#!/usr/bin/env python3
"""Experimental one-step trajectory, explicitly not an enabled mission."""
import json
import argparse
import math
import time
from collections import deque
from pathlib import Path
from motion_probe import Probe, JOINTS, pose, rclpy
from ros_gz_interfaces.msg import Contacts
from sensor_msgs.msg import JointState, Imu
from motion_guard import TiltGuard


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--roll',type=float,default=.15)
    parser.add_argument('--cycles',type=int,default=4)
    parser.add_argument('--lift',type=float,default=.65)
    parser.add_argument('--crouch-hip',type=float,default=-.15)
    args=parser.parse_args()
    if not 0<=args.roll<=.45 or not 0<=args.cycles<=8 or not .65<=args.lift<=1.1 or not -.15<=args.crouch_hip<=.05:parser.error('Outside experimental bounds')
    lift_hip=-args.lift/2+.075
    rclpy.init();node=Probe(); rows=[]
    failed=False
    guard=TiltGuard()
    telemetry=deque(maxlen=2000)
    telemetry_count=0
    last_sample=-math.inf
    current_phase='initializing'
    def stamp(msg):
        return msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
    def observe_imu(msg):
        q=msg.orientation
        guard.observe(q.x,q.y,q.z,q.w,msg.orientation_covariance[0]!=-1)
        nonlocal last_sample,telemetry_count
        sim_time=stamp(msg)
        if sim_time-last_sample >= .049 or guard.failure:
            last_sample=sim_time
            telemetry_count+=1
            telemetry.append({'sim_time':sim_time,'phase':current_phase,
                'orientation':[q.x,q.y,q.z,q.w],
                'angular_velocity':[msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z],
                'tilt_rad':guard.tilt,
                'joints':{name:{'position':v['position'],'velocity':v.get('velocity'),
                    'transmitted_effort':v.get('transmitted_effort'),'sim_time':v['sim_time']} for name,v in all_joints.items()},
                'contacts':{name:{'sim_time':v['sim_time'],'count':v['contact_count'],
                    'points':v['points'],'raw_body1_force_z':v['force_z']}
                    for name,v in contacts.items()}})
    node.create_subscription(Imu,'/raptor/imu',observe_imu,10)
    contacts={}
    all_joints={}
    def observe_joints(msg):
        if len(msg.name)==len(msg.position):
            for index,(name,value) in enumerate(zip(msg.name,msg.position)):
                if not math.isfinite(value):continue
                all_joints[name]={'position':value,'received':time.monotonic(),'sim_time':stamp(msg)}
                for field,values in [('velocity',msg.velocity),('transmitted_effort',msg.effort)]:
                    all_joints[name][field]=(values[index] if len(values)==len(msg.name)
                        and math.isfinite(values[index]) else None)
    node.create_subscription(JointState,'/joint_states',observe_joints,10)
    def observe(msg,key):
        contacts[key]={'received':time.monotonic(),'sim_time':stamp(msg),'force_z':sum(
            w.body_1_wrench.force.z for c in msg.contacts for w in c.wrenches),
            'contact_count':len(msg.contacts),
            'points':[[p.x,p.y,p.z] for c in msg.contacts for p in c.positions],
            'collision_pairs':[[c.collision1.name,c.collision2.name] for c in msg.contacts]}
    for side in ['left','right']:
        node.create_subscription(Contacts,f'/raptor/{side}_foot/contact',
                                 lambda msg,s=side:observe(msg,s+'_heel'),10)
        for digit in range(1,4):
            for part in ['proximal','distal']:
                key=f'{side}_toe_{digit}_{part}'
                node.create_subscription(Contacts,f'/raptor/{side}/toe_{digit}_{part}/contact',
                                         lambda msg,k=key:observe(msg,k),10)
    try:
        if not node.client.wait_for_server(timeout_sec=10):raise RuntimeError('No controller')
        node.settle(1)
        if 'raptor_mission_gate' in node.get_node_names():
            raise RuntimeError('Stop operator mission gate before a development probe')
        node.guard=guard
        node.check_guard()
        target=dict.fromkeys(JOINTS,0.)
        crouch={f'{s}_{j}_joint':v for s in ['left','right']
                for j,v in [('hip_pitch',args.crouch_hip),('knee_pitch',.4),('ankle_pitch',-.4-args.crouch_hip)]}
        phases=[('crouch',crouch),
            ('shift_left',{'left_hip_roll_joint':-args.roll,'right_hip_roll_joint':args.roll}),
            ('lift_right',{'right_hip_pitch_joint':lift_hip,'right_knee_pitch_joint':args.lift,'right_ankle_pitch_joint':-lift_hip-args.lift}),
            ('swing_right',{'right_hip_pitch_joint':lift_hip-.1,'right_ankle_pitch_joint':-lift_hip-args.lift+.1}),
            ('land_right',{'right_hip_pitch_joint':-.22,'right_knee_pitch_joint':.4,'right_ankle_pitch_joint':-.18}),
            ('center',{'left_hip_roll_joint':0.,'right_hip_roll_joint':0.})]
        for step in range(args.cycles):
            swing='left' if step%2==0 else 'right'
            support='right' if swing=='left' else 'left'
            roll=args.roll if support=='right' else -args.roll
            phases += [
                (f'step{step}_shift',{'left_hip_roll_joint':roll,'right_hip_roll_joint':-roll}),
                (f'step{step}_lift',{f'{swing}_hip_pitch_joint':lift_hip,
                    f'{swing}_knee_pitch_joint':args.lift,f'{swing}_ankle_pitch_joint':-lift_hip-args.lift,
                    f'{support}_hip_pitch_joint':args.crouch_hip,f'{support}_knee_pitch_joint':.4,
                    f'{support}_ankle_pitch_joint':-.4-args.crouch_hip}),
                (f'step{step}_swing',{f'{swing}_hip_pitch_joint':lift_hip-.1,f'{swing}_ankle_pitch_joint':-lift_hip-args.lift+.1}),
                (f'step{step}_land',{f'{swing}_hip_pitch_joint':-.22,f'{swing}_knee_pitch_joint':.4,
                    f'{swing}_ankle_pitch_joint':-.18}),
                (f'step{step}_center',{'left_hip_roll_joint':0.,'right_hip_roll_joint':0.})]
        for name,changes in phases:
            current_phase=name
            target.update(changes)
            row={'phase':name,'target':dict(target)};rows.append(row)
            try:row['tracking_error']=node.move(target,seconds=2)
            finally:
                row['pose']=pose()
                row['joint_positions']=dict(node.state)
            # pose() blocks on Gazebo transport: refresh ROS callbacks before contact snapshot.
            node.settle(.1)
            row['all_joint_positions']={name:{'position':v['position'],
                'age_wall_s':time.monotonic()-v['received']} for name,v in all_joints.items()}
            row['contacts']={side:{'recent_messages':any(time.monotonic()-v['received']<.5
                for key,v in contacts.items() if key.startswith(side+'_')),
                'active_sources':[key for key,v in contacts.items() if key.startswith(side+'_')
                    and time.monotonic()-v['received']<.5 and v['contact_count']>0],
                'samples':{key:{**v,'age_wall_s':time.monotonic()-v['received']}
                    for key,v in contacts.items() if key.startswith(side+'_')},
                'raw_body1_force_z':{key:v['force_z'] for key,v in contacts.items() if key.startswith(side+'_')},
                'recent_sources':[key for key,v in contacts.items()
                    if key.startswith(side+'_') and time.monotonic()-v['received']<.5]}
                for side in ['left','right']}
            q=row['pose']['orientation'];tilt=2*math.acos(min(1.,abs(q.get('w',1.))))
            row['rotation_rad']=tilt
            if tilt>.25:raise RuntimeError('Body rotation exceeded 0.25 rad; gait disabled')
        print('Trajectory completed. Foot contact and locomotion still require independent validation.')
    except RuntimeError as exc:
        failed=True
        rows.append({'error':str(exc)})
        print(str(exc))
    finally:
        Path('/raptor_ws/log/gait-telemetry.json').write_text(json.dumps({
            'arguments':vars(args),'sample_count':telemetry_count,
            'dropped_samples':telemetry_count-len(telemetry),'samples':list(telemetry)},separators=(',',':')))
        if node.cancellation_events:
            rows.append({'cancellation_events':node.cancellation_events})
        Path('/raptor_ws/log/gait-probe.json').write_text(json.dumps(rows,indent=2))
        node.destroy_node();rclpy.shutdown()
    if failed:
        raise SystemExit(1)


if __name__=='__main__':main()
