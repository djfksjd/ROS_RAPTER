"""Operator CLI: model -> semantic action -> ROS gate. --stop bypasses models."""
import argparse
import json
import subprocess
import time
import uuid
from commands import Qwen


def dispatch(action):
    payload = {'id':str(uuid.uuid4()), 'action':action, 'sent_at':time.time()}
    # Arguments, never shell interpolation: model output cannot execute code.
    message = json.dumps({'data':json.dumps(payload)})
    result = subprocess.run(['docker','exec','raptor-dev','/ros_entrypoint.sh',
        'ros2','topic','pub','--once','/raptor/mission_command','std_msgs/msg/String',message],
        capture_output=True,text=True,timeout=15)
    if result.returncode:
        raise RuntimeError('ROS publish failed; check the running raptor-dev container')
    return payload


def main():
    p=argparse.ArgumentParser()
    p.add_argument('text',nargs='?')
    p.add_argument('--backend',choices=['qwen','nanojev','laya'],default='qwen')
    p.add_argument('--adapter',help='Optional trained NanoJev decision-head safetensors')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--stop',action='store_true')
    args=p.parse_args()
    if args.stop:
        print(json.dumps(dispatch('STOP')));return
    if not args.text:p.error('text is required unless --stop is used')
    if args.backend=='qwen':model=Qwen()
    elif args.backend=='laya':
        from laya_backend import Laya
        model=Laya()
    else:
        from nanojev import NanoJev
        model=NanoJev(adapter=args.adapter)
    decision=model.predict(args.text)
    print(json.dumps(decision,ensure_ascii=False))
    if args.execute:
        print(json.dumps(dispatch(decision['action']),ensure_ascii=False))


if __name__=='__main__':main()
