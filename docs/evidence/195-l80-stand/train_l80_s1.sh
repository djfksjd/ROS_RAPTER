#!/bin/bash
# Evidence 195: stage 1a stand on the frozen 0.80 m leg (rev3_l80.json), continued from the R-03 stand policy r3_s1ho.
# Same settings as r3_s1ho (args.json) except the axis config, the realism file and the exploration std reset.
# Budget 6M steps (+ one 4M extension); 1M-step checks; kill rules in README.md (written before the run).
source .venv-sim/bin/activate
W='{"track_lin": 5, "progress": 3, "track_yaw": 4, "yaw_err": -1, "stand": -5, "gait": -8, "fold": 0.35, "extend": 0.3, "ankle_static": -0.2, "retract": 0.7, "leg_sym": -6, "heat": -2, "spring_store": 0.03, "spring_release": -0.06, "joint_limit": -30, "self_collision": -5, "hold": -3, "stand_support": -5, "slip_lin": -5, "action_rate": -2}'
python sim/rl/train_run.py --name ${NAME:-l80_s1} --steps ${STEPS:-6000000} --envs 8 --model-rev 3 --arch8 --real-mass --tail active --springs c \
  --springs-override '{"knee_pitch": [500, 0.956, "uni+", "ratchet"]}' --ankle-clutch --target-ramp \
  --rev3-config sim/rl/rev3_l80.json --realism sim/rl/realism_robstride_l80.json --contact-tc 0.005 --nominal -0.5273 1.0 -0.9997 \
  --cmd 0 0 0 --cmd-final 0 0 0 --zero-cmd 1 --no-init-speed --episode-s 20 --fixed-curriculum --level 0.5 \
  --disturb-items push pitch yaw --roll-impulse 0.1 0.1 0.5 --init-noise 0.04 --heat-limit 0.8 \
  --scale-override '{"hip_roll": 0.5}' --symmetric --action-lpf 6 --kp-err 0.2 --obs-hold '{"noise":0.02,"walk":0.005}' \
  --init ${INIT:-sim/rl/runs/r3_s1ho/model.zip} --resume-action-std ${STD:-0.15} --weights "$W"
