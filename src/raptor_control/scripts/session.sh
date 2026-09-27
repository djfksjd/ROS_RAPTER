#!/usr/bin/env bash
# One container owns all ROS, simulator and desktop processes.
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source /raptor_ws/install/setup.bash
export DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1
export GZ_SIM_RESOURCE_PATH="$(ros2 pkg prefix raptor_description)/share"
mkdir -p /raptor_ws/log/session
pids=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${pids[@]}"; do kill -INT "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM
Xvfb :99 -screen 0 1440x900x24 > /raptor_ws/log/session/xvfb.log 2>&1 & pids+=("$!")
for attempt in {1..50}; do
  if [[ -S /tmp/.X11-unix/X99 ]]; then break; fi
  sleep .1
done
[[ -S /tmp/.X11-unix/X99 ]]
fluxbox > /raptor_ws/log/session/desktop.log 2>&1 & pids+=("$!")
x11vnc -display :99 -localhost -forever -shared -nopw -rfbport 5900 > /raptor_ws/log/session/vnc.log 2>&1 & pids+=("$!")
websockify --web=/usr/share/novnc 6080 localhost:5900 > /raptor_ws/log/session/websockify.log 2>&1 & pids+=("$!")
ros2 launch raptor_control sim.launch.py detailed_visuals:=true sensors:=true > /raptor_ws/log/session/simulation.log 2>&1 &
sim_pid=$!;pids+=("$sim_pid")
python3 /raptor_ws/src/raptor_control/scripts/mission_gate.py > /raptor_ws/log/session/mission.log 2>&1 & pids+=("$!")
gz sim -g --gui-config /raptor_ws/src/raptor_control/config/gui.config > /raptor_ws/log/session/gui.log 2>&1 & pids+=("$!")
wait "$sim_pid"
