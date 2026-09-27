# Local Mac development

Use Docker Desktop's Linux ARM64 environment on the M5 Mac. The host does not
need native ROS. Start Docker Desktop before running these commands at repo root.

```sh
docker compose build
docker compose run --rm ros bash -c 'colcon build --symlink-install'
docker compose run --rm ros bash -c 'source install/setup.bash && xacro src/raptor_description/urdf/raptor.urdf.xacro > /tmp/raptor.urdf && check_urdf /tmp/raptor.urdf'
docker compose run --rm ros bash
```

Inside the shell, source `install/setup.bash` after building. Run all ROS and
Gazebo processes in the same container: ROS is restricted to localhost, domain 42.
The source directory is read-only; edit it on the Mac. Build/install/log live in
named Docker volumes, not Git. `.env` is neither mounted nor in the image build
context. The container has a 4 CPU / 4 GiB memory limit. Exit the shell to remove
the temporary container; named volumes remain. Do not remove volumes if you want
to preserve builds and logs.

The checked-in `raptor.urdf` contains an old PC's absolute controller YAML path.
Regenerate from Xacro in the current environment and spawn that generated file.
Do not infer a controller initialization fix from this path finding alone.

GUI forwarding is not configured here. Installed RViz/Gazebo GUI packages do not
mean a Mac-visible GUI has been verified. `hold_joints=true` also means an upright
model by itself is not evidence of controlled standing.

## Headless reproduction (inside the container shell)

```bash
source install/setup.bash
xacro src/raptor_description/urdf/raptor.urdf.xacro > /tmp/raptor.urdf
check_urdf /tmp/raptor.urdf
gz sim -s -r src/raptor_description/worlds/raptor_world.sdf > /tmp/gazebo.log 2>&1 &
gz_pid=$!
ros2 run robot_state_publisher robot_state_publisher /tmp/raptor.urdf > /tmp/rsp.log 2>&1 &
rsp_pid=$!
ros2 run ros_gz_bridge parameter_bridge '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock' > /tmp/clock.log 2>&1 &
clock_pid=$!
trap 'kill "$gz_pid" "$rsp_pid" "$clock_pid" 2>/dev/null; wait' EXIT
ros2 run ros_gz_sim create -world raptor_world -file /tmp/raptor.urdf -name raptor -z 0.5
ros2 node list
ros2 service list
# Proceed only after list_hardware_interfaces appears in the service list.
ros2 control list_hardware_interfaces -c /controller_manager
ros2 run controller_manager spawner joint_state_broadcaster -c /controller_manager
ros2 run controller_manager spawner raptor_joint_controller -c /controller_manager
ros2 control list_controllers -c /controller_manager
ros2 topic echo /joint_states --once
```

## Verified on 2026-09-27

- M5 / 24 GB host, Docker Linux aarch64; Ubuntu 24.04, Jazzy, Gazebo 8.15.0.
- Fresh `colcon build --symlink-install`: both packages passed.
- Generated URDF passed `check_urdf`: base_root, body, two legs and tail.
- 10 revolute joints match the 10 ros2_control joints and controller joint list.
- Headless entity creation succeeded; controller_manager and gz_ros_control ran.
- Hardware query returned 10 available position commands and 20 state interfaces.
- Both joint_state_broadcaster and raptor_joint_controller became active.
- Received joint_states with 10 finite positions/velocities. Effort is NaN because
  no effort state interface is declared; no effort measurement is claimed.
- Received ROS /clock after adding the bridge; no new missing-clock warnings
  appeared during the subsequent observation.
- Previous robot_description initialization stall did not reproduce. Its original
  root cause remains unconfirmed; the old PC and package versions were not tested.
- No commanded motion, ground-contact validation, GUI, standing or walking has
  been verified. Robot geometry and controller YAML were not changed.

## Backups

Review and commit changes, then push the current branch to GitHub without force:

```sh
git push -u origin HEAD
```

HF is backup storage only: private dataset `dannykim123/ROS_RAPTER-backup`.
No Space, GPU, or paid compute is needed. Install `huggingface_hub` in a Python
virtual environment, then run `python scripts/backup_hf.py` from that environment.
It reads the local `.env` token without printing it and uploads committed Git
history as `raptor.bundle`. It refuses dirty trees and excluded paths in history.
Review commits for secrets before uploading; path checks are not a general secret
scanner. No scheduled backup is configured: run backups after verified commits.

To restore, download `raptor.bundle` using your authenticated HF account and run:

```sh
git clone raptor.bundle raptor-restored
```

The bundle includes branches and Git history, not `.env`, Docker images, build
outputs, or untracked files. Rebuild the environment from the Dockerfile.
