#!/usr/bin/env bash
# Generate the runtime URDF with xacro inside the existing ROS image (no simulation started).
# Usage: bash sim/generate_urdf.sh OUTPUT.urdf [xacro args...]
# For offline analysis/MuJoCo only: its ros2_control parameter path points to a temporary prefix.
set -euo pipefail
out="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; shift
root="$(cd "$(dirname "$0")/.." && pwd)"
docker run --rm --platform linux/arm64 -v "$root/src:/raptor_ws/src:ro" -v "$(dirname "$out"):/out" \
  raptor:jazzy-local bash -c '
    source /opt/ros/jazzy/setup.bash
    # $(find raptor_control) only needs an ament index entry pointing at the read-only sources.
    mkdir -p /tmp/pfx/share/ament_index/resource_index/packages
    for p in raptor_control raptor_description; do
      touch /tmp/pfx/share/ament_index/resource_index/packages/$p; ln -s /raptor_ws/src/$p /tmp/pfx/share/$p
    done
    AMENT_PREFIX_PATH=/tmp/pfx:$AMENT_PREFIX_PATH xacro /raptor_ws/src/raptor_description/urdf/raptor.urdf.xacro "$@" \
      > "/out/$0"' "$(basename "$out")" "$@"
echo "wrote $out"
