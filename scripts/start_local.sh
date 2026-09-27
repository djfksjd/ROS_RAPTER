#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/.."
docker info >/dev/null
if docker container inspect raptor-dev >/dev/null 2>&1; then
  echo 'raptor-dev already exists. Use bash scripts/stop_local.sh before a fresh session.' >&2
  exit 1
fi
docker compose build
docker compose run --rm ros bash -c 'colcon build --symlink-install'
docker compose run -d --service-ports --name raptor-dev ros bash /raptor_ws/src/raptor_control/scripts/session.sh "$@"
sleep 2
if [[ $(docker inspect --format '{{.State.Running}}' raptor-dev) != true ]]; then
  docker logs raptor-dev >&2
  exit 1
fi
echo 'Gazebo desktop: http://127.0.0.1:6080/vnc.html?autoconnect=true&resize=scale'
