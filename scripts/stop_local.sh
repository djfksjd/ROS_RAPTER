#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/.."
owner=$(docker inspect --format '{{index .Config.Labels "com.docker.compose.project"}}' raptor-dev)
service=$(docker inspect --format '{{index .Config.Labels "com.docker.compose.service"}}' raptor-dev)
if [[ "$owner" != raptor || "$service" != ros ]]; then
  echo 'Refusing to stop a container not owned by this project.' >&2
  exit 1
fi
docker stop -t 10 raptor-dev
docker rm raptor-dev
