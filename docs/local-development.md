# Local Mac development

Verified on 2026-09-27: M5 / 24GB, Docker Linux ARM64, Ubuntu 24.04.5,
ROS 2 Jazzy, Gazebo Harmonic 8.15.0. See the
[installation report](INSTALLATION_REPORT.ko.md) and [project report](PROJECT_REPORT.ko.md)
for actual screenshots, measurements and remaining limitations.

## Start and stop

Start Docker Desktop, then run at the repository root:

```bash
bash scripts/start_local.sh
```

Open `http://127.0.0.1:6080/vnc.html?autoconnect=true&resize=scale`.
The script builds both ROS packages and starts Gazebo, controllers, sensors,
mission gate and a software-rendered desktop. A running container alone does not
prove controller readiness; verify:

```bash
docker exec raptor-dev /ros_entrypoint.sh ros2 control list_controllers
docker exec raptor-dev /ros_entrypoint.sh ros2 control list_hardware_interfaces
docker exec raptor-dev /ros_entrypoint.sh ros2 topic echo /joint_states --once
```

Both controllers must be `active`. To open RViz in the same desktop:

```bash
docker exec -d raptor-dev bash -c 'source /opt/ros/jazzy/setup.bash; source install/setup.bash; rviz2 -d src/raptor_description/rviz/raptor.rviz > /raptor_ws/log/session/rviz.log 2>&1'
```

Stop this project's container:

```bash
bash scripts/stop_local.sh
```

A fresh session resets the STOP latch; restart only as an intentional operator
action. Other Docker containers and persistent volumes are not removed.
The container is limited to 4 CPU / 4GiB. ROS processes share domain 42 and
localhost inside this container. The GUI port binds only to Mac localhost.
Source is mounted read-only; edit it on the Mac. Build/install/log live in named
Docker volumes. `.env` is neither mounted nor included in the Docker build context.

The launch file regenerates Xacro: the old checked-in `raptor.urdf` contains an
obsolete absolute path and is not the runtime input. The previous PC's initialization
stall did not reproduce; its original cause remains unconfirmed.

For optional passive toes and isolated contact experiments, follow
[the passive-toe report](PASSIVE_TOES.ko.md). `start_local.sh --experiment ...`
accepts custom ROS launch arguments and disables the operator mission gate.
Do not send AI commands in that development mode.

## AI setup and commands

The existing local `.venv-ai`, NanoJev inputs and Ollama model are already installed.
For a new machine, use Python 3.11+ and install the pinned environment:

```bash
python3 -m venv .venv-ai
.venv-ai/bin/python -m pip install -r ai/requirements.txt
.venv-ai/bin/python scripts/setup_ai.py
brew install ollama
```

Start Ollama in a separate terminal (not required for NanoJev):

```bash
OLLAMA_HOST=127.0.0.1:11434 ollama serve
```

For a new installation, `ollama pull qwen3:0.6b` downloads the current tag. The
exact tested manifest and pinned NanoJev revisions are recorded in `ai/models.json`;
use the private model backup for exact Qwen restoration if the tag changes.
NanoJev currently uses Apple MPS FP32, so this CLI setup targets Apple Silicon.

Preview only by default:

```bash
.venv-ai/bin/python ai/command.py '산 동쪽을 수색해'
.venv-ai/bin/python ai/command.py '제자리에서 기립해' --backend nanojev --adapter ai/artifacts/raptor-head.safetensors
```

Explicitly publish to the running simulation:

```bash
.venv-ai/bin/python ai/command.py '제자리에서 서 있어' --execute
.venv-ai/bin/python ai/command.py '제자리에서 기립해' --backend nanojev --adapter ai/artifacts/raptor-head.safetensors --execute
.venv-ai/bin/python ai/command.py --stop
docker exec raptor-dev tail -n 20 /raptor_ws/log/session/mission.log
```

The CLI's successful publication is not an execution acknowledgement: check the
matching request ID and `joint_target_reached` in the gate log. This status measures
joint tracking, not whole-body balance. Only STAND/PAUSE/RESUME/STOP are enabled;
navigation and search are rejected until walking is validated. STOP bypasses AI
and latches a simulated position hold. It is not a hardware power-off mechanism.
Do not run motion/gait development probes concurrently with this operator session.

Run the small fixed evaluation and policy tests:

```bash
.venv-ai/bin/python -m unittest discover -s tests -v
.venv-ai/bin/python ai/evaluate.py --backend qwen --holdout --output /tmp/qwen-evaluation.json
.venv-ai/bin/python ai/evaluate.py --backend nanojev --adapter ai/artifacts/raptor-head.safetensors --holdout --output /tmp/nanojev-evaluation.json
```

The published 38-command set has now been examined; use a new independent set for
future model selection. The small trained head and its 108-example training record
are in `ai/artifacts`. Original multi-GB weights stay outside Git.

## Modeling and evidence

`modeling/raptor-assembly.blend` contains named component objects.
`modeling/build_visuals.py` regenerates link-local COLLADA and a presentation render:

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --python modeling/build_visuals.py
```

The bent-leg render is an appearance prototype, not a validated standing pose.
Detailed visuals are optional in Xacro; collision/inertia and the 10 active joints
remain unchanged. The eight distal tail segments are fixed visuals, not compliant
physics. RViz currently renders these materials dark; Gazebo colors were verified.
See [evidence index](evidence/README.md) for measured results and failure records.

## Backups and restoration

GitHub holds reviewed source, small task weights, meshes, reports and evidence.
HF is backup storage only: private dataset `dannykim123/ROS_RAPTER-backup`.
No Space or paid compute is created. The token stays in ignored `.env` (mode 600).

After reviewing and committing changes:

```bash
git push -u origin HEAD
.venv-ai/bin/python scripts/backup_hf.py
```

The script rejects a dirty tree and sensitive/generated paths in Git history,
then uploads `raptor.bundle`. These path checks do not replace secret review.
For updated model inputs only, `scripts/backup_assets_hf.py` uploads an allowlist of
pinned NanoJev files, Ollama manifest/blobs and the upstream source archive; it
records remote sizes and available SHA-256 hashes in
[evidence/model-backup-assets.json](evidence/model-backup-assets.json).

To restore, authenticate to the private HF dataset and download `raptor.bundle`, then:

```bash
git clone raptor.bundle raptor-restored
git -C raptor-restored checkout feature/passive-toes
```

The bundle includes Git branches/history, not `.env`, Docker images, virtualenvs
or build outputs. Rebuild Docker and Python environments using the files above.
Restore HF `models/NanoJev/` into the checkout's `models/NanoJev/`; obtain the pinned
upstream code with `scripts/setup_ai.py` or extract the backed-up source archive to
`vendor/NanoJev/`. Restore HF `models/ollama/` under `~/.ollama/models/` while Ollama
is stopped, preserving existing unrelated models. Verify restored file hashes
against `model-backup-assets.json`. No automatic or scheduled backup is configured.
