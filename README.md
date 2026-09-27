# ROS_RAPTER

## 2026-09-27 Mac 로컬 검증 업데이트

현재 개발 브랜치: `feature/passive-toes`. 아래 기존 진행표는 이전 PC의 기록이며,
최신 실제 검증은 다음 문서를 기준으로 확인합니다.

- [ROS 2·Gazebo 설치 완료 보고서와 실제 화면](docs/INSTALLATION_REPORT.ko.md)
- [시뮬레이션 주제 보고서·외형·Qwen/NanoJev 비교](docs/PROJECT_REPORT.ko.md)
- [수동 발가락·단차·경사 접촉 검증](docs/PASSIVE_TOES.ko.md)
- [MuJoCo·MJX 강화학습 후속 순서](docs/MUJOCO_MJX_ROADMAP.ko.md)
- [로컬 실행·AI 명령·백업 복원](docs/local-development.md)

10축 제어, 제한된 static standing, 센서, 두 모델의 ROS 명령 연결을 확인했습니다.
안정적인 보행·탐색 mission·유연 꼬리 동역학은 미완료입니다.


## Project Overview
운영자의 자연어 명령을 입력받아 ROS 2 기반 랩터형 2족 로봇의 행동 명령으로 변환하고 시뮬레이션 환경에서 제어하는 10축 지상 탐사 로봇 시뮬레이션 프로젝트입니다. 로봇은 독자적으로 임의 판단을 내리지 않으며, 항상 운영자의 통제 하에 사전 정의된 안전 범위 내에서 동작합니다.

## Research Goal
- 운영자의 자연어 명령을 로봇의 정형화된 동작 명령으로 변환하는 파이프라인 검증
- 10-DOF(자유도) 랩터형 2족 로봇 모델을 ROS 2 Jazzy 및 Gazebo Harmonic 시뮬레이터 상에 구현
- 향후 대규모 언어 모델(Qwen/Ollama)과 경량 모델(NanoJev) 간의 명령 이해도 및 응답 속도 비교 평가
- 탐사 시나리오에서의 명령 전달 정확도와 시뮬레이션 제어 안정성 검증

## Robot Configuration
로봇은 양다리 각 4축과 균형 제어를 위한 꼬리 2축으로 구성된 총 10개의 Active DOF를 가집니다.

| Component | DOF | Motion | Joint Name | Joint Limit |
|---|---|---|---|---|
| Left Hip Roll | 1 | 다리 좌우 벌림 (Roll, X축) | `left_hip_roll_joint` | -0.50 ~ 0.50 rad |
| Left Hip Pitch | 1 | 허벅지 앞뒤 회전 (Pitch, Y축) | `left_hip_pitch_joint` | -1.20 ~ 0.80 rad |
| Left Knee Pitch | 1 | 무릎 굽힘 (Pitch, Y축) | `left_knee_pitch_joint` | 0.00 ~ 2.20 rad |
| Left Ankle Pitch | 1 | 발목 앞뒤 굽힘 (Pitch, Y축) | `left_ankle_pitch_joint` | -0.70 ~ 0.70 rad |
| Right Hip Roll | 1 | 다리 좌우 벌림 (Roll, X축) | `right_hip_roll_joint` | -0.50 ~ 0.50 rad |
| Right Hip Pitch | 1 | 허벅지 앞뒤 회전 (Pitch, Y축) | `right_hip_pitch_joint` | -1.20 ~ 0.80 rad |
| Right Knee Pitch | 1 | 무릎 굽힘 (Pitch, Y축) | `right_knee_pitch_joint` | 0.00 ~ 2.20 rad |
| Right Ankle Pitch | 1 | 발목 앞뒤 굽힘 (Pitch, Y축) | `right_ankle_pitch_joint` | -0.70 ~ 0.70 rad |
| Tail Yaw | 1 | 꼬리 좌우 회전 (Yaw, Z축) | `tail_yaw_joint` | -0.80 ~ 0.80 rad |
| Tail Pitch | 1 | 꼬리 상하 회전 (Pitch, Y축) | `tail_pitch_joint` | -0.60 ~ 0.60 rad |

## Previous PC Status (historical)

| Item | Status | Notes |
|---|---|---|
| ROS 2 Jazzy | Complete | Ubuntu 24.04 LTS 환경에서 ROS 2 Jazzy 기반 워크스페이스 구성 완료 |
| Gazebo Harmonic | Complete | Gazebo Sim 8.15.0 설치 및 `raptor_world.sdf` 물리 환경 구동 확인 |
| 10-DOF URDF | Complete | 10축 revolute joint, inertia, collision geometry를 포함한 Xacro/URDF 작성 완료 |
| RViz visualization | Complete | `display.launch.py`를 통해 robot_state_publisher, joint_state_publisher_gui, RViz2 연동 확인 |
| Gazebo spawn | Complete | `ros_gz_sim create` 명령을 통해 Gazebo 월드 내 로봇 모델 스폰 및 중력/충돌 확인 |
| ros2_control | Blocked | `gz_ros2_control` 플러그인 로드 후 controller_manager가 robot_description 토픽 대기로 멈춤 |
| Standing pose | Not Started | 제어기 초기 위치 유지 및 기립 제어 파라미터 미구현 |
| Walking gait | Not Started | 보행 패턴 생성 및 동역학 보행 알고리즘 미구현 |
| Camera / IMU | Not Started | 시뮬레이션 센서 xacro 링크 및 플러그인 미구현 |
| NanoJev | Not Started | 모델 연동 코드 및 액션 매핑 레이어 미구현 (향후 과제) |

## Previous PC Issue (not reproduced locally)
- **ros2_control 하드웨어 인터페이스 초기화 블로킹**:
  - `raptor.urdf.xacro`에 `gz_ros2_control::GazeboSimROS2ControlPlugin`을 적용하고 Gazebo에서 로봇을 스폰할 때, `/controller_manager` 및 `/gz_ros_control` 노드는 정상 생성됩니다.
  - 그러나 실행 로그상에서 다음과 같은 경고가 지속 발생합니다:
    ```text
    [WARN] [controller_manager]: Waiting for data on 'robot_description' topic to finish initialization
    [WARN] [gz_ros_control]: Waiting RM to load and initialize hardware...
    ```
  - Gazebo 내부의 `controller_manager`가 `/robot_description` 토픽 수신을 대기하면서 Resource Manager(RM)가 10개 관절 하드웨어 인터페이스를 로드/초기화하지 못하고 정체됩니다.
  - 이로 인해 `ros2 control list_hardware_interfaces` 서비스 호출 시 인터페이스가 조회되지 않거나 응답하지 않는 현상이 발생하고 있습니다.

## Package Structure
```text
raptor_ws/
├── .gitignore
├── README.md
└── src/
    ├── raptor_description/
    │   ├── CMakeLists.txt
    │   ├── CMakeLists.txt.backup
    │   ├── package.xml
    │   ├── launch/
    │   │   └── display.launch.py
    │   ├── urdf/
    │   │   ├── raptor.urdf
    │   │   ├── raptor.urdf.xacro
    │   │   ├── raptor.urdf.xacro.backup_4dof
    │   │   └── raptor.urdf.xacro.backup_8dof
    │   └── worlds/
    │       └── raptor_world.sdf
    └── raptor_control/
        ├── CMakeLists.txt
        ├── package.xml
        └── config/
            └── controllers.yaml
```

## Build
현재 워크스페이스에서 검증된 빌드 절차입니다:

```bash
# 1. ROS 2 환경 로드
source /opt/ros/jazzy/setup.bash

# 2. 워크스페이스 이동 및 빌드
cd ~/raptor_ws
colcon build --symlink-install

# 3. 워크스페이스 환경 반영
source install/setup.bash
```

## Run
현재 실제 코드 및 환경에서 검증된 실행 명령어입니다:

### 1. RViz2 및 관절 조작 GUI 실행 (로봇 모델 검증)
```bash
source /opt/ros/jazzy/setup.bash
source ~/raptor_ws/install/setup.bash
ros2 launch raptor_description display.launch.py
```

### 2. Gazebo 시뮬레이션 환경 실행
```bash
source /opt/ros/jazzy/setup.bash
source ~/raptor_ws/install/setup.bash
gz sim $(ros2 pkg prefix raptor_description)/share/raptor_description/worlds/raptor_world.sdf
```

### 3. Gazebo 내 로봇 모델 스폰
```bash
source /opt/ros/jazzy/setup.bash
source ~/raptor_ws/install/setup.bash
ros2 run ros_gz_sim create -file $(ros2 pkg prefix raptor_description)/share/raptor_description/urdf/raptor.urdf -name raptor -z 0.5
```

## Development Roadmap
우선순위에 따른 단계별 개발 계획입니다:

- **Phase 1: ros2_control 연동 정상화**
  - Gazebo-ros2_control 간 `robot_description` 전달 경로 해결
  - 10개 joint hardware interface 인식 확인 (`ros2 control list_hardware_interfaces`)
  - `raptor_joint_controller` (JointTrajectoryController) 활성화
- **Phase 2: Standing Pose 및 기본 자세 제어**
  - 기립 기준 관절각(Target Joint Position) 도출
  - 조인트 위치 제어(Position Control) 및 PID 게인 튜닝
  - 지면 접촉 시 정적 균형 유지 테스트
- **Phase 3: 기본 보행(Walking Gait) 생성**
  - 꼬리(Tail)를 활용한 무게중심(CoM) 보정 궤적 생성
  - 1보 전진 궤적 및 교차 발구름 패턴 테스트
  - 연속 보행 시뮬레이션 안정화
- **Phase 4: 센서 시스템 연동**
  - IMU 센서 추가 (몸통 기울기 및 가속도 계측)
  - RGB-D 카메라 추가 (전방 시야 확보)
  - 양 발 접촉 센서(Contact Sensor) 모델링
- **Phase 5: 자연어 명령 이해 파이프라인 연동**
  - Ollama 기반 Qwen 베이스라인 프롬프트 엔지니어링
  - 경량 NanoJev 액션 선택 모델 프로토타입 연동
  - 정형화된 Action ID와 ROS 2 토픽/서비스 간 브릿지 구현
- **Phase 6: 비교 평가 및 최종 데모**
  - Qwen vs NanoJev 간 명령 추론 정확도 및 추론 지연 시간(latency) 측정
  - 시뮬레이션 내 지상 탐사 미션 통합 시연

## AI Command Understanding
프로젝트에서 AI는 저수준 모터 전류/토크를 직접 제어하지 않으며, 운영자의 상위 자연어 명령을 정형화된 Action ID로 변환하는 상위 인터프리터 역할을 담당합니다.

### 제어 아키텍처 흐름
```text
Operator (운영자)
       │
       ▼ [자연어 명령 입력] (예: "산 동쪽을 수색해")
Qwen / NanoJev (명령 파서)
       │
       ▼ [구조화된 액션 출력] (예: SEARCH_EAST)
Safety Gate (안전 필터) ──▶ 비정상/비허용 명령 차단
       │
       ▼ [허가된 액션 전달]
ROS 2 Mission / Motion Controller
       │
       ▼ [관절 각도 궤적 생성]
10-DOF Raptor Robot (Gazebo Sim)
```

## Planned Command Set
현재 시스템에서 지원 예정인 정형화 명령 후보군입니다:

| Command Name | Description | Status |
|---|---|---|
| `SEARCH_EAST` | 동쪽 영역 탐사 및 이동 | Planned |
| `SEARCH_WEST` | 서쪽 영역 탐사 및 이동 | Planned |
| `SEARCH_NORTH` | 북쪽 영역 탐사 및 이동 | Planned |
| `SEARCH_SOUTH` | 남쪽 영역 탐사 및 이동 | Planned |
| `MOVE_EAST` | 동쪽 방향 단순 이동 | Planned |
| `MOVE_WEST` | 서쪽 방향 단순 이동 | Planned |
| `RETURN_BASE` | 출발 지점(베이스)으로 복귀 | Planned |
| `PAUSE` | 현재 동작 일시 정지 및 자세 유지 | Planned |
| `RESUME` | 중단된 동작 재개 | Planned |
| `STOP` | 모든 구동 즉시 정지 (최우선 순위 긴급 정지) | Planned |

## Sensors
현재 실제 코드 기준 센서 구현 상태입니다:

| Sensor | Purpose | Implementation Status | Notes |
|---|---|---|---|
| Joint Encoders | 10축 위치 및 속도 피드백 | In Progress | ros2_control state_interface 선언 완료, RM 연동 대기 중 |
| IMU | 로봇 자세(Roll/Pitch/Yaw) 추정 | Not Implemented | 향후 base_link에 장착 예정 |
| RGB-D Camera | 전방 지형 탐색 및 장애물 감지 | Not Implemented | 향후 헤드/전방부에 추가 예정 |
| Foot Contact Sensor | 지면 접촉 감지 및 보행 상태 전이 | Not Implemented | 양 발 링크 충돌 감지 플러그인 예정 |
| LiDAR | 주변 3차원 포인트클라우드 계측 | Optional / Planned | 필요 시 추가 검토 |
| Thermal Camera | 탐사 대상 열원 감지 | Optional / Planned | 필요 시 추가 검토 |

## Safety
- **운영자 통제권 보장**: 모든 미션 동작은 운영자의 명령에 의해 시작되며, 독자적인 임의 기동은 불가합니다.
- **최우선 긴급 정지 (`STOP`)**: 운영자의 `STOP` 명령은 모든 진행 중인 동작 및 궤적 생성보다 최우선 순위로 처리됩니다.
- **화이트리스트 기반 액션 수용**: AI가 출력하는 Action ID는 사전에 엄격히 정의된 명령 세트에 포함될 때만 통과됩니다.
- **Safety Gate 적용 예정**: 비정상적인 관절 각도 명령, 급격한 가속도 요구, 허용 범위를 벗어난 명령은 안전 검증 계층에서 차단됩니다.

## Limitations
- **시뮬레이션 전용**: 실제 하드웨어 로봇이 아닌 Gazebo Harmonic 시뮬레이터 환경에서만 동작이 검증되고 있습니다.
- **보행 제어 미완성**: 10축 동역학 보행 제어기 및 실시간 균형 유지 알고리즘이 아직 구현되지 않았습니다.
- **환경 한계**: 평탄한 평면 지형(`raptor_world.sdf`) 외의 비정형 험지나 실제 야외 환경에서의 물리 검증은 이루어지지 않았습니다.
- **기하 구조 한계**: 현재 URDF 외형은 기본 primitive box 형상으로 이루어진 초기 프로토타입 상태입니다.

## References
- ROS 2 Jazzy: https://docs.ros.org/en/jazzy/
- Gazebo Harmonic: https://gazebosim.org/docs/harmonic/
- ros2_control: https://control.ros.org/jazzy/
- gz_ros2_control: https://github.com/ros-controls/gz_ros2_control
- NanoJev: Planned / future integration
- Ollama / Qwen: Planned / future integration

## Repository Restore Verification
- **Fresh Clone Tested**: Verified complete build from clean environment
- **Environment**: ROS 2 Jazzy (Ubuntu 24.04 LTS)
- **colcon build**: Passed (`colcon build --symlink-install` without errors)
- **Xacro Parsing**: Passed (`xacro raptor.urdf.xacro` parses cleanly into valid URDF)
