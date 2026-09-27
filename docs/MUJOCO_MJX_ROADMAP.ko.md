# MuJoCo·MJX 강화학습 후속 작업

사용자 요청(2026-09-27): ROS 2·Gazebo 작업을 먼저 진행한 뒤 MuJoCo+MJX에서
걷기·균형 강화학습을 하고 정책을 ROS 제어와 연결한다. 현재 문서는 계획이며,
MuJoCo 설치·모델 변환·학습을 완료했다는 보고가 아니다.

## 기준과 순서

1. Gazebo에서 10 active DOF, 수동 발가락·꼬리 구분, 접촉·IMU, 기립 및 보행 기준 측정.
2. 해당 Xacro의 관절축·질량·관성·제한·collision·수동 spring을 MuJoCo MJCF로 변환.
   임의로 다른 로봇으로 교체하지 않는다. floating base를 추가하더라도 actuator는 10개.
3. 무중력 관절축, 중력 낙하, 지면 접촉, 정적 자세를 두 엔진에서 비교한다.
   엔진 간 차이를 동일한 물리라고 주장하지 않는다.
4. CPU 단일 환경 → MJX-JAX 소규모 vmap batch의 처리량·메모리·안정성을 측정한다.
5. 기립 → 작은 속도 추종 → 평지 보행 → 작은 단차·경사 curriculum 순서로 학습한다.
   보상은 속도 추종·넘어짐·미끄럼·발 접촉·동작 크기·에너지/관절 제한을 함께 평가한다.
6. 여러 seed와 미사용 지형에서 predefined controller와 비교한다. 높은 reward만으로
   보행 성공이라고 판단하지 않는다. 낙상률, 추종 오차, 미끄럼, 지형별 성공률을 기록한다.
7. ROS mission gate와 STOP 우선순위를 유지한 정책 어댑터로 Gazebo에서 재검증한다.
   Qwen/NanoJev가 직접 motor action을 생성하는 구조로 바꾸지 않는다.

## 실제 Mac과 가속 지원

확인된 Mac은 Apple M5 / 24GB RAM이며 M5 Pro로 확인된 것은 아니다.
[MuJoCo MJX 공식 문서](https://mujoco.readthedocs.io/en/latest/mjx.html)는
MJX-JAX 실행 대상으로 Apple Silicon을 포함한다. 다만
[JAX 설치 문서](https://docs.jax.dev/en/latest/installation.html)는 Apple GPU 지원을
experimental로 구분하고 macOS의 일반 설치 경로로 CPU를 안내한다.
따라서 Apple Silicon에서 실행 가능하다는 사실을 M5 GPU 가속 보장으로 해석하지 않는다.
설치 시 실제 `jax.devices()`와 workload benchmark를 기록한다. MJX-Warp의 NVIDIA
최적화 경로와 MJX-JAX 경로도 구분한다.

기존 AI 가상환경과 분리해 버전을 고정한다. 대규모 병렬 환경이나 유료 클라우드
자원을 임의로 생성하지 않는다. 학습 코드·환경 설정·seed·평가 결과는 Git,
큰 checkpoint는 비공개 HF 백업에 저장하고 복원도 검증한다.
