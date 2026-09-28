# 체크포인트

## 2026-09-28 외형 단계 (최신)

디지티그레이드 다리 설계와 morphloom 외형 완료(review 초안). DIGITIGRADE_APPEARANCE.ko.md.
재개 지점: Gazebo에서 digitigrade 흔들기 재현 또는 외형 개선. Gazebo 실행은 --experiment leg_design:=digitigrade.

## 2026-09-28 재개 결과

재개 첫 작업 2번을 기록·소스·오프라인 계산으로 수행했다. 상세 LATERAL_SUPPORT_FEASIBILITY.ko.md.
- physics 1ms/DART 6.13.2, 제어 100Hz, position 경로 = 10ms 지연 위치의 P 속도 명령(유효 gain 30/s).
- 54번 발목 속도 한도 표본 다수는 명령 포화로 설명되지 않음. shift 중 양발 모서리 접촉 확인.
- 평평한 발 정적 한 발 지지: 전체 관절 한도에서도 측방 COM 여유 음수(-0.008m).
- 떨림 기전(접촉+속도 constraint vs 지연 servo 진동)은 미확정. 1ms 단위 계측이 필요하나
  방향 결정 전에는 우선순위가 낮다.

사용자가 A(동적 흔들기 보행, 10축 유지)를 선택했다. 라이선스/revision은 기존 조사 문서를 재사용했다.
MuJoCo 모델 대응과 정적 기립 비교까지 완료: MUJOCO_MJX_ROADMAP.ko.md 진행 기록.
open-loop 흔들기 시험 완료(58): 교대 지지는 μ1에서만 안정, 보폭 전진은 깨끗한 걸음 미확립.
closed-loop 단계 완료(59, 60): 교대 지지 견고, 깨끗한 걸음은 servo kv 가정(30)에서만.
Gazebo servo 식별 완료(61, 62): DART 발목 포화 결함, kv는 고정 베이스로 결정 불가. SIMULATION_METHODOLOGY.ko.md.
다음 재개 지점: actuator 후보 사양 범위 확정(사용자) → MuJoCo actuator 모델 교체·범위 평가. 실험 컨테이너는 정지 상태.

## 이전 일시정지 체크포인트 — 2026-09-27

사용자 요청: 다음 체크포인트까지만 진행하고 잠시 중단, 모두 백업.

## 마지막 완료 작업

브랜치 `test/joint-load-observation`.
10개 position command interface를 유지하고 10개 읽기 전용 effort state를 추가했다.
관절 속도와 전달 effort를 시간 연속 계측한다. 제어 gain/토크 한도 변경 없음.
53번은 인터페이스, 54번은 toe stiffness scale=3 비교 계측 결과다.

설치: gz_ros2_control 1.2.20-1noble.20260904.032933, 실제 position gain .3.
[동일 버전 소스](https://github.com/ros-controls/gz_ros2_control/blob/1.2.20/gz_ros2_control/src/gz_system.cpp)
에서 position 오차를 update_rate와 gain으로 속도 목표로 변환하는 경로를 확인했다.
effort는 JointTransmittedWrench의 관절축 투영이며 모터 명령 토크와 동일하다고 단정하지 않는다.

54번: 80개 표본, 버퍼 초과 0. shift 중 왼발목 속도가 ±2.5rad/s에 반복 도달.
실험은 tracking error .14852rad로 실패했다. 전달 effort 일부 표본은 37.33Nm였지만,
관측만으로 모터 토크 포화나 유일한 원인을 확정하지 않는다.
강성 기본값은 1, gain .3, 속도 제한 2.5, effort 제한 80을 유지했다.

## 재개 첫 작업

1. git status, WORK_STATE 확인. 이 체크포인트에서 새 실험을 자동 재시작하지 않는다.
2. 발목 속도 포화와 접촉/수치 진동을 구분: 실제 SDF/physics 시간 간격과 설치된 physics 구현 확인.
3. 새 근거에 따라 한 변수 비교. 무작정 gain/토크/속도 한도를 높이지 않는다.
4. 이후 안정 지지 → 보행 → 지형 → MuJoCo/MJX → AI mission/evaluation 순서를 유지.

## 남은 상태

외형은 참고 이미지에 아직 미달. 안정 보행/급경사 대응/강화학습/탐색 mission은 미완료.
Qwen/NanoJev 기본 STAND/STOP 및 ROS 환경 복구는 기존 검증 기록에 있다.
프로젝트 Docker 컨테이너와 이번 세션의 Ollama 서버를 종료한 상태로 일시정지한다.
다른 사용자 Docker 자원과 앱은 종료하지 않는다.

## 백업 범위

GitHub: 소스, Blender 원본, 메시, 문서, 실제 검증 기록과 개발 브랜치.
비공개 Hugging Face: 최신 전체 Git bundle 및 기존 Qwen/NanoJev 모델 자산.
.env/토큰/가상환경/build/install/log는 백업 제외. 재현 지침은 저장소에 있다.
