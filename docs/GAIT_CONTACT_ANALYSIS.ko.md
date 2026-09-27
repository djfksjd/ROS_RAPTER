# 보행 접촉 계측 — 2026-09-27

브랜치 `fix/gait-contact-evidence`. 기존 보행 궤적을 교체하지 않고 계측부터 보완했다.

## 계측 문제

기존 `recent`는 최근 Contacts 메시지 수신 여부였으며 비어 있는 배열도 참이 될 수 있었다.
`recent_messages`와 `active_sources`를 구분하고 실제 접촉 개수·위치·충돌체 이름,
원시 힘, 수신 경과 시간, 능동/수동 관절 위치를 기록한다.
원시 body1 힘은 좌표계와 충돌 순서의 정규화 없이 지면 반력으로 해석하지 않는다.

`analyze_gait.py`는 실제 생성 URDF의 질량·관절축·관성 원점과 기록한 관절/모델 자세로
무게중심과 발 collision의 최저 z를 계산한다. 수동 관절이 없거나 오래되면 거부한다.
평지 z=0 전용이며, 모델 pose와 ROS 표본은 동기화되지 않았으므로 정밀 안정성 인증이 아니다.
접촉점의 y 최솟값/최댓값에 더해 convex hull의 signed margin을 계산한다.
음수는 COM 투영이 그 표본의 접촉 영역 바깥이라는 뜻이다.
표본 비동기·힘 정규화·동적 관성 한계로 실제 ZMP 안정성 판정과 동일하지 않다.

## 실험

동일 모델, 수동 발가락 사용, 좌우 roll 이동량 외 제어 인수 동일, lift=0.95, cycles=0.
각 실행은 초기화된 새 Gazebo 세션에서 수행했다.

- 41번: roll=0.28. lift/swing 구간에도 양발의 실제 접촉이 남음. 궤적 완료는 보행 성공 아님.
- 42~43번: roll=0.39. shift 후 오른발 collision 최저 z 약 0.0156m,
  오른발 최근 접촉 없음. 다만 COM y=0.2705m는 왼발 접촉 y=0.2300~0.2512m보다 바깥.
  이후 lift에서 총 회전각 1.20rad로 실패. 한 발 안정 지지 성공이 아님.

단계 끝의 자세 검사만으로는 동작 중 전도를 조기에 중단하지 못했다.
보행을 운영 mission으로 활성화하지 않으며 연속 IMU 감시가 후속 작업이다.

## 재현

```bash
bash scripts/stop_local.sh
RAPTOR_PASSIVE_TOES=true bash scripts/start_local.sh --experiment
# 컨테이너에서 ROS 환경 source 후:
python3 /raptor_ws/src/raptor_control/scripts/gait_probe.py --roll .39 --lift .95 --cycles 0
```

분석 입력 URDF는 실행한 passive_toes=true 모델의 xacro 출력이다.
`python analyze_gait.py MODEL.urdf REPORT.json`으로 저장된 표본을 분석한다.

## 중간 이동량과 2차원 접촉 영역

44~45번 roll=0.36: shift 후 오른발 최저 z=0.0078m, 오른발 최근 접촉 없음.
COM y=0.2391m는 왼발 y 범위 안이지만, XY convex hull margin은 -0.00529m다.
roll=0.39의 동일 margin은 -0.03663m다. 중간값도 lift에서 전도했다.
이 결과는 yaw/roll 기울기와 발 모서리 접촉을 무시한 y 구간 비교가 부족함을 보인다.
현재 판정은 특정 표본 분석이며 유일한 전도 원인을 증명한 것은 아니다.

다음은 연속 IMU 감시 후 앞뒤 COM/접촉 기하 분석이다. roll 값만 계속 탐색하지 않는다.
분석 함수는 정사각형 내부(+0.5m), 외부(-0.2m), 퇴화 선분(None) sanity 검사 통과.
기존 SafetyTests 7개도 통과했다.

## 동작 중 IMU 중단 감시

`motion_guard.py`는 world 기준 quaternion에서 중력축 기울기를 계산한다.
yaw는 기울기로 세지 않는다. 무효 quaternion, orientation 미제공, 오래된 IMU는 거부한다.
임계값 초과는 latch되며 같은 probe에서 다음 궤적을 보내지 않는다.
`Probe.move()`는 결과 대기 중 주기적으로 검사해 active action을 취소하고 응답/최종 상태를 기록한다.

46번 실제 roll=.36/lift=.95 재현: tilt=0.253780rad에서 취소 승인,
terminal status=5(CANCELED). 이전 단계 끝에 1.2rad 전도를 발견하던 경로보다 이르게 중단한다.
취소는 균형 회복이나 넘어짐 방지의 증명이 아니다. 물리는 계속 진행하며 position hold만 남는다.
센서 표본과 Python/ROS 실행 지연이 있으므로 하드 실시간 보호 장치로 간주하지 않는다.

정상/과도 기울기, yaw 분리, 무효/누락/지연 표본 회귀 검사와 기존 안전 검사 총 10개 통과.

## 앞뒤 자세 비교

기본 roll=.36/lift=.95를 유지하고 --crouch-hip 인수만 비교했다.
hip과 ankle은 합계 knee를 상쇄해 명목상 발 pitch를 유지한다.
47/48번 hip=-.05: crouch COM x=0.0474m, shift 중 pitch 증가,
IMU 0.261557rad에서 action 취소 승인/STATUS_CANCELED.
49번 hip=-.10도 shift 완료 후 settle 중 tilt=0.256586rad로 거부했다.
그 시점에는 action이 이미 성공 종료되어 취소 이벤트는 없고 다음 lift를 보내지 않았다.
앞뒤 위치 이동만으로 안정화하지 못했으며 기본 crouch 값은 -.15로 유지한다.

다음은 전환 중 시간 연속 IMU/수동 발가락/접촉 계측이다.
표본 두세 개만으로 단일 원인을 단정하거나 파라미터 탐색을 반복하지 않는다.
