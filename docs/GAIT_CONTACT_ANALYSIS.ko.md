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
