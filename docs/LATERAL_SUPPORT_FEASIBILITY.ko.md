# 발목 속도 포화 재분석과 측방 지지 가능성 — 2026-09-28

브랜치 `analysis/lateral-support-feasibility`. **새 Gazebo 실험은 실행하지 않았다.**
54번 기록 재분석, 설치 버전 소스 대조, 생성 URDF 기반 오프라인 FK 계산만 수행했다.
gain·토크·속도 한도와 10 active DOF는 변경하지 않았다.

## 1. 실제 제어·물리 경로 (evidence/55-control-path-sources.txt)

- world step 1 ms, Physics plugin에 engine 미지정 → 기본 DART(설치 6.13.2).
  라이브 로그의 엔진 이름은 아직 직접 확인하지 않았다.
- gz_ros2_control 1.2.20: `write()`는 매 physics step(1 ms), `read()/update()`는 10 ms마다.
  속도 명령 = −0.3 × (위치 − 명령) × 100. 위치는 최대 10 ms 전 값이다.
  |명령| 2.5 rad/s는 |오차| ≥ 0.0833 rad에서만 나온다.
- DART는 SERVO 명령을 속도 한도로 자르고, 속도 한도를 constraint로도 강제한다.
- effort 상태는 JointTransmittedWrench 투영이며 servo 토크가 아니다.

## 2. 54번 재분석 (evidence/55-joint-load-reanalysis.json)

`analyze_joint_load.py`로 기존 telemetry만 다시 계산했다.

- shift 중 body roll은 t=8.24까지 |0.002| rad 이하. 양발 sole은 hip roll만큼 기울었다
  (t=7.76에 15°, 8.24에 20°).
- 최신(30 ms 이내) 접촉점의 y 폭: shift 전 115~140 mm → hip roll 0.025 rad부터 9~26 mm.
  shift 내내 **양발이 모서리(선) 접촉**이었다.
- 발목 ±2.5 rad/s 표본 12개 중 **7개는 위치 명령으로 설명되지 않는다**
  (오차 0.008~0.03 rad 또는 명령과 반대 부호). 예: 오차 −0.008, P 명령 +0.23, 실측 −2.50.
- 발목 전달 effort는 대부분 7 Nm 미만, 37 Nm는 전도 중 마지막 표본뿐이다.

판정:
- **확인됨:** 이 shift 궤적은 ankle roll이 없는 다리에서 몸통을 세운 채 발바닥을 평평하게
  유지할 수 없다. 모서리 접촉에서 발목 떨림이 시작됐다.
- **기각에 가까움:** "큰 추종 오차로 속도 명령 포화"만으로는 다수 표본을 설명하지 못한다.
- **미확정(가설):** 모서리 접촉이 관절을 밀고 속도 한도 constraint가 잘랐는지, 10 ms 지연
  servo 루프 진동인지. 60 ms 희소 표본으로는 구분 불가. 모터 토크 포화는 근거 없음.

## 3. 측방 지지 가능성 (evidence/56-lateral-support-feasibility.json)

`lateral_feasibility.py`: 왼발을 평평한 지면으로 두고, 10개 능동축을 URDF 한도 안에서
무작위 20,000개 + hill-climb 3회로 COM의 측방 여유를 최대화했다.
지지 폭은 발·수동 발가락 collision 전체 y 범위(±0.07 m)로 **유리하게** 잡았다.

| 조건 | 최대 측방 여유 |
|---|---|
| 양발 crouch, 몸통 수직 | −0.110 m |
| 격자(몸통 기울기 ≤0.25 rad) | −0.038 m |
| 전체 한도, 기울기 무제한(0.9~1.5 rad) | **−0.008 m**, hip roll −0.5 한도에 고정 |

질량 17.3 kg, hip 반폭 0.18 m, 발 폭 0.12 m, ankle roll 없음.
평평한 발의 정적 한 발 지지는 **현재 기하·관절 한도에서 찾지 못했다.**
충돌·스윙 발 높이·동역학을 무시한 낙관적 조건에서도 음수다.
따라서 roll·crouch 값 탐색과 gain/토크/속도 한도 상향으로는 준정적 교대 지지를 만들 수 없다.
단, 이것은 운동학 한계이며 동적 보행 불가능을 증명하지 않는다.

## 4. 방향 선택지 (사용자 결정 필요)

| 선택 | 능동축 | 내용 | 위험 |
|---|---|---|---|
| A. 동적 측방 흔들기 보행 | 10 유지 | COM이 발 위에 정지하지 않는 보행. PlaCo/LIPM 궤적 또는 MuJoCo RL | 구현량, 접촉 모델 의존 |
| B. 기하 변경 | 10 유지 | hip 반폭 약 0.10 m 또는 발바닥 안쪽 약 7 cm 확장(추정, 재검증 필요) | 다리 간섭, 외형 변화 |
| C. 모서리 접촉 rocking | 10 유지 | 발 모서리 위에서 흔들며 이동 | 가장 불안정, 전이성 낮음 |
| D. ankle roll 추가 | **12로 증가** | 측방 지지 직접 제어 | 불변 조건 변경, 사전 승인 필요 |

B의 수치는 UNI_AI 검토 응답의 추정(COM 이동 ≈ 0.97 × hip 폭 감소)이며 FK 재계산 전이다.
Open Duck Mini도 ankle roll이 없지만 hip yaw가 있고 다른 로봇이므로 우리 보행 가능성의 증거가 아니다.

## 5. UNI_AI 사용 기록

- `/models/` 1회: 89개, `gpt-6-sol` 사용 가능.
- `gpt-6-sol` 2회(요청=응답 모델 ID): 진단 검토 total 2,893 tokens, 가능성·선택지 검토 3,562 tokens.
  둘 다 finish=stop. 응답은 제안으로만 사용했다.
- 반영: 스윙 다리·시상면 관절·지지 폭 정의에 대한 지적 → 전체 관절 무작위 탐색과
  유리한 지지 폭으로 재계산. 결론은 유지됐다.

## 재현

```bash
# URDF 생성(시뮬레이션 없음): 컨테이너에서 passive_toes:=true sensors:=true crouch_hip_pitch:=-0.1 로 xacro
python3 src/raptor_control/scripts/analyze_joint_load.py docs/evidence/54-joint-load-telemetry.json
PYTHONPATH=src/raptor_control/scripts python3 src/raptor_control/scripts/lateral_feasibility.py MODEL.urdf
```
