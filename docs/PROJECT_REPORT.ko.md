# Gazebo 로봇 시뮬레이션 주제 보고서

## 주제와 목적

**자연어 명령을 이해하는 10축 랩터형 지상 탐사 로봇**을 구현한다.
운영자의 의도를 허용된 행동으로 변환하는 것이 AI의 역할이며, AI가 작전을
스스로 결정하거나 motor command를 직접 생성하지 않는다.

```text
운영자 → Qwen / NanoJev → 정형 행동 → Safety Gate → ROS mission → controller → Raptor
```

최종 활용 목표는 재난·산악·위험 지역에서 사람이 접근하기 전에 정보를 제공하는
지상 탐사다. 현재 평면 시뮬레이션 결과를 실제 현장 성능으로 해석하지 않는다.

## 로봇 구성

| 부위 | Active DOF |
|---|---:|
| 왼쪽 Hip Roll / Hip Pitch / Knee Pitch / Ankle Pitch | 4 |
| 오른쪽 Hip Roll / Hip Pitch / Knee Pitch / Ankle Pitch | 4 |
| Tail Yaw / Tail Pitch | 2 |
| 합계 | **10** |

14축으로 변경하지 않았다. 상세 메시를 켜도 관절·질량·관성·collision이
기본 모델과 일치하는 것을 [검사](evidence/model-contract.txt)했다.

## 외형과 부품 모델

![Blender에서 렌더한 외형 프로토타입 — Gazebo 검증 화면과 구분](evidence/raptor-concept-render.png)

참고 이미지의 몸통 외장, 관절 감속기, 노출 프레임, 센서부, 발가락 및 분절
꼬리를 반영한 **시각 프로토타입**이다. 제작 가능한 최종 기계 설계가 아니다.
부품별 Blender 오브젝트와 링크별·재질별 DAE를 제공하며,
`modeling/raptor-assembly.blend`와 `modeling/build_visuals.py`로 수정·재생성한다.

꼬리는 yaw/pitch 2축 active base를 유지한다. 뒤쪽 8개 마디는 fixed visual이고
collision은 기존 단순 형상을 유지한다. 수동 관절의 탄성·케이블·스프링 동역학은
구현되지 않았다. 꼬리는 균형 보조이며 완전한 균형 제어를 보장하지 않는다.
Tendon-driven continuum tail은 후속 연구다.

## 실제 검증 결과

| 항목 | 결과와 증거 |
|---|---|
| 환경·빌드 | [설치 보고서](INSTALLATION_REPORT.ko.md) |
| 10축 개별 명령 | 공중 고정 시험에서 전 관절 0.03rad 추종. [원본](evidence/02-fixture-joint-test.json). 이 시험은 기립 증명이 아님 |
| Static standing | 고정 없는 모델에서 neutral 명령 후 자세 유지. [원본](evidence/04-static-standing.json) |
| 상세 외형 기립 | [추가 검사](evidence/09-detailed-static-standing.json) |
| Predefined gait | 26개 궤적 단계 추종은 완료했으나 정상 전진을 입증하지 못함. [원본](evidence/07-alternating-step.json) |
| 발 접촉 검증 | 초기 발 들기 구간에서도 양발 접촉 지속. [원본](evidence/10-step-contact-test.json) |
| 하중 이동 개선 실험 | 한쪽 발 접촉 해제 후 몸체 기울기 기준 초과로 실패. [원본](evidence/11-larger-shift-step.json) |
| 센서 | [IMU·RGB-D](evidence/08-sensors.txt), [왼발](evidence/08-left-foot-contact.txt), [오른발](evidence/08-right-foot-contact.txt) |

관절이 제어되거나 넘어지지 않는 한 장면만으로 보행 성공이라고 판단하지 않는다.
현재 안정적인 교대 지지·전진·외란 복구는 **미완료**다.

## Qwen과 NanoJev 비교

Qwen3-0.6B는 Ollama의 JSON schema 출력으로 행동 ID를 생성한다.
NanoJev는 공개 Qwen3-0.6B backbone과 실제 decision head로 후보 행동의 확률을
계산하며 출력 토큰을 생성하지 않는다. 원본 CUDA 실행기와 구분되는 MPS FP32
어댑터를 작성했다. 원본 게임 체크포인트가 이 명령 업무에 맞지 않아 backbone을
고정하고 108개 명령 예시로 head만 추가 학습했다.

| 모델 | 별도 38문장 정답률 | Median | P95 | 형식 오류율 |
|---|---:|---:|---:|---:|
| Qwen3-0.6B Q4_K_M / Ollama | 27/38 (71.1%) | 81.0ms | 202.2ms | 0% |
| NanoJev + Raptor head / MPS FP32 | 24/38 (63.2%) | 312.7ms | 361.2ms | 0% |

[Qwen 원본 측정](evidence/qwen-holdout.json) ·
[NanoJev 원본 측정](evidence/nanojev-adapted-holdout.json).

이는 작은 명령 평가셋 결과이며 일반 성능을 보장하지 않는다. 첫 요청의 cold 비용이
포함돼 있고, 두 모델의 정밀도·런타임·배칭 방식이 달라 순수 아키텍처 우열로
해석할 수 없다. e2e는 로컬 호출부터 파싱까지, model_ms는 각 런타임의 추론
구간이다. 원격 네트워크 측정은 하지 않았다. 형식 오류 0%는 의미 정확도 100%가 아니다.

## ROS 연결과 안전 범위

[실제 요청](evidence/05-qwen-ros-command.txt)과
[mission gate 로그](evidence/05-mission-gate.log)에서 다음을 확인했다.

- STAND → accepted → joint_target_reached.
- SEARCH_EAST → 미검증 mission이므로 거부.
- STOP → 현재 관절 위치 유지 및 stop latch.
- STOP 이후 STAND → 거부.

NanoJev도 [실제 명령](evidence/13-nanojev-ros-command.txt)에서 STAND를 선택하고
[ROS gate](evidence/13-nanojev-mission-gate.log)의 joint_target_reached까지 확인했다.
이 한 문장은 통합 동작 시험이며 별도 정확도 평가에 추가하지 않았다.

현재 실행 가능한 mission은 STAND, PAUSE, RESUME, STOP으로 제한한다.
RESUME은 이전 이동 궤적을 자동 재생하지 않는다. 별도 `--stop` 경로는 모델 추론을
거치지 않는다. STOP은 이 position-control 시뮬레이터에서 **관절 위치 유지**를
뜻하며 실제 하드웨어의 전원 차단이나 인증된 비상 정지 장치를 대신하지 않는다.
연구용 motion/gait probe는 mission gate를 거치지 않는 개발 도구이므로 운영 세션에서
함께 실행하면 안 된다. ROS 토픽 접근 자체를 인증하는 보안 경계는 아니다.

[안전 정책 테스트](evidence/safety-tests.txt) 5개 통과. 위 4개 gate 사례는
전체 탐색 mission의 실행 성공률로 환산하지 않는다.

## 남은 개발

1. IMU·접촉 기반 교대 지지와 안정적인 전진 보행.
2. 검증된 gait 위에 방향 이동·복귀·수색 mission 구현.
3. 다양한 표현·거부 사례 데이터 확장 및 새로운 평가셋에서 재검증.
4. Passive tail 재료·연결 방식 비교와 유연 동역학 검증.
5. 부품 치수·감속기·구동 토크·간섭·제작성 검토.

현재 결과는 졸업 프로젝트의 실행 가능한 중간 산출물이다. 전체 최종 목표를
완료했다고 보고하지 않는다.
