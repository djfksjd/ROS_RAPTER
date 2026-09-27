# 한발 지지 실험 — 2026-09-27

기존 실제 URDF의 관절축·질량·관성 원점에서 무게중심과 왼발 하단 모서리를
계산하는 `support_model.py`를 추가했다. 발 전체가 기울어져 모서리로 접촉하는
현재 모델에서는 지지 영역이 좁아진다는 가설을 검사하기 위한 도구다.

`balance_probe.py`는 IMU roll/각속도와 모델의 무게중심 오차로 양쪽 Hip Roll
목표를 보정하는 개발 실험이다. mission gate가 실행 중이면 거부하고, 최대
8초 simulation / 90초 wall-time 및 roll/pitch 0.25rad 제한을 둔다.

첫 실행 결과는 **실패**다. 28개 표본 후 기울기 제한을 넘었다. 약 0.431초에서
roll -0.1095rad, roll 각속도 -0.8423rad/s였으며, 발 들기 시작 전에도 불안정했다.
[원본 JSON](evidence/14-imu-balance-failure.json)과 [실행 로그](evidence/14-imu-balance-failure.log)를 보존했다.

이는 기구 변경 없이 단순한 지연 있는 위치 목표 보정만으로는 이 실험을 안정화하지
못했다는 결과다. 유일한 원인을 규명하거나 폐루프 균형 제어를 완성한 것이 아니다.
보행 mission은 계속 비활성 상태다. 이후 사용자 요청에 따라 별도 옵션에서
수동 발가락 관절과 분리된 접촉 모델을 검증한다. 기존 발 모델은 기본값으로 유지한다.

## 응답 지연 비교

14번 기록에서 목표 속도 제한과 추종 지연을 확인해, 목표 갱신 한도
0.012→0.030rad/20ms 및 trajectory horizon 80→20ms로 바꾼 비교도 수행했다.
[30번 결과](evidence/30-balance-fast-response-failure.json)는 약 0.313초까지
16개 표본 후 기울기 제한으로 실패했다. 지지 전환·충격이 동반되어 응답 속도를
높이는 것만으로 해결되지 않았다. 이 비교를 성공한 제어기 튜닝으로 채택하지 않았다.
기본값은 기존 응답이며 `--fast-response`는 실패 재현 옵션이다.

이제 프로브는 임의의 기본 Xacro 대신 실제 `/robot_description`을 받아 계산한다.
수동 발가락이나 world 고정 fixture가 있는 모델에서는 단단한 발에 대한 근사가
맞지 않으므로 실행을 거부한다. 현재 operator gait는 계속 비활성이다.

[프로브 거부 검사](evidence/34-probe-refusal-check.txt)에서 수동 발가락/운영 세션을
거부할 때 trajectory 메시지가 0개임을 확인했다. 거부 시에도 cleanup에서 hold를
발행하던 경로를 막아, 실제 제어를 시작한 실험만 종료 hold를 발행한다.
