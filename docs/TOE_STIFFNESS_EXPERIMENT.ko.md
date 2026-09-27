# 수동 발가락 강성 비교

## 근거

50번은 기본 강성에서 기존 roll=.36, crouch hip=-.10 실험을 시간 연속 계측한 기록이다.
IMU timestamp를 기준으로 약 20Hz, 최대 2,000표본으로 제한했다.
79표본, 누락(버퍼 초과) 0개. joint/contact 메시지별 simulation timestamp를 보존한다.
같은 시각으로 강제 보간하지 않으며 메시지 수신 지연을 구분해야 한다.

좌우 이동 중 접촉이 양발의 한 줄 발가락으로 좁아졌다.
실행 시작 약 4.26초에 left toe 1 proximal=-.116rad, body pitch=.096rad,
이후 pitch 각속도가 증가했다. 이 상관관계만으로 유일한 원인을 단정하지 않는다.

## 한 변수 비교

`toe_stiffness_scale` 기본값 1, launch 허용 범위 1~5.
실험값 3에서 proximal 36 / distal 24 Nm/rad. 기본값은 기존 12/8이다.
51번 검사는 SDF 변환에서도 해당 강성이 유지되며 10 active + 12 passive이고,
강성을 제외한 URDF 요소 전체가 동일함을 확인한다.

52번은 동일 궤적에서 강성만 3배로 실행했다.
pitch 증가가 줄었지만 shift 이후 왼쪽 ankle 목표 -.3rad 대비 실제 -.36213rad로
추종 오차 .06213rad에 의해 실패했다. 기본값으로 채택하지 않았다.
이는 실제 제작용 스프링 선정/부품 하중 검증이 아니다.

## 다음 조사

현재 URDF의 ankle effort=80, velocity=2.5, plugin position gain=.3,
controller_manager update_rate=100이다. 값을 올리기 전에 발목 추종 기록을 검사한다.
공식 문서는 position interface가 position error와 gain/update rate로 속도 목표를 만든다고
설명한다: [gz_ros2_control Jazzy](https://control.ros.org/jazzy/doc/gz_ros2_control/doc/index.html).
현재 바이너리 버전의 구현과 실제 하중/속도도 확인해야 하므로 문서만으로 오차 원인을 확정하지 않는다.

재현: `RAPTOR_PASSIVE_TOES=true bash scripts/start_local.sh --experiment toe_stiffness_scale:=3`
후 컨테이너에서 `gait_probe.py --roll .36 --lift .95 --cycles 0 --crouch-hip -.10`.
