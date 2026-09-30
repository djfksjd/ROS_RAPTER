1. **지금은 스윙 다리의 접힘·착지 준비·좌우 대칭을 유도하고, 타조의 지지 자세를 그대로 강제하지 않는 것이 좋습니다.** 아래 목표는 제공된 기구학에 맞춘 **초기 훈련값**이며, 최적값으로 검증된 수치는 아닙니다.

   각 다리의 시계 위상을 다음처럼 정의합니다.
   \[
   \phi_L=\operatorname{frac}(ft),\qquad
   \phi_R=\operatorname{frac}(ft+0.5)
   \]
   \[
   \text{stance}:0\le\phi<0.25,\qquad
   s=\frac{\phi-0.25}{0.75}\in[0,1)\quad\text{during swing}
   \]
   여기서 \(\alpha\)는 intertarsal included angle이며, 작을수록 접힙니다.

   | 규칙 | 지금 적용할 목표 | 피해야 할 강제 |
   |---|---|---|
   | 중간 스윙 발목 접힘 | \(s=0.30–0.60\), 우선 **80°**, 이후 **65°**, 허용 폭 약 ±15° | 처음부터 타조의 45° 요구 |
   | 착지 전 발목 펴기 | \(s=0.80–0.98\), **130°±10°**; 초기 지지 자세와 연속적으로 연결 | 168° 완전 신전 |
   | 지지 중 발목 안정 | 초기 허용 excursion **12°**, 이후 **8°**; 충격 직후·toe-off 제외 | 지지각 자체를 150–168°로 올리기 |
   | 착지 전 일정한 후방 회전 | \(s=0.75–0.95\), hip pitch 기준 우선 **+1.5 rad/s**, 허용 폭 ±0.75 rad/s | 타조 자료의 다리 전체 회전을 hip 회전과 동일시 |
   | 좌우 대칭 | 같은 다리 위상에서 발목각·hip 후방 회전속도 일치 | 같은 시간의 좌우각 일치 |

   후방 회전은 가능하면 **hip-to-foot 벡터의 sagittal angle** \(\beta\)로 평가하세요. 타조의 leg retraction에 더 가까운 변수입니다. 제공된 자료에는 \(\dot\beta\)가 없으므로 정확한 목표를 정할 수 없습니다. 현재는 측정값이 있는 hip pitch를 대리 변수로 쓰고, 왼쪽의 +1.7 rad/s 근처를 양쪽에 유도하는 것이 합리적입니다.

   **지금 강제하지 않을 항목:** subhorizontal femur, hip excursion 12.7°, stance intertarsal 168°. 현재 CoM 배치에서는 이 조합이 발을 과도하게 앞으로 보내며, 속도 유지에 필요한 자세와 충돌합니다. 타조의 MTP 탄성 분담 역시 현재 수동 발가락과 스프링 구성에 그대로 적용할 수 없습니다.

2. **새 항목은 모두 작은 음의 shaping penalty로 시작하세요.** 목표 달성 자체에 추가 양의 보상을 주지 않아 접촉·속도 보상의 의미를 유지합니다.

   50 Hz에서 \(\Delta t=0.02\) s이고,
   \[
   r_t=\Delta t\left[
   3S_v-8C_{\rm schedule}
   -\sum_j w_jP_j
   \right]
   \]
   로 둡니다. \(S_v,C_{\rm schedule}\)은 기존 정의를 유지합니다. 새 \(P_j\)는 가능한 한 \([0,1]\)에 제한합니다.

   사용할 함수는:
   \[
   G(e;\sigma)=1-\exp\!\left(-\frac{e^2}{2\sigma^2}\right)
   \]
   \[
   H(e;b,a)=\min\left[1,
   \left(\frac{\max(0,|e|-b)}{a}\right)^2\right]
   \]
   입니다. 위상 창 \(W_{a,b}(s)\)는 구간 내부에서 1이고, 양 끝 **스윙 위상 0.03 폭**에 raised-cosine ramp를 둡니다. 다리별 항목은 **두 다리의 평균**으로 계산합니다.

   | 항목 | 정확한 기본 형태 | 최종 \(w\), /s | 최대 /step |
   |---|---|---:|---:|
   | 스윙 접힘 | \(W_{.30,.60}(s)G(\alpha-65^\circ;15^\circ)\) | 0.45 | −0.009 |
   | 착지 준비 | \(W_{.80,.98}(s)G(\alpha-130^\circ;10^\circ)\) | 0.30 | −0.006 |
   | 지지 안정 | 아래 excursion penalty | 0.25 | −0.005 |
   | 일정한 retraction | \(W_{.75,.95}(s)G(\dot q_{\rm hip}-1.5;0.75)\) | 0.35 | −0.007 |
   | 좌우 대칭 | 아래 phase-matched penalty | 0.20 | −0.004 |
   | target 매끄러움 | 아래 second-difference penalty | 0.15 | −0.003 |

   새 penalty의 합은 최대 **−1.70/s**입니다. 실제 평균은 위상 창 때문에 더 작습니다. 기존 최대 속도 보상은 +0.06/step, 접촉 penalty는 −0.16/step입니다.

   **지지 excursion:** \(\phi=0.04\)에서 실제 접촉 중인 발목각을 \(\alpha_0\)로 저장하고, \(\phi=0.04–0.21\)의 running excursion을
   \[
   E=\max\{\alpha_0,\ldots,\alpha_t\}
   -\min\{\alpha_0,\ldots,\alpha_t\}
   \]
   로 계산합니다.
   \[
   P_{\rm stance}=W_{.04,.21}(\phi)
   \min\left[1,\left(\frac{\max(0,E-8^\circ)}{8^\circ}\right)^2\right]
   \]
   시작할 때는 8° 대신 12°를 사용합니다. 해당 지지 구간에 접촉이 없으면 이 항목을 1로 처리해, 접촉을 피해서 penalty를 없애는 경로를 막습니다.

   **대칭:** 현재 다리와 반대 다리의 최근 동일 위상 표본을 비교합니다.
   \[
   P_{\rm sym}=\frac12\left[
   W_{.30,.98}(s)H(\alpha_L(s)-\alpha_R(s);10^\circ,15^\circ)
   +W_{.75,.95}(s)H(\dot q_L(s)-\dot q_R(s);0.5,1.0)
   \right]
   \]
   실제 변화하는 stride clock의 위상으로 보간하고, 오래된 표본은 버립니다. roll/yaw까지 확장하면 좌우 반사에 따른 부호를 적용해야 합니다.

   **매끄러움:** 실제 각도는 동작 달성에 사용하고, position target의 2차 차분은 급격한 명령에 사용합니다.
   \[
   P_{\rm smooth}=
   \frac1{12}\sum_{k=1}^{12}
   \min\left[1,\left(
   \frac{q^*_{k,t}-2q^*_{k,t-1}+q^*_{k,t-2}}{0.05\ {\rm rad}}
   \right)^2\right]
   \]
   0.05 rad는 초기값이며, 기존 정상 정책의 명령 분포를 확인해 조정해야 합니다.

   보상 악용을 막는 핵심은 다음입니다.

   - **스윙 창은 시계로 결정합니다.** 발을 들거나 접촉을 끊었다고 접힘 보상 구간이 시작되지 않게 합니다.
   - 목표각은 actuator target이 아닌 **실제 관절각**으로 평가합니다.
   - late stance \(\phi=0.04–0.21\)에서 \(\alpha<110^\circ\)이면 \(H(\alpha-110^\circ;0,15^\circ)\)를 추가해 지지 중 과도한 접힘을 억제합니다. 이를 stance 항목 안에 합산하고 1로 제한합니다.
   - 늦은 스윙에서 \(\dot\alpha<-20^\circ/s\)인 재접힘을 extension 항목 안에서 약하게 벌점화합니다. 예를 들어 각도 penalty 80%, 재접힘 penalty 20%로 구성합니다.
   - 기존 torque/speed/power clamp와 접촉 penalty를 유지합니다. **각도 목표를 만족해도 clamp에 계속 걸리는 동작은 성공으로 판정하지 않습니다.**

3. **현재는 femur nominal pose를 바꾸지 말고 정책에 맡기세요.** nominal pose가 action offset이라면 이를 바꾸는 것만으로도 기존 정책의 모든 position target이 이동합니다. 이미 얻은 6 m/s 교대 달리기를 유지하면서 원인을 분리하려면, 먼저 위의 스윙 항목만 적용하는 편이 좋습니다.

   지지 자세는 femur 각도보다 **발의 CoM 상대 위치, 몸통 pitch, 접촉 충격, 지지 안정성**으로 평가하세요. 향후 nominal pose 변경은 새 질량 배치에서 별도 초기화하거나 기존 정책의 출력이 보존되도록 offset을 보정한 뒤 진행해야 합니다.

   필요한 질량 배치 변화는 제공된 기하에서 추정할 수 있습니다. 전방을 양수로 놓으면:
   \[
   x_{\rm foot}-x_{\rm CoM}
   =(x_{\rm foot}-x_{\rm hip})
   +(x_{\rm hip}-x_{\rm CoM})
   \]
   현재 \(x_{\rm hip}-x_{\rm CoM}\approx0.10\) m이고, 원하는 타조형 자세에서 발이 CoM 앞 0.30–0.40 m라면:
   \[
   x_{\rm foot}-x_{\rm hip}\approx0.20–0.30\ {\rm m}
   \]

   동일 자세에서 발을 CoM 근처 **0–0.05 m 앞**에 두려면, hip–CoM 상대 위치를 현재보다 대략 **0.25–0.40 m 뒤로** 옮겨야 합니다. 즉, **hip을 뒤로 옮기거나 CoM을 앞으로 옮기거나 두 변경을 조합**해야 합니다. 0.10 m만 바꾸면 꼬리에 의한 기존 후방 CoM 오프셋은 해소하지만, 자세 자체의 전방 발 위치는 남습니다.

   이는 **기하학적 필요량**이며 동적 안정성을 보장하는 설계값은 아닙니다. 먼저 배터리·몸통 질량의 전방 이동과 꼬리 질량 감소로 가능한 범위를 확인하고, hip 위치 변경을 검토하세요. 타조처럼 지지 발목을 거의 고정하며 탄성을 쓰려면, 질량 배치뿐 아니라 **발가락/MTP 부근의 실제 탄성 저장 구조**도 필요합니다.

4. **기존 교대 정책에서 이어서 학습하되, stride clock·주파수·접촉 penalty를 먼저 고정하세요.** 새 보상과 주파수 변경을 동시에 하면 원인을 분리하기 어렵습니다.

   | 단계 | 변경 |
   |---|---|
   | 기준 저장 | 현재 checkpoint와 6 m/s 평가 결과 보존 |
   | 1 | retraction 0.10/s, symmetry 0.05/s, smoothness 0.05/s |
   | 2 | fold 목표 80°, weight 0.15/s; extension 0.10/s |
   | 3 | fold 목표 80→65°, weight 0.15→0.45/s; extension 0.10→0.30/s |
   | 4 | stance excursion 12→8°, weight 0.05→0.25/s; 나머지도 표의 최종값까지 상승 |

   각 증가는 한 번에 약 **최종 weight의 20–25%**씩 적용하고, optimizer update 수보다 평가 결과로 다음 단계를 결정하세요. 조건 악화 시 마지막 weight 증가를 되돌립니다.

   단계 진행 조건의 초기 기준은 다음처럼 잡을 수 있습니다.

   - 6 m/s 평가에서 평균 속도 **5.8 m/s 이상**, 기존 대비 추종 오차 악화 **5% 이내**.
   - 좌우 위상차 **0.5±0.05 cycle**, per-leg duty **0.25±0.03**.
   - 넘어짐·비정상 접촉·contact-schedule penalty가 기존보다 지속적으로 증가하지 않음.
   - 양쪽 late-swing retraction 비율 **90% 이상**, 좌우 평균 회전속도 차 **0.5 rad/s 이하**.
   - mid-swing 최저 발목각, late-swing 재접힘 횟수, stance excursion을 **다리별 분포**로 확인.
   - 실제 각도와 target 차이, torque/speed/power clamp 점유율, 발 충격·미끄러짐, 몸통 pitch를 함께 확인.

   **첫 변경은 오른쪽 late-swing retraction과 대칭 개선**이 적절합니다. 현재 왼쪽은 이미 목표 근처라서 기존 동작을 활용할 수 있습니다. 그다음 발목 접힘을 추가하고, 현재 13–18°인 지지 excursion 축소는 마지막에 적용하세요. 속도와 교대를 잃으면서 타조 각도에 가까워진 결과는 개선으로 판정하지 않습니다.