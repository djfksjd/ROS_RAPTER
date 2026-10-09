# 113 — 회전 명령 노출2초/6초 대조 (학습 중)

112에서 실제수집중yaw명령약5.1%를확인했다. 동일source카탈로그/episode종류확률/12초turn episode길이/보상/분산/46관측/8행동/물리에서회전명령지속만2초(4–6)/6초(4–10)로비교한다. 6초가원인해결을보장한다는뜻은아니다. train_symmetric의선택CLI와CommandSequenceEnv인자로추가했고기본2초및기존함수호출동작은유지한다. savedargs에sequence_turn_seconds를기록한다. 학습을이설정으로다시실행할때명시적으로같은CLI값을사용한다. 일반RunEnv평가에는이학습명령일정옵션을주입하지않는다.

집중19검사pass:기존테스트assertion보존,2/6초yaw경계·관측갱신·다른명령종류와episodehorizon보존·불법시간거부검사. 전체135중66pass69skip(.venv-ai),집중은실제MuJoCo(.venv-sim). 초기양쪽24/24tensor source정확일치·보상가중치동일확인.

각seed0/4환경/500k요청학습을시작했다. handles52713(control2)/45690(candidate6). 모델run은turn_duration2_control_20261009/turn_duration6_candidate_20261009. 새로운표준편차/DOF/구동기변경없음. 아직평가/채택없다.

종료뒤새출발61001–61005원래4–6초회전/직진10초,62001–62003의4/7명령60초열,63001/63002기립/전환을평가한다. 회전20%·4속도±10%·무전도·직진/열회귀없음·정지조건을모두유지한다. 보행정책의새동작노출실험이며실물성능/구조채택증거가아니다. 전체goal active.

실행handle52713/45690을재poll해진행중확인했다. 최신수집스텝은 progress-observation.json에기록했다. 최종검증driver를준비·보존했으며completion.json11,841,536확인전에는실행하지않는다. 자막전체33/43검토,실제영상음성완시청0은구분한다.

후속: 양쪽503,808추가스텝exit0종료. 최종평가상태는 [114](../114-turn-duration-evaluation/README.md)에서이어간다. 아직후보미채택.
