# 130 — optimizer 이력 초기화 비교

train_symmetric에--reset-optimizer를선택형으로추가했다. 미지정시기존sourceoptimizerstate를이전하는기본행동유지, 지정시정책weight/정규화/환경/알고리즘설정은유지하고새optimizerstate로시작한다. 저장flag와24policytensor/100상태행동정확일치·defaultstate존재/freshstate비어있음을검증했다. 전체136검사66pass70skip·집중19pass. source117과같은sym_init·random명령/원보상/seed0·KL.005/65536step, 실제학습learning_rate3e-4유지.

후보학습exit0·final10395648 완료, 실패와원기준선보존. 공통개발출발67001–5의30회평가를시작했으며완료/성능은아직미확정이다. 이전Adam이력과현재gradient가거의직교였다는129는원인확정이아니며이번대조가효과를분리한다. 전체goal active.

최종30회exit0: 회전량4/20·최소속도0/20·전도1/30. 117의18/20·2/20·전도0/30보다퇴행·미채택. 초기momentum의거의직교방향만으로리셋유리함을추론할수없다. 전도/회전실패로기각했으며추가60초열·기립/정지시험은이번후보에수행하지않았다. 기본optimizer이전경로와기준선보존. 다음은동일actor/이력/KL에서학습률을한변수로줄여실제update폭을분리한다.
