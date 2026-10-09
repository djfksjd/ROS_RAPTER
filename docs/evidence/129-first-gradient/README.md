# 129 — 첫 PPO gradient의 actor/value/entropy 분리

127의random/catalogue배치각4096개와정책불변저장본을사용해실제PPO첫epoch의정규화advantage·probabilityratio·policy/value/entropy손실기울기를autograd.grad로계산했다. optimizer.step은하지않았다. rawaction/logprob재계산최대차6.53e-6/7.06e-6<1e-4이며소배치실행과전체배치행렬연산의부동소수차를보존했다. float변환warning은run.log에남겼으며weight업데이트를수행한것이아니다.

actor정책기울기노름5.471/5.225, entropy기울기0.00849, value항의actor기울기0이다. 현재MLP actor/criticparameter분리에서는value가actor에직접기울기를보내지않는다. 전체gradientnorm5.513/5.475로globalclip배수약.181/.183. 이것은전역clip상호작용이없다는뜻은아니지만value/entropy가주도해actor를망가뜨렸다는주장은이번배치로지지되지않는다.

이전optimizer의actor exp_avg노름0.17425이며현재합산actor기울기와cosine은−.01779/+.00585로거의직교다. 이평균벡터통계는각parameter의적응분모/step수/Adam실제갱신전부를설명하지않고첫배치만의결과다. 이전학습이력이퇴행의원인이라는확정이아니다.

다음은동일초기정책·보상·명령·KL.005에서optimizer이력만초기화하는선택형대조를한다. 기존optimizer이전기본값을보존하고초기정책출력/초기24tensor·새optimizerstate·학습설정저장을검증한다. 기존기준선미교체·전체goal active.
