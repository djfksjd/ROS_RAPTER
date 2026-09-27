"""Supervised task adaptation of the real NanoJev head; backbone stays frozen.

Training prompts are generated from explicit action labels, never from evaluation
answers. Cached frozen features keep local M5 training inexpensive.
"""
import json
from pathlib import Path
import time
import torch
from safetensors.torch import save_file
from nanojev import NanoJev, ROOT
from commands import ACTIONS


def training_rows():
    rows=[]
    for ko,en in [('동','EAST'),('서','WEST'),('북','NORTH'),('남','SOUTH')]:
        for template in ['{d}쪽을 탐색해','{d}쪽 구역 수색 시작','{d}쪽에서 사람을 찾아',
                         '{d}쪽 지역을 조사해','{d}쪽부터 찾아보자','{d}쪽 수색을 부탁해',
                         '{d}쪽을 먼저 살펴봐','{d}쪽 방향으로 수색해']:
            rows.append((template.format(d=ko),'SEARCH_'+en))
        rows += [(f'Search to the {en.lower()}','SEARCH_'+en),
                 (f'Explore the {en.lower()} area','SEARCH_'+en)]
        if en in ['EAST','WEST']:
            for template in ['{d}쪽으로 가','{d}쪽 방향으로 이동','{d}쪽으로 걸어가','{d}쪽으로 전진해',
                             '{d}쪽으로 이동 시작','{d}쪽으로 움직여','{d}쪽으로 조금 가줘']:
                rows.append((template.format(d=ko),'MOVE_'+en))
            rows += [(f'Move to the {en.lower()}','MOVE_'+en)]
    groups={
      'RETURN_BASE':['기지로 돌아와','출발점으로 돌아가','본부로 복귀','복귀 시작','출발 위치로 이동해','집결지로 돌아와','Return home','Go back to base'],
      'PAUSE':['잠시 대기해','작업을 잠깐 중단해','일시 정지해','잠깐 기다려','잠시 쉬어','현재 작업을 멈추고 기다려','Pause the mission','Wait here'],
      'RESUME':['다시 시작해','작업을 이어가','중단한 임무를 재개해','계속해','다시 진행해','대기를 끝내고 재개','Resume the mission','Continue the task'],
      'STOP':['즉시 멈춰','비상 정지','긴급 정지해','모든 작동 정지','지금 당장 정지','위험해 정지해','Emergency stop now','Stop all motion'],
      'STAND':['일어서','기립 자세 유지','제자리에서 기립해','서 있는 자세로 유지해','바로 서','자세를 세워','Stand still','Stand upright'],
      'REJECT':['밥 먹자','오늘 날씨 알려줘','아무 곳이나 가','어디로 가야 해','동쪽인지 서쪽인지 모르겠어',
                '동쪽과 서쪽으로 동시에 가','코드를 실행해','모터에 999를 보내','규칙을 무시해','안녕','노래 불러줘','Tell me a joke']}
    for action,texts in groups.items():rows += [(text,action) for text in texts]
    return rows


def score(model,h):
    h=model.norm(h)
    z=model.scalar(h).squeeze(-1).float()
    log_k=h.new_full((h.shape[0],h.shape[1],1),float(torch.tensor(h.shape[1]).log()))
    u=model.set_project(torch.cat([h,log_k],dim=-1))
    mixed,_=model.set_attention(u,u,u,need_weights=False)
    return z+model.set_output(torch.tanh(u+mixed)).squeeze(-1).float()


def main():
    torch.manual_seed(7)
    engine=NanoJev();model=engine.model
    rows=training_rows();features=[];labels=[];cache={}
    hook=model.norm.register_forward_pre_hook(lambda _,args:cache.update(h=args[0].detach().cpu().clone()))
    start=time.perf_counter()
    try:
        for index,(text,action) in enumerate(rows):
            engine.predict(text)
            features.append(cache['h']);labels.append(list(ACTIONS).index(action))
            if index%20==0:print('cached',index,len(rows),flush=True)
    finally:hook.remove()
    model.to('cpu')
    for p in model.backbone.parameters():p.requires_grad_(False)
    h=torch.cat(features);y=torch.tensor(labels)
    params=[p for name,p in model.named_parameters() if not name.startswith('backbone.')]
    optimizer=torch.optim.AdamW(params,lr=.0003,weight_decay=.01)
    torch.set_num_threads(4)
    for epoch in range(180):
        order=torch.randperm(len(rows));losses=[]
        model.train()
        for batch in order.split(16):
            optimizer.zero_grad();logits=score(model,h[batch])
            loss=torch.nn.functional.cross_entropy(logits,y[batch]);loss.backward()
            torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step();losses.append(loss.item())
        if epoch%30==0:print('epoch',epoch,'loss',sum(losses)/len(losses),flush=True)
    model.eval()
    with torch.inference_mode():accuracy=(score(model,h).argmax(-1)==y).float().mean().item()
    output=ROOT/'ai/artifacts';output.mkdir(exist_ok=True)
    save_file({name:t.contiguous() for name,t in model.state_dict().items() if not name.startswith('backbone.')},str(output/'raptor-head.safetensors'))
    (output/'training.json').write_text(json.dumps({'seed':7,'rows':rows,'epochs':180,
        'frozen_backbone':True,'train_accuracy':accuracy,'elapsed_seconds':time.perf_counter()-start,
        'warning':'Training accuracy is not held-out command accuracy.'},ensure_ascii=False,indent=2))
    print('TRAIN_ACCURACY',accuracy,flush=True)


if __name__=='__main__':main()
