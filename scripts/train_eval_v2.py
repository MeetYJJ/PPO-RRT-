import sys,os,json,statistics,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.ppo_rrt_experiment import run_episode
from src.ppo_torch import PPOUpdater
from src.rollout import RolloutBuffer
from src.metrics import write_csv
def train(seed):
 torch.manual_seed(seed); agent=PPOUpdater(9); logs=[]
 for ep in range(40):
  buf=RolloutBuffer()
  for d in ['simple','medium','hard']:
   _,b,_=run_episode(d,seed*1000+ep,agent); buf.data.extend(b.data)
  logs.append(dict(agent.update(buf.data,epochs=4),episode=ep,transitions=len(buf.data)))
 return agent,logs
def main():
 out=Path('results/train_eval_v2'); out.mkdir(exist_ok=False); rows=[]; trainlogs=[]
 for seed in [0,1,2]:
  agent,logs=train(seed); trainlogs += [dict(x,seed=seed) for x in logs]; agent.save(str(out/f'model_{seed}.pt'))
  for d in ['simple','medium','hard']:
   for testseed in range(10):
    _,b,_=run_episode(d,10000+testseed,agent); rows.append({'seed':seed,'difficulty':d,'success':int(any(t.done and t.reward>=5 for t in b.data)),'transitions':len(b.data)})
 write_csv(trainlogs,str(out/'training.csv')); write_csv(rows,str(out/'evaluation.csv'))
 print({s:sum(r['success'] for r in rows if r['seed']==s)/30 for s in [0,1,2]})
if __name__=='__main__':main()
