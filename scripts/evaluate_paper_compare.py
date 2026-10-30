import sys,os,statistics,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.ppo_rrt_experiment import run_episode
from src.environment import Env
from src.collision_checker import CollisionChecker
from src.rrt_star import RRTStar
from src.sampling_policies import make_sampler
from src.metrics import evaluate,write_csv
from src.ppo_torch import PPOUpdater

def soft(e):
 glob,goal,local=make_sampler('global',e),make_sampler('goal',e),make_sampler('local',e); count=[0]; stall=[0]; last=[float('inf')]
 def f(r):
  count[0]+=1; cur=r.best.cost if r.best else float('inf'); stall[0]=0 if cur<last[0]-1e-7 else stall[0]+1; last[0]=cur; s=min(1,stall[0]/35)
  if r.best is None: pg=.45+.2*s; pt=.45-.1*s
  else: pg=.15+.45*s; pt=.2
  z=r.rng.random(); return glob(r) if z<pg else goal(r) if z<pg+pt else local(r)
 return f
rows=[]; model=Path('results/train_eval_v2/model_1.pt'); agent=PPOUpdater(9); agent.load(str(model))
for d in ['simple','medium','hard']:
 for seed in range(10):
  for a in ['rrtstar','fixed_mixed','soft_hand','ppo']:
   e=Env.demo(d,20000+seed); c=CollisionChecker(e); sm=None
   if a=='fixed_mixed':sm=make_sampler('goal' if seed%2==0 else 'local',e)
   if a=='soft_hand':sm=soft(e)
   if a=='ppo':
    _,b,_=run_episode(d,20000+seed,agent); ok=any(t.done and t.reward>=5 for t in b.data); rows.append({'difficulty':d,'seed':seed,'algorithm':a,'success':int(ok),'cost':'nan','extensions':'nan'}); continue
   p=RRTStar(e,c,20000+seed,max_iter=120,sampler=sm); path=p.run(); r=evaluate(path,e,p); r.pop('sample_counts',None); r.update({'difficulty':d,'seed':seed,'algorithm':a}); rows.append(r)
write_csv(rows,'results/raw/paper_ppo_compare.csv')
for d in ['simple','medium','hard','all']:
 print(d)
 for a in ['rrtstar','fixed_mixed','soft_hand','ppo']:
  z=[r for r in rows if r['algorithm']==a and (d=='all' or r['difficulty']==d)]; print(a,round(sum(int(r['success']) for r in z)/len(z),3))
