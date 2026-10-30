import sys,os,statistics,random,json
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
from src.environment import Env
from src.collision_checker import CollisionChecker
from src.rrt_star import RRTStar
from src.sampling_policies import make_sampler
from src.metrics import evaluate,write_csv

def soft_sampler(env):
 glob,goal,local=make_sampler('global',env),make_sampler('goal',env),make_sampler('local',env)
 calls=[0]; last=[float('inf')]; stall=[0]
 def f(rrt):
  calls[0]+=1; cur=rrt.best.cost if rrt.best else float('inf')
  if cur<last[0]-1e-7: stall[0]=0; last[0]=cur
  else: stall[0]+=1
  progress=min(1,calls[0]/160); s=min(1,stall[0]/35)
  # smooth probabilities, never eliminate exploration
  if rrt.best is None: pg=.45+.20*s; pt=.45-.10*s; pl=.10
  else: pg=.15+.45*s; pt=.20; pl=.65-pg
  pg=max(.10,min(.75,pg)); pt=max(.10,min(.55,pt)); pl=max(.10,1-pg-pt)
  z=rrt.rng.random()
  if z<pg:return glob(rrt)
  if z<pg+pt:return goal(rrt)
  return local(rrt)
 return f
rows=[]
for d in ['simple','medium','hard']:
 for seed in range(8):
  for a in ['rrtstar','fixed_mixed','soft_adaptive']:
   e=Env.demo(d,seed); c=CollisionChecker(e); sm=None; gb=.05
   if a=='fixed_mixed': sm=make_sampler('goal' if seed%2==0 else 'local',e); gb=.08
   if a=='soft_adaptive': sm=soft_sampler(e)
   p=RRTStar(e,c,seed,max_iter=180,goal_bias=gb,sampler=sm); path=p.run(); r=evaluate(path,e,p); r.update({'difficulty':d,'algorithm':a,'seed':seed}); rows.append(r)
write_csv(rows,'results/raw/soft_adaptive.csv')
for d in ['simple','medium','hard']:
 print(d)
 for a in ['rrtstar','fixed_mixed','soft_adaptive']:
  x=[r for r in rows if r['difficulty']==d and r['algorithm']==a]; ok=[r['cost'] for r in x if r['success']]; print(a, 'success',sum(r['success'] for r in x)/len(x),'cost',statistics.mean(ok) if ok else None,'ext',statistics.mean(r['extensions'] for r in x))
