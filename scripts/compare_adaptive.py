import sys,os,csv,statistics
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
from src.environment import Env
from src.collision_checker import CollisionChecker
from src.rrt_star import RRTStar
from src.sampling_policies import make_sampler
from src.metrics import evaluate,write_csv

def adaptive_sampler(env):
 count=[0]; last=[float('inf')]; stall=[0]
 def f(rrt):
  count[0]+=1; cur=rrt.best.cost if rrt.best else float('inf')
  if cur < last[0]-1e-6: stall[0]=0; last[0]=cur
  else: stall[0]+=1
  # exploration early and after stagnation; exploit goal/local otherwise
  if count[0] < 70 or stall[0] > 35: mode='global'
  elif rrt.best is None: mode='goal'
  else: mode='local'
  return make_sampler(mode,env,0.08)(rrt)
 return f
rows=[]
for diff in ['simple','medium','hard']:
 for seed in range(4):
  for alg in ['rrtstar','mixed','adaptive']:
   e=Env.demo(diff,seed); c=CollisionChecker(e)
   sampler=None; gb=.05
   if alg=='mixed': sampler=make_sampler('goal' if seed%2==0 else 'local',e); gb=.08
   if alg=='adaptive': sampler=adaptive_sampler(e); gb=.05
   p=RRTStar(e,c,seed,max_iter=180,goal_bias=gb,sampler=sampler); path=p.run(); r=evaluate(path,e,p); r.update({'difficulty':diff,'algorithm':alg,'seed':seed}); rows.append(r)
write_csv(rows,'results/raw/adaptive_comparison.csv')
for d in ['simple','medium','hard']:
 print(d)
 for a in ['rrtstar','mixed','adaptive']:
  x=[r for r in rows if r['difficulty']==d and r['algorithm']==a]; ok=[r for r in x if r['success']]; print(a,'success',sum(r['success'] for r in x)/len(x),'cost',statistics.mean([r['cost'] for r in ok]) if ok else None,'ext',statistics.mean(r['extensions'] for r in x))
