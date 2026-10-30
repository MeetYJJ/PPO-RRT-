import sys,os,csv,statistics,random,math,json
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
from src.environment import Env
from src.collision_checker import CollisionChecker
from src.rrt_star import RRTStar
from src.sampling_policies import make_sampler
from src.metrics import evaluate,write_csv

def adaptive_sampler(env):
 count=[0]; last=[float('inf')]; stall=[0]; goal=make_sampler('goal',env); local=make_sampler('local',env); glob=make_sampler('global',env)
 def f(rrt):
  count[0]+=1; cur=rrt.best.cost if rrt.best else float('inf')
  if cur < last[0]-1e-8: stall[0]=0; last[0]=cur
  else: stall[0]+=1
  if count[0] < 45 or stall[0] >= 25: return glob(rrt)
  if rrt.best is None: return goal(rrt)
  return local(rrt)
 return f
rows=[]
for diff in ['simple','medium','hard']:
 for seed in range(8):
  for alg in ['rrtstar','fixed_mixed','adaptive']:
   e=Env.demo(diff,seed); c=CollisionChecker(e); sampler=None; gb=.05
   if alg=='fixed_mixed': sampler=make_sampler('goal' if seed%2==0 else 'local',e); gb=.08
   elif alg=='adaptive': sampler=adaptive_sampler(e)
   p=RRTStar(e,c,seed,max_iter=100,goal_bias=gb,sampler=sampler); path=p.run(); r=evaluate(path,e,p)
   r.update({'difficulty':diff,'algorithm':alg,'seed':seed,'budget':100}); rows.append(r)
write_csv(rows,'results/raw/adaptive_full.csv')
def boot(x,n=2000,seed=7):
 if not x:return [float('nan'),float('nan')]
 rng=random.Random(seed); means=[]
 for _ in range(n): means.append(statistics.mean(rng.choice(x) for _ in x))
 means.sort(); return [means[int(.025*n)],means[int(.975*n)]]
summary={}
for d in ['simple','medium','hard','all']:
 summary[d]={}
 for a in ['rrtstar','fixed_mixed','adaptive']:
  x=[r for r in rows if (d=='all' or r['difficulty']==d) and r['algorithm']==a]; ok=[r for r in x if r['success']==1]
  rate=sum(r['success'] for r in x)/len(x); costs=[r['cost'] for r in ok]; ext=[r['extensions'] for r in x]
  summary[d][a]={'n':len(x),'success_rate':rate,'success_ci':boot([float(r['success']) for r in x]),'cost_success_mean':statistics.mean(costs) if costs else None,'cost_success_ci':boot(costs),'extensions_mean':statistics.mean(ext)}
json.dump(summary,open('results/reports/adaptive_full_summary.json','w'),indent=2)
for d,v in summary.items():
 print('\n',d)
 for a,z in v.items(): print(a,'rate',round(z['success_rate'],3),'CI', [round(q,3) for q in z['success_ci']], 'cost', None if z['cost_success_mean'] is None else round(z['cost_success_mean'],3),'ext',round(z['extensions_mean'],1))
