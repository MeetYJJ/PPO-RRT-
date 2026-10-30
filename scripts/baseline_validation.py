import sys,os,statistics,random,math,json
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
from src.environment import Env
from src.collision_checker import CollisionChecker
from src.rrt_star import RRTStar
from src.sampling_policies import make_sampler,make_faroni_inspired_sampler
from src.metrics import evaluate,write_csv

def soft_sampler(env):
 glob,goal,local=make_sampler('global',env),make_sampler('goal',env),make_sampler('local',env); calls=[0]; last=[float('inf')]; stall=[0]
 def f(rrt):
  calls[0]+=1; cur=rrt.best.cost if rrt.best else float('inf')
  if cur<last[0]-1e-7: stall[0]=0; last[0]=cur
  else: stall[0]+=1
  s=min(1,stall[0]/35)
  if rrt.best is None: pg=.45+.20*s; pt=.45-.10*s; pl=.10
  else: pg=.15+.45*s; pt=.20; pl=.65-pg
  pg=max(.10,min(.75,pg)); pt=max(.10,min(.55,pt)); z=rrt.rng.random()
  if z<pg:return glob(rrt)
  if z<pg+pt:return goal(rrt)
  return local(rrt)
 return f

def flat(r):
 d=dict(r); sc=d.pop('sample_counts',{})
 for k,v in sc.items(): d['samples_'+k]=v
 return d
rows=[]
for diff in ['simple','medium','hard']:
 for seed in range(50):
  for alg in ['rrtstar','fixed_mixed','soft_adaptive','faroni_inspired']:
   e=Env.demo(diff,seed); c=CollisionChecker(e); sm=None; gb=.05
   if alg=='fixed_mixed': sm=make_sampler('goal' if seed%2==0 else 'local',e); gb=.08
   elif alg=='soft_adaptive': sm=soft_sampler(e)
   elif alg=='faroni_inspired': sm=make_faroni_inspired_sampler(e)
   p=RRTStar(e,c,seed,max_iter=120,goal_bias=gb,sampler=sm); path=p.run(); r=flat(evaluate(path,e,p)); r.update({'difficulty':diff,'algorithm':alg,'seed':seed,'budget':120}); rows.append(r)
  if seed%10==9: print(diff,seed+1,'scenes complete',flush=True)
write_csv(rows,'results/raw/baseline_validation.csv')
def boot(x,n=3000):
 if not x:return [float('nan'),float('nan')]
 rng=random.Random(123); z=sorted(statistics.mean(rng.choice(x) for _ in x) for _ in range(n)); return [z[int(.025*n)],z[int(.975*n)]]
summary={}
for d in ['simple','medium','hard','all']:
 summary[d]={}
 for a in ['rrtstar','fixed_mixed','soft_adaptive','faroni_inspired']:
  x=[r for r in rows if (d=='all' or r['difficulty']==d) and r['algorithm']==a]; ok=[r for r in x if r['success']==1]
  summary[d][a]={'n':len(x),'success_rate':sum(r['success'] for r in x)/len(x),'success_ci':boot([float(r['success']) for r in x]),'cost_mean_success':statistics.mean(float(r['cost']) for r in ok) if ok else None,'cost_ci_success':boot([float(r['cost']) for r in ok]),'time_to_first_mean':statistics.mean(float(r['first_solution_iter']) for r in ok) if ok else None,'extensions_mean':statistics.mean(float(r['extensions']) for r in x),'clearance_mean_success':statistics.mean(float(r['min_clearance']) for r in ok) if ok else None}
json.dump(summary,open('results/reports/baseline_validation_summary.json','w'),indent=2)
for d,v in summary.items():
 print('\n'+d)
 for a,z in v.items(): print(a,'success',round(z['success_rate'],3),'CI',[round(q,3) for q in z['success_ci']],'cost',z['cost_mean_success'])
