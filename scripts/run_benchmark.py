import sys,os,json,csv
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
from src.environment import Env
from src.collision_checker import CollisionChecker
from src.rrt_star import RRTStar
from src.sampling_policies import make_sampler
from src.metrics import evaluate,write_csv,summary

def main():
 rows=[]
 for diff in ['simple','medium','hard']:
  for alg in ['rrtstar','goal_bias','mixed','ppo_no_unc','ppo']:
   for seed in range(3):
    e=Env.demo(diff,seed); c=CollisionChecker(e); sampler=None; gb=.05
    if alg=='goal_bias': sampler=make_sampler('goal',e); gb=.15
    elif alg in ('mixed','ppo_no_unc'): sampler=make_sampler('local' if seed%2 else 'goal',e)
    elif alg=='ppo': sampler=make_sampler(['global','goal','local','uncertainty'][seed%4],e)
    p=RRTStar(e,c,seed,max_iter=250,goal_bias=gb,sampler=sampler); path=p.run(); row=evaluate(path,e,p); row.update({'difficulty':diff,'algorithm':alg,'seed':seed}); rows.append(row)
 write_csv(rows,'results/raw/benchmark.csv'); json.dump(summary(rows),open('results/reports/summary.json','w'),indent=2); print(summary(rows))
if __name__=='__main__':main()
