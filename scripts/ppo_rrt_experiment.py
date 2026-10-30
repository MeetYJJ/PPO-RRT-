import os,sys,torch
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
from src.environment import Env
from src.collision_checker import CollisionChecker
from src.rrt_star import RRTStar
from src.sampling_policies import make_sampler
from src.rl_scheduler import observation
from src.rollout import RolloutBuffer
from src.ppo_torch import PPOUpdater
from src.metrics import evaluate,write_csv
MODES=['global','goal','local','uncertainty']
class WindowController:
 def __init__(self,env,updater,buffer,window=10):
  self.env,self.updater,self.buffer,self.window=env,updater,buffer,window; self.n=0; self.action=None; self.samplers={m:make_sampler(m,env,.05) for m in MODES}; self.rows=[]; self.version=0
 def _start(self,rrt):
  o=observation(rrt,self.env,.05,.2)
  with torch.no_grad():
   logits,v=self.updater.net(torch.tensor([o],dtype=torch.float32)); d=torch.distributions.Categorical(logits=logits); a=d.sample()
  self.action=int(a); self.old_lp=float(d.log_prob(a)); self.old_v=float(v); self.obs=o; self.before_best=rrt.best.cost if rrt.best else float('inf'); self.before_dist=min((sum((n.pose[0][i]-self.env.goal[0][i])**2 for i in range(3))**0.5 for n in rrt.nodes),default=1.0); self.before_ext=rrt.extensions; self.before_calls=rrt.chk.calls; self.n=0; self.version+=1
 def sample(self,rrt):
  if self.action is None or self.n>=self.window:self._start(rrt)
  self.n+=1; return self.samplers[MODES[self.action]](rrt)
 def after_iteration(self,rrt,it):
  done=rrt.best is not None and rrt.first_solution_iter is not None and rrt.first_solution_iter==it+1
  if self.action is None:
   return bool(done or it+1>=rrt.max_iter)
  if self.n>=self.window or done or it+1>=rrt.max_iter:
   next_obs=observation(rrt,self.env,.05,.2)
   with torch.no_grad(): _,nv=self.updater.net(torch.tensor([next_obs],dtype=torch.float32))
   current=rrt.best.cost if rrt.best else float('inf'); improve=0.0 if not (self.before_best<float('inf') and current<float('inf')) else max(0.0,self.before_best-current); after_dist=min((sum((n.pose[0][i]-self.env.goal[0][i])**2 for i in range(3))**0.5 for n in rrt.nodes),default=1.0); progress=max(-.2,min(.2,self.before_dist-after_dist)); valid=(rrt.extensions-self.before_ext)/max(1,self.n)
   reward=(5.0 if done else 0.0)+2.0*improve+0.5*progress+0.1*valid-0.01*self.n; terminal=bool(done or it+1>=rrt.max_iter)
   self.buffer.add(obs=self.obs,action=self.action,log_prob=self.old_lp,value=self.old_v,reward=reward,next_obs=next_obs,next_value=float(nv),done=terminal,duration=self.n,info={'mode':MODES[self.action],'extensions':rrt.extensions-self.before_ext,'collision_checks':rrt.chk.calls-self.before_calls,'policy_version':self.version})
   self.rows.append({'window':len(self.rows),'action':self.action,'mode':MODES[self.action],'reward':reward,'duration':self.n,'done':int(terminal),'extensions':rrt.extensions-self.before_ext,'collision_checks':rrt.chk.calls-self.before_calls,'policy_version':self.version}); self.action=None
  return bool(done or it+1>=rrt.max_iter)
def run_episode(diff,seed,updater):
 e=Env.demo(diff,seed); c=CollisionChecker(e); buf=RolloutBuffer(); ctl=WindowController(e,updater,buf,10); p=RRTStar(e,c,seed,max_iter=60,sampler=ctl.sample,iteration_callback=ctl.after_iteration); path=p.run(); return evaluate(path,e,p),buf,ctl.rows
def main():
 updater=PPOUpdater(9); os.makedirs('models',exist_ok=True); logs=[]
 for ep in range(6):
  episode_buf=RolloutBuffer()
  for d in ['simple','medium','hard']:
   result,b,wr=run_episode(d,ep,updater); episode_buf.data.extend(b.data); logs.extend([dict(x,episode=ep,difficulty=d) for x in wr])
  print('train',ep,'transitions',len(episode_buf.data),updater.update(episode_buf.data,epochs=2))
 updater.save('models/ppo_rrt_planner_creditfix.pt'); write_csv(logs,'results/raw/ppo_rrt_creditfix_smoke.csv'); print('saved',len(logs),'windows')
if __name__=='__main__':main()
