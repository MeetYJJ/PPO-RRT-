import sys,torch;sys.path.insert(0,'.')
from src.rollout import DecisionTransition,RolloutBuffer
from src.ppo_torch import PPOUpdater
def test_real_action_and_ratio_one():
 p=PPOUpdater(2); o=[.1,.2]; logits,v=p.net(torch.tensor([o])); d=torch.distributions.Categorical(logits=logits); a=d.sample(); lp=float(d.log_prob(a)); b=RolloutBuffer(); b.add(obs=o,action=int(a),log_prob=lp,value=float(v),reward=1.,next_obs=o,next_value=0.,done=True,duration=3,info={'executed_action':int(a)}); logits2,_=p.net(torch.tensor([o])); assert abs(float(torch.distributions.Categorical(logits=logits2).log_prob(a))-lp)<1e-6; s=p.update(b.data); assert all(torch.isfinite(torch.tensor(x)) for x in s.values())
def test_terminal_no_bootstrap():
 p=PPOUpdater(1); b=RolloutBuffer(); b.add(obs=[0.],action=0,log_prob=0.,value=0.,reward=1.,next_obs=[0.],next_value=100.,done=True,duration=2,info={}); s=p.update(b.data); assert s['mean_reward']==1.
