import sys,torch;sys.path.insert(0,'.')
from src.rollout import RolloutBuffer
from src.ppo_torch import PPOUpdater
def test_update_finite():
 p=PPOUpdater(3); b=RolloutBuffer()
 for i in range(8):
  o=[0.1,0.2,0.3]; logits,v=p.net(torch.tensor([o])); d=torch.distributions.Categorical(logits=logits); a=d.sample(); b.add(obs=o,action=int(a),log_prob=float(d.log_prob(a)),value=float(v),reward=.1,done=i==7,info={})
 s=p.update(b.data); assert all(torch.isfinite(torch.tensor(v)) for v in s.values())
