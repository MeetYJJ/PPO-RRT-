import os,sys,math,random
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
from src.rollout import RolloutBuffer
from src.ppo_torch import PPOUpdater
b=RolloutBuffer(); rng=random.Random(0); p=PPOUpdater(9)
for i in range(64):
 obs=[rng.random() for _ in range(9)]; logits,v=p.net(__import__('torch').tensor([obs],dtype=__import__('torch').float32)); d=__import__('torch').distributions.Categorical(logits=logits); a=d.sample(); b.add(obs=obs,action=int(a),log_prob=float(d.log_prob(a)),reward=rng.random(),value=float(v),done=(i==63),info={'window':10})
print(p.update(b.data)); os.makedirs('models',exist_ok=True); p.save('models/ppo_torch_smoke.pt'); p.load('models/ppo_torch_smoke.pt'); print('saved_loaded')
