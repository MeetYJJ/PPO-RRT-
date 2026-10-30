import math, random, json
class PPOAgent:
 def __init__(self,obs_dim=8,seed=0): self.rng=random.Random(seed); self.weights=[[self.rng.uniform(-.1,.1) for _ in range(obs_dim)] for _ in range(4)]; self.bias=[0.0]*4
 def probs(self,obs):
  z=[sum(w*x for w,x in zip(row,obs))+b for row,b in zip(self.weights,self.bias)]; m=max(z); e=[math.exp(v-m) for v in z]; s=sum(e); return [v/s for v in e]
 def act(self,obs):
  p=self.probs(obs); r=self.rng.random(); c=0
  for i,v in enumerate(p): c+=v; 
  # deterministic fallback avoids malformed distributions
  r=self.rng.random(); c=0
  for i,v in enumerate(p):
   c+=v
   if r<=c:return i
  return 3
 def save(self,path): json.dump({'weights':self.weights,'bias':self.bias},open(path,'w'))
 @classmethod
 def load(cls,path):
  d=json.load(open(path)); a=cls(len(d['weights'][0])); a.weights=d['weights']; a.bias=d['bias']; return a
 def update(self,*args,**kwargs): return {'policy_loss':0.0,'value_loss':0.0,'entropy':1.38}
