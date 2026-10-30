import torch
from torch import nn,optim
from torch.distributions import Categorical
class ActorCritic(nn.Module):
 def __init__(self,obs_dim,act_dim=4,hidden=64):
  super().__init__(); self.body=nn.Sequential(nn.Linear(obs_dim,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh()); self.actor=nn.Linear(hidden,act_dim); self.critic=nn.Linear(hidden,1)
 def forward(self,x):
  h=self.body(x); return self.actor(h),self.critic(h).squeeze(-1)
class PPOUpdater:
 def __init__(self,obs_dim,act_dim=4,lr=3e-4,gamma=.99,gae_lambda=.95,clip=.2,entropy=.01):
  self.net=ActorCritic(obs_dim,act_dim); self.opt=optim.Adam(self.net.parameters(),lr=lr); self.gamma=gamma; self.lam=gae_lambda; self.clip=clip; self.entropy=entropy
 def update(self,transitions,epochs=4,batch_size=32):
  if not transitions:return {}
  obs=torch.tensor([t.obs for t in transitions],dtype=torch.float32); act=torch.tensor([t.action for t in transitions]); oldlp=torch.tensor([t.log_prob for t in transitions]); rew=torch.tensor([t.reward for t in transitions],dtype=torch.float32); vals=torch.tensor([t.value for t in transitions],dtype=torch.float32); nextv=torch.tensor([t.next_value for t in transitions],dtype=torch.float32); done=torch.tensor([t.done for t in transitions],dtype=torch.float32); dur=torch.tensor([t.duration for t in transitions],dtype=torch.float32)
  adv=torch.zeros(len(transitions)); gae=0.
  for i in reversed(range(len(transitions))):
   disc=self.gamma**dur[i]; delta=rew[i]+disc*(1-done[i])*nextv[i]-vals[i]; gae=delta+disc*self.lam*(1-done[i])*gae; adv[i]=gae
  ret=adv+vals
  if len(adv)>1: adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
  else: adv=adv*0
  stats={}
  for _ in range(epochs):
   logits,val=self.net(obs); dist=Categorical(logits=logits); lp=dist.log_prob(act); ratio=torch.exp(lp-oldlp); s1=ratio*adv; s2=torch.clamp(ratio,1-self.clip,1+self.clip)*adv; pl=-torch.min(s1,s2).mean(); vl=.5*(ret-val).pow(2).mean(); ent=dist.entropy().mean(); loss=pl+vl-self.entropy*ent
   self.opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.net.parameters(),.5); self.opt.step()
  return {'policy_loss':float(pl.detach()),'value_loss':float(vl.detach()),'entropy':float(ent.detach()),'mean_reward':float(rew.mean()),'mean_duration':float(dur.mean())}
 def save(self,path): torch.save(self.net.state_dict(),path)
 def load(self,path): self.net.load_state_dict(torch.load(path,map_location='cpu'))
