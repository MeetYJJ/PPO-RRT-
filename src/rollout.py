from dataclasses import dataclass,asdict
import json
@dataclass
class DecisionTransition:
    obs:list; action:int; log_prob:float; value:float; reward:float; next_obs:list=None; next_value:float=0.; done:bool=False; duration:int=1; info:dict=None
class RolloutBuffer:
 def __init__(self): self.data=[]
 def add(self,**kw): self.data.append(DecisionTransition(**kw))
 def clear(self): self.data.clear()
 def save_jsonl(self,path):
  with open(path,'w') as f:
   for x in self.data:f.write(json.dumps(asdict(x))+'\n')
