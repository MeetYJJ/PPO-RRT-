import sys;sys.path.insert(0,'.')
from src.ppo_agent import *
def test_probs():
 a=PPOAgent(); p=a.probs([0]*8); assert abs(sum(p)-1)<1e-9 and all(x>=0 for x in p)
