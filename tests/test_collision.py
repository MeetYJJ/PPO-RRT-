import sys;sys.path.insert(0,'.')
from src.environment import *
from src.collision_checker import *
def test_edge():
 e=Env(obstacles=(Sphere((.5,.5,.5),.15),)); c=CollisionChecker(e); a=((.1,.5,.5),(1,0,0,0));b=((.9,.5,.5),(1,0,0,0)); assert not c.edge(a,b)
