import sys;sys.path.insert(0,'.')
from src.geometry_se3 import *
def test_q(): assert abs(qdist((1,0,0,0),(-1,0,0,0)))<1e-9
def test_interp():
 a=((0,0,0),(1,0,0,0)); b=((1,0,0),(0,1,0,0)); assert interpolate(a,b,0)==a and interpolate(a,b,1)==b
