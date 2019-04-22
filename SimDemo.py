from config import Config
from simulator import Simulator
c=Config('ConfigFiles/SimG40.yml')
s=Simulator(c)
s.SimSingleGrain()
