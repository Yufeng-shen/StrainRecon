from config import Config
from reconstructor import Reconstructor
c=Config('ConfigFiles/RecG40_dL5.yml')
r=Reconstructor(c)
r.run()
c=Config('ConfigFiles/RecG40_dJ3.yml')
r=Reconstructor(c)
r.run()
c=Config('ConfigFiles/RecG40_dK1.yml')
r=Reconstructor(c)
r.run()
