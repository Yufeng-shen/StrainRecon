from config import Config
from reconstructor import Reconstructor
c=Config('ConfigFiles/RecG40.yml')
r=Reconstructor(c)
r.run()
