import numpy as np
import time
from Reconst_GPU import StrainReconstructor_GPU, ReconSingleGrain
from InitStrain import Initializer
import os

start=time.time()

outname='/home/yufengs/Results/g15_2nd/'
cfgFile='ConfigFiles/g15Ps1_2nd.yml'

if not os.path.exists(outname):
    os.makedirs(outname)

print("Start running \n Output Directory: " +outname+ "\n Configure File: "+cfgFile)
Cfg=Initializer(cfgFile)

Cfg.Simulate()


print("Initialized")

recon=StrainReconstructor_GPU( _NumG=Cfg.NumG,
        bfPath=Cfg.bfPath,
        fltPath=Cfg.fltPath,
        maxIntfn=Cfg.maxIntfn,
        _Det=Cfg.Det, _Gs=Cfg.Gs, _Info=Cfg.Info, _eng=Cfg.eng)

ReconGrain=ReconSingleGrain(grainOrien=Cfg.orien,
        micfn=Cfg.micfn)

x,y,con=ReconGrain.GetGrids()
np.save(outname+'x.npy',x)
np.save(outname+'y.npy',y)
np.save(outname+'con.npy',con)
AllMaxScore,AllMaxS=ReconGrain.ReconGrids(x,y,recon)
np.save(outname+'allMaxScore.npy',AllMaxScore)
np.save(outname+'allMaxS.npy',AllMaxS)
realO,realS=ReconGrain.Transform2RealS(AllMaxS)
np.save(outname+'realS.npy',realS)
np.save(outname+'realO.npy',realO)

end=time.time()
print("Time elapsed: {:f} seconds".format(end-start))
