import numpy as np
from Reconst_GPU import StrainReconstructor_GPU, ReconSingleGrain
from InitStrain import Initializer

outname='/home/yufengs/Strain/Results/dJ0dK0dD5dT0/g15_2nd/'

Cfg=Initializer('ConfigFiles/g15Ps1_2nd.yml')

Cfg.Simulate()

Cfg.Move(dD=0.005)

print(Cfg.NumG)

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
