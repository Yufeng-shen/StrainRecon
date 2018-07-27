import numpy as np
from Reconst_GPU import StrainReconstructor_GPU, ReconSingleGrain
from InitStrain import Det1, g13Gs1_2nd, g13Info1_2nd, g13pos2nd, eng, g13orien2nd

recon13=StrainReconstructor_GPU( _NumG=108,
        bfPath='/home/yufengs/workspace/g13Ps1_2nd_bf',
        fltPath='/home/yufengs/workspace/g13Ps1_2nd_filtered',
        maxIntfn='AuxData/MaxInt_g13Ps1_2nd.npy',
        _Det=Det1, _Gs=g13Gs1_2nd, _Info=g13Info1_2nd, _eng=eng)

ReconGrain13=ReconSingleGrain(grainOrien=[120.784, 80.9295, 246.202],
        micfn='AuxData/Ti7_WithHRM_2ndLoad_z1_.mic.LBFS')

x,y,con=ReconGrain13.GetGrids()
np.save('/home/yufengs/Strain/Results/g13_2nd/x.npy',x)
np.save('/home/yufengs/Strain/Results/g13_2nd/y.npy',y)
np.save('/home/yufengs/Strain/Results/g13_2nd/con.npy',con)
AllMaxScore,AllMaxS=ReconGrain13.ReconGrids(x,y,recon13)
np.save('/home/yufengs/Strain/Results/g13_2nd/allMaxScore.npy',AllMaxScore)
np.save('/home/yufengs/Strain/Results/g13_2nd/allMaxS.npy',AllMaxS)
realO,realS=ReconGrain13.Transform2RealS(AllMaxS)
np.save('/home/yufengs/Strain/Results/g13_2nd/realS.npy',realS)
np.save('/home/yufengs/Strain/Results/g13_2nd/realO.npy',realO)
