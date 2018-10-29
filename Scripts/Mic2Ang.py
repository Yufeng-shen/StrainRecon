from util.MicFileTool import MicFile
import numpy as np
a=MicFile('AuxData/Ti7_WithHRM_2ndLoad_z1_.mic.LBFS')

xy=a.snp[:,:2]
phi1=a.snp[:,6]
psi=a.snp[:,7]
phi2=a.snp[:,8]
conf=a.snp[:,9]

gridX=np.arange(np.min(xy[:,0]),np.max(xy[:,0]),0.005)
gridY=np.arange(np.min(xy[:,1]),np.max(xy[:,1]),0.005)
xv,yv=np.meshgrid(gridX,gridY)
SqXY=np.hstack([xv.reshape((-1,1)),yv.reshape((-1,1))])
print(SqXY.shape)

import scipy.interpolate

ConfF=scipy.interpolate.griddata(xy, conf, SqXY, method='linear', rescale=False)
phi1F=scipy.interpolate.griddata(xy, phi1, SqXY, method='linear', rescale=False)
phi2F=scipy.interpolate.griddata(xy, phi2, SqXY, method='linear', rescale=False)
psiF=scipy.interpolate.griddata(xy, psi, SqXY, method='linear', rescale=False)

idx=np.invert(np.isnan(ConfF))
output=np.hstack([SqXY[idx,:],phi1F[idx,None],psiF[idx,None],phi2F[idx,None],ConfF[idx,None]])
np.savetxt('AuxData/Ti7_WithHRM_2ndLoad_z1_.ang.txt',output,fmt='%.8f')
