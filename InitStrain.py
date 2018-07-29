import util.Simulation as Gsim
import util.RotRep as Rot
import numpy as np
import matplotlib.pyplot as plt
import yaml

with open('g15Ps1_2nd.yml') as f:
    dataMap=yaml.safe_load(f)

exp={'energy':dataMap['Setup']['Energy']}
eng=exp['energy']
etalimit=81/180.0*np.pi

########################
# Detector parameters
########################

Det1=Gsim.Detector(psize=dataMap['Setup']['Pixel-Size']/1000.0)
Det1.Move(dataMap['Setup']['J-Center'],
        dataMap['Setup']['K-Center'],
        np.array([dataMap['Setup']['Distance'],0,0]),
        Rot.EulerZXZ2Mat(np.array(dataMap['Setup']['Tilt'])/180.0*np.pi))

#########################
# LP
#########################
Ti7LP=Gsim.CrystalStr()
Ti7LP.PrimA=dataMap['Material']['Lattice'][0]*np.array([1,0,0])
Ti7LP.PrimB=dataMap['Material']['Lattice'][1]*np.array([np.cos(np.pi*2/3),np.sin(np.pi*2/3),0])
Ti7LP.PrimC=dataMap['Material']['Lattice'][2]*np.array([0,0,1])
Atoms=dataMap['Material']['Atoms']
for ii in range(len(Atoms)):
    Ti7LP.addAtom(list(map(eval,Atoms[ii][0:3])),Atoms[ii][3])
Ti7LP.getRecipVec()
Ti7LP.getGs(dataMap['Material']['MaxQ'])

##########################################
# grain 15, 2nd Load
##########################################
pos=np.array(dataMap['Grain']['Pos'])
orien=Rot.EulerZXZ2Mat(np.array(dataMap['Grain']['Orien'])/180.0*np.pi)
Ps,Gs,Info=Gsim.GetProjectedVertex(Det1,
        Ti7LP,orien,etalimit,
        pos,getPeaksInfo=True,
        omegaL=dataMap['Setup']['Omega-Range'][0],
        omegaU=dataMap['Setup']['Omega-Range'][1],**exp)

