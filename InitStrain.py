import util.Simulation as Gsim
import util.RotRep as Rot
import numpy as np
import matplotlib.pyplot as plt
import yaml

class Initializer(object):

    def __init__(self,cfgFn='ConfigFiles/g15Ps1_2nd.yml'):

        with open(cfgFn) as f:
            dataMap=yaml.safe_load(f)

        ##############
        # Files
        ##############
        self.bfPath=dataMap['Files']['bf-folder']
        self.fltPath=dataMap['Files']['filtered-folder']
        self.maxIntfn=dataMap['Files']['maxIntensity']
        self.micfn=dataMap['Files']['micFile']



        self.exp={'energy':dataMap['Setup']['Energy']}
        self.eng=self.exp['energy']
        self.etalimit=dataMap['Setup']['Eta-Limit']/180.0*np.pi
        self.omgRange=dataMap['Setup']['Omega-Range'] 
        ########################
        # Detector parameters
        ########################

        self.Det=Gsim.Detector(psize=dataMap['Setup']['Pixel-Size']/1000.0)
        self.Det.Move(dataMap['Setup']['J-Center'],
                dataMap['Setup']['K-Center'],
                np.array([dataMap['Setup']['Distance'],0,0]),
                Rot.EulerZXZ2Mat(np.array(dataMap['Setup']['Tilt'])/180.0*np.pi))

        #########################
        # LP
        #########################
        self.Ti7LP=Gsim.CrystalStr()
        self.Ti7LP.PrimA=dataMap['Material']['Lattice'][0]*np.array([1,0,0])
        self.Ti7LP.PrimB=dataMap['Material']['Lattice'][1]*np.array([np.cos(np.pi*2/3),np.sin(np.pi*2/3),0])
        self.Ti7LP.PrimC=dataMap['Material']['Lattice'][2]*np.array([0,0,1])
        Atoms=dataMap['Material']['Atoms']
        for ii in range(len(Atoms)):
            self.Ti7LP.addAtom(list(map(eval,Atoms[ii][0:3])),Atoms[ii][3])
        self.Ti7LP.getRecipVec()
        self.Ti7LP.getGs(dataMap['Material']['MaxQ'])

        ##########################################
        # Grain
        ##########################################
        self.pos=np.array(dataMap['Grain']['Pos'])
        self.orien=dataMap['Grain']['Orien']
        self.orienM=Rot.EulerZXZ2Mat(np.array(self.orien)/180.0*np.pi)


    def Simulate(self):
        self.Ps,self.Gs,self.Info=Gsim.GetProjectedVertex(self.Det,
                self.Ti7LP,self.orienM,self.etalimit,
                self.pos,getPeaksInfo=True,
                omegaL=self.omgRange[0],
                omegaU=self.omgRange[1],**(self.exp))
        self.NumG=len(self.Gs)

    def Move(self,dJ=0,dK=0,dD=0,dT=np.eye(3)):
        self.Det.Move(dJ,dK,np.array([dD,0,0]),dT)
