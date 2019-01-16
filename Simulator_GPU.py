# coding: utf-8

import pycuda.gpuarray as gpuarray
from pycuda.autoinit import context
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from scipy.linalg import polar
from scipy.ndimage import zoom
from util.MicFileTool import read_mic_file
import util.RotRep as Rot

from InitStrain import Initializer
import os
from collections import Counter
from scipy.sparse import coo_matrix
from scipy.ndimage import gaussian_filter

from pycuda.compiler import SourceModule
import h5py
class StrainSimulator_GPU(object):
    def __init__(self,_NumG,_Lim,
            _Det, _Gs, _Info, _eng):

        with open('strain_device.cu','r') as cudaSrc:
            src=cudaSrc.read()
        mod = SourceModule(src)
        self.sim_grain = mod.get_function('Simulate_for_Pos')
        self.tGref = mod.get_texref('tfG')
        self.Gs= _Gs
        self.eng= _eng
        # Number of G vectors
        self.NumG= _NumG
        ## Lim for window position
        Lim=np.array(_Lim).astype(np.int32)
        self.LimD=gpuarray.to_gpu(Lim)
        ## whichOmega for choosing between omega1 and omega2
        whichOmega=np.zeros(len(Lim),dtype=np.int32)
        for ii in range(len(Lim)):
            if _Info[ii]['WhichOmega']=='b':
                whichOmega[ii]=2
            else:
                whichOmega[ii]=1
        self.whichOmegaD=gpuarray.to_gpu(whichOmega)
        # MaxInt for normalize the weight of each spot 
        #(because different spots have very different intensity but we want them equal weighted) 

        # Det parameters
        self.Det=_Det
        afDetInfoH=np.concatenate([[2048,2048,0.001454,0.001454],
                                   _Det.CoordOrigin,_Det.Norm,_Det.Jvector,_Det.Kvector]).astype(np.float32)
        self.afDetInfoD=gpuarray.to_gpu(afDetInfoH)
        self.GsLoaded=False

    #transfer the ExpImgs and all Gs to texture memory
    def loadGs(self): 
        self.tGref.set_array(cuda.matrix_to_array(np.transpose(self.Gs).astype(np.float32),order='F'))
        self.GsLoaded=True

    def Simulate(self,xs,ys,ss,NumD):
        if self.GsLoaded==False:
            self.loadGs()
        BlockSize=256
        XD=gpuarray.empty(self.NumG*NumD,dtype=np.int32)
        YD=gpuarray.empty(self.NumG*NumD,dtype=np.int32)
        OffsetD=gpuarray.empty(self.NumG*NumD,dtype=np.int32)
        MaskD=gpuarray.empty(self.NumG*NumD,dtype=np.bool_)
        TrueMaskD=gpuarray.empty(self.NumG*NumD,dtype=np.bool_)

        xsD=gpuarray.to_gpu(xs.astype(np.float32))
        ysD=gpuarray.to_gpu(ys.astype(np.float32))
        ssD=gpuarray.to_gpu(ss.ravel(order='C').astype(np.float32))

        self.sim_grain(XD,YD,OffsetD,MaskD,TrueMaskD,
                xsD,ysD,self.afDetInfoD,ssD,
                self.whichOmegaD,np.int32(NumD),np.int32(self.NumG),
                np.float32(self.eng),np.int32(45),self.LimD,np.int32(5),
                block=(self.NumG,1,1),grid=(NumD,1))
        xtmp=XD.get().reshape((-1,self.NumG))
        ytmp=YD.get().reshape((-1,self.NumG))
        otmp=OffsetD.get().reshape((-1,self.NumG))
        maskH=MaskD.get().reshape(-1,self.NumG)
        return xtmp,ytmp,otmp,maskH

    def MoveDet(self,dJ=0,dK=0,dD=0,dT=np.eye(3)):
        self.Det.Move(dJ,dK,np.array([dD,0,0]),dT)
        afDetInfoH=np.concatenate([[2048,2048,0.001454,0.001454],
                                   self.Det.CoordOrigin,self.Det.Norm,
                                   self.Det.Jvector,self.Det.Kvector]).astype(np.float32)
        self.afDetInfoD=gpuarray.to_gpu(afDetInfoH)
    def ResetDet(self):
        self.Det.Reset()
        afDetInfoH=np.concatenate([[2048,2048,0.001454,0.001454],
                                   self.Det.CoordOrigin,self.Det.Norm,
                                   self.Det.Jvector,self.Det.Kvector]).astype(np.float32)
        self.afDetInfoD=gpuarray.to_gpu(afDetInfoH)



class SimAllGrains(object):

    def __init__(self,cfgFile,outdir,orig=[-0.256,-0.256],step=[0.002,0.002],scale=10,factor=40):
        self.Cfg=Initializer(cfgFile)
        self.FakeSample=np.load(self.Cfg.micfn)
        self.outdir=outdir
        # create a finer grid for more realistic simulation
        FakeSample=self.FakeSample
        finerDilation=zoom(FakeSample[0],zoom=scale,order=0)*factor
        finerE11=zoom(FakeSample[1],zoom=scale,order=0)*factor
        finerE12=zoom(FakeSample[2],zoom=scale,order=0)*factor
        finerE13=zoom(FakeSample[3],zoom=scale,order=0)*factor
        finerE22=zoom(FakeSample[4],zoom=scale,order=0)*factor
        finerE23=zoom(FakeSample[5],zoom=scale,order=0)*factor
        finerE33=zoom(FakeSample[6],zoom=scale,order=0)*factor
        finerGID=zoom(FakeSample[7],zoom=scale,order=0)
        finerPh1=zoom(FakeSample[8],zoom=scale,order=0)
        finerPsi=zoom(FakeSample[9],zoom=scale,order=0)
        finerPh2=zoom(FakeSample[10],zoom=scale,order=0)
        self.finerSample=np.array([finerDilation,
            finerE11,finerE12,finerE13,finerE22,finerE23,finerE33,
            finerGID,finerPh1,finerPsi,finerPh2])
        tmpx=np.arange(orig[0],step[0]/scale*finerGID.shape[0]+orig[0],step[0]/scale)
        tmpy=np.arange(orig[1],step[1]/scale*finerGID.shape[1]+orig[1],step[1]/scale)
        self.finerXV,self.finerYV=np.meshgrid(tmpx,tmpy)
        self.finerGIDLayer=self.finerSample[7].astype(int)

        # get all grains based on the grain ID (not nessasarily start from zero)
        GIDLayer=self.FakeSample[7].astype(int)
        GIDs=np.unique(GIDLayer)
        EALayer=self.FakeSample[8:11]
        EAngles=[]
        Positions=[]
        tmpx=np.arange(orig[0],step[0]*GIDLayer.shape[0]+orig[0],step[0])
        tmpy=np.arange(orig[1],step[1]*GIDLayer.shape[1]+orig[1],step[1])
        xv,yv=np.meshgrid(tmpx,tmpy)
        for gID in GIDs:
            idx=np.where(GIDLayer==gID)
            xs=xv[idx]
            ys=yv[idx]
            Positions.append([np.mean(xs),np.mean(ys),0])
            EAngles.append([np.mean(EALayer[0][idx]),np.mean(EALayer[1][idx]),np.mean(EALayer[2][idx])])
        self.Positions=np.array(Positions)
        self.EAngles=np.array(EAngles)
        self.GIDs=GIDs

    def SimSingleGrain(self,gid,outputfn=None):
        self.Cfg.SetPosOrien(self.Positions[gid],self.EAngles[gid])
        self.Cfg.Simulate()
        Lims=[]
        dx=150
        dy=80
        for ii in range(self.Cfg.NumG):
            omegid=int((180-self.Cfg.Ps[ii,2])*20)-22
            if omegid<0:
                omegid+=3600
            elif omegid>=3600:
                omegid-=3600
            x1=int(2047-self.Cfg.Ps[ii,0]-dx)
            y1=int(self.Cfg.Ps[ii,1]-dy)
            x2=x1+2*dx
            y2=y1+2*dy
            # ignore camera boundary limit, I'm just lazy, will correct it later
            Lims.append((x1,x2,y1,y2,omegid))

        simulator=StrainSimulator_GPU(self.Cfg.NumG,Lims,
                self.Cfg.Det,self.Cfg.Gs,self.Cfg.Info,self.Cfg.eng)

        idx=np.where(self.finerGIDLayer==self.GIDs[gid])
        xs=self.finerXV[idx]
        ys=self.finerYV[idx]
        tmpDil=self.finerSample[0][idx]
        tmpE11=self.finerSample[1][idx]
        tmpE12=self.finerSample[2][idx]
        tmpE13=self.finerSample[3][idx]
        tmpE22=self.finerSample[4][idx]
        tmpE23=self.finerSample[5][idx]
        tmpE33=self.finerSample[6][idx]
        # This is wrong, the strain of lattice parameters and inverse lattice parameters (Gs) are
        # different, see the Transform2RealS function. S^(-T)O=PU. And I also assume orientations are
        # the same as averaged orientation: O=U. I'm just lazy, I will correct it later.
        ss=np.zeros((len(xs),3,3))
        ss[:,0,0]=tmpE11+1
        ss[:,0,1]=tmpE12
        ss[:,0,2]=tmpE13
        ss[:,1,0]=ss[:,0,1]
        ss[:,2,0]=ss[:,0,2]
        ss[:,1,1]=tmpE22+1
        ss[:,1,2]=tmpE23
        ss[:,2,1]=ss[:,1,2]
        ss[:,2,2]=tmpE33+1

        Xs,Ys,Os,Mask=simulator.Simulate(xs,ys,ss,len(xs))

        if outputfn==None:
            outputfn=self.outdir+'/grain_{0:d}.hdf5'.format(gid)
        f=h5py.File(outputfn,'w')
        f.create_dataset("limits",data=Lims)
        f.create_dataset("Pos",data=self.Positions[gid])
        f.create_dataset("Orien",data=self.EAngles[gid])
        MaxInt=np.zeros(self.Cfg.NumG)
        grp=f.create_group('Imgs')
        for ii in range(self.Cfg.NumG):
            myMaps=np.zeros((45,Lims[ii][3]-Lims[ii][2],Lims[ii][1]-Lims[ii][0]))
            tmpMask=Mask[:,ii]
            tmpX=Xs[tmpMask,ii]
            tmpY=Ys[tmpMask,ii]
            tmpO=Os[tmpMask,ii]
            for jj in range(45):
                idx=np.where(tmpO==jj)[0]
                if len(idx)==0:
                    myMaps[jj]=0
                    continue
                myCounter=Counter(zip(tmpX[idx],tmpY[idx]))
                val=list(myCounter.values())
                xx,yy=zip(*(myCounter.keys()))
                tmp=coo_matrix((val,(yy,xx)),
                        shape=(Lims[ii][3]-Lims[ii][2],Lims[ii][1]-Lims[ii][0])).toarray()
                myMaps[jj]=self.GaussianBlur(tmp)
            MaxInt[ii]=np.max(myMaps)
            if MaxInt[ii]<1: MaxInt[ii]+=1 #some G peak are not in the window, weird
            myMaps=np.moveaxis(myMaps,0,2).astype('uint16')
            grp.create_dataset('Im{0:d}'.format(ii),data=myMaps)
        f.create_dataset("MaxInt",data=MaxInt)

    def GaussianBlur(self,myMap):
        return gaussian_filter(myMap,sigma=1,mode='nearest',truncate=4)

    def Transform2RealS(self,AllMaxS):
        AllMaxS=np.array(AllMaxS)
        realS=np.empty(AllMaxS.shape)
        realO=np.empty(AllMaxS.shape)
        for ii in range(len(realS)):
            #lattice constant I used in python nfHEDM scripts are different from the ffHEDM reconstruction used
            t=np.linalg.inv(AllMaxS[ii].T).dot(self.grainOrienM).dot([[2.95/2.9254,0,0],[0,2.95/2.9254,0],[0,0,4.7152/4.674]])
            realO[ii],realS[ii]=polar(t,'left')
        return realO, realS


