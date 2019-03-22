# coding: utf-8

#import cProfile, pstats, StringIO
import pycuda.gpuarray as gpuarray
from pycuda.autoinit import context
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from scipy.linalg import polar
from util.MicFileTool import read_mic_file
import util.RotRep as Rot
from InitStrain import Initializer
import h5py


from pycuda.compiler import SourceModule

class StrainReconstructor_GPU(object):
    def __init__(self,_NumG,
            peakFile,
            _Det, _Gs, _Info, _eng):

        with open('strain_device.cu','r') as cudaSrc:
            src=cudaSrc.read()
        mod = SourceModule(src)
        self.sim_func = mod.get_function('Simulate_for_Strain')
        self.KL_func = mod.get_function('KL_diff')
        self.hit_func = mod.get_function('Hit_Score')
        self.tExref = mod.get_texref("tcExpData")
        self.tGref = mod.get_texref('tfG')
        self.Gs= _Gs
        self.eng= _eng
        # Number of G vectors
        self.NumG= _NumG
        self.peakFile=peakFile
        ## Lim for window position
        Lim=np.array(peakFile['limits'])
#        Lim=[]
#        for ii in range(_NumG):
#            Lim.append(np.load(bfPath+'/limit{0:d}.npy'.format(ii))[0])
        Lim=np.array(Lim).astype(np.int32)
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
        MaxInt=np.array(peakFile['MaxInt'])
        self.MaxIntD=gpuarray.to_gpu(MaxInt.astype(np.float32))

        # Det parameters
        self.Det=_Det
        afDetInfoH=np.concatenate([[2048,2048,0.001454,0.001454],
                                   _Det.CoordOrigin,_Det.Norm,_Det.Jvector,_Det.Kvector]).astype(np.float32)
        self.afDetInfoD=gpuarray.to_gpu(afDetInfoH)
        self.ImLoaded=False
        self.GsLoaded=False

    #transfer the ExpImgs and all Gs to texture memory
    def loadIm(self):
        AllIm=np.zeros(shape=(160,300,self.NumG*45),dtype=np.uint32,order='F')
        for ii in range(self.NumG):
#            tmp=np.load(self.fltPath+'/Im{0:d}.npy'.format(ii))
#            tmp=np.moveaxis(tmp, 0, 2)
#            AllIm[:tmp.shape[0],:tmp.shape[1],ii*45:(ii+1)*45]=tmp
            tmp=np.array(self.peakFile['Imgs']['Im{0:d}'.format(ii)])
            AllIm[:tmp.shape[0],:tmp.shape[1],ii*45:(ii+1)*45]=tmp
        shape=AllIm.shape
        descr = cuda.ArrayDescriptor3D()
        descr.width = shape[0]
        descr.height = shape[1]
        descr.depth = shape[2]
        descr.format = cuda.dtype_to_array_format(AllIm.dtype)
        descr.num_channels = 1
        descr.flags = 0
        ary = cuda.Array(descr)
        copy = cuda.Memcpy3D()
        copy.set_src_host(AllIm)
        copy.set_dst_array(ary)
        copy.width_in_bytes = copy.src_pitch = AllIm.strides[1]
        copy.src_height = copy.height = shape[1]
        copy.depth = shape[2]
        copy()
        self.tExref.set_array(ary)
        self.ImLoaded=True
    def loadGs(self): 
        self.tGref.set_array(cuda.matrix_to_array(np.transpose(self.Gs).astype(np.float32),order='F'))
        self.GsLoaded=True

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

    def CrossEntropyMethod(self,x,y,NumD=10000,numCut=100,initStd=1e-4,MaxIter=100,S_init=np.eye(3),BlockSize=256,debug=False):
        if self.ImLoaded==False:
            self.loadIm()
        if self.GsLoaded==False:
            self.loadGs()

        S=np.random.multivariate_normal(
            np.zeros(9),np.eye(9)*initStd,size=(NumD)).reshape((NumD,3,3),order='C')+np.tile(S_init,(NumD,1,1))
        
        SD=gpuarray.to_gpu(S.ravel().astype(np.float32))

        XD=gpuarray.empty(self.NumG*NumD,dtype=np.int32)
        YD=gpuarray.empty(self.NumG*NumD,dtype=np.int32)
        OffsetD=gpuarray.empty(self.NumG*NumD,dtype=np.int32)

        MaskD=gpuarray.empty(self.NumG*NumD,dtype=np.bool_)
        TrueMaskD=gpuarray.empty(self.NumG*NumD,dtype=np.bool_)

        self.sim_func(XD,YD,OffsetD,MaskD,TrueMaskD,
                np.float32(x), np.float32(y),self.afDetInfoD,SD,
                self.whichOmegaD,np.int32(NumD),np.int32(self.NumG),np.float32(self.eng),np.int32(45),self.LimD,np.int32(5),
                 block=(self.NumG,1,1),grid=(NumD,1))
        
        scoreD=gpuarray.empty(NumD,dtype=np.float32)
        self.hit_func(scoreD,
                XD,YD,OffsetD,MaskD,TrueMaskD,
                self.MaxIntD,np.int32(self.NumG),np.int32(NumD),np.int32(45),
                block=(BlockSize,1,1),grid=(int(NumD/BlockSize+1),1))
         
        score=scoreD.get()
        args=np.argpartition(score,-numCut)[-numCut:]
        cov=np.cov(S[args].reshape((numCut,9),order='C').T)
        mean=np.mean(S[args],axis=0)
        for ii in range(MaxIter):
            S=np.random.multivariate_normal(
                np.zeros(9),cov,size=(NumD)).reshape((NumD,3,3),order='C')+np.tile(mean,(NumD,1,1))
            SD=gpuarray.to_gpu(S.ravel().astype(np.float32))
            XD=gpuarray.empty(self.NumG*NumD,dtype=np.int32)
            YD=gpuarray.empty(self.NumG*NumD,dtype=np.int32)
            OffsetD=gpuarray.empty(self.NumG*NumD,dtype=np.int32)
            MaskD=gpuarray.empty(self.NumG*NumD,dtype=np.bool_)
            TrueMaskD=gpuarray.empty(self.NumG*NumD,dtype=np.bool_)

            self.sim_func(XD,YD,OffsetD,MaskD,TrueMaskD,
                    np.float32(x), np.float32(y),self.afDetInfoD,SD,
                    self.whichOmegaD,np.int32(NumD),np.int32(self.NumG),np.float32(self.eng),np.int32(45),self.LimD,np.int32(5),
                     block=(self.NumG,1,1),grid=(NumD,1))

            scoreD=gpuarray.empty(NumD,dtype=np.float32)

            self.hit_func(scoreD,
                    XD,YD,OffsetD,MaskD,TrueMaskD,
                    self.MaxIntD,np.int32(self.NumG),np.int32(NumD),np.int32(45),
                    block=(BlockSize,1,1),grid=(int(NumD/BlockSize+1),1))


            score=scoreD.get()
            
            args=np.argpartition(score,-numCut)[-numCut:]
            cov=np.cov(S[args].reshape((numCut,9),order='C').T)
            mean=np.mean(S[args],axis=0)
            if debug:
                print(np.max(score))
            if np.trace(np.absolute(cov))<1e-9:
                break
            if np.min(score)==np.max(score):
                break

        return cov,mean,np.max(score)


class ReconSingleGrain(object):

    def __init__(self,cfgFile,outdir):
        self.Cfg=Initializer(cfgFile)
        self.peakFile=h5py.File(self.Cfg.peakfn,'r')
        self.Cfg.SetPosOrien(self.peakFile['Pos'],self.peakFile['Orien'])
        self.Cfg.Simulate()
        self.grainOrien=np.array(self.Cfg.orien)
        self.grainOrienM=Rot.EulerZXZ2Mat(self.grainOrien/180.0*np.pi)
        self.micfn=self.Cfg.micfn
        self.recon=StrainReconstructor_GPU(_NumG=self.Cfg.NumG,
                peakFile=self.peakFile,
                _Det=self.Cfg.Det, _Gs=self.Cfg.Gs, _Info=self.Cfg.Info, _eng=self.Cfg.eng)
        self.outdir=outdir

    def GetGrids(self,threshold=1,fileType='npy',orig=[-0.256,-0.256],step=[0.002,0.002],gid=40):
        if fileType=='mic':
            sw,snp=read_mic_file(self.micfn)
            t=snp[:,6:9]-np.tile(np.array(self.grainOrien),(snp.shape[0],1))
            t=np.absolute(t)<threshold
            t=t[:,0]*t[:,1]*t[:,2] #voxels that within 1 degree misorientation
            t=snp[t]
            x=t[:,0]
            y=t[:,1]
            con=t[:,9]
        elif fileType=='ang':
            snp=np.loadtxt(self.micfn,comments="#")
            t=snp[:,2:5]-np.tile(np.array(self.grainOrien),(snp.shape[0],1))
            t=np.absolute(t)<threshold
            t=t[:,0]*t[:,1]*t[:,2]
            t=snp[t]
            x=t[:,0]
            y=t[:,1]
            con=t[:,5]
        elif fileType=='npy':
            FakeSample=np.load(self.micfn)
            GIDLayer=FakeSample[7].astype(int)
            tmpx=np.arange(orig[0],step[0]*GIDLayer.shape[0]+orig[0],step[0])
            tmpy=np.arange(orig[1],step[1]*GIDLayer.shape[1]+orig[1],step[1])
            xv,yv=np.meshgrid(tmpx,tmpy)
            idx=np.where(GIDLayer==gid)
            x=xv[idx]
            y=yv[idx]
            con=np.ones(x.shape)
        else:
            print('fileType need to be mic or ang or npy')
        return x,y,con

    def test(self,tmpxx,tmpyy,reconstructor,totalTry=10000,cutTry=100,initStd=1e-4,MaxIter=1):
        idx=np.random.randint(len(tmpxx))
        reconstructor.CrossEntropyMethod(tmpxx[idx],tmpyy[idx],
                NumD=totalTry,numCut=cutTry,initStd=initStd,
                MaxIter=MaxIter,debug=True)
        return

    def ReconGrids(self,tmpxx,tmpyy,reconstructor):
        AllMaxScore=[]
        AllMaxS=[]
        for ii in range(len(tmpxx)):
            if ii==0:
                t=reconstructor.CrossEntropyMethod(tmpxx[ii],tmpyy[ii])
                print(ii,t[0])
                AllMaxScore.append(t[2])
                AllMaxS.append(t[1])
            else:
#                t=reconstructor.CrossEntropyMethod(tmpxx[ii],tmpyy[ii],S_init=AllMaxS[-1])
                t=reconstructor.CrossEntropyMethod(tmpxx[ii],tmpyy[ii])
                if ii%50==0:
                    print(ii,t[0])
                AllMaxScore.append(t[2])
                AllMaxS.append(t[1])
        return AllMaxScore, AllMaxS

    def Transform2RealS(self,AllMaxS):
        AllMaxS=np.array(AllMaxS)
        realS=np.empty(AllMaxS.shape)
        realO=np.empty(AllMaxS.shape)
        for ii in range(len(realS)):
            #lattice constant I used in python nfHEDM scripts are different from the ffHEDM reconstruction used
            t=np.linalg.inv(AllMaxS[ii].T).dot(self.grainOrienM).dot([[2.95/2.9254,0,0],[0,2.95/2.9254,0],[0,0,4.7152/4.674]])
            realO[ii],realS[ii]=polar(t,'left')
        return realO, realS

    def run(self):
        x,y,con=self.GetGrids()
        np.save(self.outdir+'/x.npy',x)
        np.save(self.outdir+'/y.npy',y)
        np.save(self.outdir+'/con.npy',con)
        AllMaxScore,AllMaxS=self.ReconGrids(x,y,self.recon)
        np.save(self.outdir+'/allMaxScore.npy',AllMaxScore)
        np.save(self.outdir+'/allMaxS.npy',AllMaxS)
        realO,realS=self.Transform2RealS(AllMaxS)
        np.save(self.outdir+'/realS.npy',realS)
        np.save(self.outdir+'/realO.npy',realO)
