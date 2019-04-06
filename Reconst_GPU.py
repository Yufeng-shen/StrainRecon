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
import os

from Simulator_GPU import StrainSimulator_GPU
from collections import Counter
from scipy.sparse import coo_matrix
from scipy.ndimage import gaussian_filter


from pycuda.compiler import SourceModule

class StrainReconstructor_GPU(object):
    def __init__(self,_NumG,
            peakFile,
            _Det, _Gs, _Info, _eng):

        with open('strain_device.cu','r') as cudaSrc:
            src=cudaSrc.read()
        mod = SourceModule(src)
        self.sim_func = mod.get_function('Simulate_for_Strain')
        self.KL_total_func = mod.get_function('KL_total')
        self.KL_diff_func = mod.get_function('KL_diff')
        self.KL_One_func = mod.get_function('KL_ChangeOne')
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

    def CrossEntropyMethod(self,x,y,
            XD,YD,OffsetD,MaskD,TrueMaskD,scoreD,S_gpu,
            NumD=10000,numCut=100,initStd=1e-4,MaxIter=100,S_init=np.eye(3),BlockSize=256,debug=False):
        if self.ImLoaded==False:
            self.loadIm()
        if self.GsLoaded==False:
            self.loadGs()

        S=np.random.multivariate_normal(
            np.zeros(9),np.eye(9)*initStd,size=(NumD)).reshape((NumD,3,3),order='C')+np.tile(S_init,(NumD,1,1))
        cuda.memcpy_htod(S_gpu,S.ravel().astype(np.float32))


        self.sim_func(XD,YD,OffsetD,MaskD,TrueMaskD,
                np.float32(x), np.float32(y),self.afDetInfoD,S_gpu,
                self.whichOmegaD,np.int32(NumD),np.int32(self.NumG),np.float32(self.eng),np.int32(45),self.LimD,np.int32(5),
                 block=(self.NumG,1,1),grid=(NumD,1))
        
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
            cuda.memcpy_htod(S_gpu,S.ravel().astype(np.float32))

            self.sim_func(XD,YD,OffsetD,MaskD,TrueMaskD,
                    np.float32(x), np.float32(y),self.afDetInfoD,S_gpu,
                    self.whichOmegaD,np.int32(NumD),np.int32(self.NumG),np.float32(self.eng),np.int32(45),self.LimD,np.int32(5),
                     block=(self.NumG,1,1),grid=(NumD,1))


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
    
    def ChangeOneVoxel_KL(self,x,y,mean,realMapsLogD,falseMapsD,
            XD,YD,OffsetD,MaskD,TrueMaskD,diffD,S_gpu,
            NumD=5000,numCut=50,cov=1e-7*np.eye(9),epsilon=1e-6,MaxIter=3,BlockSize=256,debug=False):
        if self.GsLoaded==False:
            self.loadGs()
        #remove the original hit
        S=mean
        cuda.memcpy_htod(S_gpu,S.ravel().astype(np.float32))
        self.sim_func(XD,YD,OffsetD,MaskD,TrueMaskD,
                np.float32(x), np.float32(y),self.afDetInfoD,S_gpu,
                self.whichOmegaD,np.int32(1),np.int32(self.NumG),
                      np.float32(self.eng),np.int32(45),self.LimD,np.int32(5),
                 block=(self.NumG,1,1),grid=(1,1))
        self.KL_One_func(XD,YD,OffsetD,MaskD,TrueMaskD,
                             falseMapsD,np.int32(self.NumG),np.int32(45),np.float32(epsilon),np.int32(-1), #minus one!!
                             block=(self.NumG,1,1),grid=(1,1))
        #find a better distortion matrix
        for ii in range(MaxIter):
            S=np.empty((NumD,3,3),dtype=np.float32)
            S[0,:,:]=mean
            S[1:,:,:]=np.random.multivariate_normal(
                mean.ravel(),cov,size=(NumD-1)).reshape((NumD-1,3,3),order='C')
            cuda.memcpy_htod(S_gpu,S.ravel().astype(np.float32))
            self.sim_func(XD,YD,OffsetD,MaskD,TrueMaskD,
                    np.float32(x), np.float32(y),self.afDetInfoD,S_gpu,
                    self.whichOmegaD,np.int32(NumD),np.int32(self.NumG),
                          np.float32(self.eng),np.int32(45),self.LimD,np.int32(5),
                     block=(self.NumG,1,1),grid=(NumD,1))
            self.KL_diff_func(diffD,
                        XD,YD,OffsetD,MaskD,TrueMaskD,
                        realMapsLogD,falseMapsD,
                        np.int32(self.NumG),np.int32(NumD),np.int32(45),
                        block=(BlockSize,1,1),grid=(int(NumD/BlockSize+1),1))
            diffH=diffD.get()
            args=np.argpartition(diffH,numCut)[:numCut]
            cov=np.cov(S[args].reshape((numCut,9),order='C').T)
            mean=np.mean(S[args],axis=0)
            if debug:
                print(np.min(diffH),diffH[0])
        #add the new hit
        S=mean
        cuda.memcpy_htod(S_gpu,S.ravel().astype(np.float32))
        self.sim_func(XD,YD,OffsetD,MaskD,TrueMaskD,
                np.float32(x), np.float32(y),self.afDetInfoD,S_gpu,
                self.whichOmegaD,np.int32(1),np.int32(self.NumG),
                      np.float32(self.eng),np.int32(45),self.LimD,np.int32(5),
                 block=(self.NumG,1,1),grid=(1,1))
        self.KL_One_func(XD,YD,OffsetD,MaskD,TrueMaskD,
                             falseMapsD,np.int32(self.NumG),np.int32(45),np.float32(epsilon),np.int32(+1), #plus one!!
                             block=(self.NumG,1,1),grid=(1,1))
        return mean


class ReconSingleGrain(object):

    def __init__(self,cfgFile,outdir,gid):
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
        self.gid=gid

    def GetGrids(self,threshold=1,fileType='npy',orig=[-0.256,-0.256],step=[0.002,0.002]):
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
            idx=np.where(GIDLayer==self.gid)
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

    def ReconGridsPhase1(self,tmpxx,tmpyy,reconstructor,NumD=10000,numCut=100):
        #allocate gpu memory
        XD=gpuarray.empty(self.Cfg.NumG*NumD,dtype=np.int32)
        YD=gpuarray.empty(self.Cfg.NumG*NumD,dtype=np.int32)
        OffsetD=gpuarray.empty(self.Cfg.NumG*NumD,dtype=np.int32)

        MaskD=gpuarray.empty(self.Cfg.NumG*NumD,dtype=np.bool_)
        TrueMaskD=gpuarray.empty(self.Cfg.NumG*NumD,dtype=np.bool_)
        scoreD=gpuarray.empty(NumD,dtype=np.float32)
        S_gpu=cuda.mem_alloc(NumD*9*4)

        AllMaxScore=[]
        AllMaxS=[]
        for ii in range(len(tmpxx)):
            if ii==0:
                t=reconstructor.CrossEntropyMethod(tmpxx[ii],tmpyy[ii],
                        XD,YD,OffsetD,MaskD,TrueMaskD,scoreD,S_gpu,
                        NumD=NumD,numCut=numCut)
                print(ii,t[0])
                AllMaxScore.append(t[2])
                AllMaxS.append(t[1])
            else:
#                t=reconstructor.CrossEntropyMethod(tmpxx[ii],tmpyy[ii],S_init=AllMaxS[-1])
                t=reconstructor.CrossEntropyMethod(tmpxx[ii],tmpyy[ii],
                        XD,YD,OffsetD,MaskD,TrueMaskD,scoreD,S_gpu,
                        NumD=NumD,numCut=numCut)
                if ii%50==0:
                    print(ii,t[0])
                AllMaxScore.append(t[2])
                AllMaxS.append(t[1])
        AllMaxS=np.array(AllMaxS)
        AllMaxScore=np.array(AllMaxScore)
        return AllMaxScore, AllMaxS
    
    def simfloatMap(self,tmpxx,tmpyy,AllMaxS):
        Lim=np.array(self.peakFile['limits'])

        sim=StrainSimulator_GPU( _NumG=self.Cfg.NumG,_Lim=Lim,
                _Det=self.Cfg.Det, _Gs=self.Cfg.Gs, _Info=self.Cfg.Info, _eng=self.Cfg.eng)
        sim.loadGs()

        xtmp,ytmp,otmp,maskH=sim.Simulate(tmpxx,tmpyy,AllMaxS,len(tmpxx))
        res=np.zeros(shape=(160,300,self.Cfg.NumG*45),dtype=np.uint32,order='F')
        for ii in range(self.Cfg.NumG):
            tmpMask=maskH[:,ii]
            tmpX=xtmp[tmpMask,ii]
            tmpY=ytmp[tmpMask,ii]
            tmpO=otmp[tmpMask,ii]
            myMaps=np.zeros((45,Lim[ii][3]-Lim[ii][2],Lim[ii][1]-Lim[ii][0]))
            for jj in range(45):
                idx=np.where(tmpO==jj)[0]
                if len(idx)==0:
                    myMaps[jj]=0
                    continue
                myCounter=Counter(zip(tmpX[idx],tmpY[idx]))
                val=list(myCounter.values())
                xx,yy=zip(*(myCounter.keys()))
                tmp=coo_matrix((val,(yy,xx)),shape=(Lim[ii][3]-Lim[ii][2],Lim[ii][1]-Lim[ii][0])).toarray()
    #             myMaps[jj]=gaussian_filter(tmp,sigma=1,mode='nearest',truncate=4)
                myMaps[jj]=tmp
            myMaps=np.moveaxis(myMaps,0,2)
            res[:myMaps.shape[0],:myMaps.shape[1],ii*45:(ii+1)*45]=myMaps
        return res
    
    def SimPhase1Result(self,tmpxx,tmpyy,AllMaxS,epsilon=1e-6):
        falseMaps=self.simfloatMap(tmpxx,tmpyy,AllMaxS)
        realMaps=np.zeros(shape=(160,300,self.Cfg.NumG*45),dtype=np.uint32)
        for ii in range(self.Cfg.NumG):
            tmp=np.array(self.peakFile['Imgs']['Im{0:d}'.format(ii)])
            realMaps[:tmp.shape[0],:tmp.shape[1],ii*45:(ii+1)*45]=tmp

        realMaps=realMaps/(np.sum(realMaps)/np.sum(falseMaps))
        self.falseMapsD=gpuarray.to_gpu((falseMaps.ravel()+epsilon).astype(np.float32))
        self.realMapsLogD=gpuarray.to_gpu(np.log(realMaps.ravel()+epsilon).astype(np.float32))
        return

    def ReconGridsPhase2(self,tmpxx,tmpyy,AllMaxS,recon,
            NumD=5000,numCut=50,iterN=10,shuffle=False):
        #allocate gpu memory
        XD=gpuarray.empty(self.Cfg.NumG*NumD,dtype=np.int32)
        YD=gpuarray.empty(self.Cfg.NumG*NumD,dtype=np.int32)
        OffsetD=gpuarray.empty(self.Cfg.NumG*NumD,dtype=np.int32)
        MaskD=gpuarray.empty(self.Cfg.NumG*NumD,dtype=np.bool_)
        TrueMaskD=gpuarray.empty(self.Cfg.NumG*NumD,dtype=np.bool_)
        diffD=gpuarray.empty(NumD,dtype=np.float32)
        S_gpu=cuda.mem_alloc(NumD*9*4)
        for jj in range(iterN):
            print("{0:d}/{1:d}".format(jj+1,iterN))
            if shuffle:
                order=np.random.permutation(len(tmpxx))
            else:
                order=np.arange(len(tmpxx))
            for ii in order:
                if ii==200:
                    tmp=recon.ChangeOneVoxel_KL(
                            tmpxx[ii],tmpyy[ii],AllMaxS[ii],self.realMapsLogD,self.falseMapsD,
                            XD,YD,OffsetD,MaskD,TrueMaskD,diffD,S_gpu,
                            NumD=NumD,numCut=numCut,cov=1e-7*np.eye(9),MaxIter=3,debug=True)
                    AllMaxS[ii]=tmp
                    print(AllMaxS[ii])
                else:
                    tmp=recon.ChangeOneVoxel_KL(
                            tmpxx[ii],tmpyy[ii],AllMaxS[ii],self.realMapsLogD,self.falseMapsD,
                            XD,YD,OffsetD,MaskD,TrueMaskD,diffD,S_gpu,
                            NumD=NumD,numCut=numCut,cov=1e-7*np.eye(9),MaxIter=3,debug=False)
                    AllMaxS[ii]=tmp
        return AllMaxS

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
        fn=self.outdir+"g{0:d}_rec.hdf5".format(self.gid)
        exists=os.path.isfile(fn)
        if exists:
            f=h5py.File(fn)
            x=f['x'][:]
            y=f['y'][:]
            AllMaxS=f['Phase2_S'][:]
            self.SimPhase1Result(x,y,AllMaxS)
            AllMaxS=self.ReconGridsPhase2(x,y,AllMaxS,self.recon,iterN=2,shuffle=True)
            tmp=f["Phase2_S"]
            tmp[...]=AllMaxS

            realO,realS=self.Transform2RealS(AllMaxS)
            tmp=f["realS"]
            tmp[...]=realS
            tmp=f["realO"]
            tmp[...]=realO
        else:
            with h5py.File(fn,'w') as f:
                x,y,con=self.GetGrids()
                f.create_dataset("x",data=x)
                f.create_dataset("y",data=y)
                f.create_dataset("IceNineConf",data=con)

                AllMaxScore,AllMaxS=self.ReconGridsPhase1(x,y,self.recon)
                f.create_dataset("Phase1_Conf",data=AllMaxScore)
                f.create_dataset("Phase1_S",data=AllMaxS)
                
                self.SimPhase1Result(x,y,AllMaxS)
                AllMaxS=self.ReconGridsPhase2(x,y,AllMaxS,self.recon)
                f.create_dataset("Phase2_S",data=AllMaxS)

                realO,realS=self.Transform2RealS(AllMaxS)
                f.create_dataset("realS",data=realS)
                f.create_dataset("realO",data=realO)
