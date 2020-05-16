# coding: utf-8

import pycuda.gpuarray as gpuarray
from pycuda.autoinit import context
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import util.RotRep as Rot
import util.Simulation as Gsim
from config import Config
import h5py

from collections import Counter
from scipy.sparse import coo_matrix
from scipy.ndimage import gaussian_filter

from pycuda.compiler import SourceModule


class Initializer:

    def __init__(self, Cfg):

        # GPU functions
        with open('strain_device.cu', 'r') as cudaSrc:
            src = cudaSrc.read()
        mod = SourceModule(src)
        self.sim_strain_func = mod.get_function('Simulate_for_Strain')
        self.sim_pos_func = mod.get_function('Simulate_for_Pos')
        self.KL_total_func = mod.get_function('KL_total')
        self.KL_diff_func = mod.get_function('KL_diff')
        self.L1_diff_func = mod.get_function('L1_diff')
        self.One_func = mod.get_function('ChangeOne')
        self.hit_func = mod.get_function('Hit_Score')
        self.tExref = mod.get_texref("tcExpData")
        self.tGref = mod.get_texref('tfG')

        self.Cfg = Cfg
        self.mode = Cfg.mode
        self.ImLoaded = False
        self.GsLoaded = False
        self.GsGenerated = False

        # Det parameters
        self.Det = Gsim.Detector(psizeJ=Cfg.pixelSize / 1000.0,
                                 psizeK=Cfg.pixelSize / 1000.0,
                                 J=Cfg.JCenter,
                                 K=Cfg.KCenter,
                                 trans=np.array([Cfg.Ldistance, 0, 0]),
                                 tilt=Rot.EulerZXZ2Mat(np.array(Cfg.tilt) / 180.0 * np.pi))
        afDetInfoH = np.concatenate(
            [[Cfg.JPixelNum, Cfg.KPixelNum, Cfg.pixelSize / 1000.0, Cfg.pixelSize / 1000.0],
             self.Det.CoordOrigin,
             self.Det.Norm,
             self.Det.Jvector,
             self.Det.Kvector]).astype(np.float32)
        self.afDetInfoD = gpuarray.to_gpu(afDetInfoH)
        # sample parameters 
        # hack!! only for Hexagonal
        self.sample = Gsim.CrystalStr()
        self.sample.PrimA = Cfg.lattice[0] * np.array([1, 0, 0])
        self.sample.PrimB = Cfg.lattice[1] * np.array([np.cos(np.pi * 2 / 3), np.sin(np.pi * 2 / 3), 0])
        self.sample.PrimC = Cfg.lattice[2] * np.array([0, 0, 1])
        Atoms = Cfg.atoms
        for ii in range(len(Atoms)):
            self.sample.addAtom(list(map(eval, Atoms[ii][0:3])), Atoms[ii][3])
        self.sample.getRecipVec()
        self.sample.getGs(Cfg.maxQ)

        if self.mode == 'rec':
            f = h5py.File(Cfg.peakFile, 'r')
            # Lim for window position
            self.LimH = np.array(f['limits']).astype(np.int32)
            self.LimD = gpuarray.to_gpu(self.LimH)
            # whichOmega for choosing between omega1 and omega2
            self.whichOmega = np.array(f['whichOmega']).astype(np.int32)
            self.whichOmegaD = gpuarray.to_gpu(self.whichOmega)
            # MaxInt for normalize the weight of each spot 
            # (because different spots have very different intensity but we want them equal weighted)
            self.MaxInt = np.array(f['MaxInt'], dtype=np.float32)
            self.MaxIntD = gpuarray.to_gpu(self.MaxInt)
            self.Gs = np.array(f['Gs'], dtype=np.float32)
            self.NumG = len(self.Gs)
            self.orienM = np.array(f['OrienM'])
            self.avg_distortion = np.array(f['avg_distortion'])
            self.GsGenerated = True

    # transfer the ExpImgs and all Gs to texture memory
    def loadIm(self):
        f = h5py.File(self.Cfg.peakFile, 'r')
        AllIm = np.zeros(shape=(160, 300, self.NumG * 45), dtype=np.uint32, order='F')
        for ii in range(self.NumG):
            tmp = np.array(f['Imgs']['Im{0:d}'.format(ii)])
            AllIm[:tmp.shape[0], :tmp.shape[1], ii * 45:(ii + 1) * 45] = tmp
        shape = AllIm.shape
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
        self.ImLoaded = True

    def loadGs(self):
        if not self.GsGenerated:
            raise RuntimeError('Gs are not generated yet')
        self.tGref.set_array(cuda.matrix_to_array(np.transpose(self.Gs).astype(np.float32), order='F'))
        self.GsLoaded = True

    def generateGs(self, pos, orien, avg_distortion):
        self.pos = np.array(pos)
        self.orien = np.array(orien)
        self.orienM = Rot.EulerZXZ2Mat(self.orien / 180.0 * np.pi)
        self.avg_distortion = avg_distortion

        Ps, self.Gs, Info = Gsim.GetProjectedVertex(self.Det,
                                                    self.sample, self.avg_distortion.dot(self.orienM),
                                                    self.Cfg.etalimit / 180.0 * np.pi,
                                                    self.pos, getPeaksInfo=True,
                                                    omegaL=self.Cfg.omgRange[0],
                                                    omegaU=self.Cfg.omgRange[1], energy=self.Cfg.energy)
        self.NumG = len(self.Gs)
        Lims = []
        dx = 150
        dy = 80
        for ii in range(self.NumG):
            omegid = int((180 - Ps[ii, 2]) / self.Cfg.omgInterval) - 22  # becuase store 45 frames
            if omegid < 0:
                omegid += int(180 / self.Cfg.omgInterval)
            elif omegid >= int(180 / self.Cfg.omgInterval):
                omegid -= int(180 / self.Cfg.omgInterval)
            x1 = int(2047 - Ps[ii, 0] - dx)
            y1 = int(Ps[ii, 1] - dy)
            x2 = x1 + 2 * dx
            y2 = y1 + 2 * dy
            # ignore camera boundary limit, I'm just lazy, will correct it later
            Lims.append((x1, x2, y1, y2, omegid))
        self.LimH = np.array(Lims, dtype=np.int32)
        self.LimD = gpuarray.to_gpu(self.LimH)
        # whichOmega for choosing between omega1 and omega2
        self.whichOmega = np.zeros(len(Lims), dtype=np.int32)
        for ii in range(len(Lims)):
            if Info[ii]['WhichOmega'] == 'b':
                self.whichOmega[ii] = 2
            else:
                self.whichOmega[ii] = 1
        self.whichOmegaD = gpuarray.to_gpu(self.whichOmega)
        self.GsGenerated = True

    def MoveDet(self, dJ=0, dK=0, dD=0, dT=np.eye(3)):
        self.Det.Move(dJ, dK, np.array([dD, 0, 0]), dT)
        afDetInfoH = np.concatenate(
            [[self.Cfg.JPixelNum, self.Cfg.KPixelNum,
              self.Cfg.pixelSize / 1000.0, self.Cfg.pixelSize / 1000.0],
             self.Det.CoordOrigin,
             self.Det.Norm,
             self.Det.Jvector,
             self.Det.Kvector]).astype(np.float32)
        self.afDetInfoD = gpuarray.to_gpu(afDetInfoH)

    def ResetDet(self):
        self.Det.Reset()
        afDetInfoH = np.concatenate(
            [[self.Cfg.JPixelNum, self.Cfg.KPixelNum,
              self.Cfg.pixelSize / 1000.0, self.Cfg.pixelSize / 1000.0],
             self.Det.CoordOrigin,
             self.Det.Norm,
             self.Det.Jvector,
             self.Det.Kvector]).astype(np.float32)
        self.afDetInfoD = gpuarray.to_gpu(afDetInfoH)

    def sim_pos_wrapper(self, xs, ys, ss):
        NumD = len(xs)
        if self.GsLoaded == False:
            self.loadGs()
        XD = gpuarray.empty(self.NumG * NumD, dtype=np.int32)
        YD = gpuarray.empty(self.NumG * NumD, dtype=np.int32)
        OffsetD = gpuarray.empty(self.NumG * NumD, dtype=np.int32)
        MaskD = gpuarray.empty(self.NumG * NumD, dtype=np.bool_)
        TrueMaskD = gpuarray.empty(self.NumG * NumD, dtype=np.bool_)

        xsD = gpuarray.to_gpu(xs.astype(np.float32))
        ysD = gpuarray.to_gpu(ys.astype(np.float32))
        ssD = gpuarray.to_gpu(ss.ravel(order='C').astype(np.float32))

        self.sim_pos_func(XD, YD, OffsetD, MaskD, TrueMaskD,
                          xsD, ysD, self.afDetInfoD, ssD,
                          self.whichOmegaD, np.int32(NumD), np.int32(self.NumG),
                          np.float32(self.Cfg.energy), np.int32(45), self.LimD, 
                          np.int32(5),self.Cfg.omgInterval,
                          block=(self.NumG, 1, 1), grid=(NumD, 1))
        xtmp = XD.get().reshape((-1, self.NumG))
        ytmp = YD.get().reshape((-1, self.NumG))
        otmp = OffsetD.get().reshape((-1, self.NumG))
        maskH = MaskD.get().reshape(-1, self.NumG)
        return xtmp, ytmp, otmp, maskH

    def simMap(self, tmpxx, tmpyy, AllMaxS, blur=False, dtype=np.uint32):
        if self.GsLoaded == False:
            self.loadGs()
        xtmp, ytmp, otmp, maskH = self.sim_pos_wrapper(tmpxx, tmpyy, AllMaxS)
        res = np.zeros(shape=(160, 300, self.NumG * 45), dtype=dtype)
        for ii in range(self.NumG):
            tmpMask = maskH[:, ii]
            tmpX = xtmp[tmpMask, ii]
            tmpY = ytmp[tmpMask, ii]
            tmpO = otmp[tmpMask, ii]
            myMaps = np.zeros((45, self.LimH[ii][3] - self.LimH[ii][2], self.LimH[ii][1] - self.LimH[ii][0]),
                              dtype=dtype)
            for jj in range(45):
                idx = np.where(tmpO == jj)[0]
                if len(idx) == 0:
                    myMaps[jj] = 0
                    continue
                myCounter = Counter(zip(tmpX[idx], tmpY[idx]))
                val = list(myCounter.values())
                xx, yy = zip(*(myCounter.keys()))
                tmp = coo_matrix((val, (yy, xx)),
                                 shape=(
                                 self.LimH[ii][3] - self.LimH[ii][2], self.LimH[ii][1] - self.LimH[ii][0])).toarray()
                if blur:
                    myMaps[jj] = gaussian_filter(tmp, sigma=1, mode='nearest', truncate=4)
                else:
                    myMaps[jj] = tmp
            myMaps = np.moveaxis(myMaps, 0, 2)
            res[:myMaps.shape[0], :myMaps.shape[1], ii * 45:(ii + 1) * 45] = myMaps
        return res
