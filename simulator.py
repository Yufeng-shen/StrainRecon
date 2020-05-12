# coding: utf-8

import numpy as np
from scipy.linalg import polar
from scipy.ndimage import zoom
from util.MicFileTool import read_mic_file
import util.RotRep as Rot

from initializer import Initializer
import os
import h5py


class Simulator:

    def __init__(self, Cfg, scale=10, blur=True):
        """
        scale: refine the grid by '1/scale'
        """
        self.Cfg = Cfg
        Sample = h5py.File(Cfg.micFile,'r')
        self.outFN = Cfg.peakFile
        self.scale = scale
        self.blur = blur
        # create a finer grid for more realistic simulation
        orig = Sample["origin"][:]
        step = Sample["stepSize"][:]
        
        self.finerE11 = zoom(Sample["E11"][:], zoom=scale, order=0)
        self.finerE12 = zoom(Sample["E12"][:], zoom=scale, order=0)
        self.finerE13 = zoom(Sample["E13"][:], zoom=scale, order=0)
        self.finerE22 = zoom(Sample["E22"][:], zoom=scale, order=0)
        self.finerE23 = zoom(Sample["E23"][:], zoom=scale, order=0)
        self.finerE33 = zoom(Sample["E33"][:], zoom=scale, order=0)
        self.finerGID = zoom(Sample["GrainID"][:], zoom=scale, order=0)
        self.finerPh1 = zoom(Sample["Ph1"][:], zoom=scale, order=0)
        self.finerPsi = zoom(Sample["Psi"][:], zoom=scale, order=0)
        self.finerPh2 = zoom(Sample["Ph2"][:], zoom=scale, order=0)

        tmpx = np.arange(orig[0], step[0] / scale * self.finerGID.shape[1] + orig[0], step[0] / scale)
        tmpy = np.arange(orig[1], step[1] / scale * self.finerGID.shape[0] + orig[1], step[1] / scale)
        self.finerXV, self.finerYV = np.meshgrid(tmpx, tmpy)
        self.finerGIDLayer = self.finerGID.astype(int)

        # get all grains based on the grain ID (not nessasarily start from zero)
        GIDLayer = Sample["GrainID"][:].astype(int)
        GIDs = np.unique(GIDLayer)

        EAngles = []
        Positions = []
        tmpx = np.arange(orig[0], step[0] * GIDLayer.shape[1] + orig[0], step[0])
        tmpy = np.arange(orig[1], step[1] * GIDLayer.shape[0] + orig[1], step[1])
        xv, yv = np.meshgrid(tmpx, tmpy)
        for gID in GIDs:
            idx = np.where(GIDLayer == gID)
            xs = xv[idx]
            ys = yv[idx]
            Positions.append([np.mean(xs), np.mean(ys), 0])
            EAngles.append([np.mean(Sample["Ph1"][:][idx]), np.mean(Sample["Psi"][:][idx]), np.mean(Sample["Ph2"][:][idx])])
        self.Positions = np.array(Positions)
        self.EAngles = np.array(EAngles)
        self.GIDs = GIDs
        Sample.close()

    def SimSingleGrain(self):
        gid = self.Cfg.grainID
        idx = np.where(self.finerGIDLayer == self.GIDs[gid])
        xs = self.finerXV[idx]
        ys = self.finerYV[idx]
        tmpE11 = self.finerE11[idx]
        tmpE12 = self.finerE12[idx]
        tmpE13 = self.finerE13[idx]
        tmpE22 = self.finerE22[idx]
        tmpE23 = self.finerE23[idx]
        tmpE33 = self.finerE33[idx]
        tmpPh1 = self.finerPh1[idx]
        tmpPsi = self.finerPsi[idx]
        tmpPh2 = self.finerPh2[idx]


        # v is the strain in real space
        v = np.zeros((len(xs), 3, 3))
        v[:, 0, 0] = tmpE11 + 1
        v[:, 0, 1] = tmpE12
        v[:, 0, 2] = tmpE13
        v[:, 1, 0] = v[:, 0, 1]
        v[:, 2, 0] = v[:, 0, 2]
        v[:, 1, 1] = tmpE22 + 1
        v[:, 1, 2] = tmpE23
        v[:, 2, 1] = v[:, 1, 2]
        v[:, 2, 2] = tmpE33 + 1

        # inv_avg_orien is the inverse of the average orientation in the grain
        inv_avg_orien = np.linalg.inv(Rot.EulerZXZ2Mat(self.EAngles[gid] / 180.0 * np.pi))

        # r is the orientation in real space
        r = np.zeros_like(v)
        for ii in range(len(r)):
            r[ii] = Rot.EulerZXZ2Mat(np.array([tmpPh1[ii], tmpPsi[ii], tmpPh2[ii]])/180.0*np.pi)

        # ss is the distortion in reciprocal space
        ss = np.zeros_like(v)
        for ii in range(len(ss)):
            ss[ii] = np.linalg.inv(v[ii].dot(r[ii]).dot(inv_avg_orien)).T

        avg_distortion = np.mean(ss, axis=0)

        simulator = Initializer(self.Cfg)
        simulator.generateGs(self.Positions[gid], self.EAngles[gid], avg_distortion)

        peakMap = simulator.simMap(xs, ys, ss - (avg_distortion - np.eye(3)), blur=self.blur, dtype=np.uint16)

        f = h5py.File(self.outFN, 'w')
        f.create_dataset("limits", data=simulator.LimH)
        f.create_dataset("Gs", data=simulator.Gs)
        f.create_dataset("whichOmega", data=simulator.whichOmega)
        f.create_dataset("Pos", data=simulator.pos)
        f.create_dataset("Orien", data=simulator.orien)
        f.create_dataset("OrienM", data=simulator.orienM)
        f.create_dataset("avg_distortion", data=simulator.avg_distortion)
        MaxInt = np.zeros(simulator.NumG, dtype=np.float32)
        grp = f.create_group('Imgs')
        for ii in range(simulator.NumG):
            myMaps = self._addNoise(peakMap[:, :, ii * 45:(ii + 1) * 45], simulator.Gs[ii])
            MaxInt[ii] = max(np.max(myMaps), 1)  # some G peaks are totally outside of the window, a hack
            grp.create_dataset('Im{0:d}'.format(ii), data=myMaps)
        f.create_dataset("MaxInt", data=MaxInt)

    def _addNoise(self, images, g_vector):
        if self.Cfg.noiseLevel == 0:
            return images
        else:
            PEAK = 1/(self.Cfg.noiseLevel+1e-4)
            lam = self.Cfg.noiseLevel * 4
            saltRatio = 0.7
            noisy = np.random.poisson(images  * PEAK) / PEAK + \
                    np.random.poisson(np.ones(images.shape)) * lam *(np.random.uniform(size=images.shape)>saltRatio)
            return noisy

