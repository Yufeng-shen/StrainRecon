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

    def __init__(self, Cfg, orig=[-0.256, -0.256], step=[0.002, 0.002], scale=10, factor=20, blur=True):
        """
        scale: refine the grid by 'scale' times
        factor: multiply the strain by 'factor'
        """
        self.Cfg = Cfg
        self.FakeSample = np.load(Cfg.micFile)
        self.outFN = Cfg.peakFile
        self.scale = scale
        self.factor = factor
        self.blur = blur
        # create a finer grid for more realistic simulation
        FakeSample = self.FakeSample
        finerDilation = zoom(FakeSample[0], zoom=scale, order=0) * factor
        finerE11 = zoom(FakeSample[1], zoom=scale, order=0) * factor
        finerE12 = zoom(FakeSample[2], zoom=scale, order=0) * factor
        finerE13 = zoom(FakeSample[3], zoom=scale, order=0) * factor
        finerE22 = zoom(FakeSample[4], zoom=scale, order=0) * factor
        finerE23 = zoom(FakeSample[5], zoom=scale, order=0) * factor
        finerE33 = zoom(FakeSample[6], zoom=scale, order=0) * factor
        finerGID = zoom(FakeSample[7], zoom=scale, order=0)
        finerPh1 = zoom(FakeSample[8], zoom=scale, order=0)
        finerPsi = zoom(FakeSample[9], zoom=scale, order=0)
        finerPh2 = zoom(FakeSample[10], zoom=scale, order=0)
        self.finerSample = np.array([finerDilation,
                                     finerE11, finerE12, finerE13, finerE22, finerE23, finerE33,
                                     finerGID, finerPh1, finerPsi, finerPh2])
        tmpx = np.arange(orig[0], step[0] / scale * finerGID.shape[0] + orig[0], step[0] / scale)
        tmpy = np.arange(orig[1], step[1] / scale * finerGID.shape[1] + orig[1], step[1] / scale)
        self.finerXV, self.finerYV = np.meshgrid(tmpx, tmpy)
        self.finerGIDLayer = self.finerSample[7].astype(int)

        # get all grains based on the grain ID (not nessasarily start from zero)
        GIDLayer = self.FakeSample[7].astype(int)
        GIDs = np.unique(GIDLayer)
        EALayer = self.FakeSample[8:11]
        EAngles = []
        Positions = []
        tmpx = np.arange(orig[0], step[0] * GIDLayer.shape[0] + orig[0], step[0])
        tmpy = np.arange(orig[1], step[1] * GIDLayer.shape[1] + orig[1], step[1])
        xv, yv = np.meshgrid(tmpx, tmpy)
        for gID in GIDs:
            idx = np.where(GIDLayer == gID)
            xs = xv[idx]
            ys = yv[idx]
            Positions.append([np.mean(xs), np.mean(ys), 0])
            EAngles.append([np.mean(EALayer[0][idx]), np.mean(EALayer[1][idx]), np.mean(EALayer[2][idx])])
        self.Positions = np.array(Positions)
        self.EAngles = np.array(EAngles)
        self.GIDs = GIDs

    def SimSingleGrain(self):
        gid = self.Cfg.grainID
        idx = np.where(self.finerGIDLayer == self.GIDs[gid])
        xs = self.finerXV[idx]
        ys = self.finerYV[idx]
        tmpDil = self.finerSample[0][idx]
        tmpE11 = self.finerSample[1][idx]
        tmpE12 = self.finerSample[2][idx]
        tmpE13 = self.finerSample[3][idx]
        tmpE22 = self.finerSample[4][idx]
        tmpE23 = self.finerSample[5][idx]
        tmpE33 = self.finerSample[6][idx]
        # This is wrong, the strain of lattice parameters and inverse lattice parameters (Gs) are
        # different, see the Transform2RealS function. S^(-T)O=PU. And I also assume orientations are
        # the same as averaged orientation: O=U. I'm just lazy, I will correct it later.
        ss = np.zeros((len(xs), 3, 3))
        ss[:, 0, 0] = tmpE11 + 1
        ss[:, 0, 1] = tmpE12
        ss[:, 0, 2] = tmpE13
        ss[:, 1, 0] = ss[:, 0, 1]
        ss[:, 2, 0] = ss[:, 0, 2]
        ss[:, 1, 1] = tmpE22 + 1
        ss[:, 1, 2] = tmpE23
        ss[:, 2, 1] = ss[:, 1, 2]
        ss[:, 2, 2] = tmpE33 + 1

        AvgStrain = np.mean(ss, axis=0)

        simulator = Initializer(self.Cfg)
        simulator.generateGs(self.Positions[gid], self.EAngles[gid], AvgStrain)

        peakMap = simulator.simMap(xs, ys, ss - (AvgStrain - np.eye(3)), blur=True, dtype=np.uint16)

        f = h5py.File(self.outFN, 'w')
        f.create_dataset("limits", data=simulator.LimH)
        f.create_dataset("Gs", data=simulator.Gs)
        f.create_dataset("whichOmega", data=simulator.whichOmega)
        f.create_dataset("Pos", data=simulator.pos)
        f.create_dataset("Orien", data=simulator.orien)
        f.create_dataset("OrienM", data=simulator.orienM)
        f.create_dataset("AvgStrain", data=simulator.AvgStrain)
        MaxInt = np.zeros(simulator.NumG, dtype=np.float32)
        grp = f.create_group('Imgs')
        for ii in range(simulator.NumG):
            myMaps = peakMap[:, :, ii * 45:(ii + 1) * 45]
            MaxInt[ii] = max(np.max(myMaps), 1)  # some G peaks are totally outside of the window, a hack
            grp.create_dataset('Im{0:d}'.format(ii), data=myMaps)
        f.create_dataset("MaxInt", data=MaxInt)

    def Transform2RealS(self, AllMaxS):
        AllMaxS = np.array(AllMaxS)
        realS = np.empty(AllMaxS.shape)
        realO = np.empty(AllMaxS.shape)
        for ii in range(len(realS)):
            # lattice constant I used in python nfHEDM scripts are different from the ffHEDM reconstruction used
            t = np.linalg.inv(AllMaxS[ii].T).dot(self.grainOrienM).dot(
                [[2.95 / 2.9254, 0, 0], [0, 2.95 / 2.9254, 0], [0, 0, 4.7152 / 4.674]])
            realO[ii], realS[ii] = polar(t, 'left')
        return realO, realS
