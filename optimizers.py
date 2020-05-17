import pycuda.driver as cuda
import numpy as np


def CrossEntropyMethod(recon, x, y,
                       XD, YD, OffsetD, MaskD, TrueMaskD, scoreD, S_gpu,
                       NumD=10000, numCut=100, cov=1e-6 * np.eye(9), MaxIter=50, mean=np.eye(3), BlockSize=256,
                       debug=False):
    if not recon.ImLoaded:
        recon.loadIm()
    if not recon.GsLoaded:
        recon.loadGs()

    for ii in range(MaxIter):
        S = np.random.multivariate_normal(
            np.zeros(9), cov, size=(NumD)).reshape((NumD, 3, 3), order='C') + np.tile(mean, (NumD, 1, 1))
        cuda.memcpy_htod(S_gpu, S.ravel().astype(np.float32))

        recon.sim_strain_func(XD, YD, OffsetD, MaskD, TrueMaskD,
                              np.float32(x), np.float32(y), recon.afDetInfoD, S_gpu,
                              recon.whichOmegaD, np.int32(NumD), np.int32(recon.NumG),
                              np.float32(recon.Cfg.energy), np.int32(45), recon.LimD, 
                              np.int32(5), np.float32(recon.Cfg.omgInterval),
                              block=(recon.NumG, 1, 1), grid=(NumD, 1))

        recon.hit_func(scoreD,
                       XD, YD, OffsetD, MaskD, TrueMaskD,
                       recon.MaxIntD, np.int32(recon.NumG), np.int32(NumD), np.int32(45),
                       block=(BlockSize, 1, 1), grid=(int(NumD / BlockSize + 1), 1))

        score = scoreD.get()

        args = np.argpartition(score, -numCut)[-numCut:]
        cov = np.cov(S[args].reshape((numCut, 9), order='C').T)
        mean = np.mean(S[args], axis=0)
        if debug:
            print(np.max(score))
        if np.trace(np.absolute(cov)) < 1e-8:
            break

    return cov, mean, np.max(score[args])


def ChangeOneVoxel_KL(recon, x, y, mean, realMapsLogD, falseMapsD,
                      XD, YD, OffsetD, MaskD, TrueMaskD, diffD, S_gpu,
                      NumD=10000, numCut=50, cov=1e-6 * np.eye(9), epsilon=1e-6, MaxIter=3, BlockSize=256, debug=False):
    if not recon.GsLoaded:
        recon.loadGs()
    # remove the original hit
    S = mean
    cuda.memcpy_htod(S_gpu, S.ravel().astype(np.float32))
    recon.sim_strain_func(XD, YD, OffsetD, MaskD, TrueMaskD,
                          np.float32(x), np.float32(y), recon.afDetInfoD, S_gpu,
                          recon.whichOmegaD, np.int32(1), np.int32(recon.NumG),
                          np.float32(recon.Cfg.energy), np.int32(45), recon.LimD, np.int32(5), np.float32(recon.Cfg.omgInterval),
                          block=(recon.NumG, 1, 1), grid=(1, 1))
    recon.One_func(XD, YD, OffsetD, MaskD, TrueMaskD,
                   falseMapsD, np.int32(recon.NumG), np.int32(45),
                   np.float32(epsilon), np.int32(-1),  # minus one!!
                   block=(recon.NumG, 1, 1), grid=(1, 1))
    # find a better distortion matrix
    for ii in range(MaxIter):
        S = np.empty((NumD, 3, 3), dtype=np.float32)
        S[0, :, :] = mean
        S[1:, :, :] = np.random.multivariate_normal(
            mean.ravel(), cov, size=(NumD - 1)).reshape((NumD - 1, 3, 3), order='C')
        cuda.memcpy_htod(S_gpu, S.ravel().astype(np.float32))
        recon.sim_strain_func(XD, YD, OffsetD, MaskD, TrueMaskD,
                              np.float32(x), np.float32(y), recon.afDetInfoD, S_gpu,
                              recon.whichOmegaD, np.int32(NumD), np.int32(recon.NumG),
                              np.float32(recon.Cfg.energy), np.int32(45), recon.LimD, np.int32(5), np.float32(recon.Cfg.omgInterval),
                              block=(recon.NumG, 1, 1), grid=(NumD, 1))
        recon.KL_diff_func(diffD,
                           XD, YD, OffsetD, MaskD, TrueMaskD,
                           realMapsLogD, falseMapsD,
                           np.int32(recon.NumG), np.int32(NumD), np.int32(45),
                           block=(BlockSize, 1, 1), grid=(int(NumD / BlockSize + 1), 1))
        diffH = diffD.get()
        args = np.argpartition(diffH, numCut)[:numCut]
        cov = np.cov(S[args].reshape((numCut, 9), order='C').T)
        mean = np.mean(S[args], axis=0)
        if ii == 0:
            diff_init = diffH[0]
        if debug:
            print(np.min(diffH), diffH[0])
    # add the new hit
    S = mean
    cuda.memcpy_htod(S_gpu, S.ravel().astype(np.float32))
    recon.sim_strain_func(XD, YD, OffsetD, MaskD, TrueMaskD,
                          np.float32(x), np.float32(y), recon.afDetInfoD, S_gpu,
                          recon.whichOmegaD, np.int32(1), np.int32(recon.NumG),
                          np.float32(recon.Cfg.energy), np.int32(45), recon.LimD, np.int32(5), np.float32(recon.Cfg.omgInterval),
                          block=(recon.NumG, 1, 1), grid=(1, 1))

    recon.KL_diff_func(diffD,
                       XD, YD, OffsetD, MaskD, TrueMaskD,
                       realMapsLogD, falseMapsD,
                       np.int32(recon.NumG), np.int32(1), np.int32(45),
                       block=(BlockSize, 1, 1), grid=(int(NumD / BlockSize + 1), 1))
    diffH = diffD.get()

    recon.One_func(XD, YD, OffsetD, MaskD, TrueMaskD,
                   falseMapsD, np.int32(recon.NumG), np.int32(45), np.float32(epsilon), np.int32(+1),  # plus one!!
                   block=(recon.NumG, 1, 1), grid=(1, 1))
    return cov, mean, diffH[0] - diff_init


