import numpy as np
import time
from Reconst_GPU import StrainReconstructor_GPU, ReconSingleGrain
from InitStrain import Initializer
import os
import sys
import argparse

def parse_arguments():
    parser= argparse.ArgumentParser()
    parser.add_argument('--mode',dest='mode',type=str,default='')
    parser.add_argument('--outname',dest='outname',type=str,
            default='/home/yufengs/results/g15_2nd/',
            help="Path for the output files, end with '/'")
    parser.add_argument('--cfgFile',dest='cfgFile',type=str,
            default='ConfigFiles/g15Ps1_2nd.yml',
            help="Configure file name")
    return parser.parse_args()

def main(args):
    args=parse_arguments()

    start=time.time()

    outname=args.outname

    cfgFile=args.cfgFile

    if not os.path.exists(outname):
        os.makedirs(outname)

    print("Start running \n Output Directory: " +outname+ "\n Configure File: "+cfgFile)
    Cfg=Initializer(cfgFile)

    Cfg.Simulate()


    print("Initialized")

    recon=StrainReconstructor_GPU( _NumG=Cfg.NumG,
            bfPath=Cfg.bfPath,
            fltPath=Cfg.fltPath,
            maxIntfn=Cfg.maxIntfn,
            _Det=Cfg.Det, _Gs=Cfg.Gs, _Info=Cfg.Info, _eng=Cfg.eng)

    ReconGrain=ReconSingleGrain(grainOrien=Cfg.orien,
            micfn=Cfg.micfn)
    ################################
    x,y,con=ReconGrain.GetGrids()
    np.save(outname+'/x.npy',x)
    np.save(outname+'/y.npy',y)
    np.save(outname+'/con.npy',con)
    AllMaxScore,AllMaxS=ReconGrain.ReconGrids(x,y,recon)
    np.save(outname+'/allMaxScore.npy',AllMaxScore)
    np.save(outname+'/allMaxS.npy',AllMaxS)
    realO,realS=ReconGrain.Transform2RealS(AllMaxS)
    np.save(outname+'/realS.npy',realS)
    np.save(outname+'/realO.npy',realO)

    ##############################
    #x,y,con=ReconGrain.GetGrids()
    #ReconGrain.test(x,y,recon)


    end=time.time()
    print("Time elapsed: {:f} seconds".format(end-start))


if __name__=='__main__':
    main(sys.argv)
