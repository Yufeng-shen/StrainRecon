import numpy as np
import time
from Reconst_GPU import ReconSingleGrain
from Simulator_GPU import SimAllGrains
import os
import sys
import argparse

def parse_arguments():
    parser= argparse.ArgumentParser()
    parser.add_argument('--mode',dest='mode',type=str,default='sim')
    parser.add_argument('--outdir',dest='outdir',type=str,
            default='/home/yufengs/results/g13_2nd/',
            help="Path for the output files, end with '/'")
    parser.add_argument('--cfgFile',dest='cfgFile',type=str,
            default='ConfigFiles/g13Ps1_2nd.yml',
            help="Configure file name")
    return parser.parse_args()

def main(args):
    args=parse_arguments()
    start=time.time()

    outdir=args.outdir
    cfgFile=args.cfgFile
    mode=args.mode

    if mode=='rec':
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        print("Start reconstructing \n Output Directory: " +outdir+ "\n Configure File: "+cfgFile)
        rec=ReconSingleGrain(cfgFile,outdir)
        rec.run()
    elif mode=='sim':
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        print("Start simulating \n Output Directory: " +outdir+ "\n Configure File: "+cfgFile)
        sim=SimAllGrains(cfgFile,outdir)
        sim.SimSingleGrain(40)



    end=time.time()
    print("Time elapsed: {:f} seconds".format(end-start))


if __name__=='__main__':
    main(sys.argv)
