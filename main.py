import numpy as np
import time
from Reconst_GPU import ReconSingleGrain
from Simulator_GPU import SimAllGrains
import os
import sys
import argparse

def parse_arguments():
    parser= argparse.ArgumentParser()
    parser.add_argument('--mode',dest='mode',type=str,default='sim',
            help="either 'sim' or 'rec'")
    parser.add_argument('--outdir',dest='outdir',type=str,
            default=None,
            help="Path for the output files, end with '/'")
    parser.add_argument('--cfgFile',dest='cfgFile',type=str,
            default=None,
            help="Configure file name")
    parser.add_argument('--gid',dest='gid',type=int,default=None,
            help="the grain ID to be simulated")
    return parser.parse_args()

def main(args):
    args=parse_arguments()

    outdir=args.outdir
    cfgFile=args.cfgFile
    mode=args.mode
    gid=args.gid

    if mode=='rec':
        if cfgFile==None:
            cfgFile='ConfigFiles/g40.yml'
        if outdir==None:
            outdir='/home/yufengs/SimData/'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        if gid==None:
            gid=40
        print("Start reconstructing \n Output Directory: " +outdir+ "\n Configure File: "+cfgFile)
        rec=ReconSingleGrain(cfgFile,outdir,gid)
        start=time.time()
        rec.run()
        end=time.time()
        print("Time elapsed: {:f} seconds".format(end-start))
    elif mode=='sim':
        if cfgFile==None:
            cfgFile='ConfigFiles/sim.yml'
        if outdir==None:
            outdir='/home/yufengs/SimData/'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        if gid==None:
            gid=40
        print("Start simulating \n Output Directory: " +outdir+ "\n Configure File: "+cfgFile)
        while True:
            choice = input("Proceed?(y/n) ")
            if choice == 'y' or choice == 'Y' :
                start=time.time()
                sim=SimAllGrains(cfgFile,outdir,scale=10,factor=20,blur=True)
                sim.SimSingleGrain(gid,outputfn=None)
                break
            elif choice== 'n' or choice =='N':
                break





if __name__=='__main__':
    main(sys.argv)
