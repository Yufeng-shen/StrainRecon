# coding: utf-8
import numpy as np
import h5py
f=h5py.File('Exp_G13_Ps1_2nd.hdf5','w')
limits=[]
for ii in range(108):
    limits.append(np.load('g13Ps1_2nd_bf/limit{0:d}.npy'.format(ii))[0])
    
dset=f.create_dataset("limits",data=limits)
type(dset)
MaxInt=np.load('../StrainRecon/AuxData/MaxInt_g13Ps1_2nd.npy')
dset=f.create_dataset("MaxInt",data=MaxInt)
grp=f.create_group('Imgs')
for ii in range(108):
    tmp=np.load('g13Ps1_2nd_filtered/Im{0:d}.npy'.format(ii))
    tmp=np.moveaxis(tmp,0,2)
    grp.create_dataset('Im{0:d}'.format(ii),data=tmp)
    
f.create_dataset("Pos",data=np.array([-0.253125,0.0097428,0]))
f.create_dataset("Orien",data=np.array([120.784,80.9295,246.202]))
