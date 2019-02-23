#!/usr/bin/env python
# coding: utf-8

# Since we used periodical boudary condition in synthetic microstructure generation (by Dream3D) and FFT simulation, 
# the grain pair that at the opposite boundaries will have the same grain ID, which is not we want. This script use 
# a floodfill method to relabel the grain IDs.


import numpy as np
import matplotlib.pyplot as plt
import h5py



GT=np.load('../AuxData/Ti7FFT.npy')
# row content
# 0  dilation
# 1  E11
# 2  E12
# 3  E13
# 4  E22
# 5  E23
# 6  E33
# 7  Grain ID
# 8  Phi1
# 9  Psi
# 10 Phi2


GIDLayer=GT[7].astype('int')


output=-np.ones(GIDLayer.shape,dtype='int')
def floodFill(image,target, sr, sc, newColor):
    """
    :type image: List[List[int]]
    :type sr: int
    :type sc: int
    :type newColor: int
    :rtype: List[List[int]]
    """
    R,C=len(image),len(image[0])
    color=image[sr][sc]
    if target[sr][sc]==newColor: return image
    def dfs(r,c):
        if image[r][c]==color and target[r][c]!=newColor:
            target[r][c]=newColor
            if r>0: dfs(r-1,c)
            if r+1<R: dfs(r+1,c)
            if c>0: dfs(r,c-1)
            if c+1<C: dfs(r,c+1)
    dfs(sr,sc)
    return target

curlabel=0
srs,scs=np.where(output==-1)
while len(srs)>0:
    sr,sc=srs[0],scs[0]
    output=floodFill(GIDLayer,output,sr,sc,curlabel)
    curlabel+=1
    srs,scs=np.where(output==-1)

np.save('../AuxData/Ti7FFT.npy',GT)

