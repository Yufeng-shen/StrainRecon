import util.Simulation as Gsim
import util.RotRep as Rot
import numpy as np
import matplotlib.pyplot as plt


exp={'energy':51.9957}
eng=exp['energy']
etalimit=81/180.0*np.pi

########################
# Detector parameters (With HRM)
########################

Det1=Gsim.Detector(psize=0.001454)
Det1.Move(1182.19,2026.27,np.array([7.14503,0,0]),Rot.EulerZXZ2Mat(np.array([89.1588,87.5647,0.278594])/180.0*np.pi))
#Det2=Gsim.Detector()
#Det2.Move(1199.05,2024.1,np.array([9.08727,0,0]),Rot.EulerZXZ2Mat(np.array([89.1058, 88.5764,0.268393])/180.0*np.pi))
# Det3=G.Detector()
# Det3.Move(966.576,1994.19,np.array([8.63154,0,0]),Rot.EulerZXZ2Mat(np.array([89.24, 90.545,359.188])/180.0*np.pi))

#########################
# LP15 for 2nd Load
#########################
Ti7LP15=Gsim.CrystalStr()
Ti7LP15.PrimA=2.95*np.array([1,0,0])
Ti7LP15.PrimB=2.95*np.array([np.cos(np.pi*2/3),np.sin(np.pi*2/3),0])
Ti7LP15.PrimC=4.7152*np.array([0,0,1])
Ti7LP15.addAtom([1/3.0,2/3.0,1/4.0],22)
Ti7LP15.addAtom([2/3.0,1/3.0,3/4.0],22)
Ti7LP15.getRecipVec()
Ti7LP15.getGs(13)
#########################
# Default Ti7 LP
#########################
Ti7=Gsim.CrystalStr('Ti7')
Ti7.getRecipVec()
Ti7.getGs(13)

###########################################
## grain 25, 2nd Load
###########################################
#g25pos2nd=np.array([3.7998e-07, -0.204599, 0])
#g25orien2nd=Rot.EulerZXZ2Mat(np.array([328.93, 88.8624, 11.7176])/180.0*np.pi)
#g25Ps1_2nd,g25Gs1_2nd,g25Info1_2nd=Gsim.GetProjectedVertex(Det1,Ti7LP15,g25orien2nd,etalimit,g25pos2nd,getPeaksInfo=True,
#                                           omegaL=0,omegaU=180,**exp)
#g25Ps2_2nd,g25Gs2_2nd,g25Info2_2nd=Gsim.GetProjectedVertex(Det2,Ti7LP15,g25orien2nd,etalimit,g25pos2nd,getPeaksInfo=True,
#                                           omegaL=0,omegaU=180,**exp)
#
###########################################
## grain 25, Pre Load
###########################################
#g25posPre=np.array([0.151875, -0.204599, 0])
#g25orienPre=Rot.EulerZXZ2Mat(np.array([328.678,88.6526,10.9338])/180.0*np.pi)
#g25Ps1_pre,g25Gs1_pre,g25Info1_pre=Gsim.GetProjectedVertex(Det1,Ti7,g25orienPre,etalimit,g25posPre,getPeaksInfo=True,
#                                           omegaL=0,omegaU=180,**exp)
#g25Ps2_pre,g25Gs2_pre,g25Info2_pre=Gsim.GetProjectedVertex(Det2,Ti7,g25orienPre,etalimit,g25posPre,getPeaksInfo=True,
#                                           omegaL=0,omegaU=180,**exp)
#
###########################################
## grain 25, Unload
###########################################
#g25posUn=np.array([0.104063, -0.20947, 0])
#g25orienUn=Rot.EulerZXZ2Mat(np.array([329.06, 88.7822, 11.2104])/180.0*np.pi)
#g25Ps1_un,g25Gs1_un,g25Info1_un=Gsim.GetProjectedVertex(Det1,Ti7,g25orienUn,etalimit,g25posUn,getPeaksInfo=True,
#                                           omegaL=0,omegaU=180,**exp)
#g25Ps2_un,g25Gs2_un,g25Info2_un=Gsim.GetProjectedVertex(Det2,Ti7,g25orienUn,etalimit,g25posUn,getPeaksInfo=True,
#                                           omegaL=0,omegaU=180,**exp)
#
#
###########################################
## grain 15, Pre Load
###########################################
#g15posPre=np.array([-0.109687, 0.102299, 0])
#g15orienPre=Rot.EulerZXZ2Mat(np.array([298.153, 64.7209, 42.4337])/180.0*np.pi)
#g15Ps1_pre,g15Gs1_pre,g15Info1_pre=Gsim.GetProjectedVertex(Det1,Ti7,g15orienPre,etalimit,g15posPre,
#                                                          getPeaksInfo=True,omegaL=0,omegaU=180,**exp)
#g15Ps2_pre,g15Gs2_pre,g15Info2_pre=Gsim.GetProjectedVertex(Det2,Ti7,g15orienPre,etalimit,g15posPre,
#                                                          getPeaksInfo=True,omegaL=0,omegaU=180,**exp)
#
##########################################
# grain 15, 2nd Load
##########################################
g15pos2nd=np.array([-0.163125, 0.107171, 0])
g15orien2nd=Rot.EulerZXZ2Mat(np.array([298.089, 65.4218, 42.9553])/180.0*np.pi)
g15Ps1_2nd,g15Gs1_2nd,g15Info1_2nd=Gsim.GetProjectedVertex(Det1,Ti7LP15,g15orien2nd,etalimit,g15pos2nd,getPeaksInfo=True,
                                           omegaL=0,omegaU=180,**exp)
#g15Ps2_2nd,g15Gs2_2nd,g15Info2_2nd=Gsim.GetProjectedVertex(Det2,Ti7LP15,g15orien2nd,etalimit,g15pos2nd,getPeaksInfo=True,
#                                           omegaL=0,omegaU=180,**exp)

###########################################
## grain 15, Unload
###########################################
#g15posUn=np.array([-0.0815624, 0.102299, 0])
#g15orienUn=Rot.EulerZXZ2Mat(np.array([298.35, 64.9384, 42.5896])/180.0*np.pi)
#g15Ps1_un,g15Gs1_un,g15Info1_un=Gsim.GetProjectedVertex(Det1,Ti7,g15orienUn,etalimit,g15posUn,getPeaksInfo=True,
#                                           omegaL=0,omegaU=180,**exp)
#g15Ps2_un,g15Gs2_un,g15Info2_un=Gsim.GetProjectedVertex(Det2,Ti7,g15orienUn,etalimit,g15posUn,getPeaksInfo=True,
#                                           omegaL=0,omegaU=180,**exp)
#
###########################################
## grain 24, Pre Load
###########################################
#g24posPre=np.array([-0.28125, 0.17537, 0])
#g24orienPre=Rot.EulerZXZ2Mat(np.array([341.509, 79.179, 6.53087])/180.0*np.pi)
#g24Ps1_pre,g24Gs1_pre,g24Info1_pre=Gsim.GetProjectedVertex(Det1,Ti7,g24orienPre,etalimit,g24posPre,
#                                                          getPeaksInfo=True,omegaL=0,omegaU=180,**exp)
#
###########################################
## grain 24, 2nd Load
###########################################
#g24pos2nd=np.array([-0.402187, 0.20947, 0])
#g24orien2nd=Rot.EulerZXZ2Mat(np.array([ 341.519, 79.2855, 7.2755])/180.0*np.pi)
#g24Ps1_2nd,g24Gs1_2nd,g24Info1_2nd=Gsim.GetProjectedVertex(Det1,Ti7LP15,g24orien2nd,etalimit,g24pos2nd,getPeaksInfo=True,
#                                           omegaL=0,omegaU=180,**exp)
#
###########################################
## grain 24, Unload
###########################################
#g24posUn=np.array([-0.309375, 0.165627, 0])
#g24orienUn=Rot.EulerZXZ2Mat(np.array([341.697, 79.2779, 6.79164])/180.0*np.pi)
#g24Ps1_un,g24Gs1_un,g24Info1_un=Gsim.GetProjectedVertex(Det1,Ti7,g24orienUn,etalimit,g24posUn,getPeaksInfo=True,
#                                           omegaL=0,omegaU=180,**exp)
#
#


###########################################
## grain 45, 2nd Load
###########################################
#g45pos2nd=np.array([-0.143437,  0.219213, 0])
#g45orien2nd=Rot.EulerZXZ2Mat(np.array([  117.498 ,   86.6945,  214.956])/180.0*np.pi)
#g45Ps1_2nd,g45Gs1_2nd,g45Info1_2nd=Gsim.GetProjectedVertex(Det1,Ti7LP15,g45orien2nd,etalimit,g45pos2nd,getPeaksInfo=True,
#                                           omegaL=0,omegaU=180,**exp)
#
#
###########################################
## grain 46, 2nd Load
###########################################
#g46pos2nd=np.array([-0.250312,  0.248441, 0])
#g46orien2nd=Rot.EulerZXZ2Mat(np.array([  147.248 ,   81.8513,  217.515])/180.0*np.pi)
#g46Ps1_2nd,g46Gs1_2nd,g46Info1_2nd=Gsim.GetProjectedVertex(Det1,Ti7LP15,g46orien2nd,etalimit,g46pos2nd,getPeaksInfo=True,
#                                           omegaL=0,omegaU=180,**exp)
#
##########################################
# grain 13, 2nd Load
##########################################
g13pos2nd=np.array([-0.253125,  0.0097428, 0])
g13orien2nd=Rot.EulerZXZ2Mat(np.array([   120.784 ,   80.9295,  246.202])/180.0*np.pi)
g13Ps1_2nd,g13Gs1_2nd,g13Info1_2nd=Gsim.GetProjectedVertex(Det1,Ti7LP15,g13orien2nd,etalimit,g13pos2nd,getPeaksInfo=True,
                                           omegaL=0,omegaU=180,**exp)


## for Det1
#def fetch(ii,pks,fn,offset=0,dx=100,dy=50,verbo=False,more=False):
#    omegid=int((180-pks[ii,2])*20)+offset
#    if omegid<0:
#        omegid+=3600
#    if omegid>=3600:
#        omegid-=3600
#    I=plt.imread(fn+'{0:06d}.tif'.format(omegid))
#    x1=int((2047-pks[ii,0])-dx)
#    y1=int(pks[ii,1]-dy)
#    if verbo:
#        print 'y=',pks[ii,1]
#        print 'x=',pks[ii,0]
#    x1=max(0,x1)
#    y1=max(0,y1)
#    x2=x1+2*dx
#    y2=y1+2*dy
#    x2=min(x2,2048)
#    y2=min(y2,2048)
#    if more:
#        return I[y1:y2,x1:x2],(x1,x2,y1,y2,omegid)
#    return I[y1:y2,x1:x2]
#
## for Det2
#def fetch2(ii,pks,fn,offset=0,dx=100,dy=50,verbo=False,more=False):
#    omegid=int((180-pks[ii,2])*20)+offset
#    if omegid<0:
#        omegid+=3600
#    if omegid>=3600:
#        omegid-=3600
#    I=plt.imread(fn+'{0:06d}.tif'.format(omegid+3600))
#    x1=int((2047-pks[ii,0])-dx)
#    y1=int(pks[ii,1]-dy)
#    if verbo:
#        print 'y=',pks[ii,1]
#        print 'x=',pks[ii,0]
#    x1=max(0,x1)
#    y1=max(0,y1)
#    x2=x1+2*dx
#    y2=y1+2*dy
#    x2=min(x2,2048)
#    y2=min(y2,2048)
#    if more:
#        return I[y1:y2,x1:x2],(x1,x2,y1,y2,omegid)
#    return I[y1:y2,x1:x2]


