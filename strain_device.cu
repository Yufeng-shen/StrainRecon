#include <stdio.h>
const float PI = 3.14159265359;
texture<unsigned int, cudaTextureType3D, cudaReadModeElementType> tcExpData;
texture<float, cudaTextureType2D, cudaReadModeElementType> tfG;  // texture to store scattering vectors;
//texture<int, cudaTextureType3D, cudaReadModeElementType> tlimits;// texture to store position of windows;
//int Lim[][];//position of windows (x1,x2,y1,y2,omega1)
//float MaxInten[];//max intensity of that peak
//int iWhichOmega[];// 1 if a, 2 if b

typedef struct {
    float fNPixelJ, fNPixelK;
    float fPixelJ, fPixelK;
	float afCoordOrigin[3];
	float afNorm[3];
	float afJVector[3];
	float afKVector[3];
} DetInfo;

__device__ bool GetScatteringOmegas( float &fOmegaRes1, float &fOmegaRes2,
		float &fTwoTheta,float &fEta,float &fChi,
        const float *aScatteringVec,const float &fBeamEnergy){
	////////////////////////test passed ////////////////////////
 //aScatterinVec: float[3];
	// need take PI as constant
  /// fOmegaRe
  ///  NOTE:  reciprical vectors are measured in angstrom
  ///  k  = 2pi/lambda  = E/ (h-bar c)
  ///
  ///aScatteringVec: float array,len=3, contains the scattering vecter x,y,z.
	//////////////////////////////////////////////////////////////
  float fScatteringVecMag = sqrt(aScatteringVec[0]*aScatteringVec[0] + aScatteringVec[1]*aScatteringVec[1] +aScatteringVec[2]*aScatteringVec[2]);

  float fSinTheta = fScatteringVecMag / ( (float)2.0 * 0.506773182 * fBeamEnergy);   // Bragg angle
  //float fCosTheta = sqrt( 1.f - fSinTheta * fSinTheta);
  float fCosChi = aScatteringVec[2] / fScatteringVecMag;             // Tilt angle of G relative to z-axis
  float fSinChi = sqrt( 1.f - fCosChi * fCosChi );
  //float fSinChiLaue = sin( fBeamDeflectionChiLaue );     // ! Tilt angle of k_i (+ means up)
  //float fCosChiLaue = cos( fBeamDeflectionChiLaue );

  if( fabsf( fSinTheta ) <= fabsf( fSinChi) )
  {
	//float fPhi = acosf(fSinTheta / fSinChi);
	float fSinPhi = sin(acosf(fSinTheta / fSinChi));
	//float fCosTheta = sqrt( 1.f - fSinTheta * fSinTheta);
	fEta = asinf(fSinChi * fSinPhi / sqrt( 1.f - fSinTheta * fSinTheta));
	// [-pi:pi]: angle to bring G to nominal position along +y-axis
	float fDeltaOmega0 = atan2f( aScatteringVec[0], aScatteringVec[1]);

	//  [0:pi/2] since arg >0: phi goes from above to Bragg angle
	float fDeltaOmega_b1 = asinf( fSinTheta/fSinChi );

	//float fDeltaOmega_b2 = PI -  fDeltaOmega_b1;

	fOmegaRes1 = fDeltaOmega_b1 + fDeltaOmega0;  // oScatteringVec.m_fY > 0
	fOmegaRes2 = PI - fDeltaOmega_b1 + fDeltaOmega0;  // oScatteringVec.m_fY < 0
    	//fOmegaRes1 -= 2.f * PI*(trunc(fOmegaRes1/PI));
    	//fOmegaRes2 -= 2.f * PI*(trunc(fOmegaRes2/PI));
	if ( fOmegaRes1 > PI )          // range really doesn't matter
	  fOmegaRes1 -=  2.f * PI;

	if ( fOmegaRes1 < -PI)
	  fOmegaRes1 +=  2.f * PI;

	if ( fOmegaRes2 > PI)
	  fOmegaRes2 -= 2.f * PI;

	if ( fOmegaRes2 < -PI)
	  fOmegaRes2 += 2.f * PI;
	fTwoTheta = 2.f * asinf(fSinTheta);
	fChi = acosf(fCosChi);
	return true;
  }
  else
  {
	fOmegaRes1 = fOmegaRes2 = 0;     // too close to rotation axis to be illumination
	fTwoTheta = fEta = fChi = 0;
	return false;
  }


}

__device__ void Intersection(int &iJ, int &iK, bool &bMask,
		const float *afScatterSrc, float fTwoTheta, float fEta, const float* __restrict__ afDetInfo){
	float fDist, fAngleNormScatter;
	float afScatterDir[3];
	float afInterPos[3];
	fDist = afDetInfo[7]*(afDetInfo[4] - afScatterSrc[0])
					+ afDetInfo[8]*(afDetInfo[5] - afScatterSrc[1])
					+ afDetInfo[9]*(afDetInfo[6] - afScatterSrc[2]);
	afScatterDir[0] = cos(fTwoTheta);
	afScatterDir[1] = sin(fTwoTheta) * sin(fEta);
	afScatterDir[2] = sin(fTwoTheta) * cos(fEta);
	fAngleNormScatter = afDetInfo[7]*afScatterDir[0]
	                          + afDetInfo[8]*afScatterDir[1]
	                          + afDetInfo[9]*afScatterDir[2];
	afInterPos[0] = fDist / fAngleNormScatter * afScatterDir[0] + afScatterSrc[0];
	afInterPos[1] = fDist / fAngleNormScatter * afScatterDir[1] + afScatterSrc[1];
	afInterPos[2] = fDist / fAngleNormScatter * afScatterDir[2] + afScatterSrc[2];
	iJ = floor((afDetInfo[10]*(afInterPos[0]-afDetInfo[4])
			+ afDetInfo[11]*(afInterPos[1]-afDetInfo[5])
			+ afDetInfo[12]*(afInterPos[2]-afDetInfo[6]) )/afDetInfo[2]);
	iK = floor((afDetInfo[13]*(afInterPos[0]-afDetInfo[4])
			+ afDetInfo[14]*(afInterPos[1]-afDetInfo[5])
			+ afDetInfo[15]*(afInterPos[2]-afDetInfo[6]) )/afDetInfo[3]);
	bMask = (0<=iJ ) && (iJ<(int)afDetInfo[0]) && (0<=iK) && (iK<(int)afDetInfo[1]);
}



__device__ void JK2Window(int &iX, int &iY, int &iOffset, bool &bMask,
		int iJ, int iK, float fOmega, int idx, int iNumFrame, const int* __restrict__ Lim, int LimSize){
	/*
	 * This change the pixel coordinate J, K, Omega to local window coordinate X, Y, Offset.
	 * The index idx is the peak ID (Gs ID)
	 * LimSize=5, 1:x1,2:x2,3:y1,4:y2,5:Omega1
	 */

		iX=2047-iJ-Lim[idx*LimSize+0];
		iY=iK-Lim[idx*LimSize+2];
		iOffset=int((180-fOmega*180/PI)*20)-Lim[idx*LimSize+4];
		if (iOffset<0) {iOffset+=3600;}
		bMask=(iX>=0 && iX<(Lim[idx*LimSize+1]-Lim[idx*LimSize+0]) &&
				iY>=0 && iY<(Lim[idx*LimSize+3]-Lim[idx*LimSize+2]) && iOffset<iNumFrame);

}


__global__ void Simulate_for_Strain(int *aiX, int *aiY, int *aiOffset, bool *abMask, bool *abtrueMask,
		float fx, float fy, const float* __restrict__ afDetInfo, const float *afDistortion,
		const int* __restrict__ iWhichOmega,
		int iNumD, int iNumG, float fBeamEnergy, int iNumFrame, const int* __restrict__ Lim, int LimSize){
	/*
	 * get Xs,Ys and Offsets for every G and Distortion;
	 * Number of Gs is blockDim.x,
	 * number of Distortions is gridDim.x
	 * The order of output array: same Distortion first.
	 */
	int i=threadIdx.x;// index for Gs
	int j=blockIdx.x;// index for Distortions
	float fOmegaRes1,fOmegaRes2,fTwoTheta,fEta,fChi,ftmpOmega,ftmpEta;
	float afRotatedG[3]={0,0,0};
	float aScatterSrc[3]={0,0,0};
	int iJ,iK;
	for(int ii=0;ii<3;ii++){
		for(int jj=0;jj<3;jj++){
			afRotatedG[ii]+=afDistortion[j*9+ii*3+jj]*tex2D(tfG,(float)jj,(float)i);
		}
	}
	GetScatteringOmegas( fOmegaRes1, fOmegaRes2, fTwoTheta, fEta, fChi , afRotatedG,fBeamEnergy);
	if(iWhichOmega[i]==1){
		ftmpOmega=fOmegaRes1;
		ftmpEta=fEta;
	}
	else{
		ftmpOmega=fOmegaRes2;
		ftmpEta=-fEta;
	}
	aScatterSrc[0]=cos(ftmpOmega)*fx-sin(ftmpOmega)*fy;
	aScatterSrc[1]=cos(ftmpOmega)*fy+sin(ftmpOmega)*fx;
	Intersection(iJ,iK,abtrueMask[i+j*iNumG],aScatterSrc,fTwoTheta,ftmpEta, afDetInfo);
	JK2Window(aiX[i+j*iNumG],aiY[i+j*iNumG],aiOffset[i+j*iNumG],abMask[i+j*iNumG],
			iJ,iK,ftmpOmega,i, iNumFrame, Lim, LimSize);
}

__global__ void Simulate_for_Pos(int *aiX, int *aiY, int *aiOffset, bool *abMask, bool *abtrueMask,
		float *afx, float *afy, const float* __restrict__ afDetInfo, const float *afDistortion,
		const int* __restrict__ iWhichOmega,
		int iNumD, int iNumG, float fBeamEnergy, int iNumFrame, const int* __restrict__ Lim, int LimSize){
	/*
	 * get Xs,Ys and Offsets for every G and Distortion 
     * The difference with 'Simulate_for_Strain' is this function simulate multiple xy position, each has one distortion;
	 * Number of Gs is blockDim.x,
	 * number of Distortions and position is gridDim.x
	 * The order of output array: same Distortion first.
	 */
	int i=threadIdx.x;// index for Gs
	int j=blockIdx.x;// index for Distortions and voxel
	float fOmegaRes1,fOmegaRes2,fTwoTheta,fEta,fChi,ftmpOmega,ftmpEta;
	float afRotatedG[3]={0,0,0};
	float aScatterSrc[3]={0,0,0};
	int iJ,iK;
	for(int ii=0;ii<3;ii++){
		for(int jj=0;jj<3;jj++){
			afRotatedG[ii]+=afDistortion[j*9+ii*3+jj]*tex2D(tfG,(float)jj,(float)i);
		}
	}
	GetScatteringOmegas( fOmegaRes1, fOmegaRes2, fTwoTheta, fEta, fChi , afRotatedG,fBeamEnergy);
	if(iWhichOmega[i]==1){
		ftmpOmega=fOmegaRes1;
		ftmpEta=fEta;
	}
	else{
		ftmpOmega=fOmegaRes2;
		ftmpEta=-fEta;
	}
	aScatterSrc[0]=cos(ftmpOmega)*afx[j]-sin(ftmpOmega)*afy[j];
	aScatterSrc[1]=cos(ftmpOmega)*afy[j]+sin(ftmpOmega)*afx[j];
	Intersection(iJ,iK,abtrueMask[i+j*iNumG],aScatterSrc,fTwoTheta,ftmpEta, afDetInfo);
	JK2Window(aiX[i+j*iNumG],aiY[i+j*iNumG],aiOffset[i+j*iNumG],abMask[i+j*iNumG],
			iJ,iK,ftmpOmega,i, iNumFrame, Lim, LimSize);
}

__global__ void Hit_Score(float *afscore,
		const int* __restrict__ aiX, const int*  __restrict__ aiY, 
		const int* __restrict__  aiOffset, const bool*  __restrict__ abMask, const bool*  __restrict__ abtrueMask,
		const float* __restrict__ MaxInten,
		 int iNumG, int iNumD, int iNumFrame){
	/*
	 */
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i < iNumD){
		afscore[i]=0;
		for(int jj=0;jj<iNumG;jj++){
			float ftmpScore;
			int tmpIdx=i*iNumG+jj;
			if(!abtrueMask[tmpIdx]){ftmpScore=0.5;}
			else if(!abMask[tmpIdx]){ftmpScore=0.0;}
			else{
				ftmpScore=tex3D(tcExpData,(float)aiY[tmpIdx],(float)aiX[tmpIdx],(float)(jj*iNumFrame+aiOffset[tmpIdx]));
				ftmpScore/=MaxInten[jj];
			}
			afscore[i]+=ftmpScore;
		}
	}
}

__global__ void KL_total(float *afKL,
		const float *realMapLog, const float *fakeMap,int jj, int iNumG, int iNumFrame){
	int j=threadIdx.x;
	int i=blockIdx.x;
	if(i<300*160&& j<iNumFrame){
		uint pixelIdx=i*iNumG*iNumFrame+j+jj*iNumFrame;
		float fake=fakeMap[pixelIdx];
		float realLog=realMapLog[pixelIdx];
		afKL[i+j*300*160]=fake*logf(fake)-fake*realLog;
	}
}

__global__ void L1_total(float *afL1,
		const float *realMap, const float *fakeMap,int jj, int iNumG, int iNumFrame){
	int j=threadIdx.x;
	int i=blockIdx.x;
	if(i<300*160&& j<iNumFrame){
		uint pixelIdx=i*iNumG*iNumFrame+j+jj*iNumFrame;
		float fake=fakeMap[pixelIdx];
		float real=realMap[pixelIdx];
		afL1[i+j*300*160]=fminf(fake,fabsf(fake-real));
	}
}


__global__ void ChangeOne(const int *aiX, const int *aiY, const int *aiOffset, const bool *abMask, const bool *abtrueMask,
		float* __restrict__ fakeMap,
		int iNumG, int iNumFrame,float epsilon, int one){
	int i=blockIdx.x*blockDim.x+threadIdx.x;// id of G vector
	if(i<iNumG){
		if(abMask[i] && abtrueMask[i]){
			uint pixelIdx=aiY[i]*300*iNumG*iNumFrame+aiX[i]*iNumG*iNumFrame+aiOffset[i]+iNumFrame*i;
			fakeMap[pixelIdx]=fmaxf(fakeMap[pixelIdx]+one,epsilon);
		}
	}
}


__global__ void KL_diff(float *afKLdiff,
		const int* __restrict__ aiX, const int*  __restrict__ aiY, 
		const int* __restrict__  aiOffset, const bool*  __restrict__ abMask, const bool*  __restrict__ abtrueMask,
		const float* __restrict__ realMapLog, const float* __restrict__ fakeMap,
		int iNumG, int iNumD, int iNumFrame){
	int i=blockIdx.x * blockDim.x + threadIdx.x; // id of distortion matrix
	if(i<iNumD){
		afKLdiff[i]=0;
		for(int jj=0;jj<iNumG;jj++){
			float ftmp;
			uint tmpIdx=i*iNumG+jj;
			if(!abtrueMask[tmpIdx]){ftmp=0.0;}//hit outside of Detector
			else if(!abMask[tmpIdx]){ftmp=10;}//hit outside of Window
			else{
				uint pixelIdx=aiY[tmpIdx]*300*iNumG*iNumFrame+aiX[tmpIdx]*iNumG*iNumFrame+aiOffset[tmpIdx]+iNumFrame*jj;
				float fake=fakeMap[pixelIdx];
				float realLog=realMapLog[pixelIdx];
				ftmp=(fake+1)*logf(fake+1)-realLog-fake*logf(fake);
			}
			afKLdiff[i]+=ftmp;
		}
	}
}


__global__ void L1_diff(float *afL1diff,
		const int* __restrict__ aiX, const int*  __restrict__ aiY, 
		const int* __restrict__  aiOffset, const bool*  __restrict__ abMask, const bool*  __restrict__ abtrueMask,
		const float* __restrict__ realMap, const float* __restrict__ fakeMap,
		int iNumG, int iNumD, int iNumFrame){
	int i=blockIdx.x * blockDim.x + threadIdx.x; // id of distortion matrix
	if(i<iNumD){
		afL1diff[i]=0;
		for(int jj=0;jj<iNumG;jj++){
			float ftmp;
			uint tmpIdx=i*iNumG+jj;
			if(!abtrueMask[tmpIdx]){ftmp=0.0;}//hit outside of Detector
			else if(!abMask[tmpIdx]){ftmp=10;}//hit outside of Window
			else{
				uint pixelIdx=aiY[tmpIdx]*300*iNumG*iNumFrame+aiX[tmpIdx]*iNumG*iNumFrame+aiOffset[tmpIdx]+iNumFrame*jj;
				float fake=fakeMap[pixelIdx];
				float real=realMap[pixelIdx];
				ftmp=fabsf(fake+1-real)-fabsf(fake-real);
			}
			afL1diff[i]+=ftmp;
		}
	}
}

__global__ void display_rand(float* afRandom, int iNRand){
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        printf("=%d=",i);
        if (i<iNRand){
        printf(" %f ||", afRandom[i]);
        }
}

__global__ void euler_zxz_to_mat(float* afEuler, float* afMat, int iNAngle){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<iNAngle){
        float s1 = sin(afEuler[i * 3 + 0]);
        float s2 = sin(afEuler[i * 3 + 1]);
        float s3 = sin(afEuler[i * 3 + 2]);
        float c1 = cos(afEuler[i * 3 + 0]);
        float c2 = cos(afEuler[i * 3 + 1]);
        float c3 = cos(afEuler[i * 3 + 2]);
        afMat[i * 9 + 0] = c1 * c3 - c2 * s1 * s3;
        afMat[i * 9 + 1] = -c1 * s3 - c3 * c2 * s1;
        afMat[i * 9 + 2] = s1 * s2;
        afMat[i * 9 + 3] = s1 * c3 + c2 * c1 * s3;
        afMat[i * 9 + 4] = c1 * c2 * c3 - s1 * s3;
        afMat[i * 9 + 5] = -c1 * s2;
        afMat[i * 9 + 6] = s3 * s2;
        afMat[i * 9 + 7] = s2 * c3;
        afMat[i * 9 + 8] = c2;
    }
}

__device__ void d_euler_zxz_to_mat(float* afEuler, float* afMat){
        float s1 = sin(afEuler[0]);
        float s2 = sin(afEuler[1]);
        float s3 = sin(afEuler[2]);
        float c1 = cos(afEuler[0]);
        float c2 = cos(afEuler[1]);
        float c3 = cos(afEuler[2]);
        afMat[0] = c1 * c3 - c2 * s1 * s3;
        afMat[1] = -c1 * s3 - c3 * c2 * s1;
        afMat[2] = s1 * s2;
        afMat[3] = s1 * c3 + c2 * c1 * s3;
        afMat[4] = c1 * c2 * c3 - s1 * s3;
        afMat[5] = -c1 * s2;
        afMat[6] = s3 * s2;
        afMat[7] = s2 * c3;
        afMat[8] = c2;
}

__global__ void mat_to_euler_ZXZ(float* afMatIn, float* afEulerOut, int iNAngle){
    /*
    * transform active rotation matrix to euler angles in ZXZ convention, not right(seems right now)
    * afMatIn: iNAngle * 9
    * afEulerOut: iNAngle* 3
    * TEST PASSED
    */
    float threshold = 0.9999999;
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<iNAngle){
        if(afMatIn[i * 9 + 8] > threshold){
            afEulerOut[i * 3 + 0] = 0;
            afEulerOut[i * 3 + 1] = 0;
            afEulerOut[i * 3 + 2] = atan2(afMatIn[i*9 + 3], afMatIn[i*9 + 0]);           //  atan2(m[1, 0], m[0, 0])
        }
        else if(afMatIn[i * 9 + 8] < - threshold){
            afEulerOut[i * 3 + 0] = 0;
            afEulerOut[i * 3 + 1] = PI;
            afEulerOut[i * 3 + 2] = atan2(afMatIn[i*9 + 1], afMatIn[i*9 + 0]);           //  atan2(m[0, 1], m[0, 0])
        }
        else{
            afEulerOut[i * 3 + 0] = atan2(afMatIn[i*9 + 2], - afMatIn[i*9 + 5]);          //  atan2(m[0, 2], -m[1, 2])
            afEulerOut[i * 3 + 1] = atan2( sqrt(afMatIn[i*9 + 6] * afMatIn[i*9 + 6]
                                                + afMatIn[i*9 + 7] * afMatIn[i*9 + 7]),
                                            afMatIn[i*9 + 8]);                             //     atan2(np.sqrt(m[2, 0] ** 2 + m[2, 1] ** 2), m[2, 2])
            afEulerOut[i * 3 + 2] = atan2( afMatIn[i*9 + 6], afMatIn[i*9 + 7]);           //   atan2(m[2, 0], m[2, 1])
            if(afEulerOut[i * 3 + 0] < 0){
                afEulerOut[i * 3 + 0] += 2 * PI;
            }
            if(afEulerOut[i * 3 + 1] < 0){
                afEulerOut[i * 3 + 1] += 2 * PI;
            }
            if(afEulerOut[i * 3 + 2] < 0){
                afEulerOut[i * 3 + 2] += 2 * PI;
            }
        }
    }
}

__global__ void rand_mat_neighb_from_euler(float* afEulerIn, float* afMatOut, float* afRand, float fBound){
    /* generate random matrix according to the input EulerAngle
    * afEulerIn: iNEulerIn * 3, !!!!!!!!!! in radian  !!!!!!!!
    * afMatOut: iNNeighbour * iNEulerIn * 9
    * afRand:   iNNeighbour * iNEulerIn * 3
    * fBound: the range for random angle [-fBound,+fBound]
    * iNEulerIn: number of Input Euler angles
    * iNNeighbour: number of random angle generated for EACH input
    * call:: <<(iNNeighbour,1),(iNEulerIn,1,1)>>
    * TEST PASSED
    */
    //printf("%f||",fBound);
    // keep the original input
        float afEulerTmp[3];

        afEulerTmp[0] = afEulerIn[threadIdx.x * 3 + 0] + (2 * afRand[blockIdx.x * blockDim.x * 3 + threadIdx.x * 3 + 0] - 1) * fBound;
        afEulerTmp[2] = afEulerIn[threadIdx.x * 3 + 2] + (2 * afRand[blockIdx.x * blockDim.x * 3 + threadIdx.x * 3 + 2] - 1) * fBound;
        float z = cos(afEulerIn[threadIdx.x * 3 + 1]) +
                        (afRand[blockIdx.x * blockDim.x * 3 + threadIdx.x * 3 + 1] * 2 - 1) * sin(afEulerIn[threadIdx.x * 3 + 1] * fBound);
        if(z>1){
            z = 1;
        }
        else if(z<-1){
            z = -1;
        }
        afEulerTmp[1] = acosf(z);

        if(blockIdx.x>0){
            d_euler_zxz_to_mat(afEulerTmp, afMatOut + blockIdx.x * blockDim.x * 9 + threadIdx.x * 9);
        }
        else{
            // keep the original input
            d_euler_zxz_to_mat(afEulerIn + threadIdx.x * 3, afMatOut + blockIdx.x * blockDim.x * 9 + threadIdx.x * 9);
        }
}

__device__ void mat3_transpose(float* afOut, float* afIn){
    /*
    * transpose 3x3 matrix
    */
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            afOut[i * 3 + j] = afIn[j * 3 + i];
        }
    }
}
__device__ void mat3_dot(float* afResult, float* afM0, float* afM1){
    /*
    * dot product of two 3x3 matrix
    */
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            afResult[i * 3 + j] = 0;
            for(int k=0;k<3;k++){
                afResult[i * 3 + j] += afM0[i * 3 + k] * afM1[k * 3 + j];
            }
        }
    }
}

__global__ void misorien(float* afMisOrien, float* afM0, float* afM1, float* afSymM){
    /*
    * calculate the misorientation betwen afM0 and afM1
    * afMisOrien: iNM * iNSymM
    * afM0: iNM * 9
    * afM1: iNM * 9
    * afSymM: symmetry matrix, iNSymM * 9
    * NSymM: number of symmetry matrix
    * call method: <<<(iNM,1),(iNSymM,1,1)>>>
    */
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    float afTmp0[9];
    float afTmp1[9];
    float afM1Transpose[9];
    float fCosAngle;
    mat3_transpose(afM1Transpose, afM1 + blockIdx.x * 9);
    mat3_dot(afTmp0, afSymM + threadIdx.x * 9, afM1Transpose);
    mat3_dot(afTmp1, afM0 + blockIdx.x * 9, afTmp0);
    fCosAngle = 0.5 * (afTmp1[0] + afTmp1[4] + afTmp1[8] - 1);
    fCosAngle = min(0.9999999999, fCosAngle);
    fCosAngle = max(-0.99999999999, fCosAngle);
    afMisOrien[i] = acosf(fCosAngle);
}

__device__ void d_misorien(float& fMisOrien, float* afM0, float* afM1, float* afSymM){
        /*
    * calculate the misorientation betwen afM0 and afM1
    * fMisOrien: 1
    * afM0: 9
    * afM1: 9

    * call method:
    */
    float afTmp0[9];
    float afTmp1[9];
    float afM1Transpose[9];
    float fCosAngle;
    mat3_transpose(afM1Transpose, afM1);
    mat3_dot(afTmp0, afSymM , afM1Transpose);
    mat3_dot(afTmp1, afM0, afTmp0);
    fCosAngle = 0.5 * (afTmp1[0] + afTmp1[4] + afTmp1[8] - 1);
    fCosAngle = min(0.9999999999, fCosAngle);
    fCosAngle = max(-0.99999999999, fCosAngle);
    fMisOrien = acosf(fCosAngle);
}

