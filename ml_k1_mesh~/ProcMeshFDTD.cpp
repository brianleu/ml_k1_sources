/*
 *  MeshFDTD.cpp
 *  birch1_mesh~
 *
 *  Created by Randy Jones on 8/11/08.
 *  Copyright 2008 __MyCompanyName__. All rights reserved.
 *
 */
 
 

#include "ml_jit_utils.h"		// FIXME get Jitter stuff out of here
#include "ProcMeshFDTD.h"

MeshFDTD::MeshFDTD(int cols, int rows)
{
	t_jit_object * m;
	mColumns = cols;
	mRows = rows;
	mpExcite = 0;
	mpOutL = mpOutR = 0;
	setTension(0.5);
	mExciteScale = 1.;
	mDampScale = 1.;
	mInterpolate = true;
	mMeshes = 2;
	mExciteY = mExciteX = 0.5;
	
	// mesh matrices
	mMeshMatrixArray = new t_jit_object *[mMeshes];
	mMeshMatrixNameArray = new t_symbol *[mMeshes];
	mMeshMatrixDataArray = new float *[mMeshes];
	for(int i = 0; i < mMeshes; i++)
	{
		m = ujit_matrix_2dfloat_new(cols, rows); 
		mMeshMatrixNameArray[i] = jit_symbol_unique();
		mMeshMatrixArray[i] = (t_jit_object *)jit_object_method(m, _jit_sym_register, mMeshMatrixNameArray[i]);	
		mMeshMatrixDataArray[i] = ujit_matrix_get_data(mMeshMatrixArray[i]);
	}
	mMeshMatrixRowbytes = ujit_matrix_get_rowbytes(mMeshMatrixArray[0]);

	// matrix for saved sample of excitation history
	m = ujit_matrix_2dfloat_new(cols, rows); 
	mLastExciteName = jit_symbol_unique();
	mLastExciteMatrix = (t_jit_object *)jit_object_method(m, _jit_sym_register, mLastExciteName);	
	mLastExciteData = ujit_matrix_get_data(mLastExciteMatrix);

	// matrix for RMS sum
	m = ujit_matrix_2dfloat_new(cols, rows); 
	mSumName = jit_symbol_unique();
	mSumMatrix = (t_jit_object *)jit_object_method(m, _jit_sym_register, mSumName);	
	mSumData = ujit_matrix_get_data(mSumMatrix);
	
	// mTempVec = new float[MAX_VECSIZE];

	clear();
}


MeshFDTD::~MeshFDTD(void)
{	
	delete[] mMeshMatrixArray;
	delete[] mMeshMatrixNameArray;
	delete[] mMeshMatrixDataArray;
}


void MeshFDTD::setPtrs(float ** pExciteMatrices, float * pInSig, float * pInX, float * pInY, float * pMask, float * pOutL, float * pOutR, float * pRMS)
{
	mpExcite = pExciteMatrices;
	mMask = pMask;
	mRMS = pRMS;
	mInSig = pInSig;
	mInX = pInX;
	mInY = pInY;
	mpOutL = pOutL;
	mpOutR = pOutR;
}


void MeshFDTD::setTension(float t)
{
	float k, e, c;	// korner, edge, center
	float t2 = t*t;
	mTension = t;
	
	// equal energy criterion: 4k + 4e + c = 2.0.
	// the simulation is valid up to t^2 = 3/5, at which c=0
	// and the speed of wave travel is one mesh unit per time step.
	//
	// we still get reasonable sounds past this limit, by distributing
	// more of the energy to the corners, until the
	// stability breaks down at c < -0.5.
	
	if (mInterpolate)
	{	
		k = t2 * (1. / 6.) ;
		e = t2 * (2. / 3.) ;
		c = 2. - 4.*(k + e);
	}
	else
	{
		k = 0. ;
		e = t2 * 0.5 ;
		c = 2. - 4.*(k + e);
	}
	
	mKernel[0] = k;
	mKernel[1] = e;
	mKernel[2] = k;
	mKernel[3] = e;
	mKernel[4] = c;
	mKernel[5] = e;
	mKernel[6] = k;
	mKernel[7] = e;
	mKernel[8] = k;
}

/*
void MeshFDTD::setExciteScale(float t)
{
	mExciteScale = t;
//post ("excite: %f\n", t);
}

void MeshFDTD::setDampScale(float t)
{
	mDampScale = t;
//post ("damp: %f\n", t);

}
*/


void MeshFDTD::clear()
{
	for(int i = 0; i < mMeshes; i++)
	{
		setmem(mMeshMatrixDataArray[i], mRows * mMeshMatrixRowbytes, 0);
	}
	
	setmem(mLastExciteData, mRows * mMeshMatrixRowbytes, 0);
}


void MeshFDTD::process (int samples)
{
	int i, j;
	int width = mColumns; 
	int height = mRows;
	float * pIn, * pOut, * pExcite1, * pExcite2, * pRMS, * pSum;
	long lowBit;	
	int offsetL = width * (height >> 1) + (width >> 2);
	int offsetR = width * (height >> 1) + width - (width >> 2);
	
	int offsetExc;
	
	// assert((samples & 1) == 0);
	
	// clear sums matrix
	setmem(mSumData, mRows * mMeshMatrixRowbytes, 0); // dodgy, fix
	
	for (i=0; i<samples; i++)
	{		
		// get input signal
		// mTempVec[i] = mInSig[i];
		
		// ping-pong ins and outs (assumes even signal vector size)
		lowBit = i & 1;
		pIn = mMeshMatrixDataArray[lowBit];
		pOut = mMeshMatrixDataArray[!lowBit];	
			
		// excitation matrix pair for sample		
		pExcite1 = (i == 0) ? mLastExciteData : mpExcite[i-1];
		pExcite2 = mpExcite[i];
		
		// add direct signal to excitation matrix
		// FIXME need interpolate
		offsetExc = width * (int)(mInY[i]*height) + (int)(mInX[i] * width);	
		pExcite2[offsetExc] += mInSig[i];
		
		// the heart of the matter
		runMeshStepScalar(pIn, pOut, pExcite1, pExcite2, mMask, mKernel, mSumData);
		
		// read pickup samples into output vectors
		mpOutL[i] = pOut[offsetL]; 
		mpOutR[i] = pOut[offsetR]; 
	}
	
	// calculate RMS from sum	
	// run mesh center
	for(i = 1; i < height-1; i++)
	{
		pSum = (mSumData + i*width);	
		pRMS = (mRMS + i*width);	
		for(j = 1; j < width-1; j++)
		{
			pRMS[j] = sqrt(pSum[j] / samples);
		}
	}
	
	// save last excite matrix in history
	memcpy(mLastExciteData, mpExcite[samples-1], mRows * mMeshMatrixRowbytes);
}


// run one iteration of the mesh, generating one new matrix in pOut.
inline void MeshFDTD::runMeshStepScalar(float * pIn, float * pOut, float * pExcite1, float * pExcite2, float * pMask, float * pKernel, float * pSum)
{
	register int i, j;
	register int width = mColumns; 
	register int height = mRows;
	register float kk, ke, kc, out;
	register float * pr1, * pr2, * pr3; // input row ptrs
	register float * prMask; 
	register float * prOut; 	
	register float * pSumOut; 
	register float * prE1, *prE2; 
	register float ff, f, v, damp;
	const float scale = mExciteScale;
	const float dscale = mDampScale;
	
	kk = pKernel[0];
	ke = pKernel[1];
	kc = pKernel[4];
	
	// run mesh center
	for(i = 1; i < height-1; i++)
	{
		// row ptrs
		pr1 = (pIn + i*width - width);
		pr2 = (pIn + i*width);
		pr3 = (pIn + i*width + width);
		prMask = (pMask + i*width);
		prOut = (pOut + i*width);	
		pSumOut = (pSum + i*width);	
		prE1 = (pExcite1 + i*width);				
		prE2 = (pExcite2 + i*width);	
			
		for(j = 1; j < width-1; j++)
		{
			ff = prOut[j];	

			f = ke * (pr2[j-1] + pr1[j] + pr2[j+1] + pr3[j]);
			f += kk * (pr1[j-1] + pr1[j+1] + pr3[j-1] + pr3[j+1]);
			f += kc * pr2[j];
			
			v = (prE2[j] - prE1[j]) * scale;		// p = p + dE
			
			f += v;
			f -= ff;
			
			damp = 1. - (prE2[j] * dscale);
			if (damp > 1.) damp = 1;
			if (damp < 0.) damp = 0;
			out = f * damp * prMask[j];  // + denorm_noise[nk++];
			
			prOut[j] = out;
			pSumOut[j] += out * out;
		}
	}
}



