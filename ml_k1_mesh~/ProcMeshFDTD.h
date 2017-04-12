/*
 *  MeshFDTD.h
 *  birch1_mesh~
 *
 *  Created by Randy Jones on 8/11/08.
 *  Copyright 2008 __MyCompanyName__. All rights reserved.
 *
 */

#include "jit.common.h"
#include "z_dsp.h"
#include <math.h>
#include <vector>

//--------------------------------------------------------------------------------

class MeshFDTD
{
public:
	MeshFDTD(int cols, int rows); 
    ~MeshFDTD();
	void setPtrs(float ** pExciteMatrices, float * pInSig, float * pInX, float * pInY, float * pMask, float * pOutL, float * pOutR, float * pRMS);
	void setTension(float);
//	void setExciteScale(float t); 
//	void setDampScale(float t); 
	void clear();
    void process (int samples);
	
	float mExciteScale;
	float mDampScale;
	float mExciteX, mExciteY;		// position of signal exciter on [0., 1.]

private:
	inline void runMeshStepScalar(float * pIn, float * pOut, float * pExcite1, float * pExcite2, float * pMask, float * pKernel, float * pRMS);

	long mColumns, mRows;
	long mRowBytes;
	float ** mpExcite, * mpOutL, * mpOutR;
	float * mRMS;
	float mTension;

	float * mInSig;
	float * mInX;
	float * mInY;
	
	
	float * mMask;
//	float * mTempVec;
	float mKernel[9];
	long mInterpolate;
	long mMeshes;
	
	// later, not jitter stuff.
	t_jit_object **	mMeshMatrixArray;
	t_float **		mMeshMatrixDataArray;
	t_symbol **		mMeshMatrixNameArray;
	long			mMeshMatrixRowbytes;

	t_jit_object *	mLastExciteMatrix;
	t_float *		mLastExciteData;
	t_symbol *		mLastExciteName;

	t_jit_object *	mSumMatrix;
	t_float *		mSumData;
	t_symbol *		mSumName;

};


