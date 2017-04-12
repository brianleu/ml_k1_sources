/*
 *  ProcUpsampler.h
 *  ml_k1_mesh~
 *
 *  Created by Randy Jones on 8/11/08.
 *  Copyright 2008 Madrona Labs LLC. All rights reserved.
 *
 */
 
// TEMP FIXME
#define FFT_SIZE 32
#define FFT_SIZE_BITS 5

class ProcUpsampler
{
public:
	ProcUpsampler(); 
    ~ProcUpsampler();
	void setPtrs(float * pIn, float * pOut); 
    void process (int samples);
	
	inline void setOrder(const int n) {mOrder = n;};

private:
	int mOrder;
	float *mpIn, *mpOut;
    float mx0, mx1, mx2, mx3, mx4;
	
};
