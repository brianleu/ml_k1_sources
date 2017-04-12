/*
 *  ProcUpsampler.cpp
 *
 *  Created by Randy Jones on 8/11/08.
 *  Copyright 2008 Madrona Labs LLC. All rights reserved.
 *
 */

#include "ProcUpsampler.h"


//--------------------------------------------------------------------------------
// upsampler.

ProcUpsampler::ProcUpsampler()
{
	mOrder = 0;	// default
	mpIn = mpOut = 0;
	mx0 = mx1 = mx2 = mx3 = mx4 = 0.;
}

ProcUpsampler::~ProcUpsampler(void)
{	
}

void ProcUpsampler::setPtrs(float * pIn, float * pOut)
{
	mpIn = pIn;
	mpOut = pOut;
}

// process n samples of output from n/FFT_SIZE samples of input.
// use b-spline interpolation.
/*
void ProcUpsampler::process(int n)
{
	float * pIn = mpIn;
	float * pOut = mpOut;
	float x0, x1, x2, x3, f;
	float c0, c1, c2, c3;	// polynomial coefficients
	float x02;
	const float pf = (1.0 / FFT_SIZE);
	const float onesixth = 1./6.;
	const float twothirds = 2./3.;
	const float onehalf = 1./2.;
	
	// 4 point, 3rd order b-spline
	for(int i = 0; i <= n - FFT_SIZE; i += FFT_SIZE)
	{
		x0 = mx0;
		x1 = mx1;
		x2 = mx2;
		x3 = pIn[i >> FFT_SIZE_BITS];	

		x02 = x0 + x2;
		c0 = onesixth*x02 + twothirds*x1;
		c1 = onehalf*(x2 - x0);
		c2 = onehalf*x02 - x1;
		c3 = onehalf*(x1 - x2) + onesixth*(x3 - x0);
			
		for (int j = 0; j < FFT_SIZE; j++)
		{
			f = pf*j;
			pOut[i+j] = ((c3*f + c2)*f + c1)*f + c0; 
		}
		mx0 = x1;
		mx1 = x2;
		mx2 = x3;
	}
}
*/


void ProcUpsampler::process(int n)
{
	float * pIn = mpIn;
	float * pOut = mpOut;
	float x0, x1, x2, x3, x4, x5, f;
	float c0, c1, c2, c3, c4, c5;	// polynomial coefficients
	float xx1, xx2, xx3, xx4, xx5;
	float a;
	const float pf = (1.0 / FFT_SIZE);
	
	switch(mOrder)
	{
		case (0):
		
			for(int i = 0; i <= n - FFT_SIZE; i += FFT_SIZE)
			{
				x5 = pIn[i >> FFT_SIZE_BITS];	
				for (int j = 0; j < FFT_SIZE; j++)
				{
					pOut[i+j] = x5;
				}
				mx4 = x5;
			}
			
		break;
		case (3):
			// 4 point, 3rd order b-spline
			for(int i = 0; i <= n - FFT_SIZE; i += FFT_SIZE)
			{
				x0 = mx0;
				x1 = mx1;
				x2 = mx2;
				x3 = pIn[i >> FFT_SIZE_BITS];	

				xx2 = x0 + x2;
				c0 = 1/6.0*xx2 + 2/3.0*x1;
				c1 = 1/2.0*(x2 - x0);
				c2 = 1/2.0*xx2 - x1;
				c3 = 1/2.0*(x1 - x2) + 1/6.0*(x3 - x0);
					
				for (int j = 0; j < FFT_SIZE; j++)
				{
					f = pf*j;
					pOut[i+j] = ((c3*f + c2)*f + c1)*f + c0; 
				}
				mx0 = x1;
				mx1 = x2;
				mx2 = x3;
			}
		break;
		
		default:

			// 6 point, 5th order b-spline
			for(int i = 0; i <= n - FFT_SIZE; i += FFT_SIZE)
			{
				x0 = mx0;
				x1 = mx1;
				x2 = mx2;
				x3 = mx3;
				x4 = mx4;
				x5 = pIn[i >> FFT_SIZE_BITS];	// WTF FIXME

				xx1 = x0+x4;
				xx2 = x1+x3; 
				xx3 = x4-x0;
				xx4 = x3-x1; 
				xx5 = 1/6.0*xx2; 
				
				 c0 = 1/120.0*xx1 + 13/60.0*xx2 + 11/20.0*x2; 
				 c1 = 1/24.0*xx3 + 5/12.0*xx4; 
				 c2 = 1/12.0*xx1 + xx5 - 1/2.0*x2; 
				 c3 = 1/12.0*xx3 - 1/6.0*xx4; 
				 c4 = 1/24.0*xx1 - xx5 + 1/4.0*x2; 
				 c5 = 1/120.0*(x5 - x0) + 1/24.0*(x1 - x4) + 1/12.0*(x3 - x2); 

				for (int j = 0; j < FFT_SIZE; j++)
				{
					f = pf*j;
					a = ((((c5*f + c4)*f + c3)*f + c2)*f + c1)*f + c0; 
					pOut[i+j] = a;		
				}
				
				mx0 = x1;
				mx1 = x2;
				mx2 = x3;
				mx3 = x4;
				mx4 = x5;
			}
		break;
	}
}

