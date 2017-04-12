//-------------------------------------------------------------------------------
// 	ml_k1_centroids~ - generate centroids of intensity from a 2D float matrix.
// 
// 	author: Randy Jones randy@madronalabs.com
// 	created 21 Oct 2008
//	Copyright (c) 2004-2008 Randall Jones
//
//	Permission is hereby granted, free of charge, to any person obtaining a 
//  copy of this software and associated documentation files (the "Software"), 
//  to deal in the Software without restriction, including without limitation 
//  the rights to use, copy, modify, merge, publish, distribute, sublicense, 
//  and/or sell copies of the Software, and to permit persons to whom the 
//  Software is furnished to do so, subject to the following conditions:
//
//	The above copyright notice and this permission notice shall be included in
//	all copies or substantial portions of the Software.
//
//	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
//  DEALINGS IN THE SOFTWARE.


#include "jit.common.h"
//#include "ext.h"
#include "z_dsp.h"
#include <math.h>
#include <vector>
#include "ProcUpsampler.h"
#include "ml_k1_jit_utils.h"

#define MAX_CHANS 16
#define MAX_CENTROIDS 4		// fixed number of outputs

#define	MAX_POSSIBLE_CENTROIDS	64			// number of potential centroids gathered before sorting.
#define REALLY_LARGE_NUMBER 999999999
#define Z_BIAS	1.0f						// multiplier for z component in distance calc

// e_pixdata-- stores which adjacent pixels are less in intensity. 
// 
typedef unsigned char e_pixdata;
#define	PIX_RIGHT 	0x01
#define	PIX_UP		0x02
#define	PIX_LEFT	0x04
#define	PIX_DOWN	0x08
#define	PIX_ALL		0x0F
#define PIX_DONE	0x10

typedef struct _centroid_info
{
	float					fp_sum;
	float					fx;
	float					fy;
	float					x_sum;
	float					y_sum;
	float					matchDist;
	long					match;
	long					exists;
	long					matchesPrevious;
	long					index;
}	t_centroid_info;

t_centroid_info gZeroCentroid = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0, 0, 0, 0};


//--------------------------------------------------------------------------------

typedef struct _birch1centroids
{
	t_pxobject	xObj;
	void		*obex;		
	long		lock;	
	long		initialized;
	long		fftOffset;				
	long		in_dim[2];
	long		meshes;
	long		mframesPerVector;

	float					oneOverPixels;		
	float					oneOverWidth;		
	float					oneOverHeight;		
	float					threshold;				// only include pixels above this value
	float					mMatchDistance;			// maximum distance to match centroids
	
	long					match;					// match old centroids with new at each frame
	int						width;
	int						height;
	e_pixdata *				mInputMap;
	
	t_centroid_info	*		mpCurrentCentroids;
	t_centroid_info *		mpNewCentroids;
	t_centroid_info *		mpPreviousCentroids;
	
	long					mSubtractThreshold;
	long					mNewCentroids;			
	long					mCurrentCentroids;			
	int						mPreviousCentroids;
	long					mpInData;				// ptr to input matrix base addr

		
	float **				mInSigsLo;
	
	t_jit_object *			mpTempMatrix;
	t_float *				mpTempMatrixData;
							
	t_jit_object **	mExciteMatrixArray;
	t_float **		mExciteMatrixDataArray;
	t_symbol **		mExciteMatrixNameArray;
//	long			mExciteMatrixRowbytes;
	long					in_rowbytes;
	
	float **				mCentroidSigsLo;
	float **				mCentroidSigsHi;
	ProcUpsampler *			mUpsamplers;
	long					mInterpXY;
	
 	t_float		fs;
	t_float		oneOverFs;
	
	float mPreviousX[MAX_CENTROIDS + 1];
	float mPreviousY[MAX_CENTROIDS + 1];

}	t_birch1centroids;

void *birch1centroids_class;

// methods
void *birch1centroids_new(t_symbol *s, short ac, t_atom *av);
void birch1centroids_free(t_birch1centroids *x);

e_pixdata * new_map(int width, int height);
void free_map(e_pixdata *x, int width, int height);
void birch1centroids_do_resize(t_birch1centroids *x);

void mapImage(t_birch1centroids *x);
void gatherCentroids(t_birch1centroids *x);
void gather_centroid(t_birch1centroids *x, int i, int j, t_centroid_info * c);
void matchCentroids(t_birch1centroids *x);


void birch1centroids_assist(t_birch1centroids *x, void *b, long m, long a, char *s);
void birch1centroids_dsp(t_birch1centroids *x, t_signal **sp, short *count);
t_int *birch1centroids_perform(t_int *w);

inline void blurMatrix(float * pIn, float * pOut, long width, long height );

int main(void)
{
	long attrflags;
	void *classex, *attr;

	setup((t_messlist **)&birch1centroids_class, (method)birch1centroids_new, (method)birch1centroids_free, (short)sizeof(t_birch1centroids), 0L, A_GIMME, 0);
	addmess((method)birch1centroids_dsp, (char *)"dsp", A_CANT, 0);

	dsp_initclass();
	
	addmess((method)birch1centroids_assist, (char *)"assist", A_CANT, 0);

	jit_class_typedwrapper_get(NULL,NULL); // guarantee Jitter 1.5 or later

	// add attributes
	classex = max_jit_classex_setup(calcoffset(t_birch1centroids, obex));
	attrflags = JIT_ATTR_GET_DEFER_LOW | JIT_ATTR_SET_USURP_LOW;

	attr = jit_object_new(_jit_sym_jit_attr_offset, "threshold", _jit_sym_float32, attrflags,
		(method)0L, (method)0L, calcoffset(t_birch1centroids, threshold));	
	max_jit_classex_addattr(classex, attr);	
	
	attr = jit_object_new(_jit_sym_jit_attr_offset, "interp_xy", _jit_sym_long, attrflags,
		(method)0L, (method)0L, calcoffset(t_birch1centroids, mInterpXY));	
	max_jit_classex_addattr(classex, attr);	
	
	attr = jit_object_new(_jit_sym_jit_attr_offset, "match_distance", _jit_sym_float32, attrflags,
		(method)0L, (method)0L, calcoffset(t_birch1centroids, mMatchDistance));	
	max_jit_classex_addattr(classex, attr);	
	
	attr = jit_object_new(_jit_sym_jit_attr_offset, "match", _jit_sym_long, attrflags,
		(method)0L, (method)0L, calcoffset(t_birch1centroids, match));	
	max_jit_classex_addattr(classex, attr);	
	
	attr = jit_object_new(_jit_sym_jit_attr_offset, "subtract_threshold", _jit_sym_long, attrflags,
		(method)0L, (method)0L, calcoffset(t_birch1centroids, mSubtractThreshold));	
	max_jit_classex_addattr(classex, attr);	

	max_jit_classex_standard_wrap(classex, NULL, 0);
	
	return(0);
}

void *birch1centroids_new(t_symbol *s, short argc, t_atom *argv)
{
	t_birch1centroids *x;
	long i;
	const int m = MAX_POSSIBLE_CENTROIDS + 1;

	if (x = (t_birch1centroids *) max_jit_obex_new(birch1centroids_class, NULL))
	{
		x->in_dim[0] = x->in_dim[1] = 8; 
		x->fftOffset = 0;

		x->initialized = FALSE;

	// init vars
	x->threshold = 0.01f;
	x->mMatchDistance = 2.0f;
	x->mSubtractThreshold = true;
	x->match = 1;
	x->mInputMap = 0;
	x->mpInData = 0;
	x->in_rowbytes = 0;
	x->width = 0;
	x->height = 0;
	x->mPreviousCentroids = 0;
	x->mCurrentCentroids = 0;
	x->mNewCentroids = 0;
	x->mInterpXY = 1;	
	
	// allocate data for gatherAndReportCentroids
	x->mpCurrentCentroids = (t_centroid_info *)jit_getbytes(sizeof(t_centroid_info) * m);
	x->mpNewCentroids = (t_centroid_info *)jit_getbytes(sizeof(t_centroid_info) * m);
	x->mpPreviousCentroids = (t_centroid_info *)jit_getbytes(sizeof(t_centroid_info) * m);
	
	if ((!x->mpCurrentCentroids) || (!x->mpNewCentroids) || (!x->mpPreviousCentroids)) 
	{
		error((char *)"t_birch1centroids: out of memory!");
		x = 0;
		goto out;
	}
	
	for (i=0;i<m;i++)
	{
		x->mpCurrentCentroids[i] = gZeroCentroid;
		x->mpNewCentroids[i] = gZeroCentroid;
		x->mpPreviousCentroids[i] = gZeroCentroid;
	}
		
	for (i=0; i<MAX_CENTROIDS + 1; i++)	
	{
		x->mPreviousX[i] = 0.;
		x->mPreviousY[i] = 0.;
	}



		// does initial resize for dim args. 
		max_jit_attr_args(x,argc,argv);					
		
		// inputs: columns + input signal
		dsp_setup((t_pxobject *)x, x->in_dim[0]);		
		
		
		// 3 outlets for each centroid
		for(i=0; i<MAX_CENTROIDS*3; i++)
		{
			outlet_new(x, (char *)"signal");
		}
			
			
		birch1centroids_do_resize(x);
		
		x->initialized = TRUE;
		x->lock = 0;			
	}
	else
	{
		error((char*)"birch1_process~: could not allocate object");
		freeobject((t_object *)x);
	}
out:
	return ((void *)x);
}


void birch1centroids_free(t_birch1centroids *x)
{
	const int m = MAX_POSSIBLE_CENTROIDS + 1;
	x->lock = 1;
	
	jit_freebytes(x->mpNewCentroids, sizeof(t_centroid_info)*m);
	jit_freebytes(x->mpCurrentCentroids, sizeof(t_centroid_info)*m);
	jit_freebytes(x->mpPreviousCentroids, sizeof(t_centroid_info)*m);
	free_map(x->mInputMap, x->width, x->height);
	dsp_free((t_pxobject *)x);
	
	// FIX -- clean up!
	delete [] x->mUpsamplers;
	
	max_jit_obex_free(x);

}


void free_map(e_pixdata * pMap, int width, int height)
{
	if (pMap)
	{	
		jit_freebytes(pMap, width * height * sizeof(e_pixdata));
		pMap = 0;
	}
}


e_pixdata * new_map(int width, int height)
{
	// allocate image map
	e_pixdata * pNew = 0;
	if (!(pNew = (e_pixdata *)jit_getbytes(width*height*sizeof(e_pixdata))))
	{
		error((char *)"2up.jit.centroids: couldn't make image map!");
	}
	return pNew;
}




// make everything.
void birch1centroids_do_resize(t_birch1centroids *x)
{
	t_jit_object * m = 0;

	long inCols = x->in_dim[0];
	long inRows = x->in_dim[1];
	long in_signals = inCols * inRows;

	long framesPerVector = MAX_VECSIZE / FFT_SIZE;  // max framespervector.
//	int n;
	
post((char *)"setting dim: %d, %d, %d \n", inCols, inRows, framesPerVector);
	
	
	// input signals lo: (cols * rows * framesPerVector)
	x->mInSigsLo = new float *[in_signals];
	for(int i = 0; i < in_signals; i++)
	{
		x->mInSigsLo[i] = new float[framesPerVector];
	}
	
	// input signals lo: (cols * rows * framesPerVector)
	x->mCentroidSigsLo = new float *[in_signals];
	for(int i = 0; i < in_signals; i++)
	{
		x->mCentroidSigsLo[i] = new float[framesPerVector];
	}

	// input sigs hi: cols * rows * vecsize
	x->mCentroidSigsHi = new float *[in_signals];
	for(int i = 0; i < in_signals; i++)
	{
		x->mCentroidSigsHi[i] = new float[MAX_VECSIZE];
	}
	
	
	// temp matrix
	x->mpTempMatrix = ujit_matrix_2dfloat_new(inCols, inRows);
	x->mpTempMatrixData = ujit_matrix_get_data(x->mpTempMatrix);
	
	
	// matrices: one per fft in vector.
	x->mExciteMatrixArray = new t_jit_object *[framesPerVector];
	x->mExciteMatrixNameArray = new t_symbol *[framesPerVector];
	x->mExciteMatrixDataArray = new float *[framesPerVector];
	for(int i = 0; i < framesPerVector; i++)
	{
		m = ujit_matrix_2dfloat_new(inCols, inRows);
		x->mExciteMatrixNameArray[i] = jit_symbol_unique();
		x->mExciteMatrixArray[i] = (t_jit_object *)jit_object_method(m, _jit_sym_register, x->mExciteMatrixNameArray[i]);	
		x->mExciteMatrixDataArray[i] = ujit_matrix_get_data(x->mExciteMatrixArray[i]);
	}
	x->in_rowbytes = ujit_matrix_get_rowbytes(m);
	
	// biquad filter / upsampler processors
	x->mUpsamplers = new ProcUpsampler [in_signals];
	for (int i = 0; i < in_signals; i++)
	{
		x->mUpsamplers[i].setPtrs(x->mCentroidSigsLo[i], x->mCentroidSigsHi[i]);
	}
	
	if ((x->mInputMap = new_map(inCols, inRows)))
	{
		x->width = inCols; 
		x->height = inRows;
	}
		
	x->oneOverWidth = 1.f / (float)(inCols);
	x->oneOverHeight = 1.f / (float)(inRows);
	x->oneOverPixels = 1.f / (float)(inCols * inRows);	
	

}


void birch1centroids_assist(t_birch1centroids *x, void *b, long m, long a, char *s)
{

}


void mapImage(t_birch1centroids *x)
{
	int						i, j;
	float *					pf_data = 0;
	float *					pf_data_prev = 0;
	float *					pf_data_next = 0;
	e_pixdata *				p_map, * p_map_prev = 0, * p_map_next = 0;
	int						firstrow, lastrow, firstcol, lastcol;
	float					pixel;
	
	const int width = x->width;
	const int height = x->height;
	const int rowbytes = x->in_rowbytes;

	// clear map
	if (x->mInputMap)
	{
		memset(x->mInputMap, 0, width*height*sizeof(e_pixdata));
	}
	else
		return;

	// set direction ptrs for each pixel to point to neighboring pixels of less intensity.
	// PIX_DOWN means that pixel in down direction has less intensity, etc. 

	for (i=0; i < height; i++)
	{
		firstrow = (i==0);
		lastrow = (i==height-1);
		
		pf_data = (float *)(x->mpInData + i*rowbytes);
		pf_data_prev = (float *)(x->mpInData + (i-1)*rowbytes);
		pf_data_next = (float *)(x->mpInData + (i+1)*rowbytes);
		
		p_map = x->mInputMap + i*width;
		p_map_prev = x->mInputMap + (i-1)*width;
		p_map_next = x->mInputMap + (i+1)*width;

		for (j=0; j < width; j++)
		{		
			firstcol = (j==0);
			lastcol = (j==width-1);
			pixel = pf_data[j];

			// right
			if (!lastcol)
			{
				if (pf_data[j+1] > pixel)
					p_map[j+1] |= PIX_LEFT;
			}
			else
			{
				p_map[j] |= PIX_RIGHT;
			}
			// up
			if (!firstrow)
			{
				if (pf_data_prev[j] > pixel)
					p_map_prev[j] |= PIX_DOWN;
			}
			else
			{
				p_map[j] |= PIX_UP;
			}
			// left
			if (!firstcol)
			{
				if (pf_data[j-1] > pixel)
					p_map[j-1] |= PIX_RIGHT;
			}
			else
			{
				p_map[j] |= PIX_LEFT;
			}
			// down
			if (!lastrow)
			{
				if (pf_data_next[j] > pixel)
					p_map_next[j] |= PIX_UP;
			}
			else
			{
				p_map[j] |= PIX_DOWN;
			} 
		}
	}
}




void gatherCentroids(t_birch1centroids *x)
{
	int i, j;

	unsigned char * p_map;
	t_centroid_info			c;

	int n = 0;			
	
	// clear new
	for (i=1; i <= MAX_POSSIBLE_CENTROIDS; i++)
	{
		x->mpNewCentroids[i] = gZeroCentroid;
	}
	
	// gathering from peaks collects pixels into new centroids
	for (i=0; i< x->height; i++)
	{
		p_map = x->mInputMap + i*x->width;
		for (j=0; j< x->width; j++)
		{
			// if peak
			if (p_map[j] == PIX_ALL)
			{
				// zero temp
				c = gZeroCentroid;
				
				// gather
				gather_centroid(x, i, j, &c);
				
				// if big enough, calc x and y and add to list
				// first centroid index is 1. 
				if (c.fp_sum > 0.)
				{
					n++;	
					if (n > MAX_POSSIBLE_CENTROIDS) goto done;
					
					// calculate center of mass
					c.fx = (float)c.x_sum / c.fp_sum;
					c.fy = (float)c.y_sum / c.fp_sum;
									
					x->mpNewCentroids[n] = c;
					x->mpNewCentroids[n].exists = true;
				}
			}
		}
	}
	
done:

	// sort new centroids by intensity
	if (n > 1)
	{
		for (i=1; i <= n; i++)
		{
			for (j=i; j <= n; j++)
			{
				if (x->mpNewCentroids[j].fp_sum > x->mpNewCentroids[i].fp_sum)
				{
					c = x->mpNewCentroids[j];
					x->mpNewCentroids[j] = x->mpNewCentroids[i];
					x->mpNewCentroids[i] = c;
				}
			}
		}
	}
	
	x->mNewCentroids = n;
	return;
}


void gather_centroid(t_birch1centroids *x, int i, int j, t_centroid_info * c)
{
	register float h;				// pressure
	register float * pf_data;		// ptr to float data row
	unsigned char * p_map;
	int width = x->width;
	int firstrow, lastrow, firstcol, lastcol;
		
	p_map = x->mInputMap + i*width;
	
	// mark pixel as read
	p_map[j] |= PIX_DONE;

	pf_data = ((float *)(x->mpInData + i*x->in_rowbytes));
	
	h = pf_data[j];
	if (x->mSubtractThreshold)
	{
		// subtracting threshold from pressure prevents discontinuity at zero for controllers
		// but we don't want it in all applications
		h -= x->threshold;		
	}
	
	if (h <= 0.)
	{
		return;
	}
	
	// add pixel to centroid
	c->fp_sum += h;
	c->x_sum += (float)j*h;
	c->y_sum += (float)i*h;
	
	// recurse to any unmarked adjacent pixels of lesser intensity
	firstrow = (i==0);
	lastrow = (i==x->height-1);
	firstcol = (j==0);
	lastcol = (j==x->width-1);
	
	// right
	if (!lastcol)
		if (p_map[j] & PIX_RIGHT)
			if (!(p_map[j+1] & PIX_DONE))
				gather_centroid(x, i, j+1, c);
	// up
	if (!firstrow)
		if (p_map[j] & PIX_UP)
			if (!(p_map[j-width] & PIX_DONE))
				gather_centroid(x, i-1, j, c);
	// left
	if (!firstcol)
		if (p_map[j] & PIX_LEFT)
			if (!(p_map[j-1] & PIX_DONE))
				gather_centroid(x, i, j-1, c);
	// down
	if (!lastrow)
		if (p_map[j] & PIX_DOWN)
			if (!(p_map[j+width] & PIX_DONE))
				gather_centroid(x, i+1, j, c);

}


inline float distSquared(const t_centroid_info * a, const t_centroid_info * b);
inline float distSquared(const t_centroid_info * a, const t_centroid_info * b)
{
	float h, v, z;
	h = fabsf(a->fx - b->fx);
	v = fabsf(a->fy - b->fy);
	z = fabsf(a->fp_sum - b->fp_sum);
	return (h*h + v*v + Z_BIAS*z*z);
}


// swap new centroids to match order of current centroids close to them. 
void matchCentroids(t_birch1centroids *x)
{
	int i, j, m;
	float d, dMin;
	t_centroid_info * centroidList[MAX_POSSIBLE_CENTROIDS];
	int reserved[MAX_POSSIBLE_CENTROIDS];
	t_centroid_info	* t;		
	t_centroid_info * pCurrent, * pNew, * pClosest;
	const int newCentroids = MIN(MAX_CENTROIDS, x->mNewCentroids);
	const int current = x->mCurrentCentroids;

	x->mNewCentroids = newCentroids;
	
	// copy current to previous
	for (i=1; i <= MAX_CENTROIDS; i++)
	{
		x->mpPreviousCentroids[i] = x->mpCurrentCentroids[i];
	}
	x->mPreviousCentroids = x->mCurrentCentroids;

	if ((x->match) && (current > 0))
	{
		// clear matches from current
		for (i=1; i <= MAX_CENTROIDS; i++)
		{
			x->mpCurrentCentroids[i].match = 0;
		}
		
		// for each new centroid, get closest current centroid
		for (i=1; i <= newCentroids; i++)
		{
			pNew = &x->mpNewCentroids[i];
			pNew->index = i;
			pNew->match = i; // default match for no current
			dMin = REALLY_LARGE_NUMBER;
			for(j=1; j<= MAX_CENTROIDS; j++)
			{
				pCurrent = &x->mpCurrentCentroids[j];
				if(pCurrent->exists)
				{
					d = distSquared(pNew, pCurrent);
					if (d < dMin)
					{
						dMin = d;
						pNew->matchDist = d;
						pNew->match = j;
					}
				}
			}
			pClosest = &x->mpCurrentCentroids[pNew->match];
		}

		// sort centroids into a list by distance to closest current value
		for (i=1; i <= newCentroids; i++)
		{
			pNew = &x->mpNewCentroids[i];
			centroidList[i] = pNew;
		}
		if (newCentroids > 1)
		{
			for (i=1; i <= newCentroids; i++)
			{
				pNew = &x->mpNewCentroids[i];
				
				for (j=i; j <= newCentroids; j++)
				{
					if (centroidList[i]->matchDist > centroidList[j]->matchDist)
					{
						t = centroidList[j];
						centroidList[j] = centroidList[i];
						centroidList[i] = t;
					}
				}
			}
		}

		// mark existing as free
		for (i=1; i <= MAX_CENTROIDS; i++)
		{
			reserved[i] = false;
		}


		// in order of distance, try to put centroids where they match 
		for(i=1; i<=newCentroids; i++)
		{
			m = centroidList[i]->match;
			d = centroidList[i]->matchDist;
			if (!reserved[m]) // desired match is not reserved
			{			
				// centroid matches with previous.
				reserved[m] = true;
				x->mpCurrentCentroids[m].match = centroidList[i]->index;
			
				// mark continuous centroids.
				centroidList[i]->matchesPrevious = (x->mpCurrentCentroids[m].exists) && (d < x->mMatchDistance);
			}
			else	// not a continued centroid, just find room anywhere. 
			{
				centroidList[i]->matchesPrevious = false; 
				for(j=1; j<=MAX_CENTROIDS; j++)
				{
					if (!reserved[j])
					{
						reserved[j] = true;
						x->mpCurrentCentroids[j].match = centroidList[i]->index;

						centroidList[i]->match = j; // reassign
						break;
					}
				}
			}
		}		
	}
	else // no matching
	{
		for (i=1; i <= newCentroids; i++)
		{
			x->mpCurrentCentroids[i].match = i;
			// to avoid note offs every frame, we assume connected centroids when not doing matching 	
			x->mpNewCentroids[i].matchesPrevious = x->mpCurrentCentroids[i].exists;	
		}
	}
		
	// copy new centroids to current
	for (i=1; i <= MAX_CENTROIDS; i++)
	{
		m = x->mpCurrentCentroids[i].match;
		if (m)
		{
			x->mpCurrentCentroids[i] = x->mpNewCentroids[m];
		}
		else
		{
			x->mpCurrentCentroids[i] = gZeroCentroid;
		}
	}
	x->mCurrentCentroids = x->mNewCentroids;
}	
	


void birch1centroids_dsp(t_birch1centroids *x, t_signal **sp, short *count)
{
	int i;
	int columns = x->in_dim[0];
	int outs = MAX_CENTROIDS * 3;
	int ins = columns;
	long n_signals = (ins + outs);
	long n_args = n_signals + 2;
	long size = n_args * sizeof(long);
	long * vecArray;

	x->fs = sp[0]->s_sr;
	x->oneOverFs = 1.0/x->fs;
	x->mframesPerVector =  sp[0]->s_n / FFT_SIZE;
			
//post ((char *)"blocks per vector: %d\n", x->mframesPerVector);	
	vecArray = (long *)t_getbytes(size);
	vecArray[0] = (long)x;
	vecArray[1] = sp[0]->s_n;		
		
	for (i = 0; i < n_signals; i++) 
	{
		vecArray[2 + i] = (long)(sp[i]->s_vec);
	}

	dsp_addv(birch1centroids_perform, n_args, (void **)vecArray);
	t_freebytes(vecArray, size);
	
//	birch1centroids_clear(x);
}



// pressure values are copied from input signals to input matrix. 
// every time an input matrix is filled, it gets processed to output matrix. 
// calibrated values are copied from output matrix to output.
t_int *birch1centroids_perform(t_int *w)
{
	t_birch1centroids *x = (t_birch1centroids *)(w[1]);
	int vecsize = w[2];

	float *p_ins[MAX_CHANS];
	float *p_outs[MAX_CENTROIDS * 3];
	
	int i, j, k, n, frame_start;
	int framesPerVector = x->mframesPerVector;
	int cols = x->in_dim[0];
	int rows = x->in_dim[1];
	
	int ins = cols;	// column signals 
	int outs = MAX_CENTROIDS * 3;
	long dspSignals = (ins + outs);
	long dspArgs = dspSignals + 2;
	float v;
	
	if (x->lock || x->xObj.z_disabled)
		goto bail;
		
	// need a fifo in case FFT_SIZE < vecsize. FIXME
		
	// set up signal vector pointers. input and output pointers may be identical!
	for (i = 0; i < ins; i++)
	{
		p_ins[i] = (t_float *)(w[3 + i]);
	}
	for (i = 0; i < outs; i++)
	{
		p_outs[i] = (t_float *)(w[3 + ins + i]);
	}
	
	// loop for each FFT_SIZE block.  
	// assumes that fft frames start on signal vector.

	// 1: copy input signals to mInSigsLo.
	for(k = 0; k < framesPerVector; k++)
	{
		frame_start = k*FFT_SIZE;		
		for (i = 0; i < cols; i++)
		{
			for(j = 0; j < rows; j++)
			{
				n = i*rows + j;
				x->mInSigsLo[n][k] = (p_ins[i])[frame_start + j];
			}
		}
	}
			
	// 2: copy mInSigsLo to matrices. // rotate 90deg.  
	{
		float * pOutRow;

		for (i = 0; i < framesPerVector; i++)
		{
			for (j = 0; j < rows; j++)
			{
				pOutRow = x->mExciteMatrixDataArray[i] + j*rows;
				
				for (k = 0; k < cols; k++)
				{ 
					v = x->mInSigsLo[k * rows + j][i];			
					pOutRow[k] = v;	
				}
			}	
		}	
	}
	
	// 2.5: blur matrices!
	for (i = 0; i < framesPerVector; i++)
	{
		// copy matrix to temp
		memcpy(x->mpTempMatrixData, x->mExciteMatrixDataArray[i], sizeof(float)*rows*cols);
		blurMatrix(x->mpTempMatrixData, x->mExciteMatrixDataArray[i], cols, rows);
		
	}	
	
	
	// 3: get centroids for each matrix.  write to centroidSigsLo.
	{	
		int c, cm, cc;
		float fx, fy, fp;
		
		for (i = 0; i < framesPerVector; i++)
		{
			x->mpInData = (long)x->mExciteMatrixDataArray[i];
			mapImage(x);
			gatherCentroids(x);
			matchCentroids(x);	
			
			// copy centroids to centroidSigsLo
			for (c = 1; c <= MAX_CENTROIDS + 1; c++)
			{
				if (x->mpCurrentCentroids[c].exists)
				{
					fp = (x->mpCurrentCentroids[c]).fp_sum * x->oneOverPixels;
					fx = ((x->mpCurrentCentroids[c]).fx + 0.5f) * x->oneOverWidth;
					fy = ((x->mpCurrentCentroids[c]).fy + 0.5f) * x->oneOverHeight;
					x->mPreviousX[c] = fx;
					x->mPreviousY[c] = fy;
					
				}
				else
				{
					fp = 0.;
					fx = x->mPreviousX[c];//((x->mpCurrentCentroids[c]).fx + 0.5f) * x->oneOverWidth;
					fy = x->mPreviousY[c];//((x->mpCurrentCentroids[c]).fy + 0.5f) * x->oneOverHeight;
				}
				
				cm = c-1;
				cc = cm*3;
				x->mCentroidSigsLo[cc][i] = fx;
				x->mCentroidSigsLo[cc+1][i] = fy;
				x->mCentroidSigsLo[cc+2][i] = fp;
			
			}
		}
	}
			
	// 4: upsample centroidSigsLo to centroidSigsHi.
	for (i = 0; i < MAX_CENTROIDS*3; i+=3)
	{
		if (x->mInterpXY)
		{
			x->mUpsamplers[i].setOrder(5);		
			x->mUpsamplers[i+1].setOrder(5);
		}
		else
		{
			x->mUpsamplers[i].setOrder(0);		
			x->mUpsamplers[i+1].setOrder(0);
		}
		
		x->mUpsamplers[i+2].setOrder(5);	// always interpolation for z
		
		x->mUpsamplers[i].process(vecsize);
		x->mUpsamplers[i+1].process(vecsize);
		x->mUpsamplers[i+2].process(vecsize);
	}
	
	// 5: copy centroidSigsHi to outs.
	for (i = 0; i < MAX_CENTROIDS*3; i++)
	{
		for(j=0; j<vecsize; j++)
		{
			p_outs[i][j] =  x->mCentroidSigsHi[i][j];
		}
	}		
	
	// TEST copy one upsampled sig to output	
//	for(i=0; i<vecsize; i++)
//	{
//		p_outs[1][i] = x->mInSigsHi[((inRows * inCols) >> 1) + ((inCols) >> 1)][i];
//	}
				
bail:
	return (w + dspArgs + 1);
}

void gaussianKernel()
{
	// someday
}



inline void blurMatrix(float * pIn, float * pOut, long width, long height )
{
	register int i, j;
	register float f, kk, ke, kc;
	register float * pr1, * pr2, * pr3; // input row ptrs
	register float * prOut; 	
	
	kk = 0.25 * 0.25;
	ke = 0.5 * 0.25;
	kc = 1 * 0.25;
	
//	kk = 0.5 * 0.2;
//	ke = 0.5 * 0.2;
//	kc = 0.5 * 0.2;
	
	i = 0;	// top row
	{
		// row ptrs
		pr2 = (pIn + i*width);
		pr3 = (pIn + i*width + width);
		prOut = (pOut + i*width);
		
		j = 0; // left corner
		{
			f = ke * (pr2[j+1] + pr3[j]);
			f += kk * (pr3[j+1]);
			f += kc * pr2[j];
			prOut[j] = f;		
		}
			
		for(j = 1; j < width - 1; j++) // top side
		{
			f = ke * (pr2[j-1] + pr2[j+1] + pr3[j]);
			f += kk * (pr3[j-1] + pr3[j+1]);
			f += kc * pr2[j];
			prOut[j] = f;
		}
		
		j = width - 1; // right corner
		{
			f = ke * (pr2[j-1] + pr3[j]);
			f += kk * (pr3[j-1]);
			f += kc * pr2[j];
			prOut[j] = f;		
		}
	}
	for(i = 1; i < height - 1; i++) // center rows
	{
		// row ptrs
		pr1 = (pIn + i*width - width);
		pr2 = (pIn + i*width);
		pr3 = (pIn + i*width + width);
		prOut = (pOut + i*width);
		
		j = 0; // left side
		{
			f = ke * (pr1[j] + pr2[j+1] + pr3[j]);
			f += kk * (pr1[j+1] + pr3[j+1]);
			f += kc * pr2[j];
			prOut[j] = f;		
		}
			
		for(j = 1; j < width - 1; j++) // center
		{
			f = ke * (pr2[j-1] + pr1[j] + pr2[j+1] + pr3[j]);
			f += kk * (pr1[j-1] + pr1[j+1] + pr3[j-1] + pr3[j+1]);
			f += kc * pr2[j];
			prOut[j] = f;
		}
		
		j = width - 1; // right side
		{
			f = ke * (pr2[j-1] + pr1[j] + pr3[j]);
			f += kk * (pr1[j-1] + pr3[j-1]);
			f += kc * pr2[j];
			prOut[j] = f;		
		}
	}
	i = height;	// bottom row
	{
		// row ptrs
		pr1 = (pIn + i*width - width);
		pr2 = (pIn + i*width);
		prOut = (pOut + i*width);
		
		j = 0; // left corner
		{
			f = ke * (pr1[j] + pr2[j+1]);
			f += kk * (pr1[j+1]);
			f += kc * pr2[j];
			prOut[j] = f;		
		}
			
		for(j = 1; j < width - 1; j++) // bottom side
		{
			f = ke * (pr2[j-1] + pr1[j] + pr2[j+1]);
			f += kk * (pr1[j-1] + pr1[j+1]);
			f += kc * pr2[j];
			prOut[j] = f;
		}
		
		j = width - 1; // right corner
		{
			f = ke * (pr2[j-1] + pr1[j]);
			f += kk * (pr1[j-1]);
			f += kc * pr2[j];
			prOut[j] = f;		
		}
	}
}


