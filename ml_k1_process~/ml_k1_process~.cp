//
// ml_k1_process~.c
// preprocessing for k1 8x8 controller.
// who	when 		what
// rej	24 July 08	created
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

#include <iostream>
#include <fstream>
using namespace std;


#define MAX_CHANS 16
#define FFT_SIZE 32
#define BQ_SIZE 4
#define CAL_SAMPLES 256
#define CAL_SKIP 10		
#define CAL_PARAMS 4	// calibration params: mean, stdEDev, max.

typedef enum eCalState
{
	eUncalibrated = 0,
	eGatherNoise = 1,
	eGatherNoiseDone = 2,
	eGatherSignal = 3,
	eGatherSignalDone = 4,
	eCalibrated = 5
};

// meaning of matrices in mCalParamsDataArray[]
typedef enum eCalParams
{
	eParamsMin = 0,
	eParamsDynMin = 1,
	eParamsStdDev = 2,
	eParamsMax = 3,
};


typedef struct _ml_k1proc
{
	t_pxobject		xObj;
	void			*obex;		
	long			lock;	
	
	// calibration stuff (should be separate object)
	long			mCalibrate;		// run calibration, or not
	eCalState		mCalState;
	long			mCalIndex;
	long			mCalSkipCtr;
	t_jit_object **	mCalMatrixArray;
	t_float **		mCalMatrixDataArray;
	t_symbol **		mCalMatrixNameArray;
	t_jit_object **	mCalParamsArray;
	t_float **		mCalParamsDataArray;
	t_symbol **		mCalParamsNameArray;
	
	long			initialized;
	long			fftOffset;				
	long			dim[2];
	t_jit_object *	matrixOutlet;
	
	t_jit_object *	mInMatrix;
	t_float *		mInMatrixData;
	t_symbol *		mInMatrixName;
	t_jit_object *	mOutMatrix;
	t_float *		mOutMatrixData;
	t_symbol *		mOutMatrixName;
	long			mMatrixRowbytes;
	
	// biquad history
	t_jit_object **	mBiquadMatrixArray;
	t_float **		mBiquadMatrixDataArray;
	t_symbol **		mBiquadMatrixNameArray;
	
	// biquad coeffs
 	t_float			fs;
	t_float			oneOverFs;
	float			mF0;	// cutoff
	float			mQ;
	double			mW0;
	double			mAlpha;
	double			mCosW0;
	float			mB0;
    float			mB1;
	float			mB2;
    float			mA1;
    float			mA2;

// 	t_float		hpfThresh;
// 	t_float		hpfFreq;
	float			threshold;  // std devs. to clip

}	t_ml_k1proc;

void *ml_k1proc_class;

// methods
void *ml_k1proc_new(t_symbol *s, short ac, t_atom *av);
void ml_k1proc_free(t_ml_k1proc *x);
void ml_k1proc_set_dim(t_ml_k1proc *x, void *attr, long argc, t_atom *argv);
void ml_k1proc_do_resize(t_ml_k1proc *x);
void ml_k1proc_clear(t_ml_k1proc *x);
void ml_k1proc_matrix_out(t_ml_k1proc *x);

void ml_k1proc_calib_getmin(t_ml_k1proc *x);
void ml_k1proc_calib_getstddev(t_ml_k1proc *x);
void ml_k1proc_calib_getmax(t_ml_k1proc *x);
void ml_k1proc_calib_start(t_ml_k1proc *x);
void ml_k1proc_calib_end(t_ml_k1proc *x);
void ml_k1proc_calib_read(t_ml_k1proc *x);
void ml_k1proc_calib_write(t_ml_k1proc *x);
void ml_k1proc_set_calib_cutoff(t_ml_k1proc *x, void *attr, long argc, t_atom *argv);

void ml_k1proc_assist(t_ml_k1proc *x, void *b, long m, long a, char *s);
void ml_k1proc_dsp(t_ml_k1proc *x, t_signal **sp, short *count);
void ml_k1proc_biquad_coeffs(t_ml_k1proc *x);
t_int *ml_k1proc_perform(t_int *w);

// utilities
t_jit_object * ujit_matrix_2dfloat_new(long width, long height);
t_float * ujit_matrix_get_data(t_jit_object * m);
unsigned long ujit_matrix_get_rowbytes(t_jit_object * m);

int main(void)
{
	long attrflags;
	void *classex, *attr;

	setup((t_messlist **)&ml_k1proc_class, (method)ml_k1proc_new, (method)ml_k1proc_free, (short)sizeof(t_ml_k1proc), 0L, A_GIMME, 0);
	addmess((method)ml_k1proc_dsp, (char *)"dsp", A_CANT, 0);

	dsp_initclass();
	
	addmess((method)ml_k1proc_assist, (char *)"assist", A_CANT, 0);

	jit_class_typedwrapper_get(NULL,NULL); // guarantee Jitter 1.5 or later

	// add attributes
	classex = max_jit_classex_setup(calcoffset(t_ml_k1proc, obex));
	attrflags = JIT_ATTR_GET_DEFER_LOW | JIT_ATTR_SET_USURP_LOW;
	// dim
	attr = jit_object_new(_jit_sym_jit_attr_offset_array, "dim", _jit_sym_long, 2, attrflags,
		(method)0L,(method)ml_k1proc_set_dim, 0, calcoffset(t_ml_k1proc, dim));
	max_jit_classex_addattr(classex, attr);
	// row offset
	attr = jit_object_new(_jit_sym_jit_attr_offset, "offset", _jit_sym_long, attrflags,
		(method)0L,(method)0L, calcoffset(t_ml_k1proc, fftOffset));
	max_jit_classex_addattr(classex, attr);
	
	// HPF threshold
	attr = jit_object_new(_jit_sym_jit_attr_offset, "threshold", _jit_sym_float32, attrflags,
		(method)0L,(method)0L, calcoffset(t_ml_k1proc, threshold));
	max_jit_classex_addattr(classex, attr);
	
	// dynamic calibration cutoff freq.
	attr = jit_object_new(_jit_sym_jit_attr_offset, "calib_cutoff", _jit_sym_float32, attrflags,
		(method)0L,(method)ml_k1proc_set_calib_cutoff, calcoffset(t_ml_k1proc, mF0));
	max_jit_classex_addattr(classex, attr);
	// calibration switch
	attr = jit_object_new(_jit_sym_jit_attr_offset, "calibrate", _jit_sym_long, attrflags,
		(method)0L,(method)0L, calcoffset(t_ml_k1proc, mCalibrate));
	max_jit_classex_addattr(classex, attr);
	
	max_jit_classex_standard_wrap(classex, NULL, 0);
	
	// add other messages
	addmess((method)ml_k1proc_clear, (char *)"clear", 0);
	addmess((method)ml_k1proc_calib_start, (char *)"calib_start", 0);
	addmess((method)ml_k1proc_calib_end, (char *)"calib_end", 0);
	
	addmess((method)ml_k1proc_calib_read, (char *)"calib_read", 0);
	addmess((method)ml_k1proc_calib_write, (char *)"calib_write", 0);

	addbang((method)ml_k1proc_matrix_out);
	addmess((method)ml_k1proc_calib_getmin, (char *)"calib_getmin", 0);
	addmess((method)ml_k1proc_calib_getstddev, (char *)"calib_getstddev", 0);
	addmess((method)ml_k1proc_calib_getmax, (char *)"calib_getmax", 0);
	
	return(0);
}

void *ml_k1proc_new(t_symbol *s, short argc, t_atom *argv)
{
	int i;
	t_ml_k1proc *x;
	if (x = (t_ml_k1proc *) max_jit_obex_new(ml_k1proc_class, NULL))
	{
		x->dim[0] = x->dim[1] = 8; 
		x->fftOffset = 0;
		x->mInMatrix = x->mOutMatrix = 0;
		x->initialized = FALSE;
		x->mCalState = eUncalibrated;
		x->mCalibrate = TRUE;
		x->mCalIndex = 0;
		x->threshold = 10.;
		
		x->lock = 1;
		x->mF0 = 1.;		// default: 1 Hz
		x->mQ = 0.77;		// flattish passband
		
	//	x->x_obj.z_misc = Z_NO_INPLACE;		does not work.

		max_jit_attr_args(x,argc,argv);					// does resize for dim arg. 
		
		dsp_setup((t_pxobject *)x, x->dim[0] + 1);		// inputs: columns + FFT index
		
		// outlets
		x->matrixOutlet = (t_jit_object *)outlet_new(x,0L); 	
		for (i = 0; i < x->dim[0] + 1; i++)
		{
			outlet_new((t_pxobject *)x, (char *)"signal");
		}
		
		ml_k1proc_do_resize(x);
		ml_k1proc_clear(x);
		
		x->initialized = TRUE;
		x->lock = 0;		// allow DSP		
	}
	else
	{
		error((char*)"ml_k1_process~: could not allocate object");
		freeobject((t_object *)x);
	}

	return ((void *)x);
}


void ml_k1proc_free(t_ml_k1proc *x)
{
	x->lock = 1;
	dsp_free((t_pxobject *)x);
	jit_object_free(x->mInMatrix);
	jit_object_free(x->mOutMatrix);
	max_jit_obex_free(x);
}


// set new dimensions.  the needs_resize flag is handled in the perform method. 
void ml_k1proc_set_dim(t_ml_k1proc *x, void *attr, long argc, t_atom *argv)
{
	if (x->initialized)
	{
		post((char *)"ml_k1proc: remake object to set new size.");
	}
	else
	{
		if (argc&&argv) 
		{
			x->dim[0] = MAX(jit_atom_getlong(&argv[0]), 4);	
			x->dim[1] = MAX(jit_atom_getlong(&argv[1]), 4);	
		}
	}
}



void ml_k1proc_do_resize(t_ml_k1proc *x)
{
	t_jit_object * m = 0;
	long bytes;
	
	post((char *)"setting dim: %d, %d \n", x->dim[0], x->dim[1]);
	
	// new matrix
	m = ujit_matrix_2dfloat_new(x->dim[1], x->dim[0]); // sic
	x->mInMatrixName = jit_symbol_unique();
	m = (t_jit_object *)jit_object_method(m, _jit_sym_register, x->mInMatrixName);
	
	if (m)
	{
		x->mInMatrixData = ujit_matrix_get_data(m);
		x->mMatrixRowbytes = ujit_matrix_get_rowbytes(m);
		bytes = x->dim[0] * x->mMatrixRowbytes;
		setmem(x->mInMatrixData, bytes, 0);
		x->mInMatrix = m;
	}
	
	m = ujit_matrix_2dfloat_new(x->dim[1], x->dim[0]);
	x->mOutMatrixName = jit_symbol_unique();
	m = (t_jit_object *)jit_object_method(m, _jit_sym_register, x->mOutMatrixName);
	
	if (m)
	{
		x->mOutMatrixData = ujit_matrix_get_data(m);
		bytes = x->dim[0] * x->mMatrixRowbytes;
		setmem(x->mOutMatrixData, bytes, 0);
		x->mOutMatrix = m;
	}
	
	// obviously we need a MatrixArray object.
		
	x->mBiquadMatrixArray = new t_jit_object *[BQ_SIZE];
	x->mBiquadMatrixNameArray = new t_symbol *[BQ_SIZE];
	x->mBiquadMatrixDataArray = new float *[BQ_SIZE];
	for(int i = 0; i < BQ_SIZE; i++)
	{
		m = ujit_matrix_2dfloat_new(x->dim[1], x->dim[0]);
		x->mBiquadMatrixNameArray[i] = jit_symbol_unique();
		x->mBiquadMatrixArray[i] = (t_jit_object *)jit_object_method(m, _jit_sym_register, x->mBiquadMatrixNameArray[i]);	
		x->mBiquadMatrixDataArray[i] = ujit_matrix_get_data(x->mBiquadMatrixArray[i]);
	}

	x->mCalMatrixArray = new t_jit_object *[CAL_SAMPLES];
	x->mCalMatrixNameArray = new t_symbol *[CAL_SAMPLES];
	x->mCalMatrixDataArray = new float *[CAL_SAMPLES];
	for(int i = 0; i < CAL_SAMPLES; i++)
	{
		m = ujit_matrix_2dfloat_new(x->dim[1], x->dim[0]);
		x->mCalMatrixNameArray[i] = jit_symbol_unique();
		x->mCalMatrixArray[i] = (t_jit_object *)jit_object_method(m, _jit_sym_register, x->mCalMatrixNameArray[i]);	
		x->mCalMatrixDataArray[i] = ujit_matrix_get_data(x->mCalMatrixArray[i]);
	}
	
	x->mCalParamsArray = new t_jit_object *[CAL_PARAMS];
	x->mCalParamsNameArray = new t_symbol *[CAL_PARAMS];
	x->mCalParamsDataArray = new float *[CAL_PARAMS];
	for(int i = 0; i < CAL_PARAMS; i++)
	{
		m = ujit_matrix_2dfloat_new(x->dim[1], x->dim[0]);
		x->mCalParamsNameArray[i] = jit_symbol_unique();
		x->mCalParamsArray[i] = (t_jit_object *)jit_object_method(m, _jit_sym_register, x->mCalParamsNameArray[i]);	
		x->mCalParamsDataArray[i] = ujit_matrix_get_data(x->mCalParamsArray[i]);
	}
}


void ml_k1proc_clear(t_ml_k1proc *x)
{
	const long bytes = x->dim[0] * x->mMatrixRowbytes;
	
	x->lock = 1;	// need critical region
	setmem(x->mInMatrixData, bytes, 0);
	setmem(x->mOutMatrixData, bytes, 0);
	
	for(int i = 0; i < BQ_SIZE; i++)
	{
		setmem(x->mBiquadMatrixDataArray[i], bytes, 0);
	}
	
	x->lock = 0;
}


// send out both raw and calibrated matrices
void ml_k1proc_matrix_out(t_ml_k1proc *x)
{
	Atom av[2];
	if (x->lock) return;	
	
	if (x->mCalibrate)
	{
		jit_atom_setsym(av, _jit_sym_jit_matrix);
		jit_atom_setsym(av+1, x->mOutMatrixName);
		outlet_anything(x->matrixOutlet, gensym((char *)"cal"), 2, av);	
	}
	else
	{
		jit_atom_setsym(av, _jit_sym_jit_matrix);
		jit_atom_setsym(av+1, x->mInMatrixName);
		outlet_anything(x->matrixOutlet, gensym((char *)"raw"), 2, av);	
	}
}



void ml_k1proc_calib_getmin(t_ml_k1proc *x)
{
	Atom av[2];
	jit_atom_setsym(av, _jit_sym_jit_matrix);
	jit_atom_setsym(av+1, x->mCalParamsNameArray[eParamsMin]);
	outlet_anything(x->matrixOutlet, gensym((char *)"min"), 2, av);	
}

void ml_k1proc_calib_getstddev(t_ml_k1proc *x)
{
	Atom av[2];
	jit_atom_setsym(av, _jit_sym_jit_matrix);
	jit_atom_setsym(av+1, x->mCalParamsNameArray[eParamsStdDev]);
	outlet_anything(x->matrixOutlet, gensym((char *)"stddev"), 2, av);	
}

void ml_k1proc_calib_getmax(t_ml_k1proc *x)
{
	Atom av[2];
	jit_atom_setsym(av, _jit_sym_jit_matrix);
	jit_atom_setsym(av+1, x->mCalParamsNameArray[eParamsMax]);
	outlet_anything(x->matrixOutlet, gensym((char *)"max"), 2, av);	
}





void ml_k1proc_calib_start(t_ml_k1proc *x)
{
	post ((char *)"starting calibration:\n");
	x->mCalState = eGatherNoise;
	x->mCalSkipCtr = 0;
	x->mCalIndex = 0;
}

void ml_k1proc_calib_end(t_ml_k1proc *x)
{
	x->mCalState = eGatherSignalDone;
	x->mCalSkipCtr = 0;
	x->mCalIndex = 0;
}


void pathname_slash_to_unix(char * in, char * out);
void pathname_slash_to_unix(char * in, char * out)
{
	int i, j, h;
	const char * pre = "/Volumes/";
	
	// copy preamble
	for(i = 0; pre[i]; i++)
	{
		out[i] = pre[i];
	}
	// copy rest
	h = 0;
	for(j = 0; in[j]; j++)
	{
//post ("char %d : %c (%d)\n", j, in[j], in[j]);
		if (in[j] == ':')
		{
			h--;
		}
		else
		{
			out[i+j+h] = in[j];
		}		
	}
	// terminate
	out[i+j+h] = 0;
}

static char * fileShort = (char *)"ml_k1_calib.txt";


// very primitive read function
void ml_k1proc_calib_read(t_ml_k1proc *x)
{
	char fileLong[512];
	char file2[512];
	char inBuf[512];
	short ret, ret2, pathID;
	ifstream in;
	int i = 0, j = 0;
	float a;
	float & rA = a;
	long columns = x->dim[0];
	long rows = x->dim[1];
	long cells = columns * rows;
	float *pMin, *pDynMin, *pStdDev, * pMax;
	pathID = path_getdefault();
	ret = path_topotentialname(pathID, fileShort, fileLong, FALSE);
	ret2 = path_nameconform(fileLong, file2, PATH_STYLE_SLASH, PATH_TYPE_ABSOLUTE);
	
	ret = locatefiletype(fileShort, &pathID, 0L, 0L); 
//post((char *)"locatefiletype returned %d\n", ret);
	path_topathname(pathID, fileShort, fileLong);
//post ((char *)"reading calibration from: %s\n", fileLong);
	
//post((char *)"path_topotentialname returned %d\n", ret);
//post((char *)"conform to: %s\n", fileLong);

	pathname_slash_to_unix(fileLong, file2);
	post((char *)"ml_k1_process: reading calibration from: %s\n", file2);

	in.open(file2, ifstream::in);
	if (!in.good())
	{
		error((char *)"ml_k1_process: couldn't open file %s\n", fileShort);
	}
	else
	{
//		c = in.peek();
//		post ("got %c ", c);
		while(!isdigit(in.peek())) in.getline(inBuf, 512);
		for ( i=0; i < rows; i++)
		{
			pMin = (t_float *)(x->mCalParamsDataArray[eParamsMin] + i*rows);
			pDynMin = (t_float *)(x->mCalParamsDataArray[eParamsDynMin] + i*rows);
			for ( j=0; j < columns; j++)
			{
				in >> rA;
				pMin[j] = a;
				pDynMin[j] = a;
			}	
			if (!in.good()) break;
		}
		while(!isdigit(in.peek())) in.getline(inBuf, 512);
		for ( i=0; i < rows; i++)
		{
			pStdDev = (t_float *)(x->mCalParamsDataArray[eParamsStdDev] + i*rows);
			for ( j=0; j < columns; j++)
			{
				in >> rA;
				pStdDev[j] = a;
			}
			if (!in.good()) break;
		}
		
		while(!isdigit(in.peek())) in.getline(inBuf, 512);
		for ( i=0; i < rows; i++)
		{
			pMax = (t_float *)(x->mCalParamsDataArray[eParamsMax] + i*rows);
			for ( j=0; j < columns; j++)
			{
				in >> rA;
				pMax[j] = a;
			}
			if (!in.good()) break;
		}
		
		if ((i == rows) && (j == columns))
		{
			post((char *)"ml_k1_process: read calibration OK: %d cells.\n", cells);
			x->mCalState = eCalibrated;
		}
		else
		{
			error((char *)"ml_k1_process: bad calibration file %s\n", fileShort);
		}
	}
	in.close();
}


// very primitive write function
void ml_k1proc_calib_write(t_ml_k1proc *x)
{
	char fileLong[512];
	char file2[512];
	float a;
	float & rA = a;
	short ret, ret2, pathID;
	long columns = x->dim[0];
	long rows = x->dim[1];
	float *pMin, *pStdDev, * pMax;
	long cells = columns * rows;
	ofstream f;
	
	pathID = path_getdefault();
	ret = path_topotentialname(pathID, fileShort, fileLong, FALSE);
	ret2 = path_nameconform(fileLong, file2, PATH_STYLE_SLASH, PATH_TYPE_ABSOLUTE);
	
//post((char *)"path_topotentialname returned %d\n", ret);
//post((char *)"conform to: %s\n", fileLong);

	pathname_slash_to_unix(fileLong, file2);
	post((char *)"ml_k1_process: writing calibration to: %s\n", file2);

	f.open(file2);
	f << "// " << "ml_k1 calibration file.\n";
	f.setf(ios::fixed,ios::floatfield); 
  	f.precision(5);
	f << "// " << cells << " means:\n";
	for (int i=0; i < rows; i++)
	{
		pMin = (t_float *)(x->mCalParamsDataArray[eParamsMin] + i*rows);
		for (int j=0; j < columns; j++)
		{
			a = pMin[j];
			f << rA;
			f << " ";
		}
		f << "\n";
	}
	f << "// " << cells << " standard deviations:\n";
 	f.precision(7);
	for (int i=0; i < rows; i++)
	{
		pStdDev = (t_float *)(x->mCalParamsDataArray[eParamsStdDev] + i*rows);		
		for (int j=0; j < columns; j++)
		{
			a = pStdDev[j];
			f << rA;
			f << " ";		
		}
		f << "\n";
	}
	f << "// " << cells << " maxima:\n";
 	f.precision(5);
	for (int i=0; i < rows; i++)
	{
		pMax = (t_float *)(x->mCalParamsDataArray[eParamsMax] + i*rows);
		for (int j=0; j < columns; j++)
		{
			a = pMax[j];
			f << rA;
			f << " ";
				}
		f << "\n";
	}
	
	f.close();
}

void ml_k1proc_set_calib_cutoff(t_ml_k1proc *x, void *attr, long argc, t_atom *argv)
{
	if (argc&&argv) 
	{
		x->mF0 = jit_atom_getfloat(&argv[0]);
		ml_k1proc_biquad_coeffs(x);
//post ("cutoff %f\n", x->mF0);
	}
}




void ml_k1proc_assist(t_ml_k1proc *x, void *b, long m, long a, char *s)
{

}


void ml_k1proc_dsp(t_ml_k1proc *x, t_signal **sp, short *count)
{
	int i;
	int c = x->dim[0];
	long n_signals = (c + 1 + c + 1);
	long n_args = n_signals + 2;
	long size = n_args * sizeof(long);
	long * vecArray;

	x->fs = sp[0]->s_sr;
	x->oneOverFs = 1.0/x->fs;
	
	vecArray = (long *)t_getbytes(size);
	vecArray[0] = (long)x;
	vecArray[1] = sp[0]->s_n;		
		
	for (i = 0; i < n_signals; i++) 
	{
		vecArray[2 + i] = (long)(sp[i]->s_vec);
	}

	dsp_addv(ml_k1proc_perform, n_args, (void **)vecArray);
	t_freebytes(vecArray, size);
	
	// precalculate biquad values that are constant over frequency.
	
	ml_k1proc_biquad_coeffs(x);
	ml_k1proc_clear(x);
}


void ml_k1proc_biquad_coeffs(t_ml_k1proc *x)
{
	double oneOverA0;
	double controlRate = (double)x->fs / (double)FFT_SIZE;

	x->mW0 = (double)TWOPI * ((double)x->mF0 / controlRate);
	x->mAlpha = sin(x->mW0)/(2.*x->mQ);
	x->mCosW0 = cos(x->mW0); 
	
//post ("f0 coeff =  %f\n", x->mCosW0);
	
    oneOverA0 	= 1. / ( 1. + x->mAlpha);
    
	// HPF
	//x->mB0 	= oneOverA0 * ((1. + x->mCosW0) * 0.5);
    //x->mB1 	= oneOverA0 * -(1. + x->mCosW0);
	//x->mB2 	= x->mB0;
    //x->mA1 	= oneOverA0 * (-2. * x->mCosW0);
    //x->mA2 	= oneOverA0 * (1. - x->mAlpha);
	
	// LPF
	x->mB0 	= oneOverA0 * ((1. + x->mCosW0) * 0.5);
    x->mB1 	= oneOverA0 * (1. + x->mCosW0);
	x->mB2 	= x->mB0;
    x->mA1 	= oneOverA0 * (-2. * x->mCosW0);
    x->mA2 	= oneOverA0 * (1. - x->mAlpha);
}

inline void _process_matrix(t_ml_k1proc *x);

// pressure values are copied from input signals to input matrix. 
// every time an input matrix is filled, it gets processed to output matrix. 
// calibrated values are copied from output matrix to output.
t_int *ml_k1proc_perform(t_int *w)
{
	t_ml_k1proc *x = (t_ml_k1proc *)(w[1]);
	float *p_ins[MAX_CHANS];
	float *p_inmatrix[MAX_CHANS];
	float *p_outmatrix[MAX_CHANS];
	float *p_outs[MAX_CHANS];
	
	int vecsize = w[2];
	int i, j, frame_start;
	int offset = x->fftOffset;
	int columns = x->dim[0];
	int chans = columns + 1;
	int rows = x->dim[1];
	long n_signals = (chans * 2);
	long n_args = n_signals + 2;
	float f;
	
	if (x->lock || x->xObj.z_disabled)
		goto bail;
		
	// set up signal vector pointers. input and output pointers may be identical!
	for (i = 0; i < chans; i++)
	{
		p_ins[i] = (t_float *)(w[3 + i]);
		p_outs[i] = (t_float *)(w[3 + chans + i]);
	}
	
	// set up matrix row pointers. data is flipped x<->y in the matrix for speed.
	// each signal column is a matrix row.
	for (i = 0; i < columns; i++)
	{
		p_inmatrix[i] = (t_float *)(x->mInMatrixData + i*rows);
		p_outmatrix[i] = (t_float *)(x->mOutMatrixData + i*rows);
	}

	// assumes that fft frames start on signal vector.
	for(frame_start = 0; frame_start <= vecsize - FFT_SIZE; frame_start += FFT_SIZE)
	{
		// copy input signals to input matrix.
		for (i = 0; i < columns; i++)
		{
			for(j = 0; j < rows; j++)
			{
				(p_inmatrix[i])[j] = (p_ins[i])[frame_start + offset + j];
			}
		}
		
		_process_matrix(x);
				
		// copy output matrix to output signals.
		for (i = 0; i < columns; i++)
		{
			for(j = 0; j < rows; j++)
			{
				f = (p_outmatrix[i])[j];
				(p_outs[i])[frame_start + j] = f; //(p_outmatrix[i])[j];
			}
			for(j = rows; j < FFT_SIZE; j++)
			{
				(p_outs[i])[frame_start + j] = 0.; // zero out unused data
			}
		}		

		// make row index signal
		for(j = 0; j < rows; j++)
		{
			(p_outs[columns])[frame_start + j] = j;
		}
		for(j = rows; j < FFT_SIZE; j++)
		{
			(p_outs[columns])[frame_start + j] = -1.;
		}
			
	}
	
bail:
	return (w + n_args + 1);
}



inline void _process_matrix(t_ml_k1proc *x)
{
	int i, j, k;
	const int columns = x->dim[0];
	const int rows = x->dim[1];
	float * pIn, * pOut, * pXMinus1, * pXMinus2, * pYMinus1, * pYMinus2;
	float * pCal, * pMin, * pDynMin, * pStdDev, * pMax;
	float f, fm, fm2, min, dynMin, max, sum, dev, dev2Sum, thresh;
	/*
	const float b0= x->mB0;
	const float b1= x->mB1;
	const float b2= x->mB2;
	const float a1= x->mA1;
	const float a2= x->mA2;
	const long bytes = x->mMatrixRowbytes * rows;
	*/
	

	float in, out;
	long doPassthru = 0;
	
	// if calibrating, gather calibration .  Store calculated params in mCalMatrixDataArray.
	// mCalMatrixDataArray[0] = mean of rest (no pressure)
	// mCalMatrixDataArray[1] = stdDev of rest
	// mCalMatrixDataArray[2] = max pressure
	
	switch(x->mCalState) // calibrate data or gather samples. 
	{
		case(eGatherNoise):	// gather noise samples
		{
			doPassthru = 1;
			x->mCalSkipCtr++;
			if (x->mCalSkipCtr >= CAL_SKIP)
			{
				x->mCalSkipCtr = 0;
			}
			else break;
			
			if (x->mCalIndex == 0)
			{
				post ((char *)"gathering noise...\n");
			}
	//		pIn = (t_float *)(x->mInMatrixData);
	//		pCal = (t_float *)(x->mCalMatrixDataArray[x->mCalIndex]);
	//		memcpy(pCal, pIn, bytes);
				
			
			for (i = 0; i < columns; i++)
			{
				pIn = (t_float *)(x->mInMatrixData + i*rows);
				pCal = (t_float *)(x->mCalMatrixDataArray[x->mCalIndex] + i*rows);
				for(j = 0; j < rows; j++)
				{
					pCal[j] = pIn[j];
				}
			}
		
			x->mCalIndex++;
			if (x->mCalIndex >= CAL_SAMPLES)
			{
				x->mCalState = eGatherNoiseDone;
			}
		}
		break;
		
		case(eGatherNoiseDone): // reduce noise data
		{
			// compute mean and std. dist. of noise for each cell.
			doPassthru = 1;
			for (i = 0; i < columns; i++)
			{
				pMin = (t_float *)(x->mCalParamsDataArray[eParamsMin] + i*rows);
				pDynMin = (t_float *)(x->mCalParamsDataArray[eParamsDynMin] + i*rows);
				pStdDev = (t_float *)(x->mCalParamsDataArray[eParamsStdDev] + i*rows);
				for(j = 0; j < rows; j++)
				{
					sum = 0.;
					dev2Sum = 0;
					for (k=0; k<CAL_SAMPLES; k++)
					{
						pCal = (t_float *)(x->mCalMatrixDataArray[k] + i*rows);
						sum += pCal[j];
					}
					pMin[j] = pDynMin[j] = sum / CAL_SAMPLES;
					for (k=0; k<CAL_SAMPLES; k++)
					{
						pCal = (t_float *)(x->mCalMatrixDataArray[k] + i*rows);
						dev = pCal[j] - pMin[j];
						dev2Sum += dev * dev;
					}
					pStdDev[j] = sqrt(dev2Sum / CAL_SAMPLES);
				}
			}
			post ((char *)"gathering noise done.\n");
			x->mCalIndex = -1;
			x->mCalState = eGatherSignal;
		}
		break;
		
		case(eGatherSignal): // for now, write maximum for each cell directly to params
		{			
			doPassthru = 1;
			if (x->mCalIndex == -1) // first time here?
			{
				post ((char *)"gathering signal: (calib_end to stop)\n");
				// clear maxima
				const long bytes = x->dim[0] * x->mMatrixRowbytes;
				setmem(x->mCalParamsDataArray[eParamsMax], bytes, 0);
			}			
			else for (i = 0; i < columns; i++)
			{
				pIn = (t_float *)(x->mInMatrixData + x->mCalIndex*rows);
				pMax = (t_float *)(x->mCalParamsDataArray[eParamsMax] + x->mCalIndex*rows);

				for(j = 0; j < rows; j++)
				{
					f = pIn[j];
					max = pMax[j];
					if (f > max)
					{
						pMax[j] = f;
					}
				}
			}
			x->mCalIndex++;
			if (x->mCalIndex >= columns) // wrap
			{
				x->mCalIndex = 0;
			}			
		}
		break;
		
		case(eGatherSignalDone): 
		{
			doPassthru = 1;
			// nothing to do
			post ((char *)"gathering signal done.\n");
			x->mCalState = eCalibrated;
		}
		break;
		
		case(eUncalibrated):
		{
			doPassthru = 1;
		}
		break;
		
		case(eCalibrated):
		default: 
		{
			if (!x->mCalibrate)
			{
				doPassthru = 1;
			}
			else for (i = 0; i < columns; i++)
			{
				pIn = (t_float *)(x->mInMatrixData + i*rows);
				pOut = (t_float *)(x->mOutMatrixData + i*rows);
				pXMinus1 = (t_float *)(x->mBiquadMatrixDataArray[0] + i*rows);
				pXMinus2 = (t_float *)(x->mBiquadMatrixDataArray[1] + i*rows);
				pYMinus1 = (t_float *)(x->mBiquadMatrixDataArray[2] + i*rows);
				pYMinus2 = (t_float *)(x->mBiquadMatrixDataArray[3] + i*rows);
				pMin = (t_float *)(x->mCalParamsDataArray[eParamsMin] + i*rows);
				pDynMin = (t_float *)(x->mCalParamsDataArray[eParamsDynMin] + i*rows);
				pStdDev = (t_float *)(x->mCalParamsDataArray[eParamsStdDev] + i*rows);
				pMax = (t_float *)(x->mCalParamsDataArray[eParamsMax] + i*rows);
				
				for(j = 0; j < rows; j++)
				{
					min = pMin[j];
					dynMin = pDynMin[j];
					max = pMax[j];
					dev = pStdDev[j];
					in = pIn[j];
					
					// something new: thresh is percentage of max.
					thresh = dynMin + ((max - min) * x->threshold);
					
					// filter frequency multiplier: [1, 0] on [min, max].
					fm = ((in - min) / (max - min));
					fm *= 2.;
					fm = 1. - fm;
					if (fm < 0.) fm = 0;
					if (fm > 1.) fm = 1.;
					
					// multiply filter frequency. 
					// cos is near 1. at low frequencies, so this is a good approximation
					// of multiplying the frequency.
					fm2 = 1. - ((1. - x->mCosW0) * fm * fm);
					
					// filter dynamically calibrated mins			
					// using one-zero filter
					pDynMin[j]  = (in * (1. - fm2)) + dynMin * fm2;	

					// if below threshold, assume not touching. 
					// adjust mean value, return 0
					if (in < thresh)
					{
						// return 0
						out = 0.;
					}
					else // else assume touching, return calibrated value
					{
						// can store divide
						out = (in - thresh) / (max - thresh);
					}	
									
					pOut[j] = out;
					
				}
			}
		}
		break;
	
	}	// end (switch)
	if (doPassthru)
	{
		for (i = 0; i < columns; i++)
		{
			pIn = (t_float *)(x->mInMatrixData + i*rows);
			pOut = (t_float *)(x->mOutMatrixData + i*rows);
			for(j = 0; j < rows; j++)
			{
				pOut[j] = pIn[j]; 
			}
		}
	}
}




//--------------------------------------------------------------------------------


t_jit_object * ujit_matrix_2dfloat_new(long width, long height)
{
	t_jit_matrix_info info;
	t_jit_object * newMatrix = 0;

	jit_matrix_info_default(&info);
	info.type = _jit_sym_float32;
	info.dimcount = 2;
	info.planecount = 1;
	info.dim[0] = width;		
	info.dim[1] = height;
	
	newMatrix = (t_jit_object *)jit_object_new(gensym((char *)"jit_matrix"), &info);

	if (!newMatrix)
	{
		error((char *)"ml_k1_process: couldn't allocate matrix!"); 
	}
	return newMatrix;
}


t_float * ujit_matrix_get_data(t_jit_object * m)
{
	t_float * pData;
	jit_object_method(m, _jit_sym_getdata, &pData);
	return(pData);
}


unsigned long ujit_matrix_get_rowbytes(t_jit_object * m)
{
	t_jit_matrix_info info;
	jit_object_method(m, _jit_sym_getinfo, &info);
	return (info.dimstride[1]);
}



