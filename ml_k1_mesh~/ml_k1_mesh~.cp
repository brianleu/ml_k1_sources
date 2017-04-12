//
// ml_k1_mesh~.c
// waveguide mesh synthesis for k1 8x8 controller.
// who	when 		what
// rej	24 July 08	created


#include "jit.common.h"
//#include "ext.h"
#include "z_dsp.h"
#include <math.h>
#include <vector>
#include "ProcMeshFDTD.h"
#include "ProcMeshWGM.h"
#include "ProcUpsampler.h"

#define MAX_CHANS 16



#include "ml_jit_utils.h"


//--------------------------------------------------------------------------------

typedef struct _birch1mesh
{
	t_pxobject	xObj;
	void		*obex;		
	long		lock;	
	long		initialized;
	long		fftOffset;				
	long		in_dim[2];
	long		mesh_dim[2];
	float		excite_pos[2];
	long		meshes;
	long		mframesPerVector;
	float		excite_scale;
	float		damp_scale;		// damping from applied force
	float		mDamp;			// all-over damping every step
	
	t_jit_object *	matrixOutlet;
	
	float **		mInSigsLo;
//	float **		mDiffSigsLo;
	float **		mInSigsHi;
	ProcUpsampler *	mUpsamplers;
//	ProcSmoother *	mSmoothers;
	
	t_jit_object *	mMaskMatrix;
	t_float *		mMaskMatrixData;
	t_symbol *		mMaskMatrixName;
	long			mMaskMatrixRowbytes;
	
	t_jit_object *	mMaskDampMatrix;
	t_float *		mMaskDampMatrixData;
	t_symbol *		mMaskDampMatrixName;
	long			mMaskDampMatrixRowbytes;
	
	t_jit_object *	mRMSMatrix;
	t_float *		mRMSMatrixData;
	t_symbol *		mRMSMatrixName;
	long			mRMSMatrixRowbytes;
	
	t_jit_object **	mExciteMatrixArray;
	t_float **		mExciteMatrixDataArray;
	t_symbol **		mExciteMatrixNameArray;
	long			mExciteMatrixRowbytes;
	
	MeshFDTD *		mMesh;
	float			tension;
	float			v_thresh;

	t_jit_object *	mOutMatrix;
	t_float *		mOutMatrixData;
	t_symbol *		mOutMatrixName;

 	t_float		fs;
	t_float		oneOverFs;

}	t_birch1mesh;

void *birch1mesh_class;

// methods
void *birch1mesh_new(t_symbol *s, short ac, t_atom *av);
void birch1mesh_free(t_birch1mesh *x);
void birch1mesh_set_in_dim(t_birch1mesh *x, void *attr, long argc, t_atom *argv);
void birch1mesh_set_mesh_dim(t_birch1mesh *x, void *attr, long argc, t_atom *argv);
t_jit_err birch1mesh_set_excite_scale(t_birch1mesh *x, void *attr, long argc, t_atom *argv);
t_jit_err birch1mesh_set_damp_scale(t_birch1mesh *x, void *attr, long argc, t_atom *argv);
t_jit_err birch1mesh_set_damp(t_birch1mesh *x, void *attr, long argc, t_atom *argv);
t_jit_err birch1mesh_set_excite_pos(t_birch1mesh *x, void *attr, long argc, t_atom *argv);
t_jit_err birch1mesh_set_tension(t_birch1mesh *x, void *attr, long argc, t_atom *argv);
void birch1mesh_mask(t_birch1mesh *x, t_symbol *s, int argc, t_atom *argv);
void birch1mesh_add_mask_border(t_birch1mesh *x);

void birch1mesh_do_resize(t_birch1mesh *x);
void birch1mesh_clear(t_birch1mesh *x);
void birch1mesh_matrix_out(t_birch1mesh *x);
void birch1mesh_assist(t_birch1mesh *x, void *b, long m, long a, char *s);
void birch1mesh_dsp(t_birch1mesh *x, t_signal **sp, short *count);
t_int *birch1mesh_perform(t_int *w);

int main(void)
{
	long attrflags;
	void *classex, *attr;

	setup((t_messlist **)&birch1mesh_class, (method)birch1mesh_new, (method)birch1mesh_free, (short)sizeof(t_birch1mesh), 0L, A_GIMME, 0);
	addmess((method)birch1mesh_dsp, (char *)"dsp", A_CANT, 0);

	dsp_initclass();
	
	addmess((method)birch1mesh_assist, (char *)"assist", A_CANT, 0);

	jit_class_typedwrapper_get(NULL,NULL); // guarantee Jitter 1.5 or later

	// add attributes
	classex = max_jit_classex_setup(calcoffset(t_birch1mesh, obex));
	attrflags = JIT_ATTR_GET_DEFER_LOW | JIT_ATTR_SET_USURP_LOW;
	// in_dim
	attr = jit_object_new(_jit_sym_jit_attr_offset_array, "in_dim", _jit_sym_long, 2, attrflags,
		(method)0L,(method)birch1mesh_set_in_dim, 0, calcoffset(t_birch1mesh, in_dim));
	max_jit_classex_addattr(classex, attr);
	// mesh_dim
	attr = jit_object_new(_jit_sym_jit_attr_offset_array, "mesh_dim", _jit_sym_long, 2, attrflags,
		(method)0L,(method)birch1mesh_set_mesh_dim, 0, calcoffset(t_birch1mesh, mesh_dim));
	max_jit_classex_addattr(classex, attr);
	// direct signal excite position
	attr = jit_object_new(_jit_sym_jit_attr_offset_array, "excite_pos", _jit_sym_float32, 2, attrflags,
		(method)0L,(method)birch1mesh_set_excite_pos, 0, calcoffset(t_birch1mesh, excite_pos));
	max_jit_classex_addattr(classex, attr);
	// tension
	attr = jit_object_new(_jit_sym_jit_attr_offset, "tension", _jit_sym_float32, attrflags,
	(method)0L, (method)birch1mesh_set_tension, calcoffset(t_birch1mesh, tension));	
	max_jit_classex_addattr(classex, attr);	
	// v_thresh
	attr = jit_object_new(_jit_sym_jit_attr_offset, "v_thresh", _jit_sym_float32, attrflags,
	(method)0L, (method)0L, calcoffset(t_birch1mesh, v_thresh));	
	max_jit_classex_addattr(classex, attr);	
	// excite scale
	attr = jit_object_new(_jit_sym_jit_attr_offset, "excite_scale", _jit_sym_float32, attrflags,
	(method)0L, (method)birch1mesh_set_excite_scale, calcoffset(t_birch1mesh, excite_scale));	
	max_jit_classex_addattr(classex, attr);	
	// damp scale
	attr = jit_object_new(_jit_sym_jit_attr_offset, "damp_scale", _jit_sym_float32, attrflags,
	(method)0L, (method)birch1mesh_set_damp_scale, calcoffset(t_birch1mesh, damp_scale));	
	max_jit_classex_addattr(classex, attr);	
	// damp scale
	attr = jit_object_new(_jit_sym_jit_attr_offset, "damp", _jit_sym_float32, attrflags,
	(method)0L, (method)birch1mesh_set_damp, calcoffset(t_birch1mesh, mDamp));	
	max_jit_classex_addattr(classex, attr);	

	max_jit_classex_standard_wrap(classex, NULL, 0);
	
	// add other messages
	addmess((method)birch1mesh_clear, (char *)"clear", 0);
	max_addmethod_usurp_low((method)birch1mesh_mask, (char *)"mask");
	addbang((method)birch1mesh_matrix_out);
	
	return(0);
}

void *birch1mesh_new(t_symbol *s, short argc, t_atom *argv)
{
	t_birch1mesh *x;
	if (x = (t_birch1mesh *) max_jit_obex_new(birch1mesh_class, NULL))
	{
		x->in_dim[0] = x->in_dim[1] = 8; 
		x->mesh_dim[0] = x->mesh_dim[1] = 8; 
		x->fftOffset = 0;
		x->mMaskMatrix = x->mOutMatrix = 0;
		x->meshes = 2;
		x->initialized = FALSE;
		x->tension = 0.5;
		x->excite_scale = 1.0;
		x->damp_scale = 1.0;
		x->v_thresh = 0.;
		x->lock = 1;
		x->mDamp = 1.;
		x->excite_pos[0] = x->excite_pos[1] = 0.5;

		// does initial resize for dim args. 
		max_jit_attr_args(x,argc,argv);					
		
		// inputs: columns + input signal, x, y
		dsp_setup((t_pxobject *)x, x->in_dim[0] + 3);		
		
		// matrix outlet
		x->matrixOutlet = (t_jit_object *)outlet_new(x,0L);
		 	
		// sound outlets
		outlet_new((t_pxobject *)x, (char *)"signal");		
		outlet_new((t_pxobject *)x, (char *)"signal");
		
		x->mMesh = new MeshFDTD(x->mesh_dim[0], x->mesh_dim[1]);
		birch1mesh_do_resize(x);

		
		x->initialized = TRUE;
		x->lock = 0;			
	}
	else
	{
		error((char*)"birch1_process~: could not allocate object");
		freeobject((t_object *)x);
	}

	return ((void *)x);
}


void birch1mesh_free(t_birch1mesh *x)
{
	x->lock = 1;
	dsp_free((t_pxobject *)x);
	
	// FIX -- clean up!
	delete [] x->mUpsamplers;
	delete x->mMesh;
	jit_object_free(x->mMaskMatrix);
	jit_object_free(x->mOutMatrix);
	max_jit_obex_free(x);
}


// set new input dimensions.  
void birch1mesh_set_in_dim(t_birch1mesh *x, void *attr, long argc, t_atom *argv)
{
	if (x->initialized)
	{
		post((char *)"birch1mesh: remake object to set new size.");
	}
	else
	{
		if (argc&&argv) 
		{
			x->in_dim[0] = MAX(jit_atom_getlong(&argv[0]), 4);	
			x->in_dim[1] = MAX(jit_atom_getlong(&argv[1]), 4);	
		}
	}
}


// set new input dimensions.  
void birch1mesh_set_mesh_dim(t_birch1mesh *x, void *attr, long argc, t_atom *argv)
{
	if (x->initialized)
	{
		post((char *)"birch1mesh: remake object to set new mesh size.");
	}
	else
	{
		if (argc&&argv) 
		{
			x->mesh_dim[0] = MAX(jit_atom_getlong(&argv[0]), 4);	
			x->mesh_dim[1] = MAX(jit_atom_getlong(&argv[1]), 4);	
		}
	}
}


t_jit_err birch1mesh_set_tension(t_birch1mesh *x, void *attr, long argc, t_atom *argv)
{
	if (argc&&argv) 
	{
		x->tension = MAX(MIN(1.0, jit_atom_getfloat(argv)), 0.);		
		x->mMesh->setTension(x->tension);
	}
	return JIT_ERR_NONE;
}


t_jit_err birch1mesh_set_excite_scale(t_birch1mesh *x, void *attr, long argc, t_atom *argv)
{
	if (argc&&argv) 
	{
		x->excite_scale = jit_atom_getfloat(argv);		
		x->mMesh->mExciteScale = (x->excite_scale);
	}
	return JIT_ERR_NONE;
}

t_jit_err birch1mesh_set_damp_scale(t_birch1mesh *x, void *attr, long argc, t_atom *argv)
{
	if (argc&&argv) 
	{
		x->damp_scale = jit_atom_getfloat(argv);		
		x->mMesh->mDampScale = (x->damp_scale);
	}
	return JIT_ERR_NONE;
}

t_jit_err birch1mesh_set_damp(t_birch1mesh *x, void *attr, long argc, t_atom *argv)
{
	if (argc&&argv) 
	{
		x->mDamp = jit_atom_getfloat(argv);		
		birch1mesh_add_mask_border(x);		// remult mask with new damping 
	}
	return JIT_ERR_NONE;
}

t_jit_err birch1mesh_set_excite_pos(t_birch1mesh *x, void *attr, long argc, t_atom *argv)
{
	if (argc&&argv) 
	{
	
		x->excite_pos[0] = CLAMP(jit_atom_getfloat(&argv[0]), 0., 1.);		
		x->excite_pos[1] = CLAMP(jit_atom_getfloat(&argv[1]), 0., 1.);		
		x->mMesh->mExciteX = x->excite_pos[0];		
		x->mMesh->mExciteY = x->excite_pos[1];		
	}
	return JIT_ERR_NONE;
}



// set mask for multiply frfom incoming jitter matrix.
void birch1mesh_mask(t_birch1mesh *x, t_symbol *s, int argc, t_atom *argv)
{
	t_jit_matrix_info 		info;
	t_symbol * 				matrix_name = 0;
	void *					matrix = 0;
	void *					p_data = 0;
//	long					rowbytes;
	long					height, width;
	t_matrix_conv_info 		mcinfo = {0};

	// check for "jit_matrix"
	matrix_name = jit_atom_getsym(&argv[0]);
	if (matrix_name != _jit_sym_jit_matrix)
	{
		error ((char *)"birch1mesh: mask must be a matrix.");
		return;
	}
	// get matrix
	matrix_name = jit_atom_getsym(&argv[1]);
	if (matrix_name != _jit_sym_nothing) 
	{
		matrix = jit_object_findregistered(matrix_name);
	}
	if (!matrix) 
	{
		error ((char *)"birch1mesh: couldn't get matrix object %s!", matrix_name->s_name);
		return;
	}
	
	jit_object_method(matrix,_jit_sym_getinfo,&info);
	jit_object_method(matrix,_jit_sym_getdata,&p_data);
	if (p_data == 0)
	{
		error((char *)"birch1mesh: null data ptr for matrix!");
		return;
	}	
	if (info.dimcount != 2)
	{
		error((char *)"birch1mesh: input matrix must be 2D.");
		return;
	}	
	if (info.planecount != 1)
	{
		error((char *)"birch1mesh: input matrix must be 1 plane.");
		return;
	}	
	if (info.type != _jit_sym_float32)
	{
		error((char *)"birch1mesh: sorry, float32 matrix needed.");
		return;
	}
	height = info.dim[1];
	width = info.dim[0];
	
	// copy input to mask
	{
		mcinfo.flags |= JIT_MATRIX_CONVERT_INTERP;
		jit_object_method(x->mMaskMatrix, _jit_sym_frommatrix, matrix, &mcinfo);
	}
	
	birch1mesh_add_mask_border(x);
}



// make everything.
void birch1mesh_do_resize(t_birch1mesh *x)
{
	t_jit_object * m = 0;
	long bytes;
	long inCols = x->in_dim[0];
	long inRows = x->in_dim[1];
	long in_signals = inCols * inRows;
	long meshCols = x->mesh_dim[0];
	long meshRows = x->mesh_dim[1];
	long framesPerVector = MAX_VECSIZE / FFT_SIZE;
//	int n;
	
post((char *)"setting dim: %d, %d, %d \n", inCols, inRows, framesPerVector);
	
	// mask matrix
	m = ujit_matrix_2dfloat_new(meshCols, meshRows); 
	x->mMaskMatrixName = jit_symbol_unique();
	m = (t_jit_object *)jit_object_method(m, _jit_sym_register, x->mMaskMatrixName);	
	if (m)
	{
		x->mMaskMatrixData = ujit_matrix_get_data(m);
		x->mMaskMatrixRowbytes = ujit_matrix_get_rowbytes(m);
		bytes = x->mesh_dim[0] * x->mMaskMatrixRowbytes;
		setmem(x->mMaskMatrixData, bytes, 0);
		x->mMaskMatrix = m;
	setmem(x->mMaskMatrixData, x->mesh_dim[0] * x->mMaskMatrixRowbytes, 0);
	}
	
	// mask matrix
	m = ujit_matrix_2dfloat_new(meshCols, meshRows); 
	x->mMaskDampMatrixName = jit_symbol_unique();
	m = (t_jit_object *)jit_object_method(m, _jit_sym_register, x->mMaskDampMatrixName);	
	if (m)
	{
		x->mMaskDampMatrixData = ujit_matrix_get_data(m);
		x->mMaskDampMatrixRowbytes = ujit_matrix_get_rowbytes(m);
		bytes = x->mesh_dim[0] * x->mMaskDampMatrixRowbytes;
		setmem(x->mMaskDampMatrixData, bytes, 0);
		x->mMaskDampMatrix = m;
	}
	
	// average matrix
	m = ujit_matrix_2dfloat_new(meshCols, meshRows);
	x->mRMSMatrixName = jit_symbol_unique();
	m = (t_jit_object *)jit_object_method(m, _jit_sym_register, x->mRMSMatrixName);	
	if (m)
	{
		x->mRMSMatrixData = ujit_matrix_get_data(m);
		x->mRMSMatrixRowbytes = ujit_matrix_get_rowbytes(m);
		bytes = x->mesh_dim[0] * x->mRMSMatrixRowbytes;
		setmem(x->mRMSMatrixData, bytes, 0);
		x->mRMSMatrix = m;
	}
	
	// input signals lo: (cols * rows * framesPerVector)
	x->mInSigsLo = new float *[in_signals];
	for(int i = 0; i < in_signals; i++)
	{
		x->mInSigsLo[i] = new float[framesPerVector];
	}
	
/*	// diff sigs lo: cols * rows * framesPerVector
	x->mDiffSigsLo = new float *[in_signals];
	for(int i = 0; i < in_signals; i++)
	{
		x->mDiffSigsLo[i] = new float[framesPerVector];
	}
*/		
	// input sigs hi: cols * rows * vecsize
	x->mInSigsHi = new float *[in_signals];
	for(int i = 0; i < in_signals; i++)
	{
		x->mInSigsHi[i] = new float[MAX_VECSIZE];
	}

	// excite matrices: one per sample in vector.
	x->mExciteMatrixArray = new t_jit_object *[MAX_VECSIZE];
	x->mExciteMatrixNameArray = new t_symbol *[MAX_VECSIZE];
	x->mExciteMatrixDataArray = new float *[MAX_VECSIZE];
	for(int i = 0; i < MAX_VECSIZE; i++)
	{
		m = ujit_matrix_2dfloat_new(meshCols, meshRows);
		x->mExciteMatrixNameArray[i] = jit_symbol_unique();
		x->mExciteMatrixArray[i] = (t_jit_object *)jit_object_method(m, _jit_sym_register, x->mExciteMatrixNameArray[i]);	
		x->mExciteMatrixDataArray[i] = ujit_matrix_get_data(x->mExciteMatrixArray[i]);
	}
	x->mExciteMatrixRowbytes = ujit_matrix_get_rowbytes(m);
	
/*	// difference processors
	x->mDiffers = new ProcDiff [in_signals];
	for (int i = 0; i < in_signals; i++)
	{
		x->mDiffers[i].setPtrs(x->mInSigsLo[i], x->mDiffSigsLo[i]);
	}
*/
	
	// biquad filter / upsampler processors
	x->mUpsamplers = new ProcUpsampler [in_signals];
	for (int i = 0; i < in_signals; i++)
	{
		x->mUpsamplers[i].setPtrs(x->mInSigsLo[i], x->mInSigsHi[i]);
	}
	
	birch1mesh_clear(x);
}


void birch1mesh_add_mask_border(t_birch1mesh *x)
{
	long meshCols = x->mesh_dim[0];
	long meshRows = x->mesh_dim[1];
	// set damp matrix to default mask
	float * pIn, * pOut;
	long onEdge;
	for (int i = 0; i < meshRows; i++)
	{
		pIn = x->mMaskMatrixData + i*meshRows;	// assuming packed floats
		pOut = x->mMaskDampMatrixData + i*meshRows;	// assuming packed floats
		for (int j = 0; j < meshCols; j++)
		{
			onEdge = ((j == 0)||(j==meshCols-1)||(i == 0)||(j==meshRows-1));
			if (!onEdge)
			{
				pOut[j] = pIn[j] * x->mDamp;
			}
			else
			{
				pOut[j] = 0.;
			}			
		}
	}	
}


void birch1mesh_clear(t_birch1mesh *x)
{
//	long bytes = x->mesh_dim[0] * x->mMatrixRowbytes;
	long in_cols = x->in_dim[0];
	long in_rows = x->in_dim[1];
	long in_signals = in_cols * in_rows;
	long framesPerVector = MAX_VECSIZE / FFT_SIZE;
	int i;
	
	for(i = 0; i < in_signals; i++)
	{
		setmem(x->mInSigsLo[i] , sizeof(float)*framesPerVector, 0);
		setmem(x->mInSigsHi[i] , sizeof(float)*MAX_VECSIZE, 0);
	}
	for(i = 0; i < MAX_VECSIZE; i++)
	{
		setmem(x->mExciteMatrixDataArray[i], x->mesh_dim[0] * x->mExciteMatrixRowbytes, 0);
	}

	
	// clear avg matrix
	setmem(x->mRMSMatrixData, x->mesh_dim[0] * x->mRMSMatrixRowbytes, 0);
	
	x->mMesh->clear();
	
//	setmem(x->mOutMatrixData, bytes, 0);

}


// send out model matrix
void birch1mesh_matrix_out(t_birch1mesh *x)
{
	Atom av[1];
	
	jit_atom_setsym(av, x->mRMSMatrixName);
	outlet_anything(x->matrixOutlet, _jit_sym_jit_matrix, 1, av);	
}


void birch1mesh_assist(t_birch1mesh *x, void *b, long m, long a, char *s)
{

}


void birch1mesh_dsp(t_birch1mesh *x, t_signal **sp, short *count)
{
	int i;
	int columns = x->in_dim[0];
	int outs = 2;
	int ins = columns + 3;	// column signals + direct excitation signal, x, y
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

	dsp_addv(birch1mesh_perform, n_args, (void **)vecArray);
	t_freebytes(vecArray, size);
	
	birch1mesh_clear(x);
}


inline void _process_matrix(t_birch1mesh *x);


// pressure values are copied from input signals to input matrix. 
// every time an input matrix is filled, it gets processed to output matrix. 
// calibrated values are copied from output matrix to output.
t_int *birch1mesh_perform(t_int *w)
{
	t_birch1mesh *x = (t_birch1mesh *)(w[1]);
	int vecsize = w[2];

	float *p_ins[MAX_CHANS];
//	float *p_DampMatrix[MAX_CHANS];
//	float *p_outmatrix[MAX_CHANS];
	float *p_outs[2];
	
	int i, j, k, n, frame_start;
	int framesPerVector = x->mframesPerVector;
	int inCols = x->in_dim[0];
	int inRows = x->in_dim[1];
	int inSignals = inCols * inRows;
	int meshCols = x->mesh_dim[0];
	int meshRows = x->mesh_dim[1];
	int ins = inCols + 3;	// column signals + direct excitation signal, x, y
	int outs = 2;
	long dspSignals = (ins + outs);
	long dspArgs = dspSignals + 2;
//	float thresh = x->v_thresh;
	float v;
	
	if (x->lock || x->xObj.z_disabled)
		goto bail;
		
	// set up signal vector pointers. input and output pointers may be identical!
	for (i = 0; i < ins; i++)
	{
		p_ins[i] = (t_float *)(w[3 + i]);
	}
	p_outs[0] = (t_float *)(w[3 + ins]);
	p_outs[1] = (t_float *)(w[3 + ins + 1]);
	
	// loop for each FFT_SIZE block.  
	// assumes that fft frames start on signal vector.

	// 1: copy input signals to inputSignalsLo.
	for(k = 0; k < framesPerVector; k++)
	{
		frame_start = k*FFT_SIZE;		
		for (i = 0; i < inCols; i++)
		{
			for(j = 0; j < inRows; j++)
			{
				n = i*inRows + j;
				x->mInSigsLo[n][k] = (p_ins[i])[frame_start + j];
			}
		}
	}
	
	// 1.5: smooth inSigsLo.
//	for (i = 0; i < inSignals; i++)
//	{
//		x->mSmoothers[i].process(framesPerVector);
//	}
		
	// 2: upsample inSigsLo to inSigsHi.
	for (i = 0; i < inSignals; i++)
	{
		x->mUpsamplers[i].process(vecsize);
	}
		
	// 4: interpolate inSigsHi to exciteMatrices. // rotate 90deg.  
	{
		float fdx, fdy;
		float * pOutRow;
		long inRow, inCol;
		fdy = (float)(inRows) / (float)(meshRows);
		fdx = (float)(inCols) / (float)(meshCols);
		
		for (i = 0; i < vecsize; i++)
		{
			for (j = 0; j < meshRows; j++)
			{
				pOutRow = x->mExciteMatrixDataArray[i] + j*meshRows;
				inRow = floor((float)j * fdy);
				
				for (k = 0; k < meshCols; k++)
				{ 
					inCol = floor((float)k * fdx);
					
					// zeroth order interpolate.
					v = x->mInSigsHi[inCol * inRows + inRow][i];			
					pOutRow[k] = v;	
				}
			}	
		}	
	}
			
	// 5: run mesh.
	x->mMesh->setPtrs(x->mExciteMatrixDataArray, p_ins[ins-3], p_ins[ins-2], p_ins[ins-1], x->mMaskDampMatrixData, p_outs[0], p_outs[1], x->mRMSMatrixData);
	x->mMesh->process(vecsize);
	
	
	// TEST copy one upsampled sig to output	
	//for(i=0; i<vecsize; i++)
	//{
	//	p_outs[1][i] = x->mInSigsHi[((inRows * inCols) >> 1) + ((inCols) >> 1)][i];
	//}

				
bail:
	return (w + dspArgs + 1);
}

/*
// calibrate matrix. 
inline void _process_matrix(t_birch1mesh *x)
{
	int i, j;
	const int columns = x->dim[0];
	const int rows = x->dim[1];
	float * pIn, * pOut;
	
	// if calibrating, run calibration
	
	//...
	
	// otherwise, calibrate and copy to output matrix
		
	for (i = 0; i < columns; i++)
	{
		pIn = (t_float *)(x->mMaskMatrixData + i*rows);
		pOut = (t_float *)(x->mOutMatrixData + i*rows);

		for(j = 0; j < rows; j++)
		{
			pOut[j] = pIn[j] + 1.; // * scale + offset 
		}
	}
}

*/



