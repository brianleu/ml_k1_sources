/*
 *  ml_jit_utils.h
 *
 */


#include "jit.common.h"
//#include "ext.h"
#include "z_dsp.h"

//--------------------------------------------------------------------------------
// constants

#define MAX_VECSIZE	1024

//--------------------------------------------------------------------------------
// utilities

t_jit_object * ujit_matrix_2dfloat_new(long width, long height);
t_float * ujit_matrix_get_data(t_jit_object * m);
unsigned long ujit_matrix_get_rowbytes(t_jit_object * m);