/*
 *  ml_jit_utils.cpp
 *
 *  Created by Randy Jones on 8/11/08.
 *
 */

#include "ml_jit_utils.h"


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


