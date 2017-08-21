/** \file cuda_functions.h

    Some helper functions for CUDA,
    callable from standard C without CUDA compiler

    Copyright (C) 2010 Bastian Goldluecke,
    <first name>AT<last name>.net

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
   
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __CUDA_FUNCTIONS_H
#define __CUDA_FUNCTIONS_H

#include <assert.h>
#include <vector>
#include <iostream>
#include <string>
#include <math.h>

#include "../common/debug.h"
#include "../common/gsl_image.h"


// Return types for CUDA objects
typedef unsigned char* gpu_uchar_array;
typedef float*         gpu_float_array;

namespace coco {

/********************************************************
  Memory allocation
*********************************************************/

// standard float array
float *cuda_alloc_floats( size_t nfloats );


/********************************************************
  Create various float buffers for texture transfer
*********************************************************/

// Float buffer from double array
float *make_float_buffer( const double *data, size_t N );
// Interleaved float buffer from double array
float *make_interleaved_float_buffer( std::vector<double*> &data, size_t N );



/********************************************************
  Write CUDA buffers as test images to disk
*********************************************************/

// write cuda array to image file, unsigned version
bool write_test_image_unsigned( size_t W, size_t H, float *data, const std::string &pattern, int hint, bool normalize=true );
// write cuda array to image file, signed version
bool write_test_image_signed( size_t W, size_t H, float *data, const std::string &pattern, int hint, bool normalize=true );
// write float cuda array to pfm image file
bool write_pfm_image_signed( size_t W, size_t H, float *data, const std::string &pattern, int hint );
// write color float cuda array to pfm image file
bool write_pfm_image_signed( size_t W, size_t H, float *r, float *g, float *b, const std::string &spattern, int hint);

// write cuda array to image file, unsigned version
bool write_test_image_unsigned( size_t W, size_t H, int *data, const std::string &pattern, int hint, bool normalize=true );
// write cuda array to image file, signed version
bool write_test_image_signed( size_t W, size_t H, int *data, const std::string &pattern, int hint, bool normalize=true );
// write cuda array to image file, bool version
bool write_test_image_bool( size_t W, size_t H, bool *data, const std::string &pattern, int hint, bool normalize=true );

// write cuda array to image file, rgb version
bool write_test_image_rgb( size_t W, size_t H, const float *r, const float *g, const float *b,
			   const std::string &pattern, int hint, bool normalize=true );



/********************************************************
  MemCpy wrappers
*********************************************************/

// Copy data from gsl matrix to CUDA float array
bool cuda_memcpy( float* gpu_target, const gsl_matrix *M );
// Copy data from CUDA to gsl matrix
bool cuda_memcpy( gsl_matrix *M, const float* gpu_source );
// Copy data from gsl vector to CUDA float array
bool cuda_memcpy( float* gpu_target, const gsl_vector *v );
// Copy data from CUDA to gsl vector
bool cuda_memcpy( gsl_vector *v, const float* gpu_source );
// Copy data from double array to CUDA float array
bool cuda_memcpy( float* gpu_target, const double *d, size_t number_of_doubles );


#ifdef CUDA_DOUBLE
// Copy data from gsl matrix to CUDA float array
bool cuda_memcpy( double* gpu_target, gsl_matrix *M );
// Copy data from CUDA to gsl matrix
bool cuda_memcpy( gsl_matrix *M, double* gpu_source );
// Copy data from gsl vector to CUDA float array
bool cuda_memcpy( double* gpu_target, gsl_vector *v );
// Copy data from CUDA to gsl vector
bool cuda_memcpy( gsl_vector *v, double* gpu_source );
// Copy data from gsl matrix to CUDA float array
bool cuda_memcpy( double* gpu_target, double *d, size_t number_of_doubles );
#endif


} // namespace
#endif
