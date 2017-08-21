/** \file cuda_convolutions.h

    Convolution functions for CUDA.

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

#ifndef __CUDA_CONVOLUTIONS_H
#define __CUDA_CONVOLUTIONS_H

#include "../defs.h"
#include "../common/gsl_matrix_helper.h"

#include "cuda_interface.h"

namespace coco {

  /********************************************************
  Convolution kernel structure
  *********************************************************/
  
  // Convolution kernel
  struct cuda_kernel;

  // Init arbitrary convolution kernel
  cuda_kernel *cuda_kernel_alloc( const gsl_matrix *m );

  // Init separable convolution kernel
  cuda_kernel *cuda_kernel_alloc_separable( const gsl_vector *vx, const gsl_vector *vy );

  // Release convolution kernel
  void cuda_kernel_free( cuda_kernel *kernel );


  /********************************************************
  Convolution functions
  *********************************************************/

  // Convolve array with kernel
  bool cuda_convolution( const cuda_kernel *kernel, 
			 size_t W, size_t H,
			 const float* in, float* out );


  // Fast convolution for Row-3 kernel
  bool cuda_convolution_row( float k1, float k2, float k3,
			     size_t W, size_t H,
			     const float* in, float* out );

  // Fast convolution for Column-3 kernel
  bool cuda_convolution_column( float k1, float k2, float k3,
				size_t W, size_t H,
				const float* in, float* out );

}
  
#endif
