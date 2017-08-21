/** \file cuda_image_processing.h

    Image processing algorithms for CUDA

    Copyright (C) 2012 Bastian Goldluecke,
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

#ifndef __CUDA_IMAGE_PROCESSING_H
#define __CUDA_IMAGE_PROCESSING_H

#include "../defs.h"
#include "cuda_interface.h"
#include "cuda_convolutions.h"

#include <vector>

namespace coco {

  /********************************************************
  Special derivative filters
    Required workspace size is W*H floats
  *********************************************************/

  // Rotation invariant derivative x-Direction, Dirichlet
  bool cuda_dx_roi_dirichlet( size_t W, size_t H,
			      float* u, float* ux,
			      float* workspace );

  // Rotation invariant derivative x-Direction, Neumann
  bool cuda_dx_roi_neumann( size_t W, size_t H,
			    float* u, float* ux,
			    float* workspace );

  // Rotation invariant derivative y-Direction, Dirichlet
  bool cuda_dy_roi_dirichlet( size_t W, size_t H,
			      float* u, float* uy,
			      float* workspace );

  // Rotation invariant derivative y-Direction, Neumann
  bool cuda_dy_roi_neumann( size_t W, size_t H,
			    float* u, float* uy,
			    float* workspace );


  /********************************************************
  Structure tensor and related algorithms
  *********************************************************/

  // Generate structure tensor kernels from scale parameters
  bool cuda_alloc_structure_tensor_kernels( float outer_scale, float inner_scale,
					    cuda_kernel* &outer_kernel, cuda_kernel* &inner_kernel );

  // Compute structure tensor for an image
  /* Boundary conditions are chosen so that slopes at boundaries are computed correctly
     (or as correctly as possible). Not consistent with any differentiation/PDE methods.

     a,c: main diagonal components
     b  : off-diagonal component
  */
  bool cuda_structure_tensor( size_t W, size_t H,
			      float outer_scale, float inner_scale,
			      float* image,
			      float* a, float* b, float* c,
			      float* workspace );

  // Compute structure tensor for an image
  /* Boundary conditions are chosen so that slopes at boundaries are computed correctly
     (or as correctly as possible). Not consistent with any differentiation/PDE methods.

     a,c: main diagonal components
     b  : off-diagonal component
  */
  bool cuda_structure_tensor( size_t W, size_t H,
			      cuda_kernel *outer_kernel, cuda_kernel *inner_kernel,
			      float* image,
			      float* a, float* b, float* c,
			      float* workspace );

  // Compute structure tensor for a vector-valued image
  /* Boundary conditions are chosen so that slopes at boundaries are computed correctly
     (or as correctly as possible). Not consistent with any differentiation/PDE methods.

     a,c: main diagonal components
     b  : off-diagonal component
  */
  bool cuda_multichannel_structure_tensor( size_t W, size_t H,
					   cuda_kernel *outer_kernel, cuda_kernel *inner_kernel,
					   const std::vector<float*> &image,
					   float* a, float* b, float* c,
					   float* workspace );

  // Compute structure tensor for an image (fast and approximate, 3x3 kernels)
  /* Boundary conditions are chosen so that slopes at boundaries are computed correctly
     (or as correctly as possible). Not consistent with any differentiation/PDE methods.

     a,c: main diagonal components
     b  : off-diagonal component
  */
  bool cuda_structure_tensor_3x3( size_t W, size_t H,
				  float outer_scale, float inner_scale,
				  float* image,
				  float* a, float* b, float* c,
				  float* workspace );


  // Compute slopes and coherence values from Eigenvector decomposition of structure tensor
  bool cuda_structure_tensor_slope_and_coherence( size_t W, size_t H,
						  float* a, float* b, float* c,
						  float* slope, float* coherence );



  /********************************************************
  Second order structure tensor and related algorithms
  *********************************************************/

  // Compute multichannel structure tensor for an image up to second order
  /* Boundary conditions are chosen so that slopes at boundaries are computed correctly
     (or as correctly as possible). Not consistent with any differentiation/PDE methods.

     outer, inner kernel: first order kernels
     outer, inner kernel multi: second order kernels

     image : input image

     a,c: main diagonal components
     b  : off-diagonal component
     
     a11, a12, a13,
     a22, a23,
     a33:   second order tensor

     workspace: at least 30 times image size floats
  */
  bool cuda_second_order_multichannel_structure_tensor( size_t W, size_t H,
							cuda_kernel *outer_kernel,
							cuda_kernel *inner_kernel,
							cuda_kernel *outer_kernel_multi,
							cuda_kernel *inner_kernel_multi,
							std::vector<float*> image,
							float* a, float* b, float* c,
							float* a11, float* a12, float* a13,
							float* a22, float* a23,
							float* a33,
							float* workspace );

}
  
#endif
