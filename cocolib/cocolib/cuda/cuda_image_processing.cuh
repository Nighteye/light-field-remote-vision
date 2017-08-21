/* -*-c++-*- */
/** \file cuda_image_processing.cuh

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

#ifndef __CUDA_IMAGE_PROCESSING_CUH
#define __CUDA_IMAGE_PROCESSING_CUH


// Useful kernels

// First order structure tensors
__global__ void structure_tensor_dx( int W, int H,
				     float *I, float *dx );
__global__ void structure_tensor_conv_x( int W, int H,
					 float k0, float k1,
					 float *in, float *out );
__global__ void structure_tensor_conv_y( int W, int H,
					 float k0, float k1,
					 float *in, float *out );
__global__ void structure_tensor_dy( int W, int H,
				     float *I, float *dy );
__global__ void structure_tensor_components( int W, int H,
					     float *a, float *b, float *c );



// Compute edge slope and coherence from structure tensor
__global__ void structure_tensor_slope_and_coherence( int W, int H,
						      float *a, float *b, float *c,
						      float *slope,
						      float *coherence );

__global__ void postprocess_slope_and_coherence( int W, int H,
						 float dmin, float dmax,
						 float *D, float *C );




// Higher order structure tensors

// Compute higher order structure tensor for overlaid orientations from second derivatives
__global__ void cuda_multidim_structure_tensor_device( int W, int H,
						       float *dxx, float *dxy, float *dyy,
						       float *a11, float *a12, float *a13,
						       float *a22, float *a23,
						       float *a33 );


// Compute Eigenvalues and Eigenvector for smallest Eigenvalue for 3x3 symmetric matrix
__global__ void smallest_eigenvalue_vector_3x3_symm_device( int W, int H,
							    float *A11, float *A12, float *A13,
							    float *A22, float *A23,
							    float *A33 );

// compute directions from MOP vector
__global__ void disparity_from_mop_device( int W, int H,
					   float disp_min, float disp_max,
					   float *m0, float *m1, float *m2,
					   float *a, float *b, float *c,
					   float *eig0, float *eig1, float *eig2,
					   float *dir0, float *dir1 );

#endif
