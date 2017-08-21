/* -*-c++-*- */
/** \file cuda_image_processing.cu

    CUDA image processing algorithms

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

#include <stdio.h>

#include "cuda_image_processing.h"
#include "cuda_convolutions.h"
#include "cuda_helper.h"
#include "cuda_kernels.cuh"

#include "../common/gsl_matrix_convolutions.h"
#include "../defs.h"



/********************************************************
  Structure tensor and related algorithms
*********************************************************/

__global__ void structure_tensor_dx( int W, int H,
				     float *I, float *dx )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;
  if ( ox==0 ) {
    dx[o] = I[o+1] - I[o];
  }
  else if ( ox==W-1 ) {
    dx[o] = I[o] - I[o-1];
  }
  else {
    dx[o] = 0.5f * (I[o+1] - I[o-1]);
  }
}


__global__ void structure_tensor_conv_x( int W, int H,
					 float k0, float k1,
					 float *in, float *out )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;
  if ( ox==0 ) {
    out[o] = (k1 * in[o+1] + k0 * in[o]) / (k0+k1);
  }
  else if ( ox==W-1 ) {
    out[o] = (k0 * in[o] + k1 * in[o-1]) / (k0+k1);
  }
  else {
    out[o] = k1 * in[o+1] + k1 * in[o-1] + k0 * in[o];
  }
}



__global__ void structure_tensor_conv_y( int W, int H,
					 float k0, float k1,
					 float *in, float *out )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;
  if ( oy==0 ) {
    out[o] = (k1 * in[o+W] + k0 * in[o]) / (k0+k1);
  }
  else if ( oy==H-1 ) {
    out[o] = (k0 * in[o] + k1 * in[o-W]) / (k0+k1);
  }
  else {
    out[o] = k1 * in[o+W] + k1 * in[o-W] + k0 * in[o];
  }
}



__global__ void structure_tensor_dy( int W, int H,
				     float *I, float *dy )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;
  if ( oy==0 ) {
    dy[o] = I[o+W] - I[o];
  }
  else if ( oy==H-1 ) {
    dy[o] = I[o] - I[o-W];
  }
  else {
    dy[o] = 0.5f * (I[o+W] - I[o-W]);
  }
}



__global__ void structure_tensor_components( int W, int H,
					     float *a, float *b, float *c )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;

  // before: dx is stored in b, dy in c
  float dx = b[o];
  float dy = c[o];
  a[o] = dx*dx;
  b[o] = dx*dy;
  c[o] = dy*dy;
}



// Generate structure tensor kernels from scale parameters
bool coco::cuda_alloc_structure_tensor_kernels( float outer_scale, float inner_scale,
						cuda_kernel* &outer_kernel, cuda_kernel* &inner_kernel )
{
  // Alloc convolution kernels
  outer_kernel = NULL;
  inner_kernel = NULL;
  // Filter size is 2*sigma
  if ( outer_scale > 0.0f ) {
    size_t outer_size = 4 * int( ceil( outer_scale ) ) + 1;
    assert( outer_size > 0 );
    gsl_vector *outer_gaussian = gsl_kernel_gauss_1xn( outer_size, outer_scale );
    outer_kernel = cuda_kernel_alloc_separable( outer_gaussian, outer_gaussian );
    gsl_vector_free( outer_gaussian );
  }
  if ( inner_scale > 0.0f ) {
    size_t inner_size = 4 * int( ceil( inner_scale ) ) + 1;
    assert( inner_size > 0 );
    gsl_vector *inner_gaussian = gsl_kernel_gauss_1xn( inner_size, inner_scale );
    inner_kernel = cuda_kernel_alloc_separable( inner_gaussian, inner_gaussian );
    gsl_vector_free( inner_gaussian );
  }
  return true;
}


// Compute structure tensor for an image
/* Boundary conditions are chosen so that slopes at boundaries are computed correctly
   (or as correctly as possible). Not consistent with any differentiation/PDE methods.

   a,c: main diagonal components
   b  : off-diagonal component
*/
bool coco::cuda_structure_tensor( size_t W, size_t H,
				  float outer_scale, float inner_scale,
				  float* image,
				  float* a, float* b, float* c,
				  float* workspace )
{
  // Alloc convolution kernels
  cuda_kernel *outer_kernel = NULL;
  cuda_kernel *inner_kernel = NULL;
  cuda_alloc_structure_tensor_kernels( outer_scale, inner_scale,
					  outer_kernel, inner_kernel );

  cuda_structure_tensor( W,H, outer_kernel, inner_kernel,
			 image, a,b,c, workspace );

  cuda_kernel_free( outer_kernel );
  cuda_kernel_free( inner_kernel );
  return true;
}


// Compute structure tensor for an image
/* Boundary conditions are chosen so that slopes at boundaries are computed correctly
   (or as correctly as possible). Not consistent with any differentiation/PDE methods.

   a,c: main diagonal components
   b  : off-diagonal component
*/
bool coco::cuda_structure_tensor( size_t W, size_t H,
				  cuda_kernel *outer_kernel, cuda_kernel *inner_kernel,
				  float* image,
				  float* a, float* b, float* c,
				  float* workspace )
{
  // Alloc helper arrays for convolutions
  size_t nbytes = W*H*sizeof(float);
  float *ic = workspace;
  float *dx = workspace + W*H;
  float *dy = workspace + 2*W*H;

  // Image pre-convolution
  dim3 dimGrid;
  dim3 dimBlock;
  cuda_default_grid( W,H, dimGrid, dimBlock );

  if ( outer_kernel != NULL ) {
    cuda_convolution( outer_kernel, W,H, image, ic );
  }
  else {
    CUDA_SAFE_CALL( cudaMemcpy( ic, image, nbytes, cudaMemcpyDeviceToDevice ));    
  }

  // Compute derivatives (in b,c)
  // Special boundary behaviour to obtain (hopefully) valid slopes
  structure_tensor_dx<<<dimGrid, dimBlock>>> ( W,H, ic, a );
  cuda_convolution_column( 3.0 / 16.0, 10.0 / 16.0, 3.0 / 16.0, W,H, a,dx );
  structure_tensor_dy<<<dimGrid, dimBlock>>> ( W,H, ic, b );
  cuda_convolution_row( 3.0 / 16.0, 10.0 / 16.0, 3.0 / 16.0, W,H, b,dy );

  // Compute structure tensor components
  structure_tensor_components<<<dimGrid, dimBlock>>> ( W,H, ic, dx,dy );

  // Inner scale
  if ( inner_kernel != NULL ) {
    cuda_convolution( inner_kernel, W,H, ic, a );
    cuda_convolution( inner_kernel, W,H, dx, b );
    cuda_convolution( inner_kernel, W,H, dy, c );
  }
  else {
    CUDA_SAFE_CALL( cudaMemcpy( a, ic, nbytes, cudaMemcpyDeviceToDevice ));    
    CUDA_SAFE_CALL( cudaMemcpy( b, dx, nbytes, cudaMemcpyDeviceToDevice ));    
    CUDA_SAFE_CALL( cudaMemcpy( c, dy, nbytes, cudaMemcpyDeviceToDevice ));    
  }

  return true;
}



// Compute structure tensor for a multichannel image
/*
   a,c: main diagonal components
   b  : off-diagonal component
*/
bool coco::cuda_multichannel_structure_tensor( size_t W, size_t H,
					       cuda_kernel *outer_kernel, cuda_kernel *inner_kernel,
					       const std::vector<float*> &image,
					       float* a, float* b, float* c,
					       float* workspace )
{
  // Alloc helper arrays for convolutions
  size_t nbytes = W*H*sizeof(float);
  float *at = workspace;
  float *bt = workspace + nbytes;
  float *ct = workspace + 2*nbytes;
  workspace += 3*W*H;
  CUDA_SAFE_CALL( cudaMemset( a, 0, sizeof(float)*W*H ));
  CUDA_SAFE_CALL( cudaMemset( b, 0, sizeof(float)*W*H ));
  CUDA_SAFE_CALL( cudaMemset( c, 0, sizeof(float)*W*H ));

  // Image pre-convolution
  dim3 dimGrid;
  dim3 dimBlock;
  cuda_default_grid( W,H, dimGrid, dimBlock );
  
  // Loop over all channels, add structure tensor components
  int N = image.size();
  for ( int n=0; n<N; n++ ) {
    cuda_structure_tensor( W,H, outer_kernel, inner_kernel,
			   image[n], at,bt,ct, workspace );
    cuda_add_to_device<<< dimGrid, dimBlock >>> ( W,H, at,a );
    cuda_add_to_device<<< dimGrid, dimBlock >>> ( W,H, bt,b );
    cuda_add_to_device<<< dimGrid, dimBlock >>> ( W,H, ct,c );
  }

  cuda_scale_device<<< dimGrid, dimBlock >>> ( W,H, a, 1.0f / float(N) );
  cuda_scale_device<<< dimGrid, dimBlock >>> ( W,H, b, 1.0f / float(N) );
  cuda_scale_device<<< dimGrid, dimBlock >>> ( W,H, c, 1.0f / float(N) );
  return true;
}




// Compute structure tensor for an image
/* Boundary conditions are chosen so that slopes at boundaries are computed correctly
   (or as correctly as possible). Not consistent with any differentiation/PDE methods.

   a,c: main diagonal components
   b  : off-diagonal component
*/
bool coco::cuda_structure_tensor_3x3( size_t W, size_t H,
				      float outer_scale, float inner_scale,
				      float* image,
				      float* a, float* b, float* c,
				      float* workspace )
{
  // Alloc helper arrays for convolutions
  size_t nbytes = W*H*sizeof(float);
  float *ic = workspace;
  float *dx = workspace + W*H;
  float *dy = workspace + 2*W*H;

  // Image pre-convolution
  dim3 dimGrid;
  dim3 dimBlock;
  cuda_default_grid( W,H, dimGrid, dimBlock );

  if ( outer_scale > 0.0f ) {
    gsl_vector *outer_gaussian = gsl_kernel_gauss_1xn( 3, outer_scale );
    structure_tensor_conv_x<<< dimGrid, dimBlock >>> ( W,H,
						       outer_gaussian->data[1],
						       outer_gaussian->data[0],
						       image, a );
    structure_tensor_conv_y<<< dimGrid, dimBlock >>> ( W,H,
						       outer_gaussian->data[1],
						       outer_gaussian->data[0],
						       a, ic );
    gsl_vector_free( outer_gaussian );
  }
  else {
    CUDA_SAFE_CALL( cudaMemcpy( ic, image, nbytes, cudaMemcpyDeviceToDevice ));    
  }

  // Compute derivatives (in b,c)
  // Special boundary behaviour to obtain (hopefully) valid slopes
  structure_tensor_dx<<<dimGrid, dimBlock>>> ( W,H, ic, a );
  cuda_convolution_column( 3.0 / 16.0, 10.0 / 16.0, 3.0 / 16.0, W,H, a,dx );
  structure_tensor_dy<<<dimGrid, dimBlock>>> ( W,H, ic, b );
  cuda_convolution_row( 3.0 / 16.0, 10.0 / 16.0, 3.0 / 16.0, W,H, b,dy );

  // Compute structure tensor components
  structure_tensor_components<<<dimGrid, dimBlock>>> ( W,H, ic, dx,dy );

  // Inner scale
  if ( inner_scale > 0.0f ) {
    gsl_vector *inner_gaussian = gsl_kernel_gauss_1xn( 3, inner_scale );

    structure_tensor_conv_x<<< dimGrid, dimBlock >>> ( W,H,
						       inner_gaussian->data[1],
						       inner_gaussian->data[0],
						       ic, b );
    structure_tensor_conv_y<<< dimGrid, dimBlock >>> ( W,H,
						       inner_gaussian->data[1],
						       inner_gaussian->data[0],
						       b, a );

    structure_tensor_conv_x<<< dimGrid, dimBlock >>> ( W,H,
						       inner_gaussian->data[1],
						       inner_gaussian->data[0],
						       dx, ic );
    structure_tensor_conv_y<<< dimGrid, dimBlock >>> ( W,H,
						       inner_gaussian->data[1],
						       inner_gaussian->data[0],
						       ic, b );

    structure_tensor_conv_x<<< dimGrid, dimBlock >>> ( W,H,
						       inner_gaussian->data[1],
						       inner_gaussian->data[0],
						       dy, ic );
    structure_tensor_conv_y<<< dimGrid, dimBlock >>> ( W,H,
						       inner_gaussian->data[1],
						       inner_gaussian->data[0],
						       ic, c );
    gsl_vector_free( inner_gaussian );
  }
  else {
    CUDA_SAFE_CALL( cudaMemcpy( a, ic, nbytes, cudaMemcpyDeviceToDevice ));    
    CUDA_SAFE_CALL( cudaMemcpy( b, dx, nbytes, cudaMemcpyDeviceToDevice ));    
    CUDA_SAFE_CALL( cudaMemcpy( c, dy, nbytes, cudaMemcpyDeviceToDevice ));    
  }

  return true;
}





__global__ void postprocess_slope_and_coherence( int W, int H,
						 float dmin, float dmax,
						 float *D, float *C )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;

  // read current slope and coherence
  float d = D[o];
  float c = C[o];

  if ( d<dmin ) {
    d = dmin;
    c = 0.0f;
  }
  if ( d>dmax ) {
    d = dmax;
    c = 0.0f;
  }
  
  D[o] = d;
  C[o] = c;
}



__global__ void structure_tensor_slope_and_coherence( int W, int H,
						      float *a, float *b, float *c,
						      float *slope,
						      float *coherence )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;

  // Get structure tensor matrix
  cuflt av = a[o];
  cuflt bv = b[o];
  cuflt cv = c[o];

  // Compute larger Eigenvalue (= square of largest singular value)
  /*
  cuflt trace = av + cv;
  cuflt det = av*cv - bv*bv;
  cuflt d = sqrtf( 0.25f*trace*trace - det );
  cuflt lmax = max( 0.0f, 0.5f * trace + d );
  cuflt lmin = max( 0.0f, 0.5f * trace - d );
  cuflt v12, v22;
  if ( bv == 0.0f ) {
    if ( av >= cv ) {
      v12 = 0.0f; v22 = 1.0f;
    }
    else {
      v12 = 1.0f; v22 = 0.0f;
    }
  }
  else {
    v12 = lmin - bv; v22 = bv;
    cuflt l2 = hypotf( v12, v22 );
    v12 /= l2; v22 /= l2;
  }

  // Compute direction
  cuflt dir_x = v12;
  cuflt dir_y = v22;
  */
  float angle = 0.5f * atan2f( 2.0f * bv, av-cv );
  
  // Compute coherence
  //float coh = (ac==0.0f) ? 0.0f : ( dir_x * dir_x + dir_y * dir_y ) / ( ac*ac );
  float coh = hypotf( av-cv, 2.0f * bv ) / (av + cv + 1e-16);

  // Compute slope
  float slp = tanf( -angle );

  // Compute coherence
  slope[o] = slp;
  coherence[o] = coh;
}

// Compute slopes and coherence values from Eigenvector decomposition of structure tensor
bool coco::cuda_structure_tensor_slope_and_coherence( size_t W, size_t H,
						      float* a, float* b, float* c,
						      float* slope, float* coherence )
{
  dim3 dimGrid;
  dim3 dimBlock;
  cuda_default_grid( W,H, dimGrid, dimBlock );

  structure_tensor_slope_and_coherence<<< dimGrid, dimBlock >>> ( W,H,
								  a,b,c,
								  slope,
								  coherence );

  return true;
}




/*********************************************************************
 ** DERIVATIVE FILTERS
 *********************************************************************/

static __global__ void x_convolution2_dirichlet( int W, int H,
						 float k1, float k3,
						 float *u, float *r )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;

  float rv = 0.0f;
  if ( ox>0 ) {
    rv += k1 * u[o-1];
  }
  if ( ox<W-1 ) {
    rv += k3 * u[o+1];
  }

  r[o] = rv;
}

static __global__ void x_convolution3_dirichlet( int W, int H,
						 float k1, float k2, float k3,
						 float *u, float *r )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;

  float rv = k2 * u[o];
  if ( ox>0 ) {
    rv += k1 * u[o-1];
  }
  if ( ox<W-1 ) {
    rv += k3 * u[o+1];
  }

  r[o] = rv;
}

static __global__ void y_convolution2_dirichlet( int W, int H,
						 float k1, float k3,
						 float *u, float *r )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;

  float rv = 0.0f;
  if ( oy>0 ) {
    rv += k1 * u[o-W];
  }
  if ( oy<H-1 ) {
    rv += k3 * u[o+W];
  }

  r[o] = rv;
}


static __global__ void y_convolution3_dirichlet( int W, int H,
						 float k1, float k2, float k3,
						 float *u, float *r )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;

  float rv = k2 * u[o];
  if ( oy>0 ) {
    rv += k1 * u[o-W];
  }
  if ( oy<H-1 ) {
    rv += k3 * u[o+W];
  }

  r[o] = rv;
}


// Rotation invariant derivative x-Direction, Dirichlet
bool coco::cuda_dx_roi_dirichlet( size_t W, size_t H,
				  float* u, float* ux,
				  float* workspace )
{
  dim3 dimGrid;
  dim3 dimBlock;
  cuda_default_grid( W,H, dimGrid, dimBlock );

  x_convolution2_dirichlet<<< dimGrid, dimBlock >>> ( W,H, -1.0f, 1.0f, u, workspace );
  y_convolution3_dirichlet<<< dimGrid, dimBlock >>> ( W,H, 3.0f / 32.0f, 10.0f / 32.0f, 3.0f / 32.0f,
						      workspace, ux );
  
  return true;
}

// Rotation invariant derivative y-Direction, Dirichlet
bool coco::cuda_dy_roi_dirichlet( size_t W, size_t H,
				  float* u, float* uy,
				  float* workspace )
{
  dim3 dimGrid;
  dim3 dimBlock;
  cuda_default_grid( W,H, dimGrid, dimBlock );

  y_convolution2_dirichlet<<< dimGrid, dimBlock >>> ( W,H, -1.0f, 1.0f, u, workspace );
  x_convolution3_dirichlet<<< dimGrid, dimBlock >>> ( W,H, 3.0f / 32.0f, 10.0f / 32.0f, 3.0f / 32.0f,
						      workspace, uy );
  
  return true;
}




static __global__ void x_convolution3_neumann( int W, int H,
					       float k1, float k2, float k3,
					       float *u, float *r )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;

  float uv = u[o];
  float rv = k2 * uv;
  if ( ox>0 ) {
    rv += k1 * u[o-1];
  }
  else {
    rv += k1 * uv;
  }

  if ( ox<W-1 ) {
    rv += k3 * u[o+1];
  }
  else {
    rv += k3 * uv;
  }

  r[o] = rv;
}


static __global__ void y_convolution3_neumann( int W, int H,
					       float k1, float k2, float k3,
					       float *u, float *r )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;

  float uv = u[o];
  float rv = k2 * uv;
  if ( oy>0 ) {
    rv += k1 * u[o-W];
  }
  else {
    rv += k1 * uv;
  }

  if ( oy<H-1 ) {
    rv += k3 * u[o+W];
  }
  else {
    rv += k3 * uv;
  }

  r[o] = rv;
}


/*
static __global__ void x_convolution3_periodic( int W, int H,
						float k1, float k2, float k3,
						float *u, float *r )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;

  float uv = u[o];
  float rv = k2 * uv;
  if ( ox>0 ) {
    rv += k1 * u[o-1];
  }
  else {
    rv += k1 * u[o+W-1];
  }

  if ( ox<W-1 ) {
    rv += k3 * u[o+1];
  }
  else {
    rv += k3 * u[o-W+1];
  }

  r[o] = rv;
}


static __global__ void y_convolution3_periodic( int W, int H,
						float k1, float k2, float k3,
						float *u, float *r )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;

  float uv = u[o];
  float rv = k2 * uv;
  if ( oy>0 ) {
    rv += k1 * u[o-W];
  }
  else {
    rv += k1 * u[o-W+W*H];
  }

  if ( oy<H-1 ) {
    rv += k3 * u[o+W];
  }
  else {
    rv += k3 * u[o+W-W*H];
  }

  r[o] = rv;
}
*/



// Rotation invariant derivative x-Direction, Neumann
bool coco::cuda_dx_roi_neumann( size_t W, size_t H,
				float* u, float* ux,
				float* workspace )
{
  dim3 dimGrid;
  dim3 dimBlock;
  cuda_default_grid( W,H, dimGrid, dimBlock );

  x_convolution3_neumann<<< dimGrid, dimBlock >>> ( W,H, -1.0f, 0.0f, 1.0f, u, workspace );
  y_convolution3_neumann<<< dimGrid, dimBlock >>> ( W,H, 3.0f / 32.0f, 10.0f / 32.0f, 3.0f / 32.0f,
						    workspace, ux );
  
  return true;
}


// Rotation invariant derivative y-Direction, Neumann
bool coco::cuda_dy_roi_neumann( size_t W, size_t H,
				float* u, float* uy,
				float* workspace )
{
  dim3 dimGrid;
  dim3 dimBlock;
  cuda_default_grid( W,H, dimGrid, dimBlock );

  y_convolution3_neumann<<< dimGrid, dimBlock >>> ( W,H, -1.0f, 0.0f, 1.0f, u, workspace );
  x_convolution3_neumann<<< dimGrid, dimBlock >>> ( W,H, 3.0f / 32.0f, 10.0f / 32.0f, 3.0f / 32.0f,
						    workspace, uy );
  return true;
}





// Compute higher order structure tensor from second derivatives
static __global__ void cuda_multidim_structure_tensor_device( int W, int H,
							      float *dxx, float *dxy, float *dyy,
							      float *a11, float *a12, float *a13,
							      float *a22, float *a23,
							      float *a33 )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;

  float dxxv = dxx[o];
  float dxyv = dxy[o];
  float dyyv = dyy[o];
  
  a11[o] = dxxv*dxxv;
  a12[o] = dxxv*dxyv;
  a13[o] = dxxv*dyyv;
  a22[o] = dxyv*dxyv;
  a23[o] = dxyv*dyyv;
  a33[o] = dyyv*dyyv;
}



__global__ void smallest_eigenvalue_vector_3x3_symm_device( int W, int H,
							    float *A11, float *A12, float *A13,
							    float *A22, float *A23,
							    float *A33 )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;

  const float a11 = A11[o];
  const float a12 = A12[o];
  const float a13 = A13[o];
  const float a22 = A22[o];
  const float a23 = A23[o];
  const float a33 = A33[o];

  float p = a12*a12 + a13*a13 + a23*a23;
  float eig1;
  float eig2;
  float eig3;
  float tmp;
  float b11, b12, b13, b22, b23, b33;

  if (p == 0.0f) {
    // A is diagonal.
    eig1 = a11;
    eig2 = a22;
    eig3 = a33;
    // sort
    if ( eig3>eig1 ) {
      tmp = eig3; eig3 = eig1; eig1 = tmp;
    }
    if ( eig2>eig1 ) {
      tmp = eig2; eig2 = eig1; eig1 = tmp;
    }
    if ( eig3>eig2 ) {
      tmp = eig3; eig3 = eig2; eig2 = tmp;
    }
  }
  else {

    const float q = (a11 + a22 + a33) / 3.0f;
    b11 = a11 - q; b12 = a12; b13 = a13;
    b22 = a22 - q; b23 = a23;
    b33 = a33 - q;
    p = b11*b11 + b22*b22 + b33*b33 + 2.0f*p;
    p = sqrtf(p / 6.0f);

    const float detB = b11*(b22*b33 - b23*b23) + b12*(b23*b13-b12*b33) + b13*(b12*b23-b22*b13);
    float r = detB / (2.0f*p*p*p);

    // In exact arithmetic for a symmetric matrix  -1 <= r <= 1
    // but computation error can leave it slightly outside this range.
    float phi;
    if (r <= -1.0f) {
      phi = 3.14159265f / 3.0f;
    }
    else if (r >= 1.0f) {
      phi = 0.0f;
    }
    else {
      phi = acosf(r) / 3.0f;
    }
 
    // the eigenvalues satisfy eig3 <= eig2 <= eig1
    eig1 = q + 2.0f * p * cosf(phi);
    eig3 = q + 2.0f * p * cosf(phi +  3.14159265f * (2.0f/3.0f));
    eig2 = 3.0f * q - eig1 - eig3; // since trace(A) = eig1 + eig2 + eig3
  }

  // return Eigenvalues
  A22[o] = eig1;
  A23[o] = eig2;
  A33[o] = eig3;

  // compute eigenvector
  b11 = (a11-eig1)*(a11-eig2) + a12*a12 + a13*a13;
  b12 = (a11-eig1)*a12 + a12*(a22-eig2) + a13*a23;
  b13 = (a11-eig1)*a13 + a12*a23 + a13*(a33-eig2);
  b22 = a12*a12 + (a22-eig1)*(a22-eig2) + a23*a23;
  b23 = a12*a13 + (a22-eig1)*a23 + a23*(a33-eig2);
  b33 = a13*a13 + a23*a23 + (a33-eig1)*(a33-eig2);

  // largest column of (A-lambda1 I) (A-lambda2 I)
  float len1 = b11*b11 + b12*b12 + b13*b13;
  float len2 = b12*b12 + b22*b22 + b23*b23;
  float len3 = b13*b13 + b23*b23 + b33*b33;
  if ( len1 >= len2 && len1 >= len3 ) {
    len1 = sqrtf( len1 );
    A11[o] = b11 / len1;
    A12[o] = b12 / len1;
    A13[o] = b13 / len1;
  }
  else if ( len2 >= len3 ) {
    len1 = sqrtf( len2 );
    A11[o] = b12 / len2;
    A12[o] = b22 / len2;
    A13[o] = b23 / len2;
  }
  else {
    len1 = sqrtf( len3 );
    A11[o] = b13 / len3;
    A12[o] = b23 / len3;
    A13[o] = b33 / len3;
  }
}



__global__ void disparity_from_mop_device( int W, int H,
					   float disp_min, float disp_max,
					   float *m0, float *m1, float *m2,
					   float *a, float *b, float *c,
					   float *eig0, float *eig1, float *eig2,
					   float *dir0, float *dir1 )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;

  const float mop0 = m0[o];
  const float mop1 = m1[o];
  const float mop2 = m2[o];

  float u0, v0;
  if ( fabs(mop0) < 1e-10f ) {
    u0 = 0.0f;
    v0 = 0.0f;
  }
  else {
    // temporary matrix
    const float m00 =  mop1 / mop0;
    const float m01 = -mop2 / mop0;
    const float m10 = 1.0f;
    const float m11 = 0.0f;
    // Eigenvalues give second coordinate of direction
    const float T = m00 + m11;
    const float D = m00 * m11 - m10 * m01;
    float dc = T*T/4.0f - D;
    u0 = 0.0f;
    v0 = 0.0f;
    if ( dc >= 0.0f ) {
      dc = sqrtf( dc );
      u0 = T/2.0f + dc;
      v0 = T/2.0f - dc;
    }
  }

  dir0[o] = u0;
  dir1[o] = v0; 
}




// Compute multichannel, multidirectional structure tensor for an image
/* Boundary conditions are chosen so that slopes at boundaries are computed correctly
   (or as correctly as possible). Not consistent with any differentiation/PDE methods.

   a,c: main diagonal components
   b  : off-diagonal component

   a11, a12, a13,
        a22, a23,
             a33:   second order tensor
*/
bool coco::cuda_second_order_multichannel_structure_tensor( size_t W, size_t H,
							    cuda_kernel *outer_kernel,
							    cuda_kernel *inner_kernel,
							    cuda_kernel *outer_kernel_multi,
							    cuda_kernel *inner_kernel_multi,
							    std::vector<float*> image,
							    float* a, float* b, float* c,
							    float* a11, float* a12, float* a13,
							    float* a22, float* a23,
							    float* a33,
							    float* workspace )
{
  // Alloc helper arrays for convolutions
  size_t nbytes = W*H*sizeof(float);

  float *ic = workspace;
  float *dx = workspace + 1*W*H;
  float *dy = workspace + 2*W*H;
  float *tmp = workspace + 3*W*H;
  //float *tmp2 = workspace + 4*W*H;
  float *atmp = workspace + 5*W*H;
  float *btmp = workspace + 6*W*H;
  float *ctmp = workspace + 7*W*H;
  float *dxx = workspace + 8*W*H;
  float *dxy = workspace + 9*W*H;
  float *dyy = workspace + 10*W*H;
  float *t[6];
  t[0] = workspace + 15*W*H;
  t[1] = workspace + 16*W*H;
  t[2] = workspace + 17*W*H;
  t[3] = workspace + 18*W*H;
  t[4] = workspace + 19*W*H;
  t[5] = workspace + 20*W*H;
  float *t_conv[6];
  t_conv[0] = workspace + 21*W*H;
  t_conv[1] = workspace + 22*W*H;
  t_conv[2] = workspace + 23*W*H;
  t_conv[3] = workspace + 24*W*H;
  t_conv[4] = workspace + 25*W*H;
  t_conv[5] = workspace + 26*W*H;

  // Image pre-convolution
  dim3 dimGrid;
  dim3 dimBlock;
  cuda_default_grid( W,H, dimGrid, dimBlock );

  // Clear accum components
  CUDA_SAFE_CALL( cudaMemset( a,0, nbytes ));
  CUDA_SAFE_CALL( cudaMemset( b,0, nbytes ));
  CUDA_SAFE_CALL( cudaMemset( c,0, nbytes ));

  CUDA_SAFE_CALL( cudaMemset( a11,0, nbytes ));
  CUDA_SAFE_CALL( cudaMemset( a12,0, nbytes ));
  CUDA_SAFE_CALL( cudaMemset( a13,0, nbytes ));
  CUDA_SAFE_CALL( cudaMemset( a22,0, nbytes ));
  CUDA_SAFE_CALL( cudaMemset( a23,0, nbytes ));
  CUDA_SAFE_CALL( cudaMemset( a33,0, nbytes ));

  for ( size_t nchannel=0; nchannel<image.size(); nchannel++ ) {
    if ( outer_kernel != NULL ) {
      cuda_convolution( outer_kernel, W,H, image[nchannel], ic );
    }
    else {
      CUDA_SAFE_CALL( cudaMemcpy( ic, image[nchannel], nbytes, cudaMemcpyDeviceToDevice ));    
    }

    // Compute derivatives (in b,c)
    // Special boundary behaviour to obtain (hopefully) valid slopes
    structure_tensor_dx<<<dimGrid, dimBlock>>> ( W,H, ic, atmp );
    cuda_convolution_column( 3.0 / 16.0, 10.0 / 16.0, 3.0 / 16.0, W,H, atmp,dx );
    structure_tensor_dy<<<dimGrid, dimBlock>>> ( W,H, ic, btmp );
    cuda_convolution_row( 3.0 / 16.0, 10.0 / 16.0, 3.0 / 16.0, W,H, btmp,dy );

    //structure_tensor_dx<<<dimGrid, dimBlock>>> ( W,H, ic, dx );
    //structure_tensor_dy<<<dimGrid, dimBlock>>> ( W,H, ic, dy );
    
    // Second order derivatives
    cuda_convolution( outer_kernel_multi, W,H, dx, tmp );
    structure_tensor_dx<<<dimGrid, dimBlock>>> ( W,H, tmp, atmp );
    cuda_convolution_column( 3.0 / 16.0, 10.0 / 16.0, 3.0 / 16.0, W,H, atmp,dxx );
    structure_tensor_dy<<<dimGrid, dimBlock>>> ( W,H, tmp, btmp );
    cuda_convolution_row( 3.0 / 16.0, 10.0 / 16.0, 3.0 / 16.0, W,H, btmp,dxy );

    //structure_tensor_dx<<<dimGrid, dimBlock>>> ( W,H, tmp, dxx );
    //structure_tensor_dy<<<dimGrid, dimBlock>>> ( W,H, tmp, dxy );
    cuda_convolution( outer_kernel_multi, W,H, dy, tmp );
    structure_tensor_dy<<<dimGrid, dimBlock>>> ( W,H, tmp, ctmp );
    cuda_convolution_row( 3.0 / 16.0, 10.0 / 16.0, 3.0 / 16.0, W,H, ctmp,dyy );

    //structure_tensor_dy<<<dimGrid, dimBlock>>> ( W,H, tmp, dyy );


    // Compute structure tensor components
    structure_tensor_components<<<dimGrid, dimBlock>>> ( W,H, ic, dx,dy );

    // Inner scale
    if ( inner_kernel != NULL ) {
      cuda_convolution( inner_kernel, W,H, ic, atmp );
      cuda_convolution( inner_kernel, W,H, dx, btmp );
      cuda_convolution( inner_kernel, W,H, dy, ctmp );
    }
    else {
      CUDA_SAFE_CALL( cudaMemcpy( atmp, ic, nbytes, cudaMemcpyDeviceToDevice ));    
      CUDA_SAFE_CALL( cudaMemcpy( btmp, dx, nbytes, cudaMemcpyDeviceToDevice ));    
      CUDA_SAFE_CALL( cudaMemcpy( ctmp, dy, nbytes, cudaMemcpyDeviceToDevice ));    
    }

    // Accumulate multichannel result
    cuda_add_to_device<<< dimGrid, dimBlock >>>
      ( W,H, atmp, a );
    cuda_add_to_device<<< dimGrid, dimBlock >>>
      ( W,H, btmp, b );
    cuda_add_to_device<<< dimGrid, dimBlock >>>
      ( W,H, ctmp, c );


    // Compute second order structure tensor components
    cuda_multidim_structure_tensor_device<<< dimGrid, dimBlock >>>
      ( W, H,
	dxx, dxy, dyy,
	t[0], t[1], t[2],
	t[3], t[4],
	t[5] );

    // Inner scale
    if ( inner_kernel_multi != NULL ) {
      for ( int i=0; i<6; i++ ) {
	cuda_convolution( inner_kernel_multi, W,H, t[i], t_conv[i] );
      }
    }
    else {
      for ( int i=0; i<6; i++ ) {
	CUDA_SAFE_CALL( cudaMemcpy( t_conv[i], t[i], nbytes, cudaMemcpyDeviceToDevice ));    
      }
    }

    // Accumulate multichannel result
    cuda_add_to_device<<< dimGrid, dimBlock >>>
      ( W,H, t_conv[0], a11 );
    cuda_add_to_device<<< dimGrid, dimBlock >>>
      ( W,H, t_conv[1], a12 );
    cuda_add_to_device<<< dimGrid, dimBlock >>>
      ( W,H, t_conv[2], a13 );
    cuda_add_to_device<<< dimGrid, dimBlock >>>
      ( W,H, t_conv[3], a22 );
    cuda_add_to_device<<< dimGrid, dimBlock >>>
      ( W,H, t_conv[4], a23 );
    cuda_add_to_device<<< dimGrid, dimBlock >>>
      ( W,H, t_conv[5], a33 );
  }

  return true;
}





