/* -*-c++-*- */
/** \file kernels_vtv.cu

    VTV kernels on grids

    Copyright (C) 2011-2014 Bastian Goldluecke,
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

#include "../compute_api/kernels_reprojections.h"
#include "compute_api_implementation_cuda.h"


// Reprojection for RGB, TV_F
static __global__ void kernel_reproject_euclidean_1D( int W, int H,
							 const float r,
							 float *p )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>= H ) {
    return;
  }
  int o = oy*W + ox;

  // Equivalent to clamping to range [-r,r]
  p[o] = max( -r, min( r,p[o] ));
}

// Reprojection to Euclidean ball with given radius
void coco::kernel_reproject_euclidean_1D( const compute_grid *G,
					     const float radius,
					     compute_buffer &px1 )
{
  dim3 dimGrid, dimBlock;
  kernel_configure( G, dimGrid, dimBlock );
  ::kernel_reproject_euclidean_1D<<< dimGrid, dimBlock >>>
      ( G->W(), G->H(),
	radius,
	px1 );
}



// Reprojection for RGB, TV_F
static __global__ void kernel_reproject_euclidean_2D( int W, int H,
							 const float r,
							 float *px1, float *py1 )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>= H ) {
    return;
  }
  int o = oy*W + ox;

  // Local vars
  float a1 = px1[o];
  float a2 = py1[o];

  // Total norm
  float n = hypotf( a1, a2 );

  // Project
  if ( n > r ) {
    n = r/n;
    px1[o] = a1 * n;
    py1[o] = a2 * n;
  }
}

// Reprojection to Euclidean ball with given radius
void coco::kernel_reproject_euclidean_2D( const compute_grid *G,
					     const float radius,
					     compute_buffer &px1, compute_buffer &py1 )
{
  dim3 dimGrid, dimBlock;
  kernel_configure( G, dimGrid, dimBlock );
  ::kernel_reproject_euclidean_2D<<< dimGrid, dimBlock >>>
      ( G->W(), G->H(),
	radius,
	px1, py1 );
}



// Reprojection for RGB, TV_F
static __global__ void kernel_reproject_euclidean_4D( int W, int H,
							 const float r,
							 float *px1, float *py1,
							 float *px2, float *py2 )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>= H ) {
    return;
  }
  int o = oy*W + ox;

  // Local vars
  float a11 = px1[o];
  float a21 = px2[o];
  float a12 = py1[o];
  float a22 = py2[o];

  // Total norm
  float n = powf( a11, 2.0f ) + powf( a21, 2.0f );
  n += powf( a12, 2.0f ) + powf( a22, 2.0f );
  n = sqrtf( n );

  // Project
  if ( n > r ) {
    n = r/n;
    px1[o] = a11 * n;
    py1[o] = a12 * n;
    px2[o] = a21 * n;
    py2[o] = a22 * n;
  }
}


// Reprojection to Euclidean ball with given radius
void coco::kernel_reproject_euclidean_4D( const compute_grid *G,
					     const float radius,
					     compute_buffer &px1, compute_buffer &py1,
					     compute_buffer &px2, compute_buffer &py2 )
{
  dim3 dimGrid, dimBlock;
  kernel_configure( G, dimGrid, dimBlock );
  ::kernel_reproject_euclidean_4D<<< dimGrid, dimBlock >>>
      ( G->W(), G->H(),
	radius,
	px1, py1, px2, py2 );
}




// Reprojection for RGB, TV_F
static __global__ void kernel_reproject_euclidean_6D( int W, int H,
							 const float r,
							 float *px1, float *py1,
							 float *px2, float *py2,
							 float *px3, float *py3 )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>= H ) {
    return;
  }
  int o = oy*W + ox;

  // Local vars
  float a11 = px1[o];
  float a21 = px2[o];
  float a31 = px3[o];
  float a12 = py1[o];
  float a22 = py2[o];
  float a32 = py3[o];

  // Total norm
  float n = powf( a11, 2.0f ) + powf( a21, 2.0f ) + powf( a31, 2.0f );
  n += powf( a12, 2.0f ) + powf( a22, 2.0f ) + powf( a32, 2.0f );
  n = sqrtf( n );

  // Project
  if ( n > r ) {
    n = r/n;
    px1[o] = a11 * n;
    py1[o] = a12 * n;
    px2[o] = a21 * n;
    py2[o] = a22 * n;
    px3[o] = a31 * n;
    py3[o] = a32 * n;
  }
}


// Reprojection to Euclidean ball with given radius
void coco::kernel_reproject_euclidean_6D( const compute_grid *G,
					     const float radius,
					     compute_buffer &px1, compute_buffer &py1,
					     compute_buffer &px2, compute_buffer &py2,
					     compute_buffer &px3, compute_buffer &py3 )
{
  dim3 dimGrid, dimBlock;
  kernel_configure( G, dimGrid, dimBlock );
  ::kernel_reproject_euclidean_6D<<< dimGrid, dimBlock >>>
      ( G->W(), G->H(),
	radius,
	px1, py1, px2, py2, px3, py3 );
}




// Reprojection for RGB, TV_J
__global__ void kernel_reproject_nuclear_6D( int W, int H,
						const float radius,
						float *px1, float *py1,
						float *px2, float *py2,
						float *px3, float *py3 )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>= H ) {
    return;
  }
  int o = oy*W + ox;

  // Local vars
  float a11 = px1[o];
  float a21 = px2[o];
  float a31 = px3[o];
  float a12 = py1[o];
  float a22 = py2[o];
  float a32 = py3[o];

  // Compute A^T A
  float d11 = a11*a11 + a21*a21 + a31*a31;
  float d12 = a12*a11 + a22*a21 + a32*a31;
  float d22 = a12*a12 + a22*a22 + a32*a32;

  // Compute larger Eigenvalue (= square of largest singular value)
  float trace = d11 + d22;
  float det = d11*d22 - d12*d12;
  float d = sqrt( 0.25*trace*trace - det );
  float lmax = max( 0.0, 0.5 * trace + d );
  float lmin = max( 0.0, 0.5 * trace - d );
  float smax = sqrt( lmax );
  float smin = sqrt( lmin );

  // If smax + smin > 1:
  // Project (smax,smin) to line (0,1) + tau * (1,-1), 0<=tau<=1.
  if ( smax + smin > radius ) {

    float v11, v12, v21, v22;
    if ( d12 == 0.0 ) {
      if ( d11 >= d22 ) {
	v11 = 1.0; v21 = 0.0; v12 = 0.0; v22 = 1.0;
      }
      else {
	v11 = 0.0; v21 = 1.0; v12 = 1.0; v22 = 0.0;
      }
    }
    else {
      v11 = lmax - d22; v21 = d12;
      float l1 = hypotf( v11, v21 );
      v11 /= l1; v21 /= l1;
      v12 = lmin - d22; v22 = d12;
      float l2 = hypot( v12, v22 );
      v12 /= l2; v22 /= l2;
    }

    // Compute projection of Eigenvalues
    float tau = 0.5f * (smax - smin + 1.0f);
    float s1 = min( 1.0f, tau );
    float s2 = 1.0f - s1;
    // Compute \Sigma^{-1} * \Sigma_{new}
    s1 /= smax;
    s2 = (smin > 0.0f) ? s2 / smin : 0.0f;

    // A_P = A * \Sigma^{-1} * \Sigma_{new}
    float t11 = s1*v11*v11 + s2*v12*v12;
    float t12 = s1*v11*v21 + s2*v12*v22;
    float t21 = s1*v21*v11 + s2*v22*v12;
    float t22 = s1*v21*v21 + s2*v22*v22;

    // Result
    px1[o] = radius * (a11 * t11 + a12 * t21);
    px2[o] = radius * (a21 * t11 + a22 * t21);
    px3[o] = radius * (a31 * t11 + a32 * t21);
    
    py1[o] = radius * (a11 * t12 + a12 * t22);
    py2[o] = radius * (a21 * t12 + a22 * t22);
    py3[o] = radius * (a31 * t12 + a32 * t22);
  }
}



// Reprojection to nuclear norm ball
void coco::kernel_reproject_nuclear_6D( const compute_grid *G,
					   const float radius,
					   compute_buffer &px1, compute_buffer &py1,
					   compute_buffer &px2, compute_buffer &py2,
					   compute_buffer &px3, compute_buffer &py3 )
{
  dim3 dimGrid, dimBlock;
  kernel_configure( G, dimGrid, dimBlock );
  ::kernel_reproject_nuclear_6D<<< dimGrid, dimBlock >>>
      ( G->W(), G->H(),
	radius,
	px1, py1, px2, py2, px3, py3 );
}


