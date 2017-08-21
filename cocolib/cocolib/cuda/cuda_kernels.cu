/* -*-c++-*- */
/** \file cuda_kernels.cu

    Some standard CUDA kernels.

    Copyright (C) 2011 Bastian Goldluecke,
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


#include "cuda_helper.h"
#include "cuda_inline_device_functions.cu"



////////////////////////////////////////////////////////////////////////////////
// Array initialization
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_set_all_device( int W, int H, cuflt *dst, float value )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;
  dst[o] = value;
}

__global__ void cuda_set_all_bool_device( int W, int H, bool *dst, bool value )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;
  dst[o] = value;
}



////////////////////////////////////////////////////////////////////////////////
// Transpose matrix
// Grid layout given for target
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_transpose_device( int W, int H, cuflt *src, cuflt *dst )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int d = IMUL( oy,W ) + ox;
  const int s = IMUL( ox,H ) + oy;
  dst[d] = src[s];
}


////////////////////////////////////////////////////////////////////////////////
// Add first argument to second
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_add_to_device( int W, int H, cuflt *src, cuflt *dst )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;
  dst[o] += src[o];
}

__global__ void cuda_add_scalar_to_device( int W, int H, cuflt v, cuflt *dst )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;
  dst[o] += v;
}



////////////////////////////////////////////////////////////////////////////////
// Subtract first argument from second
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_subtract_from_device( int W, int H, const cuflt *src, cuflt *dst )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;
  dst[o] -= src[o];
}



////////////////////////////////////////////////////////////////////////////////
// Divide first argument by second
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_divide_by_device( int W, int H, cuflt *src, cuflt *dst )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;
  float v = src[o];
  if ( v != 0.0f ) {
    dst[o] /= v;
  }
}


////////////////////////////////////////////////////////////////////////////////
// Multiply first argument with second, store in third
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_multiply_device( int W, int H, cuflt *m1, cuflt *m2, cuflt *r )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;
  r[o] = m1[o] * m2[o];
}


////////////////////////////////////////////////////////////////////////////////
// Multiply first argument with second
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_multiply_with_device( int W, int H, cuflt *dst, cuflt *src )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;
  dst[o] *= src[o];
}



////////////////////////////////////////////////////////////////////////////////
// Add scaled first argument to second
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_add_scaled_to_device( int W, int H, const cuflt *src, cuflt t, cuflt *dst )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;
  dst[o] += t*src[o];
}

__global__ void cuda_add_scaled_to_device( int W, int H, const cuflt *src, cuflt *t, cuflt *dst )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;
  dst[o] += t[o]*src[o];
}


////////////////////////////////////////////////////////////////////////////////
// Compute linear combination of two arguments
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_linear_combination_device( int W, int H,
						cuflt w0, cuflt *src0,
						cuflt w1, cuflt *src1,
						cuflt *dst )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;
  dst[o] = w0*src0[o] + w1*src1[o];
}


////////////////////////////////////////////////////////////////////////////////
// Scale array by argument
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_scale_device( int W, int H, cuflt *dst, cuflt t )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;
  dst[o] *= t;
}


////////////////////////////////////////////////////////////////////////////////
// Square array element-wise
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_square_device( int W, int H, cuflt *dst )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;
  dst[o] = powf( dst[o], 2.0f );
}


////////////////////////////////////////////////////////////////////////////////
// Arbitrary power element-wise
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_pow_device( int W, int H, cuflt *dst, cuflt exponent )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;
  dst[o] = powf( max( 0.0f, dst[o] ), exponent );
}



////////////////////////////////////////////////////////////////////////////////
// Pointwise absolute value
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_abs_device( int W, int H, cuflt *dst )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;
  dst[o] = fabs( dst[o] );
}


////////////////////////////////////////////////////////////////////////////////
// Arbitrary power element-wise
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_max_truncate_device( int W, int H, cuflt *dst, cuflt value )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;
  dst[o] = max( value, dst[o] );
}


////////////////////////////////////////////////////////////////////////////////
// Clamp to range element-wise
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_clamp_device( int W, int H, cuflt *dst, cuflt m, cuflt M )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;
  dst[o] = min( M, max( m, dst[o] ));
}



////////////////////////////////////////////////////////////////////////////////
// Normalize element-wise (zero weight target is set to zero)
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_normalize_device( int W, int H,
				       float *target, const float *weight )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }

  int o = ox + oy*W;
  float w = weight[o];
  if ( w != 0.0f ) {
    target[o] /= w;
  }
}



////////////////////////////////////////////////////////////////////////////////
// Threshold element-wise
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_threshold_device( int W, int H,
				       float *target, float threshold,
				       float min_val, float max_val )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }

  int o = ox + oy*W;
  float w = target[o];
  if ( w >= threshold ) {
    target[o] = max_val;
  }
  else {
    target[o] = min_val;
  }
}





/////////////////////////////////////////////////////////////
//  STANDARD DERIVATIVES
/////////////////////////////////////////////////////////////

__global__ void cuda_compute_gradient_device( int W, int H, 
					      cuflt *u,
					      cuflt *px, cuflt *py )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  int o = oy*W + ox;

  // Step for each p equals gradient component of phi
  // Forward differences, Neumann
  // X
  cuflt gradX = 0.0f;
  if ( ox < W-1 ) {
    gradX = u[o+1] - u[o];
  }
  px[o] = gradX;
  // Y
  cuflt gradY = 0.0f;
  if ( oy < H-1 ) {
    gradY = u[o+W] - u[o];
  }
  py[o] = gradY;
}

__global__ void cuda_compute_divergence_device( int W, 
						int H,
						cuflt *px, cuflt *py,
						cuflt *d )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  int o = oy*W + ox;

  // Step equals divergence of p, backward differences, dirichlet
  cuflt div = px[o] + py[o];
  if ( ox>0 ) {
    div -= px[o-1];
  }
  if ( oy>0 ) {
    div -= py[o-W];
  }
  d[o] = div;
}

__global__ void cuda_compute_norm_device( int W, int H,
                cuflt *px1, cuflt *py1,
                cuflt *norm)
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  int o = oy*W + ox;

  // Total norm
  cuflt n = 0.0f;
  n += powf( px1[o], 2.0f ) + powf( py1[o], 2.0f );
  norm[o] = sqrtf( n );
}

__global__ void cuda_compute_norm_device( int W, int H,
                cuflt *px1, cuflt *px2, cuflt *px3,
                cuflt *norm)
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  int o = oy*W + ox;

  // Total norm
  cuflt n = 0.0;
  n += powf( px1[o], 2.0f ) + powf( px2[o], 2.0f ) + powf( px3[o], 2.0f );
  norm[o] = sqrtf( n );
}

__global__ void cuda_compute_norm_device( int W, int H,
                cuflt *x, cuflt *y,
                cuflt *z, cuflt *w,
                cuflt *norm)
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  int o = oy*W + ox;

  // Total norm
  cuflt n = 0.0;
  n += powf( x[o], 2.0f ) + powf( y[o], 2.0f ) + powf( z[o], 2.0f ) + pow( w[o], 2.0f );
  norm[o] = sqrtf( n );
}
__global__ void cuda_compute_norm_device( int W, int H,
                cuflt *px1, cuflt *py1,
                cuflt *px2, cuflt *py2,
                cuflt *px3, cuflt *py3,
                cuflt *norm)
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  int o = oy*W + ox;

  // Total norm
  cuflt n = 0.0;
  n += powf( px1[o], 2.0f ) + powf( px2[o], 2.0f ) + powf( px3[o], 2.0f );
  n += powf( py1[o], 2.0f ) + powf( py2[o], 2.0f ) + powf( py3[o], 2.0f );
  norm[o] = sqrtf( n );
}


/////////////////////////////////////////////////////////////
//  STANDARD ROF KERNELS
/////////////////////////////////////////////////////////////
__global__ void cuda_rof_primal_prox_step_device( int W, 
						  int H,
						  cuflt tau,
						  cuflt lambda,
						  cuflt *u,
						  cuflt *uq,
						  cuflt *f,
						  cuflt *px, cuflt *py )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  int o = oy*W + ox;

  // Step equals divergence of p, backward differences, dirichlet
  cuflt step = px[o] + py[o];
  if ( ox>0 ) {
    step -= px[o-1];
  }
  if ( oy>0 ) {
    step -= py[o-W];
  }
  // Projection onto allowed range
  cuflt unew = u[o] + tau * step;
  uq[o] = ( tau * f[o] + lambda * unew ) / ( tau + lambda );
}


__global__ void cuda_rof_primal_descent_step_device( int W, int H,
							    cuflt tau,
							    cuflt lambda,
							    cuflt *u,
							    cuflt *uq,
							    cuflt *f,
							    cuflt *px, cuflt *py )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  int o = oy*W + ox;

  // Step equals divergence of p, backward differences, dirichlet
  cuflt step = px[o] + py[o];
  if ( ox>0 ) {
    step -= px[o-1];
  }
  if ( oy>0 ) {
    step -= py[o-W];
  }
  // Data term step
  cuflt uv = u[o];
  step -= ( uv - f[o] ) / lambda;
  uq[o] = uv + tau * step;
}



__global__ void tv_l2_dual_step_device( int W, int H, float tstep,
					       float *u,
					       float *px, float *py )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  int o = oy*W + ox;

  // Copy to shared memory.
  extern __shared__ float bd[];
  int SW = blockDim.x+1;
  int ol = threadIdx.x + threadIdx.y*SW;
  bd[ol] = u[o];
  // Last row & col copies overlap
  if ( threadIdx.x==blockDim.x-1 ) {
    bd[blockDim.x + threadIdx.y*SW] = u[o+1];
  }
  if ( threadIdx.y==blockDim.y-1 ) {
    bd[threadIdx.x + blockDim.y*SW] = u[o+W];
  }
  __syncthreads();

  // Step for each p equals gradient component of phi
  // Forward differences, Neumann
  // X
  float px_new = px[o] + tstep*(ox<W-1)*(bd[ol+1] - bd[ol]);
  // Y
  float py_new = py[o] + tstep*(oy<H-1)*(bd[ol+SW] - bd[ol]);
  // Reproject X,Y
  float n = max( 1.0f, hypotf( px_new, py_new ));
  px[o] = px_new / n;
  py[o] = py_new / n;
}



__global__ void tv_primal_descent_step_device( int W, int H,
						      cuflt tau,
						      cuflt *u,
						      cuflt *v,
						      cuflt *px, cuflt *py )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  int o = oy*W + ox;

  // Step equals divergence of p, backward differences, dirichlet
  cuflt step = px[o] + py[o];
  if ( ox>0 ) {
    step -= px[o-1];
  }
  if ( oy>0 ) {
    step -= py[o-W];
  }
  // Projection onto allowed range
  v[o] = u[o] + tau * step;
}



/////////////////////////////////////////////////////////////
//  Full primal-dual algorithm kernel (one full iteration)
//  Adaptive maximum step size
/////////////////////////////////////////////////////////////
__global__ void cuda_linear_dual_prox_device( int W, int H,
					      float lambda,
					      float *uq,
					      float *px, float *py )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;

  // Perform dual step with extragradient variable
  // Step for each p equals gradient component of phi
  // Forward differences, Neumann
  // X
  float gradX = 0.0f;
  if ( ox < W-1 ) {
    gradX = uq[o+1] - uq[o];
  }
  float px_new = px[o] + 0.5f * gradX;
  // Y
  float gradY = 0.0f;
  if ( oy < H-1 ) {
    gradY = uq[o+W] - uq[o];
  }
  float py_new = py[o] + 0.5f * gradY;

  // Reproject (px,py)
  float n = hypotf( px_new, py_new );
  if ( n>lambda ) {
    px_new *= lambda / n;
    py_new *= lambda / n;
  }    
  px[o] = px_new;
  py[o] = py_new;
}


__global__ void cuda_linear_dual_prox_weighted_device( int W, int H,
						       float *g,
						       float *uq,
						       float *px, float *py )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;

  // weight determines local step size
  float w = g[o];
  float sigma = 0.5f / max( 0.1f, w );

  // Perform dual step with extragradient variable
  // Step for each p equals gradient component of phi
  // Forward differences, Neumann
  // X
  float gradX = 0.0f;
  if ( ox < W-1 ) {
    gradX = uq[o+1] - uq[o];
  }
  float px_new = px[o] + sigma * w * gradX;
  // Y
  float gradY = 0.0f;
  if ( oy < H-1 ) {
    gradY = uq[o+W] - uq[o];
  }
  float py_new = py[o] + sigma * w * gradY;

  // Reproject (px,py)
  float n = max( 1.0f, hypotf( px_new, py_new ));
  px[o] = px_new / n;
  py[o] = py_new / n;
}


__global__ void cuda_linear_primal_prox_extragradient_device( int W, int H,
							      float *u,
							      float *uq,
							      float *a,
							      float *px, float *py )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;

  // Step for u equals divergence of p, backward differences, dirichlet
  float step = px[o] + py[o];
  if ( ox>0 ) {
    step -= px[o-1];
  }
  if ( oy>0 ) {
    step -= py[o-W];
  }

  // Compute new u, projected
  float u_old = u[o];
  float u_new = max( 0.0f, min( 1.0f, u_old + 0.25f * (step - a[o]) ));
  // New extragradient variable
  uq[o] = 2.0f * u_new - u_old;
  u[o] = u_new;
}



__global__ void cuda_linear_primal_prox_weighted_extragradient_device( int W, int H,
								       float *g,
								       float *u,
								       float *uq,
								       float *a,
								       float *px, float *py )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;

  // weight determines local step size
  float w = g[o];
  float tau = 0.25f / max( 0.1f, w );

  // Step for u equals divergence of p, backward differences, dirichlet
  float step = px[o] + py[o];
  if ( ox>0 ) {
    step -= px[o-1];
  }
  if ( oy>0 ) {
    step -= py[o-W];
  }

  // Compute new u, projected
  float u_old = u[o];
  float u_new = max( 0.0f, min( 1.0f, u_old + tau * ( w * step - a[o] ) ));
  // New extragradient variable
  uq[o] = 2.0f * u_new - u_old;
  u[o] = u_new;
}




/////////////////////////////////////////////////////////////
//  STANDARD LINEAR KERNELS
/////////////////////////////////////////////////////////////
__global__ void cuda_linear_primal_prox_step_device( int W, int H,
						     float tau,
						     float *u,
						     float *uq,
						     float *a,
						     float *px, float *py )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;

  // Step equals divergence of p, backward differences, dirichlet
  float step = px[o] + py[o];
  if ( ox>0 ) {
    step -= px[o-1];
  }
  if ( oy>0 ) {
    step -= py[o-W];
  }
  // Projection onto allowed range
  uq[o] = u[o] + tau * ( step - a[o] );
}



///////////////////////////////////////////////////////////////////////////////////////////
// Multi-channel reprojections
///////////////////////////////////////////////////////////////////////////////////////////

__global__ void cuda_reproject_to_unit_ball_1d( int W, int H, cuflt *p )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  int o = IMUL( oy,W ) + ox;
  // Compute 1D reprojection
  cuflt pv = p[o];
  p[o] = pv / max( 1.0f, fabs( pv ));
}

__global__ void cuda_reproject_to_unit_ball_2d( int W, int H, cuflt *px, cuflt *py )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  int o = IMUL( oy,W ) + ox;
  // Compute 2D reprojection
  cuflt pxv = px[o];
  cuflt pyv = py[o];
  cuflt n = max( 1.0f, hypot( pxv, pyv ));
  px[o] = pxv / n;
  py[o] = pyv / n;
}

__global__ void cuda_reproject_to_ball_3d( int W, int H,
						  cuflt r, cuflt *p1, cuflt *p2, cuflt *p3 )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  int o = IMUL( oy,W ) + ox;
  // Compute 3D reprojection
  cuflt p1v = p1[o];
  cuflt p2v = p2[o];
  cuflt p3v = p3[o];
  cuflt n = max( 1.0f, r * sqrtf(powf( p1v, 2.0f ) + powf( p2v, 2.0f ) + powf( p3v, 2.0f )) );
  p1[o] = p1v / n;
  p2[o] = p2v / n;
  p3[o] = p3v / n;
}

__global__ void cuda_reproject_to_ball_2d( int W, int H,
						  cuflt r, cuflt *p1, cuflt *p2 )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  int o = IMUL( oy,W ) + ox;
  // Compute 2D reprojection
  cuflt p1v = p1[o];
  cuflt p2v = p2[o];
  cuflt n = hypotf( p1v, p2v );
  if ( n>r ) {
    p1[o] = p1v * r / n;
    p2[o] = p2v * r / n;
  }
}

__global__ void cuda_reproject_to_ball_1d( int W, int H,
					       cuflt r, cuflt *p1 )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  int o = IMUL( oy,W ) + ox;
  // Compute 1D reprojection
  cuflt p1v = p1[o];
  cuflt n = fabs(p1v);
  if ( n > r ) {
    p1[o] = r * p1v / n;
  }
}



__global__ void cuda_reproject_to_ball_2d( int W, int H, cuflt *g, cuflt *p1, cuflt *p2 )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy >= H ) {
    return;
  }
  int o = IMUL( oy,W ) + ox;
  // Compute 2D reprojection
  cuflt p1v = p1[o];
  cuflt p2v = p2[o];
  cuflt r = g[o];
  cuflt n = hypotf( p1v, p2v );
  if ( n>r ) {
    p1[o] = p1v * r / n;
    p2[o] = p2v * r / n;
  }
}


__global__ void cuda_reproject_to_ellipse( int W, int H,
						  float nx, float ny, // main axis direction
						  float r,            // small axis scale
						  float *px, float *py ) 
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy >= H ) {
    return;
  }
  int o = IMUL( oy,W ) + ox;

  // Rotate direction into x-Axis
  float pxv = px[o];
  float pyv = py[o];
  float qx =  nx*pxv + ny*pyv;
  float qy = -ny*pxv + nx*pyv;
  // Project onto unit ball
  qx = min( 1.0f, max( -1.0f, qx ));
  qy = 0.0f;
  // Rotate back
  px[o] = nx*qx - ny*qy;
  py[o] = ny*qx + nx*qy;
}


__global__ void cuda_reproject_to_ellipse( int W, int H,
						  float nx, float ny, // main axis direction
						  float r,            // small axis scale
						  float *a,            // main axis length (variable)
						  float *px, float *py ) 
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy >= H ) {
    return;
  }
  int o = IMUL( oy,W ) + ox;

  // Rotate direction into x-Axis
  float pxv = px[o];
  float pyv = py[o];
  float av = a[o];
  float qx =  nx*pxv + ny*pyv;
  float qy = -ny*pxv + nx*pyv;
  // Project onto unit ball
  qx = min( 1.0, max( -1.0, qx ));
  qy = min( av, max( -av, qy ));
  // Rotate back
  px[o] = nx*qx - ny*qy;
  py[o] = ny*qx + nx*qy;
}



__global__ void cuda_compute_largest_singular_value_device( int W, int H,
								   cuflt *px1, cuflt *py1,
								   cuflt *px2, cuflt *py2,
								   cuflt *px3, cuflt *py3,
								   cuflt *lsv )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  int o = IMUL( oy,W ) + ox;

  // Local vars
  cuflt a00 = px1[o];
  cuflt a10 = px2[o];
  cuflt a20 = px3[o];
  cuflt a01 = py1[o];
  cuflt a11 = py2[o];
  cuflt a21 = py3[o];

  // Compute A^T A
  cuflt d00 = a00*a00 + a10*a10 + a20*a20;
  cuflt d01 = a01*a00 + a11*a10 + a21*a20;
  cuflt d11 = a01*a01 + a11*a11 + a21*a21;

  // Compute larger Eigenvalue (= square of largest singular value)
  cuflt diff = d11 - d00;
  cuflt d = hypotf( diff, 2.0f*d01 );
  lsv[o] = sqrt( 0.5f * (d00 + d11 + d ));
}







///////////////////////////////////////////////////////////////////////////////////////////
// Interpolation and resampling
///////////////////////////////////////////////////////////////////////////////////////////

// Matrix upsampling
__global__ void cuda_upsample_matrix_device( int W, int H, // Hi-res size
					     int w, int h, // Lo-res size
					     float F,        // Scale factor
					     float *m,     // lo-res matrix
					     float *M )    // hi-res result
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }

  float px = float(ox) / F;
  float py = float(oy) / F;

  int o = ox + oy*W;
  M[o] = bilinear_interpolation( w,h, m, px, py );
}



////////////////////////////////////////////////////////////////////////////////
// Copy inside mask region
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_masked_copy_to_device( int W, int H, cuflt *s, cuflt *mask, cuflt *r )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;
  if ( mask[o] != 0.0f ) {
    r[o] = s[o];
  }
}


////////////////////////////////////////////////////////////////////////////////
// Compute weighted average of two channels
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_weighted_average_device( int W, int H, cuflt *s1, cuflt *s2, cuflt *mask, cuflt *r )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;
  float m = mask[o];
  r[o] = s1[o] * ( 1.0f - m ) + s2[o] * m;
}





/*********************************************************************
 ** EIGENSYSTEMS
 *********************************************************************/
__global__ void cuda_eigenvalues_symm_device( int W, int H,
					      float *a, float *b, float *c,
					      float *lmin, float *lmax )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;

  // Get structure tensor matrix
  cuflt d11 = a[o];
  cuflt d12 = b[o];
  cuflt d22 = c[o];

  // Compute Eigenvalues
  cuflt trace = d11 + d22;
  cuflt det = d11*d22 - d12*d12;
  cuflt d = sqrtf( 0.25f*trace*trace - det );
  lmax[o] = max( 0.0f, 0.5f * trace + d );
  lmin[o] = max( 0.0f, 0.5f * trace - d );
}






////////////////////////////////////////////////////////////////////////////////
// SLICES OF IMAGE STACKS
////////////////////////////////////////////////////////////////////////////////

// Horizontal slice, size W x N at scanline y
// Attn: set block and grid size to accomodate W x N threads
__global__ void cuda_stack_slice_H_device( int W, int H, int N,
					   int y,
					   float *stack, float *slice )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= N ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;
  slice[o] = stack[ ox + W * ( y + oy*H ) ];
}



// Vertical slice, size N x H at column x
// Attn: set block and grid size to accomodate N x H threads
__global__ void cuda_stack_slice_W_device( int W, int H, int N,
					   int x,
					   float *stack, float *slice )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= N || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,N ) + ox;
  slice[o] = stack[ x + W * ( oy + ox*H ) ];
}


