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

#include "../compute_api/kernels_vtv.h"
#include "compute_api_implementation_cuda.h"

static __global__ void kernel_extragradient_step( int W, 
						  int H,
						  float theta,
						  float *u,
						  float *uq )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;
  float uv = u[o];
  float uqv = uq[o];
  uq[o] = uv + theta * ( uqv - uv );
  u[o] = uqv;
}


/// Kernel for extragradient step
void coco::kernel_extragradient_step( const compute_grid *G,
				      const float theta,
				      compute_buffer &uq,
				      compute_buffer &u )
{
  dim3 dimGrid, dimBlock;
  kernel_configure( G, dimGrid, dimBlock );
  ::kernel_extragradient_step<<< dimGrid, dimBlock >>>
      ( G->W(), G->H(),
	theta, uq, u );
}


  /// Kernel for ROF functional, compute exact solution for primal variable
  /*
static __global__ void kernel_rof_functional_primal_exact_solution( int W, int H,
								   float lambda,
								   float *u,
								   const float *f,
								   const float *px, const float *py )
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
  // Projecton onto allowed range
  u[o] = f[o] + lambda * step;
}

  void kernel_rof_functional_primal_exact_solution( const compute_grid *G,
						    const float lambda,
						    const float *u,
						    const float *f,
						    float *px, float *py );
  */



static __global__ void kernel_rof_primal_prox( int W, 
					       int H,
					       float tau,
					       float *u,
					       float lambda,
					       const float *f )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  int o = oy*W + ox;

  // Projection onto allowed range
  float unew = ( tau * f[o] + lambda * u[o] ) / ( tau + lambda );
  u[o] = min( 1.0f, max( 0.0f, unew ));
}



/// Kernel for ROF functional, compute primal prox operator
void coco::kernel_rof_primal_prox( const compute_grid *G,
				   const float tau,
				   compute_buffer &u,
				   const float lambda,
				   const compute_buffer &f )
{
  dim3 dimGrid, dimBlock;
  kernel_configure( G, dimGrid, dimBlock );
  ::kernel_rof_primal_prox<<< dimGrid, dimBlock >>>
    ( G->W(), G->H(),
      tau, u, lambda, f );
}


static __global__ void kernel_gradient_operator_primal_step( int W, 
							     int H,
							     float tau,
							     float *u,
							     const float *px, const float *py )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox >= W || oy >= H ) {
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
  u[o] += tau * step;
}


// Gradient operator step kernels
void coco::kernel_gradient_operator_primal_step( const compute_grid *G,
						 float tau,
						 compute_buffer &u,
						 const compute_buffer &px, const compute_buffer &py )
{
  dim3 dimGrid, dimBlock;
  kernel_configure( G, dimGrid, dimBlock );
  ::kernel_gradient_operator_primal_step<<< dimGrid, dimBlock >>>
      ( G->W(), G->H(),
	tau, u, px, py );
}


static __global__ void kernel_gradient_operator_dual_step( int W, int H, float tstep,
							   const float *u,
							   float *px, float *py )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>= H ) {
    return;
  }
  int o = oy*W + ox;

  // Step for each p equals gradient component of phi
  // Forward differences, Neumann
  // X
  float grad = 0.0;
  if ( ox < W-1 ) {
    grad = u[o+1] - u[o];
  }
  float px_new = px[o] + tstep * grad;
  // Y
  grad = 0.0;
  if ( oy < H-1 ) {
    grad = u[o+W] - u[o];
  }
  float py_new = py[o] + tstep * grad;
  // Reprojection is combined for all channels
  px[o] = px_new;
  py[o] = py_new;
}


void coco::kernel_gradient_operator_dual_step( const compute_grid *G, float tstep,
					       const compute_buffer &u,
					       compute_buffer &px, compute_buffer &py )
{
  dim3 dimGrid, dimBlock;
  kernel_configure( G, dimGrid, dimBlock );
  ::kernel_gradient_operator_dual_step<<< dimGrid, dimBlock >>>
      ( G->W(), G->H(),
	tstep, u, px, py );
}
