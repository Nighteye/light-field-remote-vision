/* -*-c++-*- */
/** \file kernels_vtv.cpp

    VTV kernels on grids

    Copyright (C) 2014 Bastian Goldluecke,
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
#include "compute_api_implementation_opencl.h"


const char *kernel_extragradient_step = "\n" \
  "__kernel void kernel_extragradient_step(                               \n" \
  "   const unsigned int W,                                               \n" \
  "   const unsigned int H,                                               \n" \
  "   const float theta,                                                  \n" \
  "   __global float* u,                                                  \n" \
  "   __global float* uq )                                                \n" \
  "{                                                                      \n" \
  "   int ox = get_global_id(0);                                          \n" \
  "   int oy = get_global_id(1);                                          \n" \
  "   if ( ox>=W || oy>=H ) return;                                       \n" \
  "   int o = oy*W + ox;                                                  \n" \
  "                                                                       \n" \
  "   float uv = u[o];                                                    \n" \
  "   float uqv = uq[o];                                                  \n" \
  "   uq[o] = uv + theta * ( uqv - uv );                                  \n" \
  "   u[o] = uqv;                                                         \n" \
  "}                                                                      \n" \
  "                                                                       \n";

/// Kernel for extragradient step
void coco::kernel_extragradient_step( const compute_grid *G,
				      const float theta,
				      compute_buffer &uq,
				      compute_buffer &u )
{
  // Create kernel
  static cl_kernel kernel = NULL;
  if ( !kernel ) {
    kernel = kernel_compile( G->engine(),
			     "kernel_extragradient_step",
			     ::kernel_extragradient_step );
    assert( kernel != NULL );
  }

  // Launch kernel
  dim3 dimGrid;
  dim3 dimBlock;
  kernel_configure( G, dimGrid, dimBlock );
  
  CL_COMMAND_QUEUE( G->engine() );

  int W = G->W();
  int H = G->H();
  cl_mem m_uq( uq );
  cl_mem m_u( u );

  CL_SAFE_CALL( clSetKernelArg(kernel, 0, sizeof(int), &W ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 1, sizeof(int), &H ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 2, sizeof(cl_float), &theta ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 3, sizeof(cl_mem), &m_uq ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 4, sizeof(cl_mem), &m_u ));

  CL_SAFE_CALL( clEnqueueNDRangeKernel(commands, kernel, 2, NULL,
				       dimGrid, dimBlock, 0, NULL, NULL) );
}





const char *kernel_rof_primal_prox = "\n" \
  "__kernel void kernel_rof_primal_prox(                                  \n" \
  "   const unsigned int W,                                               \n" \
  "   const unsigned int H,                                               \n" \
  "   const float tau,                                                    \n" \
  "   __global float* u,                                                  \n" \
  "   const float lambda,                                                 \n" \
  "   __global const float* f )                                           \n" \
  "{                                                                      \n" \
  "   int ox = get_global_id(0);                                          \n" \
  "   int oy = get_global_id(1);                                          \n" \
  "   if ( ox>=W || oy>=H ) return;                                       \n" \
  "   int o = oy*W + ox;                                                  \n" \
  "                                                                       \n" \
  "   // Projection onto allowed range                                    \n" \
  "   float unew = ( tau * f[o] + lambda * u[o] ) / ( tau + lambda );     \n" \
  "   u[o] = min( 1.0f, max( 0.0f, unew ));                               \n" \
  "}                                                                      \n" \
  "                                                                       \n";

/// Kernel for ROF functional, compute primal prox operator
void coco::kernel_rof_primal_prox( const compute_grid *G,
				   const float tau,
				   compute_buffer &u,
				   const float lambda,
				   const compute_buffer &f )
{
  // Create kernel
  static cl_kernel kernel = NULL;
  if ( !kernel ) {
    kernel = kernel_compile( G->engine(),
			     "kernel_rof_primal_prox",
			     ::kernel_rof_primal_prox );
    assert( kernel != NULL );
  }

  // Launch kernel
  dim3 dimGrid;
  dim3 dimBlock;
  kernel_configure( G, dimGrid, dimBlock );
  
  CL_COMMAND_QUEUE( G->engine() );

  int W = G->W();
  int H = G->H();
  cl_mem m_u( u );
  cl_mem m_f( f );

  CL_SAFE_CALL( clSetKernelArg(kernel, 0, sizeof(int), &W ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 1, sizeof(int), &H ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 2, sizeof(cl_float), &tau ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 3, sizeof(cl_mem), &m_u ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 4, sizeof(cl_float), &lambda ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 5, sizeof(cl_mem), &m_f ));

  CL_SAFE_CALL( clEnqueueNDRangeKernel(commands, kernel, 2, NULL,
				       dimGrid, dimBlock, 0, NULL, NULL) );
}





const char *kernel_gradient_operator_primal_step = "\n" \
  "__kernel void kernel_function(                                         \n" \
  "   const unsigned int W,                                               \n" \
  "   const unsigned int H,                                               \n" \
  "   const float tau,                                                    \n" \
  "   __global float* u,                                                  \n" \
  "   __global const float* px,                                           \n" \
  "   __global const float* py )                                          \n" \
  "{                                                                      \n" \
  "  int ox = get_global_id(0);                                          \n" \
  "  int oy = get_global_id(1);                                          \n" \
  "  if ( ox>=W || oy>=H ) return;                                       \n" \
  "  int o = oy*W + ox;                                                  \n" \
  "                                                                      \n" \
  "  // Step equals divergence of p, backward differences, dirichlet     \n" \
  "  float step = px[o] + py[o];                                         \n" \
  "  if ( ox>0 ) {                                                       \n" \
  "    step -= px[o-1];                                                  \n" \
  "  }                                                                   \n" \
  "  if ( oy>0 ) {                                                       \n" \
  "    step -= py[o-W];                                                  \n" \
  "  }                                                                   \n" \
  "  u[o] += tau * step;                                                 \n" \
  "}                                                                     \n" \
  "                                                                      \n";

// Gradient operator step kernels
void coco::kernel_gradient_operator_primal_step( const compute_grid *G,
						 float tau,
						 compute_buffer &u,
						 const compute_buffer &px, const compute_buffer &py )
{
  // Create kernel
  static cl_kernel kernel = NULL;
  if ( !kernel ) {
    kernel = kernel_compile( G->engine(),
			     "kernel_function",
			     ::kernel_gradient_operator_primal_step );
    assert( kernel != NULL );
  }

  // Launch kernel
  dim3 dimGrid;
  dim3 dimBlock;
  kernel_configure( G, dimGrid, dimBlock );
  
  CL_COMMAND_QUEUE( G->engine() );

  int W = G->W();
  int H = G->H();
  cl_mem m_u( u );
  cl_mem m_px( px );
  cl_mem m_py( py );

  CL_SAFE_CALL( clSetKernelArg(kernel, 0, sizeof(int), &W ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 1, sizeof(int), &H ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 2, sizeof(cl_float), &tau ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 3, sizeof(cl_mem), &m_u ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 4, sizeof(cl_mem), &m_px ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 5, sizeof(cl_mem), &m_py ));

  CL_SAFE_CALL( clEnqueueNDRangeKernel(commands, kernel, 2, NULL,
				       dimGrid, dimBlock, 0, NULL, NULL) );
}


const char *kernel_gradient_operator_dual_step = "\n" \
  "__kernel void kernel_function(                                        \n" \
  "   const unsigned int W,                                              \n" \
  "   const unsigned int H,                                              \n" \
  "   const float tstep,                                                 \n" \
  "   __global const float* u,                                           \n" \
  "   __global float* px,                                                \n" \
  "   __global float* py )                                               \n" \
  "{                                                                     \n" \
  "  int ox = get_global_id(0);                                          \n" \
  "  int oy = get_global_id(1);                                          \n" \
  "  if ( ox>=W || oy>=H ) return;                                       \n" \
  "  int o = oy*W + ox;                                                  \n" \
  "                                                                      \n" \
  "  // Step for each p equals gradient component of phi                 \n" \
  "  // Forward differences, Neumann                                     \n" \
  "  // X                                                                \n" \
  "  float grad = 0.0;                                                   \n" \
  "  if ( ox < W-1 ) {                                                   \n" \
  "    grad = u[o+1] - u[o];                                             \n" \
  "  }                                                                   \n" \
  "  float px_new = px[o] + tstep * grad;                                \n" \
  "  // Y                                                                \n" \
  "  grad = 0.0;                                                         \n" \
  "  if ( oy < H-1 ) {                                                   \n" \
  "    grad = u[o+W] - u[o];                                             \n" \
  "  }                                                                   \n" \
  "  float py_new = py[o] + tstep * grad;                                \n" \
  "  // Reprojection is combined for all channels                        \n" \
  "  px[o] = px_new;                                                     \n" \
  "  py[o] = py_new;                                                     \n" \
  "}                                                                     \n" \
  "                                                                      \n";


void coco::kernel_gradient_operator_dual_step( const compute_grid *G, float tstep,
					       const compute_buffer &u,
					       compute_buffer &px, compute_buffer &py )
{
  // Create kernel
  static cl_kernel kernel = NULL;
  if ( !kernel ) {
    kernel = kernel_compile( G->engine(),
			     "kernel_function",
			     ::kernel_gradient_operator_dual_step );
    assert( kernel != NULL );
  }

  // Launch kernel
  dim3 dimGrid;
  dim3 dimBlock;
  kernel_configure( G, dimGrid, dimBlock );
  
  CL_COMMAND_QUEUE( G->engine() );

  int W = G->W();
  int H = G->H();
  cl_mem m_u( u );
  cl_mem m_px( px );
  cl_mem m_py( py );

  CL_SAFE_CALL( clSetKernelArg(kernel, 0, sizeof(int), &W ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 1, sizeof(int), &H ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 2, sizeof(cl_float), &tstep ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 3, sizeof(cl_mem), &m_u ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 4, sizeof(cl_mem), &m_px ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 5, sizeof(cl_mem), &m_py ));

  CL_SAFE_CALL( clEnqueueNDRangeKernel(commands, kernel, 2, NULL,
				       dimGrid, dimBlock, 0, NULL, NULL) );
}

