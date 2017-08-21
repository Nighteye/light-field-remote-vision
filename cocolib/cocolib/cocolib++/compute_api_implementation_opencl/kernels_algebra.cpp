/* -*-c++-*- */
/** \file kernels_algebra.cpp

    Basic algebraic computations on grids

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

#include "../compute_api/kernels_algebra.h"
#include "compute_api_implementation_opencl.h"


////////////////////////////////////////////////////////////////////////////////
// Grid initialization
////////////////////////////////////////////////////////////////////////////////
void coco::kernel_set_all( const compute_grid *G, compute_buffer &dst, const float value );

////////////////////////////////////////////////////////////////////////////////
// Add first argument to second
////////////////////////////////////////////////////////////////////////////////
void coco::kernel_add_to( const compute_grid *G, const compute_buffer &src, compute_buffer &dst );
void coco::kernel_add_to( const compute_grid* G, const float v, compute_buffer &dst );

////////////////////////////////////////////////////////////////////////////////
// Subtract first argument from second
////////////////////////////////////////////////////////////////////////////////

const char *kernel_subtract_from = "\n" \
  "__kernel void kernel_subtract_from(                                    \n" \
  "   const unsigned int W,                                               \n" \
  "   const unsigned int H,                                               \n" \
  "   __global const float* src,                                          \n" \
  "   __global float* dst )                                               \n" \
  "{                                                                      \n" \
  "   int ox = get_global_id(0);                                          \n" \
  "   int oy = get_global_id(1);                                          \n" \
  "   if ( ox>=W || oy>=H ) return;                                       \n" \
  "   int o = oy*W + ox;                                                  \n" \
  "   dst[o] -= src[o];                                                   \n" \
  "}                                                                      \n" \
  "\n";


void coco::kernel_subtract_from( const compute_grid* G, const compute_buffer &src, compute_buffer &dst )
{
  // Create kernel
  static cl_kernel kernel = NULL;
  if ( !kernel ) {
    kernel = kernel_compile( G->engine(),
			     "kernel_subtract_from",
			     ::kernel_subtract_from );
    assert( kernel != NULL );
  }

  // Launch kernel
  dim3 dimGrid;
  dim3 dimBlock;
  kernel_configure( G, dimGrid, dimBlock );
  
  CL_COMMAND_QUEUE( G->engine() );

  int W = G->W();
  int H = G->H();
  cl_mem m_src( src );
  cl_mem m_dst( dst );

  CL_SAFE_CALL( clSetKernelArg(kernel, 0, sizeof(int), &W ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 1, sizeof(int), &H ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 2, sizeof(cl_mem), &m_src ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 3, sizeof(cl_mem), &m_dst ));

  CL_SAFE_CALL( clEnqueueNDRangeKernel(commands, kernel, 2, NULL,
				       dimGrid, dimBlock, 0, NULL, NULL) );
}


////////////////////////////////////////////////////////////////////////////////
// Divide first argument by second
////////////////////////////////////////////////////////////////////////////////
void coco::kernel_divide_by( const compute_grid* G, compute_buffer &dst, const compute_buffer &src );

////////////////////////////////////////////////////////////////////////////////
// Multiply first argument with second
////////////////////////////////////////////////////////////////////////////////
void coco::kernel_multiply_with( const compute_grid* G, compute_buffer &dst, const compute_buffer &src );

////////////////////////////////////////////////////////////////////////////////
// Multiply first argument with second, store in third
////////////////////////////////////////////////////////////////////////////////
void coco::kernel_multiply( const compute_grid* G, const compute_buffer &m1, const compute_buffer &m2, compute_buffer &r );

////////////////////////////////////////////////////////////////////////////////
// Add scaled first argument to second
////////////////////////////////////////////////////////////////////////////////
void coco::kernel_multiply_and_add_to( const compute_grid* G, const compute_buffer &src, const compute_buffer &t, compute_buffer &dst );



////////////////////////////////////////////////////////////////////////////////
// Compute linear combination of two arguments
////////////////////////////////////////////////////////////////////////////////
const char *kernel_linear_combination = "\n" \
  "__kernel void kernel_linear_combination(                               \n" \
  "   const unsigned int W,                                               \n" \
  "   const unsigned int H,                                               \n" \
  "   const float w0,                                                     \n" \
  "   __global const float* src0,                                         \n" \
  "   const float w1,                                                     \n" \
  "   __global const float* src1,                                         \n" \
  "   __global float* dst )                                               \n" \
  "{                                                                      \n" \
  "   int ox = get_global_id(0);                                          \n" \
  "   int oy = get_global_id(1);                                          \n" \
  "   if ( ox>=W || oy>=H ) return;                                       \n" \
  "   int o = oy*W + ox;                                                  \n" \
  "   dst[o] = w0*src0[o] + w1*src1[o];                                   \n" \
  "}                                                                      \n" \
  "\n";


void coco::kernel_linear_combination( const compute_grid* G,
				      const float w0, const compute_buffer &src0,
				      const float w1, const compute_buffer &src1,
				      compute_buffer &dst )
{
  // Create kernel
  static cl_kernel kernel = NULL;
  if ( !kernel ) {
    kernel = kernel_compile( G->engine(),
			     "kernel_linear_combination",
			     ::kernel_linear_combination );
    assert( kernel != NULL );
  }

  // Launch kernel
  dim3 dimGrid;
  dim3 dimBlock;
  kernel_configure( G, dimGrid, dimBlock );
  
  CL_COMMAND_QUEUE( G->engine() );

  int W = G->W();
  int H = G->H();
  cl_mem m_src0( src0 );
  cl_mem m_src1( src1 );
  cl_mem m_dst( dst );

  CL_SAFE_CALL( clSetKernelArg(kernel, 0, sizeof(int), &W ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 1, sizeof(int), &H ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 2, sizeof(cl_float), &w0 ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 3, sizeof(cl_mem), &m_src0 ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 4, sizeof(cl_float), &w1 ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 5, sizeof(cl_mem), &m_src1 ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 6, sizeof(cl_mem), &m_dst ));

  CL_SAFE_CALL( clEnqueueNDRangeKernel(commands, kernel, 2, NULL,
				       dimGrid, dimBlock, 0, NULL, NULL) );
}


////////////////////////////////////////////////////////////////////////////////
// Scale array by argument
////////////////////////////////////////////////////////////////////////////////
const char *kernel_scale = "\n" \
  "__kernel void kernel_scale(                                            \n" \
  "   const unsigned int W,                                               \n" \
  "   const unsigned int H,                                               \n" \
  "   __global float* dst,                                                \n" \
  "   const float t )                                                     \n" \
  "{                                                                      \n" \
  "   int ox = get_global_id(0);                                          \n" \
  "   int oy = get_global_id(1);                                          \n" \
  "   if ( ox>=W || oy>=H ) return;                                       \n" \
  "   int o = oy*W + ox;                                                  \n" \
  "   dst[o] *= t;                                                        \n" \
  "}                                                                      \n" \
  "\n";


void coco::kernel_scale( const compute_grid* G, compute_buffer &dst, const float t )
{
  // Create kernel
  static cl_kernel kernel = NULL;
  if ( !kernel ) {
    kernel = kernel_compile( G->engine(),
			     "kernel_scale",
			     ::kernel_scale );
    assert( kernel != NULL );
  }

  // Launch kernel
  dim3 dimGrid;
  dim3 dimBlock;
  kernel_configure( G, dimGrid, dimBlock );
  
  CL_COMMAND_QUEUE( G->engine() );

  int W = G->W();
  int H = G->H();
  cl_mem m_dst( dst );

  CL_SAFE_CALL( clSetKernelArg(kernel, 0, sizeof(int), &W ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 1, sizeof(int), &H ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 2, sizeof(cl_mem), &m_dst ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 3, sizeof(cl_float), &t ));

  CL_SAFE_CALL( clEnqueueNDRangeKernel(commands, kernel, 2, NULL,
				       dimGrid, dimBlock, 0, NULL, NULL) );
}



////////////////////////////////////////////////////////////////////////////////
// Square array element-wise
////////////////////////////////////////////////////////////////////////////////
void coco::kernel_square( const compute_grid* G, compute_buffer &dst );


////////////////////////////////////////////////////////////////////////////////
// Arbitrary power element-wise
////////////////////////////////////////////////////////////////////////////////
void coco::kernel_pow( const compute_grid* G, compute_buffer &dst, float exponent );


////////////////////////////////////////////////////////////////////////////////
// Pointwise absolute value
////////////////////////////////////////////////////////////////////////////////
void coco::kernel_abs( const compute_grid* G, compute_buffer &dst );
  

////////////////////////////////////////////////////////////////////////////////
// Clamp to range element-wise
////////////////////////////////////////////////////////////////////////////////
const char *kernel_clamp = "\n" \
  "__kernel void kernel_clamp(                                            \n" \
  "   const unsigned int W,                                               \n" \
  "   const unsigned int H,                                               \n" \
  "   __global float* dst,                                                \n" \
  "   const float m,                                                      \n" \
  "   const float M )                                                     \n" \
  "{                                                                      \n" \
  "   int ox = get_global_id(0);                                          \n" \
  "   int oy = get_global_id(1);                                          \n" \
  "   if ( ox>=W || oy>=H ) return;                                       \n" \
  "   int o = oy*W + ox;                                                  \n" \
  "   dst[o] = max( m, min( M,dst[o] ));                                  \n" \
  "}                                                                      \n" \
  "\n";


void coco::kernel_clamp( const compute_grid* G, compute_buffer &dst, float m, float M )
{
  // Create kernel
  static cl_kernel kernel = NULL;
  if ( !kernel ) {
    kernel = kernel_compile( G->engine(),
			     "kernel_clamp",
			     ::kernel_clamp );
    assert( kernel != NULL );
  }

  // Launch kernel
  dim3 dimGrid;
  dim3 dimBlock;
  kernel_configure( G, dimGrid, dimBlock );
  
  CL_COMMAND_QUEUE( G->engine() );

  int W = G->W();
  int H = G->H();
  cl_mem m_dst( dst );

  CL_SAFE_CALL( clSetKernelArg(kernel, 0, sizeof(int), &W ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 1, sizeof(int), &H ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 2, sizeof(cl_mem), &m_dst ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 3, sizeof(cl_float), &m ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 4, sizeof(cl_float), &M ));

  CL_SAFE_CALL( clEnqueueNDRangeKernel(commands, kernel, 2, NULL,
				       dimGrid, dimBlock, 0, NULL, NULL) );
}

  

////////////////////////////////////////////////////////////////////////////////
// Threshold element-wise
////////////////////////////////////////////////////////////////////////////////
void coco::kernel_threshold( const compute_grid* G,
			     compute_buffer &target,
			     float threshold,
			     float min_val, float max_val );


  

  /* WILL PROBABLY NOT BE IMPLEMENTED, OR SOMEWHERE ELSE

  /////////////////////////////////////////////////////////////
  //  Standard derivative kernels
  /////////////////////////////////////////////////////////////
  void kernel_compute_gradient( const compute_grid* G, 
				compute_buffer &u,
				compute_buffer &px, compute_buffer &py );

  /////////////////////////////////////////////////////////////
  //  Norms
  /////////////////////////////////////////////////////////////
  void kernel_compute_norm( const compute_grid* G,
			    compute_buffer &x, compute_buffer &y,
			    compute_buffer &norm);
  void kernel_compute_norm( const compute_grid* G,
			    compute_buffer &x, compute_buffer &y, compute_buffer &z,
			    compute_buffer &norm);
  void kernel_compute_norm( const compute_grid* G,
			    compute_buffer &x, compute_buffer &y,
			    compute_buffer &z, compute_buffer &w,
			    compute_buffer &norm);
  void kernel_compute_norm( const compute_grid* G,
			    compute_buffer &px1, compute_buffer &py1,
			    compute_buffer &px2, compute_buffer &py2,
			    compute_buffer &px3, compute_buffer &py3,
			    compute_buffer &norm);
  
  /////////////////////////////////////////////////////////////
  //  STANDARD ROF KERNELS
  /////////////////////////////////////////////////////////////
  void kernel_rof_primal_prox_step( int W, 
				    int H,
				    float tau,
				    float lambda,
				    compute_buffer &u,
				    compute_buffer &uq,
				    compute_buffer &f,
				    compute_buffer &px, compute_buffer &py );
  
  void kernel_rof_primal_descent_step( const compute_grid* G,
				       float tau,
				       float lambda,
				       compute_buffer &u,
				       compute_buffer &uq,
				       compute_buffer &f,
				       compute_buffer &px, compute_buffer &py );


  void tv_l2_dual_step( const compute_grid* G, float tstep,
			compute_buffer &u,
			compute_buffer &px, compute_buffer &py );
  
  
  void tv_primal_descent_step( const compute_grid* G,
			       float tau,
			       compute_buffer &u,
			       compute_buffer &v,
			       compute_buffer &px, compute_buffer &py );
  
  
  /////////////////////////////////////////////////////////////
  //  STANDARD LINEAR KERNELS
  /////////////////////////////////////////////////////////////
  void kernel_linear_primal_prox_step( const compute_grid* G,
				       float tau,
				       compute_buffer &u,
				       compute_buffer &uq,
				       compute_buffer &a,
				       compute_buffer &px, compute_buffer &py );



  ///////////////////////////////////////////////////////////////////////////////////////////
  // Multi-channel reprojections
  ///////////////////////////////////////////////////////////////////////////////////////////
  
  void kernel_reproject_to_unit_ball_1d( const compute_grid* G, compute_buffer &p );
  void kernel_reproject_to_unit_ball_2d( const compute_grid* G, compute_buffer &px, compute_buffer &py );
  void kernel_reproject_to_ball_3d( const compute_grid* G,
				    float r, compute_buffer &p1, compute_buffer &p2, compute_buffer &p3 );
  void kernel_reproject_to_ball_2d( const compute_grid* G,
				    float r, compute_buffer &p1, compute_buffer &p2 );
  void kernel_reproject_to_ball_1d( const compute_grid* G,
				    float r, compute_buffer &p1 );
  void kernel_reproject_to_ball_2d( const compute_grid* G, compute_buffer &g, compute_buffer &p1, compute_buffer &p2 );
  void kernel_reproject_to_ellipse( const compute_grid* G,
				    float nx, float ny, // main axis direction
				    float r,            // small axis scale
				    compute_buffer &px, compute_buffer &py );
  void kernel_reproject_to_ellipse( const compute_grid* G,
				    float nx, float ny, // main axis direction
				    float r,            // small axis scale
				    compute_buffer &a,            // main axis length (variable)
				    compute_buffer &px, compute_buffer &py );
  void kernel_compute_largest_singular_value( const compute_grid* G,
					      compute_buffer &px1, compute_buffer &py1,
					      compute_buffer &px2, compute_buffer &py2,
					      compute_buffer &px3, compute_buffer &py3,
					      compute_buffer &lsv );
  

  /////////////////////////////////////////////////////////////
  //  Full primal-dual algorithm kernel (one full iteration)
  //  Adaptive maximum step size
  /////////////////////////////////////////////////////////////
  void kernel_linear_dual_prox( const compute_grid* G,
				float lambda,
				compute_buffer &uq,
				compute_buffer &px, compute_buffer &py );
  void kernel_linear_dual_prox_weighted( const compute_grid* G,
					 compute_buffer &g,
					 compute_buffer &uq,
					 compute_buffer &px, compute_buffer &py );
  void kernel_linear_primal_prox_extragradient( const compute_grid* G,
						compute_buffer &u,
						compute_buffer &uq,
						compute_buffer &a,
						compute_buffer &px, compute_buffer &py );
  void kernel_linear_primal_prox_weighted_extragradient( const compute_grid* G,
							 compute_buffer &g,
							 compute_buffer &u,
							 compute_buffer &uq,
							 compute_buffer &a,
							 compute_buffer &px, compute_buffer &py );



  ///////////////////////////////////////////////////////////////////////////////////////////
  // Interpolation and resampling
  ///////////////////////////////////////////////////////////////////////////////////////////
  
  // Matrix upsampling
  void kernel_upsample_matrix( const compute_grid* G, // Hi-res size
			       int w, int h, // Lo-res size
			       float F,        // Scale factor
			       compute_buffer &m,     // lo-res matrix
			       compute_buffer &M );    // hi-res result
  

  ////////////////////////////////////////////////////////////////////////////////
  // Copy inside mask region
  ////////////////////////////////////////////////////////////////////////////////
  void kernel_masked_copy_to( const compute_grid* G, compute_buffer &s, compute_buffer &mask, compute_buffer &r );
  

  ////////////////////////////////////////////////////////////////////////////////
  // Compute weighted average of two channels
  ////////////////////////////////////////////////////////////////////////////////
  void kernel_weighted_average( const compute_grid* G, compute_buffer &s1, compute_buffer &s2, compute_buffer &mask, compute_buffer &r );
  



  ////////////////////////////////////////////////////////////////////////////////
  // EIGENSYSTEMS
  ////////////////////////////////////////////////////////////////////////////////
  void kernel_eigenvalues_symm( const compute_grid* G,
				compute_buffer &a, compute_buffer &b, compute_buffer &c,
				compute_buffer &lmin, compute_buffer &lmax );




  ////////////////////////////////////////////////////////////////////////////////
  // SLICES OF IMAGE STACKS
  ////////////////////////////////////////////////////////////////////////////////
  
  // Horizontal slice, size W x N at scanline y
  // Attn: set block and grid size to accomodate W x N threads
  void kernel_stack_slice_H( const compute_grid* G, int N,
			     int y,
			     compute_buffer &stack, compute_buffer &slice );
  
  
  // Vertical slice, size N x H at column x
  // Attn: set block and grid size to accomodate N x H threads
  void kernel_stack_slice_W( const compute_grid* G, int N,
			     int x,
			     compute_buffer &stack, compute_buffer &slice );
  */

 

