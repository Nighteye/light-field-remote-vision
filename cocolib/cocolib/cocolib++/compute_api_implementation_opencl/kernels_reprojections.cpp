/* -*-c++-*- */
/** \file kernels_reprojections.cu

    Reprojection kernels on grids

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

#include "../compute_api/kernels_reprojections.h"
#include "compute_api_implementation_opencl.h"


const char *kernel_reproject_euclidean_1D = "\n" \
  "__kernel void kernel_function(                                        \n" \
  "   const unsigned int W,                                              \n" \
  "   const unsigned int H,                                              \n" \
  "   const float r,                                                     \n" \
  "   __global float* p )                                                \n" \
  "{                                                                     \n" \
  "  int ox = get_global_id(0);                                          \n" \
  "  int oy = get_global_id(1);                                          \n" \
  "  if ( ox>=W || oy>=H ) return;                                       \n" \
  "  int o = oy*W + ox;                                                  \n" \
  "                                                                      \n" \
  "  // Equivalent to clamping to range [-r,r]                           \n" \
  "  p[o] = max( -r, min( r,p[o] ));                                     \n" \
  "}                                                                     \n";


// Reprojection to Euclidean ball with given radius
void coco::kernel_reproject_euclidean_1D( const compute_grid *G,
					     const float radius,
					     compute_buffer &px1 )
{
  // Create kernel
  static cl_kernel kernel = NULL;
  if ( !kernel ) {
    kernel = kernel_compile( G->engine(),
			     "kernel_function",
			     ::kernel_reproject_euclidean_1D );
    assert( kernel != NULL );
  }

  // Launch kernel
  dim3 dimGrid;
  dim3 dimBlock;
  kernel_configure( G, dimGrid, dimBlock );
  
  CL_COMMAND_QUEUE( G->engine() );

  int W = G->W();
  int H = G->H();
  cl_mem m_px1( px1 );

  CL_SAFE_CALL( clSetKernelArg(kernel, 0, sizeof(int), &W ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 1, sizeof(int), &H ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 2, sizeof(cl_float), &radius ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 3, sizeof(cl_mem), &m_px1 ));

  CL_SAFE_CALL( clEnqueueNDRangeKernel(commands, kernel, 2, NULL,
				       dimGrid, dimBlock, 0, NULL, NULL) );
}



// Reprojection for RGB, TV_F
const char *kernel_reproject_euclidean_2D = "\n" \
  "__kernel void kernel_function(                                        \n" \
  "   const unsigned int W,                                              \n" \
  "   const unsigned int H,                                              \n" \
  "   const float r,                                                     \n" \
  "   __global float* px1,                                               \n" \
  "   __global float* py1 )                                              \n" \
  "{                                                                     \n" \
  "  int ox = get_global_id(0);                                          \n" \
  "  int oy = get_global_id(1);                                          \n" \
  "  if ( ox>=W || oy>=H ) return;                                       \n" \
  "  int o = oy*W + ox;                                                  \n" \
  "                                                                      \n" \
  "  // Local vars                                                       \n" \
  "  float a1 = px1[o];                                                  \n" \
  "  float a2 = py1[o];                                                  \n" \
  "                                                                      \n" \
  "  // Project                                                          \n" \
  "  float n = hypot( a1, a2 );                                          \n" \
  "  if ( n > r ) {                                                      \n" \
  "    n = r/n;                                                          \n" \
  "    px1[o] = a1 * n;                                                  \n" \
  "    py1[o] = a2 * n;                                                  \n" \
  "  }                                                                   \n" \
  "}                                                                     \n";

// Reprojection to Euclidean ball with given radius
void coco::kernel_reproject_euclidean_2D( const compute_grid *G,
					  const float radius,
					  compute_buffer &px1, compute_buffer &py1 )
{
  // Create kernel
  static cl_kernel kernel = NULL;
  if ( !kernel ) {
    kernel = kernel_compile( G->engine(),
			     "kernel_function",
			     ::kernel_reproject_euclidean_2D );
    assert( kernel != NULL );
  }

  // Launch kernel
  dim3 dimGrid;
  dim3 dimBlock;
  kernel_configure( G, dimGrid, dimBlock );
  
  CL_COMMAND_QUEUE( G->engine() );

  int W = G->W();
  int H = G->H();
  cl_mem m_px1( px1 );
  cl_mem m_py1( py1 );

  CL_SAFE_CALL( clSetKernelArg(kernel, 0, sizeof(int), &W ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 1, sizeof(int), &H ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 2, sizeof(cl_float), &radius ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 3, sizeof(cl_mem), &m_px1 ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 4, sizeof(cl_mem), &m_py1 ));

  CL_SAFE_CALL( clEnqueueNDRangeKernel(commands, kernel, 2, NULL,
				       dimGrid, dimBlock, 0, NULL, NULL) );
}



// Reprojection for RGB, TV_F
const char *kernel_reproject_euclidean_4D = "\n" \
  "__kernel void kernel_function(                                        \n" \
  "   const unsigned int W,                                              \n" \
  "   const unsigned int H,                                              \n" \
  "   const float r,                                                     \n" \
  "   __global float* px1,                                               \n" \
  "   __global float* py1,                                               \n" \
  "   __global float* px2,                                               \n" \
  "   __global float* py2 )                                              \n" \
  "{                                                                     \n" \
  "  int ox = get_global_id(0);                                          \n" \
  "  int oy = get_global_id(1);                                          \n" \
  "  if ( ox>=W || oy>=H ) return;                                       \n" \
  "  int o = oy*W + ox;                                                  \n" \
  "                                                                      \n" \
  "  // Local vars                                                       \n" \
  "  float a11 = px1[o];                                                 \n" \
  "  float a21 = px2[o];                                                 \n" \
  "  float a12 = py1[o];                                                 \n" \
  "  float a22 = py2[o];                                                 \n" \
  "                                                                      \n" \
  "  // Total norm                                                       \n" \
  "  float n = pow( a11, 2.0f ) + pow( a21, 2.0f );                      \n" \
  "  n += pow( a12, 2.0f ) + pow( a22, 2.0f );                           \n" \
  "  n = sqrt( n );                                                      \n" \
  "                                                                      \n" \
  "  // Project                                                          \n" \
  "  if ( n > r ) {                                                      \n" \
  "    n = r/n;                                                          \n" \
  "    px1[o] = a11 * n;                                                 \n" \
  "    py1[o] = a12 * n;                                                 \n" \
  "    px2[o] = a21 * n;                                                 \n" \
  "    py2[o] = a22 * n;                                                 \n" \
  "  }                                                                   \n" \
  "}                                                                     \n";


// Reprojection to Euclidean ball with given radius
void coco::kernel_reproject_euclidean_4D( const compute_grid *G,
					     const float radius,
					     compute_buffer &px1, compute_buffer &py1,
					     compute_buffer &px2, compute_buffer &py2 )
{
  // Create kernel
  static cl_kernel kernel = NULL;
  if ( !kernel ) {
    kernel = kernel_compile( G->engine(),
			     "kernel_function",
			     ::kernel_reproject_euclidean_4D );
    assert( kernel != NULL );
  }

  // Launch kernel
  dim3 dimGrid;
  dim3 dimBlock;
  kernel_configure( G, dimGrid, dimBlock );
  
  CL_COMMAND_QUEUE( G->engine() );

  int W = G->W();
  int H = G->H();
  cl_mem m_px1( px1 );
  cl_mem m_py1( py1 );
  cl_mem m_px2( px2 );
  cl_mem m_py2( py2 );

  CL_SAFE_CALL( clSetKernelArg(kernel, 0, sizeof(int), &W ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 1, sizeof(int), &H ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 2, sizeof(cl_float), &radius ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 3, sizeof(cl_mem), &m_px1 ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 4, sizeof(cl_mem), &m_py1 ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 5, sizeof(cl_mem), &m_px2 ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 6, sizeof(cl_mem), &m_py2 ));

  CL_SAFE_CALL( clEnqueueNDRangeKernel(commands, kernel, 2, NULL,
				       dimGrid, dimBlock, 0, NULL, NULL) );
}




// Reprojection for RGB, TV_F
const char *kernel_reproject_euclidean_6D = "\n" \
  "__kernel void kernel_function(                                        \n" \
  "   const unsigned int W,                                              \n" \
  "   const unsigned int H,                                              \n" \
  "   const float r,                                                     \n" \
  "   __global float* px1,                                               \n" \
  "   __global float* py1,                                               \n" \
  "   __global float* px2,                                               \n" \
  "   __global float* py2,                                               \n" \
  "   __global float* px3,                                               \n" \
  "   __global float* py3 )                                              \n" \
  "{                                                                     \n" \
  "  int ox = get_global_id(0);                                          \n" \
  "  int oy = get_global_id(1);                                          \n" \
  "  if ( ox>=W || oy>=H ) return;                                       \n" \
  "  int o = oy*W + ox;                                                  \n" \
  "                                                                      \n" \
  "  // Local vars                                                       \n" \
  "  float a11 = px1[o];                                                 \n" \
  "  float a21 = px2[o];                                                 \n" \
  "  float a31 = px3[o];                                                 \n" \
  "  float a12 = py1[o];                                                 \n" \
  "  float a22 = py2[o];                                                 \n" \
  "  float a32 = py3[o];                                                 \n" \
  "                                                                      \n" \
  "  // Total norm                                                       \n" \
  "  float n = a11*a11 + a21*a21 + a31*a31;                              \n" \
  "  n += a12*a12 + a22*a22 + a32*a32;                                   \n" \
  "  n = sqrt( n );                                                      \n" \
  "                                                                      \n" \
  "  // Project                                                          \n" \
  "  if ( n > r ) {                                                      \n" \
  "    n = r/n;                                                          \n" \
  "    px1[o] = a11 * n;                                                 \n" \
  "    py1[o] = a12 * n;                                                 \n" \
  "    px2[o] = a21 * n;                                                 \n" \
  "    py2[o] = a22 * n;                                                 \n" \
  "    px3[o] = a31 * n;                                                 \n" \
  "    py3[o] = a32 * n;                                                 \n" \
  "  }                                                                   \n" \
  "}                                                                     \n";


// Reprojection to Euclidean ball with given radius
void coco::kernel_reproject_euclidean_6D( const compute_grid *G,
					     const float radius,
					     compute_buffer &px1, compute_buffer &py1,
					     compute_buffer &px2, compute_buffer &py2,
					     compute_buffer &px3, compute_buffer &py3 )
{
  // Create kernel
  static cl_kernel kernel = NULL;
  if ( !kernel ) {
    kernel = kernel_compile( G->engine(),
			     "kernel_function",
			     ::kernel_reproject_euclidean_6D );
    assert( kernel != NULL );
  }

  // Launch kernel
  dim3 dimGrid;
  dim3 dimBlock;
  kernel_configure( G, dimGrid, dimBlock );
  
  CL_COMMAND_QUEUE( G->engine() );

  int W = G->W();
  int H = G->H();
  cl_mem m_px1( px1 );
  cl_mem m_py1( py1 );
  cl_mem m_px2( px2 );
  cl_mem m_py2( py2 );
  cl_mem m_px3( px3 );
  cl_mem m_py3( py3 );

  CL_SAFE_CALL( clSetKernelArg(kernel, 0, sizeof(int), &W ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 1, sizeof(int), &H ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 2, sizeof(cl_float), &radius ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 3, sizeof(cl_mem), &m_px1 ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 4, sizeof(cl_mem), &m_py1 ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 5, sizeof(cl_mem), &m_px2 ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 6, sizeof(cl_mem), &m_py2 ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 7, sizeof(cl_mem), &m_px3 ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 8, sizeof(cl_mem), &m_py3 ));

  CL_SAFE_CALL( clEnqueueNDRangeKernel(commands, kernel, 2, NULL,
				       dimGrid, dimBlock, 0, NULL, NULL) );
}




// Reprojection for RGB, TV_J
const char *kernel_reproject_nuclear_6D = "\n" \
  "__kernel void kernel_function(                                        \n" \
  "   const unsigned int W,                                              \n" \
  "   const unsigned int H,                                              \n" \
  "   const float radius,                                                \n" \
  "   __global float* px1,                                               \n" \
  "   __global float* py1,                                               \n" \
  "   __global float* px2,                                               \n" \
  "   __global float* py2,                                               \n" \
  "   __global float* px3,                                               \n" \
  "   __global float* py3 )                                              \n" \
  "{                                                                     \n" \
  "  int ox = get_global_id(0);                                          \n" \
  "  int oy = get_global_id(1);                                          \n" \
  "  if ( ox>=W || oy>=H ) return;                                       \n" \
  "  int o = oy*W + ox;                                                  \n" \
  "                                                                      \n" \
  "  // Local vars                                                       \n" \
  "  float a11 = px1[o];                                                 \n" \
  "  float a21 = px2[o];                                                 \n" \
  "  float a31 = px3[o];                                                 \n" \
  "  float a12 = py1[o];                                                 \n" \
  "  float a22 = py2[o];                                                 \n" \
  "  float a32 = py3[o];                                                 \n" \
  "                                                                      \n" \
  "  // Compute A^T A                                                    \n" \
  "  float d11 = a11*a11 + a21*a21 + a31*a31;                            \n" \
  "  float d12 = a12*a11 + a22*a21 + a32*a31;                            \n" \
  "  float d22 = a12*a12 + a22*a22 + a32*a32;                            \n" \
  "                                                                      \n" \
  "  // Compute larger Eigenvalue (= square of largest singular value)   \n" \
  "  float trace = d11 + d22;                                            \n" \
  "  float det = d11*d22 - d12*d12;                                      \n" \
  "  float d = sqrt( 0.25*trace*trace - det );                           \n" \
  "  float lmax = max( 0.0, 0.5 * trace + d );                           \n" \
  "  float lmin = max( 0.0, 0.5 * trace - d );                           \n" \
  "  float smax = sqrt( lmax );                                          \n" \
  "  float smin = sqrt( lmin );                                          \n" \
  "                                                                      \n" \
  "  // If smax + smin > 1:                                              \n" \
  "  // Project (smax,smin) to line (0,1) + tau * (1,-1), 0<=tau<=1.     \n" \
  "  if ( smax + smin > radius ) {                                       \n" \
  "                                                                      \n" \
  "    float v11, v12, v21, v22;                                         \n" \
  "    if ( d12 == 0.0 ) {                                               \n" \
  "      if ( d11 >= d22 ) {                                             \n" \
  "        v11 = 1.0; v21 = 0.0; v12 = 0.0; v22 = 1.0;                   \n" \
  "      }                                                               \n" \
  "      else {                                                          \n" \
  "        v11 = 0.0; v21 = 1.0; v12 = 1.0; v22 = 0.0;                   \n" \
  "      }                                                               \n" \
  "    }                                                                 \n" \
  "    else {                                                            \n" \
  "      v11 = lmax - d22; v21 = d12;                                    \n" \
  "      float l1 = hypot( v11, v21 );                                   \n" \
  "      v11 /= l1; v21 /= l1;                                           \n" \
  "      v12 = lmin - d22; v22 = d12;                                    \n" \
  "      float l2 = hypot( v12, v22 );                                   \n" \
  "      v12 /= l2; v22 /= l2;                                           \n" \
  "    }                                                                 \n" \
  "                                                                      \n" \
  "    // Compute projection of Eigenvalues                              \n" \
  "    float tau = 0.5f * (smax - smin + 1.0f);                          \n" \
  "    float s1 = min( 1.0f, tau );                                      \n" \
  "    float s2 = 1.0f - s1;                                             \n" \
  "    // Compute \Sigma^{-1} * \Sigma_{new}                             \n" \
  "    s1 /= smax;                                                       \n" \
  "    s2 = (smin > 0.0f) ? s2 / smin : 0.0f;                            \n" \
  "                                                                      \n" \
  "    // A_P = A * \Sigma^{-1} * \Sigma_{new}                           \n" \
  "    float t11 = s1*v11*v11 + s2*v12*v12;                              \n" \
  "    float t12 = s1*v11*v21 + s2*v12*v22;                              \n" \
  "    float t21 = s1*v21*v11 + s2*v22*v12;                              \n" \
  "    float t22 = s1*v21*v21 + s2*v22*v22;                              \n" \
  "                                                                      \n" \
  "    // Result                                                         \n" \
  "    px1[o] = radius * (a11 * t11 + a12 * t21);                        \n" \
  "    px2[o] = radius * (a21 * t11 + a22 * t21);                        \n" \
  "    px3[o] = radius * (a31 * t11 + a32 * t21);                        \n" \
  "                                                                      \n" \    
  "    py1[o] = radius * (a11 * t12 + a12 * t22);                        \n" \
  "    py2[o] = radius * (a21 * t12 + a22 * t22);                        \n" \
  "    py3[o] = radius * (a31 * t12 + a32 * t22);                        \n" \
  "  }                                                                   \n" \
  "}                                                                     \n";



// Reprojection to nuclear norm ball
void coco::kernel_reproject_nuclear_6D( const compute_grid *G,
					const float radius,
					compute_buffer &px1, compute_buffer &py1,
					compute_buffer &px2, compute_buffer &py2,
					compute_buffer &px3, compute_buffer &py3 )
{
  // Create kernel
  static cl_kernel kernel = NULL;
  if ( !kernel ) {
    kernel = kernel_compile( G->engine(),
			     "kernel_function",
			     ::kernel_reproject_nuclear_6D );
    assert( kernel != NULL );
  }

  // Launch kernel
  dim3 dimGrid;
  dim3 dimBlock;
  kernel_configure( G, dimGrid, dimBlock );
  
  CL_COMMAND_QUEUE( G->engine() );

  int W = G->W();
  int H = G->H();
  cl_mem m_px1( px1 );
  cl_mem m_py1( py1 );
  cl_mem m_px2( px2 );
  cl_mem m_py2( py2 );
  cl_mem m_px3( px3 );
  cl_mem m_py3( py3 );

  CL_SAFE_CALL( clSetKernelArg(kernel, 0, sizeof(int), &W ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 1, sizeof(int), &H ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 2, sizeof(cl_float), &radius ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 3, sizeof(cl_mem), &m_px1 ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 4, sizeof(cl_mem), &m_py1 ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 5, sizeof(cl_mem), &m_px2 ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 6, sizeof(cl_mem), &m_py2 ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 7, sizeof(cl_mem), &m_px3 ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 8, sizeof(cl_mem), &m_py3 ));

  CL_SAFE_CALL( clEnqueueNDRangeKernel(commands, kernel, 2, NULL,
				       dimGrid, dimBlock, 0, NULL, NULL) );
}


