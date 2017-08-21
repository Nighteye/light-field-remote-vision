/* -*-c++-*- */
/** \file convolutions.cpp

    OpenCL convolution implementation.

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

#include <stdio.h>

#include "../compute_api/convolutions.h"
#include "compute_api_implementation_opencl.h"

using namespace std;
using namespace coco;


////////////////////////////////////////////////////////////////////////////////
// Convolution configuration
// Size of tiles (blocks) for convolution operations
// Larger block sizes = less overhead for apron
////////////////////////////////////////////////////////////////////////////////

// Assuming ROW_TILE_W, KERNEL_RADIUS_ALIGNED and dataW 
// are multiples of coalescing granularity size,
// all global memory operations are coalesced in convolutionRowGPU()
////////////////////////////////////////////////////////////////////////////////
// Convolution configuration
// Size of tiles (blocks) for convolution operations
// Larger block sizes = less overhead for apron
////////////////////////////////////////////////////////////////////////////////

// Assuming ROW_TILE_W, KERNEL_RADIUS_ALIGNED and dataW 
// are multiples of coalescing granularity size,
// all global memory operations are coalesced in convolutionRowGPU()
#define ROW_TILE_W 128

// Assuming COLUMN_TILE_W and dataW are multiples
// of coalescing granularity size, all global memory operations 
// are coalesced in convolutionColumnGPU()
#define COLUMN_TILE_W 16
#define COLUMN_TILE_H 48


////////////////////////////////////////////////////////////////////////////////
// Row convolution filter (from nVidia SDK)
////////////////////////////////////////////////////////////////////////////////
const char *kernel_convolution_row_src = "\n" \
  "__kernel void kernel_function(                                        \n" \
  "   __global float* d_Result,                                          \n" \
  "   __global const float* d_Data,                                      \n" \
  "   __global const float* d_Kernel,                                    \n" \
  "   const int KERNEL_RADIUS,                                           \n" \
  "   const int KERNEL_RADIUS_ALIGNED,                                   \n" \
  "   const int dataW,                                                   \n" \
  "   const int dataH,                                                   \n" \
  "   __local float *data )                                              \n" \
  "{                                                                     \n" \
  "  //Current tile and apron limits, relative to row start              \n" \
  "  const int         tileStart = get_group_id(0) * 128;                \n" \
  "  const int           tileEnd = tileStart + 128 - 1;                  \n" \
  "  const int        apronStart = tileStart - KERNEL_RADIUS;            \n" \
  "  const int          apronEnd = tileEnd   + KERNEL_RADIUS;            \n" \
  "                                                                      \n" \
  "  //Clamp tile and apron limits by image borders                      \n" \
  "  const int    tileEndClamped = min(tileEnd, dataW - 1);              \n" \
  "  const int apronStartClamped = max(apronStart, 0);                   \n" \
  "  const int   apronEndClamped = min(apronEnd, dataW - 1);             \n" \
  "                                                                      \n" \
  "  //Row start index in d_Data[]                                       \n" \
  "  const int          rowStart = get_group_id(1) * dataW;              \n" \
  "                                                                      \n" \
  "  //Aligned apron start. Assuming dataW and ROW_TILE_W are multiples  \n" \
  "  //of half-warp size, rowStart + apronStartAligned is also a         \n" \
  "  //multiple of half-warp size, thus having proper alignment          \n" \
  "  //for coalesced d_Data[] read.                                      \n" \
  "  const int apronStartAligned = tileStart - KERNEL_RADIUS_ALIGNED;    \n" \
  "                                                                      \n" \
  "  const int loadPos = apronStartAligned + get_local_id(0);                \n" \
  "  //Set the entire data cache contents                                \n" \
  "  //Load global memory values, if indices are within the image        \n" \
  "  //borders, or initialize with zeroes otherwise                      \n" \
  "  if(loadPos >= apronStart){                                          \n" \
  "    const int smemPos = loadPos - apronStart;                         \n" \
  "                                                                      \n" \
  "    data[smemPos] =                                                                                   \n" \
  "      (loadPos < apronStartClamped) ? d_Data[rowStart + apronStartClamped] :                          \n" \
  "      ( (loadPos > apronEndClamped) ? d_Data[rowStart + apronEndClamped] :                            \n" \
  "      d_Data[rowStart + loadPos] );                                                                   \n" \
  "  }                                                                                                   \n" \
  "                                                                      \n" \
  "  //Ensure the completness of the loading stage                       \n" \
  "  //because results, emitted by each thread depend on the data,       \n" \
  "  //loaded by another threads                                         \n" \
  "  barrier(CLK_LOCAL_MEM_FENCE);                                       \n" \
  "                                                                      \n" \
  "  const int writePos = tileStart + get_local_id(0);                       \n" \
  "  //Assuming dataW and ROW_TILE_W are multiples of half-warp size,    \n" \
  "  //rowStart + tileStart is also a multiple of half-warp size,        \n" \
  "  //thus having proper alignment for coalesced d_Result[] write.      \n" \
  "  if(writePos <= tileEndClamped){                                     \n" \
  "    const int smemPos = writePos - apronStart;                        \n" \
  "    float sum = 0;                                                    \n" \
  "    for(int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++) {            \n" \
  "      sum += data[smemPos + k] * d_Kernel[KERNEL_RADIUS - k];         \n" \
  "    }                                                                 \n" \
  "    d_Result[rowStart + writePos] = sum;                              \n" \
  "  }                                                                   \n" \
  "}                                                                     \n";



// Row convolution entry function
static void kernel_convolution_row( const compute_engine *CE,
				    const dim3 &dimGrid,
				    const dim3 &dimBlock,
				    compute_buffer &d_Result,
				    const compute_buffer &d_Data,
				    const compute_buffer &d_Kernel,
				    int KERNEL_RADIUS,
				    int KERNEL_RADIUS_ALIGNED,
				    int dataW,
				    int dataH,
				    size_t memsize )
{
  // Create kernel
  static cl_kernel kernel = NULL;
  if ( !kernel ) {
    kernel = kernel_compile( CE,
			     "kernel_function",
			     ::kernel_convolution_row_src );
    assert( kernel != NULL );
  }

  CL_COMMAND_QUEUE( CE );

  cl_mem m_Result( d_Result );
  cl_mem m_Data( d_Data );
  cl_mem m_Kernel( d_Kernel );

  CL_SAFE_CALL( clSetKernelArg(kernel, 0, sizeof(cl_mem), &m_Result ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 1, sizeof(cl_mem), &m_Data ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 2, sizeof(cl_mem), &m_Kernel ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 3, sizeof(int), &KERNEL_RADIUS ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 4, sizeof(int), &KERNEL_RADIUS_ALIGNED ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 5, sizeof(int), &dataW ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 6, sizeof(int), &dataH ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 7, memsize, NULL ));

  dim3 dimGrid2;
  dimGrid2[0] = dimGrid[0] * dimBlock[0];
  dimGrid2[1] = dimGrid[1] * dimBlock[1];
  dimGrid2[2] = 1;
  int err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,
				   dimGrid2, dimBlock, 0, NULL, NULL);

  if ( err == CL_INVALID_KERNEL_ARGS ) {
    ERROR( "arguments not set." << endl );
  }
  else if ( err == CL_INVALID_WORK_GROUP_SIZE ) {
    ERROR( "invalid work group size." << endl );
  }
  else if ( err == CL_INVALID_WORK_ITEM_SIZE ) {
    ERROR( "invalid work item size." << endl );
  }
  else if ( err != CL_SUCCESS ) {
    ERROR( "generic error." );
  }
}


////////////////////////////////////////////////////////////////////////////////
// Column convolution filter (from nVidia SDK)
////////////////////////////////////////////////////////////////////////////////

const char *kernel_convolution_column_src = "\n" \
  "__kernel void kernel_function(                                        \n" \
  "   __global float* d_Result,                                          \n" \
  "   __global const float* d_Data,                                      \n" \
  "   __global const float* d_Kernel,                                    \n" \
  "   const int KERNEL_RADIUS,                                           \n" \
  "   const int dataW,                                                   \n" \
  "   const int dataH,                                                   \n" \
  "   const int smemStride,                                              \n" \
  "   const int gmemStride,                                              \n" \
  "   __local float *data )                                              \n" \
  "{                                                                     \n" \
  "  //Current tile and apron limits, in rows                            \n" \
  "  const int         tileStart = get_group_id(1) * 48;      \n" \
  "  const int           tileEnd = tileStart + 48 - 1;        \n" \
  "  const int        apronStart = tileStart - KERNEL_RADIUS;            \n" \
  "  const int          apronEnd = tileEnd   + KERNEL_RADIUS;            \n" \
  "                                                                      \n" \
  "  //Clamp tile and apron limits by image borders                      \n" \
  "  const int    tileEndClamped = min(tileEnd, dataH - 1);              \n" \
  "  const int apronStartClamped = max(apronStart, 0);                   \n" \
  "  const int   apronEndClamped = min(apronEnd, dataH - 1);             \n" \
  "                                                                      \n" \
  "  //Current column index                                              \n" \
  "  const int       columnStart = get_group_id(0) * 16 + get_local_id(0);                      \n" \
  "                                                                      \n" \
  "  //Shared and global memory indices for current column               \n" \
  "  int smemPos = get_local_id(1) * 16 + get_local_id(0);       \n" \
  "  int gmemPos = (apronStart + get_local_id(1)) * dataW + columnStart;  \n" \
  "  //Cycle through the entire data cache                               \n" \
  "  //Load global memory values, if indices are within the image borders,                             \n" \
  "  //or initialize with zero otherwise                                                               \n" \
  "  for(int y = apronStart + get_local_id(1); y <= apronEnd; y += get_local_size(1)) {                           \n" \
  "    data[smemPos] = (y < apronStartClamped) ? d_Data[apronStartClamped*dataW + columnStart] : \n" \
  "      ((y > apronEndClamped) ? d_Data[apronEndClamped*dataW + columnStart] :                  \n" \
  "       d_Data[gmemPos]);                                             \n" \
  "    smemPos += smemStride;                                           \n" \
  "    gmemPos += gmemStride;                                           \n" \
  "  }                                                                  \n" \
  "                                                                     \n" \
  "  //Ensure the completness of the loading stage                      \n" \
  "  //because results, emitted by each thread depend on the data,      \n" \
  "  //loaded by another threads                                        \n" \
  "  barrier(CLK_LOCAL_MEM_FENCE);                                      \n" \
  "  //Shared and global memory indices for current column              \n" \
  "  smemPos = (get_local_id(1) + KERNEL_RADIUS) * 16 + get_local_id(0);       \n" \
  "  gmemPos = (tileStart + get_local_id(1)) * dataW + columnStart;                  \n" \
  "  //Cycle through the tile body, clamped by image borders                         \n" \
  "  //Calculate and output the results                                              \n" \
  "  for(int y = tileStart + get_local_id(1); y <= tileEndClamped; y += get_local_size(1)) {    \n" \
  "    float sum = 0;                                                                \n" \
  "    for(int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++) {                        \n" \
  "      sum +=                                                                      \n" \
  "        data[smemPos + k * 16] *                                  \n" \
  "        d_Kernel[KERNEL_RADIUS - k];                                              \n" \
  "     }                                                                            \n" \
  "     d_Result[gmemPos] = sum;                                                     \n" \
  "     smemPos += smemStride;                                           \n" \
  "     gmemPos += gmemStride;                                           \n" \
  "  }                                                                   \n" \
  "}                                                                     \n";




// Column convolution entry function
static void kernel_convolution_column( const compute_engine *CE,
				       const dim3 &dimGrid, const dim3 &dimBlock,
				       compute_buffer &d_Result,
				       const compute_buffer &d_Data,
				       const compute_buffer &d_Kernel,
				       int KERNEL_RADIUS,
				       int dataW,
				       int dataH,
				       int smemStride,
				       int gmemStride,
				       size_t memsize )
{
  // Create kernel
  static cl_kernel kernel = NULL;
  if ( !kernel ) {
    kernel = kernel_compile( CE,
			     "kernel_function",
			     ::kernel_convolution_column_src );
    assert( kernel != NULL );
  }

  CL_COMMAND_QUEUE( CE );

  cl_mem m_Result( d_Result );
  cl_mem m_Data( d_Data );
  cl_mem m_Kernel( d_Kernel );

  CL_SAFE_CALL( clSetKernelArg(kernel, 0, sizeof(cl_mem), &m_Result ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 1, sizeof(cl_mem), &m_Data ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 2, sizeof(cl_mem), &m_Kernel ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 3, sizeof(int), &KERNEL_RADIUS ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 4, sizeof(int), &dataW ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 5, sizeof(int), &dataH ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 6, sizeof(int), &smemStride ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 7, sizeof(int), &gmemStride ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 8, memsize, NULL ));

  dim3 dimGrid2;
  dimGrid2[0] = dimGrid[0] * dimBlock[0];
  dimGrid2[1] = dimGrid[1] * dimBlock[1];
  dimGrid2[2] = 1;

  CL_SAFE_CALL( clEnqueueNDRangeKernel(commands, kernel, 2, NULL,
				       dimGrid2, dimBlock, 0, NULL, NULL) );
}



const char *kernel_convolution_nonsep = "\n" \
  "__kernel void kernel_function(                                        \n" \
  "   const unsigned int W,                                              \n" \
  "   const unsigned int H,                                              \n" \
  "   __global const float* k,                                           \n" \
  "   const unsigned int w,                                              \n" \
  "   const unsigned int h,                                              \n" \
  "   const unsigned int w2,                                             \n" \
  "   const unsigned int h2,                                             \n" \
  "   __global const float* s,                                           \n" \
  "   __global float* d )                                                \n" \
  "{                                                                     \n" \
  "  int ox = get_global_id(0);                                          \n" \
  "  int oy = get_global_id(1);                                          \n" \
  "  if ( ox>=W || oy>=H ) return;                                       \n" \
  "  int o = oy*W + ox;                                                  \n" \
  "                                                                      \n" \
  "  // Compute local convolution                                        \n" \
  "  float v = 0.0;                                                      \n" \
  "  float n = 0.0;                                                      \n" \
  "  int index=0;                                                        \n" \
  "  for ( int j=0; j<h; j++ ) {                                         \n" \
  "    for ( int i=0; i<w; i++ ) {                                       \n" \
  "                                                                      \n" \
  "      int xx = ox - w2 + i;                                           \n" \
  "      int yy = oy - h2 + j;                                           \n" \
  "                                                                      \n" \
  "      if ( xx>=0 && xx<W && yy>=0 && yy<H ) {                         \n" \
  "        float kv = k[index];                                          \n" \
  "        n += kv;                                                      \n" \
  "        v += kv * s[ yy * W + xx ];                                   \n" \
  "      }                                                               \n" \
  "                                                                      \n" \
  "      index++;                                                        \n" \
  "    }                                                                 \n" \
  "  }                                                                   \n" \
  "                                                                      \n" \
  "  if ( n>0.0 ) {                                                      \n" \
  "    v /= n;                                                           \n" \
  "  }                                                                   \n" \
  "                                                                      \n" \
  "  d[o] = v;                                                           \n" \
  "}                                                                     \n";



// Slow nonseparable version
static bool cuda_convolution_nonsep( const coco::compute_grid *G,
				     const coco::convolution_kernel *k, 
				     const coco::compute_buffer &in, coco::compute_buffer &out )
{
  // Create kernel
  static cl_kernel kernel = NULL;
  if ( !kernel ) {
    kernel = kernel_compile( G->engine(),
			     "kernel_function",
			     ::kernel_convolution_nonsep );
    assert( kernel != NULL );
  }

  // Launch kernel
  dim3 dimGrid;
  dim3 dimBlock;
  kernel_configure( G, dimGrid, dimBlock );
  
  CL_COMMAND_QUEUE( G->engine() );

  int W = G->W();
  int H = G->H();
  cl_mem m_data( *k->_data );
  int w = k->_w;
  int h = k->_h;
  int w2 = ( k->_w -1 ) / 2;
  int h2 = ( k->_h - 1 ) / 2;
  cl_mem m_in( in );
  cl_mem m_out( out );

  CL_SAFE_CALL( clSetKernelArg(kernel, 0, sizeof(int), &W ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 1, sizeof(int), &H ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 2, sizeof(cl_mem), &m_data ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 3, sizeof(int), &w ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 4, sizeof(int), &h ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 5, sizeof(int), &w2 ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 6, sizeof(int), &h2 ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 7, sizeof(cl_mem), &m_in ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 8, sizeof(cl_mem), &m_out ));

  CL_SAFE_CALL( clEnqueueNDRangeKernel(commands, kernel, 2, NULL,
				       dimGrid, dimBlock, 0, NULL, NULL) );
  return true;
}




// interger division with rounding to next highest
inline int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}


// Convolve array with kernel
bool coco::convolution( const compute_grid *grid,
			const convolution_kernel *kernel, 
			const compute_buffer &in, compute_buffer &out )
{
  if ( !kernel->_separable ) {
    return cuda_convolution_nonsep( grid, kernel, in, out );
  }

  // Needs a temp array
  compute_buffer *tmp = grid->alloc_layers(1);
  int W = grid->W();
  int H = grid->H();

  // Compute radius
  const int KERNEL_RADIUS_X = kernel->_w / 2;
  assert( kernel->_w == KERNEL_RADIUS_X*2 + 1 );
  const int KERNEL_RADIUS_Y = kernel->_h / 2;
  assert( kernel->_h == KERNEL_RADIUS_Y*2 + 1 );
  // Compute alignment radius: must be multiple of 16 (half warp size)
  // for maximum performance.
  const int KERNEL_RADIUS_ALIGNED = ((KERNEL_RADIUS_X-1) / 16 + 1) * 16;

  // Call CUDA kernels
  dim3 blockGridRows;
  blockGridRows[0] = iDivUp(W, ROW_TILE_W);
  blockGridRows[1] = H;
  blockGridRows[2] = 1;
  dim3 blockGridColumns;
  blockGridColumns[0] = iDivUp(W, COLUMN_TILE_W);
  blockGridColumns[1] = iDivUp(H, COLUMN_TILE_H);
  blockGridColumns[2] = 1;

  dim3 threadBlockRows;
  threadBlockRows[0] = KERNEL_RADIUS_ALIGNED + ROW_TILE_W + KERNEL_RADIUS_X;
  threadBlockRows[1] = 1;
  threadBlockRows[2] = 1;
  dim3 threadBlockColumns;
  threadBlockColumns[0] = COLUMN_TILE_W;
  threadBlockColumns[1] = 8;
  threadBlockColumns[2] = 1;

  size_t memsize_row = sizeof(float) * (KERNEL_RADIUS_X + ROW_TILE_W + KERNEL_RADIUS_X);
  kernel_convolution_row
    (grid->engine(),
     blockGridRows, threadBlockRows,
     *tmp,
     in,
     *kernel->_data_x,
     KERNEL_RADIUS_X,
     KERNEL_RADIUS_ALIGNED,
     W,H,
     memsize_row );

  size_t memsize_column = sizeof(float) * COLUMN_TILE_W * (KERNEL_RADIUS_Y + COLUMN_TILE_H + KERNEL_RADIUS_Y);
  kernel_convolution_column
    ( grid->engine(),
      blockGridColumns, threadBlockColumns,
      out,
      *tmp,
      *kernel->_data_y, KERNEL_RADIUS_Y,
      W,H,
      COLUMN_TILE_W * threadBlockColumns[1],
      W * threadBlockColumns[1],
      memsize_column );

  delete tmp;
  return true;
}



/*
static __global__ void convolution_row3_device( int W, int H,
						float k0, float k1, float k2,
						const float *in, float *out )
{
  // Global thread index
  const int ox = IMUL( get_local_size(0), get_group_id(0) ) + get_local_id(0);
  const int oy = IMUL( get_local_size(1), get_group_id(1) ) + get_local_id(1);
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;
  if ( ox==0 ) {
    out[o] = (k2 * in[o+1] + k1 * in[o]) / (k1+k2);
  }
  else if ( ox==W-1 ) {
    out[o] = (k1 * in[o] + k0 * in[o-1]) / (k0+k1);
  }
  else {
    out[o] = k2 * in[o+1] + k0 * in[o-1] + k1 * in[o];
  }
}


static __global__ void convolution_column3_device( int W, int H,
						   float k0, float k1, float k2,
						   const float *in, float *out )
{
  // Global thread index
  const int ox = IMUL( get_local_size(0), get_group_id(0) ) + get_local_id(0);
  const int oy = IMUL( get_local_size(1), get_group_id(1) ) + get_local_id(1);
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;
  if ( oy==0 ) {
    out[o] = (k2 * in[o+W] + k1 * in[o]) / (k1+k2);
  }
  else if ( oy==H-1 ) {
    out[o] = (k1 * in[o] + k0 * in[o-W]) / (k0+k1);
  }
  else {
    out[o] = k2 * in[o+W] + k0 * in[o-W] + k1 * in[o];
  }
}


// Fast convolution for Row-3 kernel
bool coco::convolution_row( float k0, float k1, float k2,
			    size_t W, size_t H,
			    const float* in, float* out )
{
  dim3 dimBlock;
  dim3 dimGrid;
  cuda_default_grid( W,H, dimGrid, dimBlock );
  convolution_row3_device<<< dimGrid, dimBlock >>>
    ( W,H, k0,k1,k2,
      in, out );
  return true;
}


// Fast convolution for Column-3 kernel
bool coco::convolution_column( float k0, float k1, float k2,
			       size_t W, size_t H,
			       const float* in, float* out )
{
  dim3 dimBlock;
  dim3 dimGrid;
  cuda_default_grid( W,H, dimGrid, dimBlock );
  convolution_column3_device<<< dimGrid, dimBlock >>>
    ( W,H, k0,k1,k2,
      in, out );
  return true;
}

*/
