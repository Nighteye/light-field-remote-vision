/* -*-c++-*- */
/** \file cuda_convolution.cu

    CUDA convolution implementation.

    Copyright (C) 2010 Bastian Goldluecke,
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

#include "cuda_convolutions.h"
#include "cuda_convolutions.cuh"
#include "cuda_interface.h"
#include "cuda_helper.h"

#include "../defs.h"

/********************************************************
  Convolution kernel structure
*********************************************************/

// Init arbitrary convolution kernel
coco::cuda_kernel *coco::cuda_kernel_alloc( const gsl_matrix *m )
{
  cuda_kernel *k = new cuda_kernel;
  k->_w = m->size2;
  k->_h = m->size1;
  k->_separable = false;
  CUDA_SAFE_CALL( cudaMalloc( &k->_data, k->_w * k->_h * sizeof(float) ));
  cuda_memcpy( k->_data, m );
  k->_data_x = NULL;
  k->_data_y = NULL;
  return k;
}


// Init separable convolution kernel
coco::cuda_kernel *coco::cuda_kernel_alloc_separable( const gsl_vector *vx, const gsl_vector *vy )
{
  cuda_kernel *k = new cuda_kernel;
  k->_w = vx->size;
  k->_h = vy->size;
  k->_separable = true;
  k->_data = NULL;
  CUDA_SAFE_CALL( cudaMalloc( &k->_data_x, k->_w * sizeof(float) ));
  CUDA_SAFE_CALL( cudaMalloc( &k->_data_y, k->_h * sizeof(float) ));
  cuda_memcpy( k->_data_x, vx );
  cuda_memcpy( k->_data_y, vy );
  return k;
}

// Release convolution kernel
void coco::cuda_kernel_free( cuda_kernel *k )
{
  if ( k==NULL ) {
    return;
  }
  if ( k->_data != NULL ) {
    CUDA_SAFE_CALL( cudaFree( k->_data ));
  }
  if ( k->_data_x != NULL ) {
    CUDA_SAFE_CALL( cudaFree( k->_data_x ));
  }
  if ( k->_data_y != NULL ) {
    CUDA_SAFE_CALL( cudaFree( k->_data_y ));
  }
  delete k;
}


/**********************************************************
  Convolution functions (adapted from nVidia SDK examples)
***********************************************************/

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
__global__ void convolutionRowGPU( float *d_Result, const float *d_Data, const float *d_Kernel,
				   int KERNEL_RADIUS,
				   int KERNEL_RADIUS_ALIGNED,
				   int dataW,
				   int dataH )
{
  // Data cache
  extern __shared__ float data[];

  //Current tile and apron limits, relative to row start
  const int         tileStart = IMUL(blockIdx.x, ROW_TILE_W);
  const int           tileEnd = tileStart + ROW_TILE_W - 1;
  const int        apronStart = tileStart - KERNEL_RADIUS;
  const int          apronEnd = tileEnd   + KERNEL_RADIUS;

  //Clamp tile and apron limits by image borders
  const int    tileEndClamped = min(tileEnd, dataW - 1);
  const int apronStartClamped = max(apronStart, 0);
  const int   apronEndClamped = min(apronEnd, dataW - 1);

  //Row start index in d_Data[]
  const int          rowStart = IMUL(blockIdx.y, dataW);

  //Aligned apron start. Assuming dataW and ROW_TILE_W are multiples 
  //of half-warp size, rowStart + apronStartAligned is also a 
  //multiple of half-warp size, thus having proper alignment 
  //for coalesced d_Data[] read.
  const int apronStartAligned = tileStart - KERNEL_RADIUS_ALIGNED;
  
  const int loadPos = apronStartAligned + threadIdx.x;
  //Set the entire data cache contents
  //Load global memory values, if indices are within the image borders,
  //or initialize with zeroes otherwise
  if(loadPos >= apronStart){
    const int smemPos = loadPos - apronStart;
    
    data[smemPos] = 
      (loadPos < apronStartClamped) ? d_Data[rowStart + apronStartClamped] : 
      ( (loadPos > apronEndClamped) ? d_Data[rowStart + apronEndClamped] :
	d_Data[rowStart + loadPos] );
  }

  //Ensure the completness of the loading stage
  //because results, emitted by each thread depend on the data,
  //loaded by another threads
  __syncthreads();

  const int writePos = tileStart + threadIdx.x;
  //Assuming dataW and ROW_TILE_W are multiples of half-warp size,
  //rowStart + tileStart is also a multiple of half-warp size,
  //thus having proper alignment for coalesced d_Result[] write.
  if(writePos <= tileEndClamped){
    const int smemPos = writePos - apronStart;
    float sum = 0;
    for(int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++) {
      sum += data[smemPos + k] * d_Kernel[KERNEL_RADIUS - k];
    }
    d_Result[rowStart + writePos] = sum;
  }
}



////////////////////////////////////////////////////////////////////////////////
// Column convolution filter (from nVidia SDK)
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumnGPU(
    float *d_Result,
    const float *d_Data,
    const float *d_Kernel,
    int KERNEL_RADIUS,
    int dataW,
    int dataH,
    int smemStride,
    int gmemStride )
{
  //Data cache
  extern __shared__ float data[];

  //Current tile and apron limits, in rows
  const int         tileStart = IMUL(blockIdx.y, COLUMN_TILE_H);
  const int           tileEnd = tileStart + COLUMN_TILE_H - 1;
  const int        apronStart = tileStart - KERNEL_RADIUS;
  const int          apronEnd = tileEnd   + KERNEL_RADIUS;

  //Clamp tile and apron limits by image borders
  const int    tileEndClamped = min(tileEnd, dataH - 1);
  const int apronStartClamped = max(apronStart, 0);
  const int   apronEndClamped = min(apronEnd, dataH - 1);

  //Current column index
  const int       columnStart = IMUL(blockIdx.x, COLUMN_TILE_W) + threadIdx.x;

  //Shared and global memory indices for current column
  int smemPos = IMUL(threadIdx.y, COLUMN_TILE_W) + threadIdx.x;
  int gmemPos = IMUL(apronStart + threadIdx.y, dataW) + columnStart;
  //Cycle through the entire data cache
  //Load global memory values, if indices are within the image borders,
  //or initialize with zero otherwise
  for(int y = apronStart + threadIdx.y; y <= apronEnd; y += blockDim.y) {
    data[smemPos] = (y < apronStartClamped) ? d_Data[IMUL(apronStartClamped,dataW) + columnStart] :
      ((y > apronEndClamped) ? d_Data[IMUL(apronEndClamped,dataW) + columnStart] :
       d_Data[gmemPos]);
    smemPos += smemStride;
    gmemPos += gmemStride;
  }

  //Ensure the completness of the loading stage
  //because results, emitted by each thread depend on the data, 
  //loaded by another threads
  __syncthreads();
  //Shared and global memory indices for current column
  smemPos = IMUL(threadIdx.y + KERNEL_RADIUS, COLUMN_TILE_W) + threadIdx.x;
  gmemPos = IMUL(tileStart + threadIdx.y , dataW) + columnStart;
  //Cycle through the tile body, clamped by image borders
  //Calculate and output the results
  for(int y = tileStart + threadIdx.y; y <= tileEndClamped; y += blockDim.y) {
    float sum = 0;
    for(int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++) {
      sum += 
	data[smemPos + IMUL(k, COLUMN_TILE_W)] *
	d_Kernel[KERNEL_RADIUS - k];
    }
    d_Result[gmemPos] = sum;
    smemPos += smemStride;
    gmemPos += gmemStride;
  }
}



__global__ void cuda_convolution_nonsep_device( int W, int H,
						float *k,
						int w, int h,
						int w2, int h2,
						const float *s, float *d )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;

  // Compute local convolution
  float v = 0.0f;
  float n = 0.0f;
  int index=0;
  for ( int j=0; j<h; j++ ) {
    for ( int i=0; i<w; i++ ) {
      
      int xx = ox - w2 + i;
      int yy = oy - h2 + j;

      if ( xx>=0 && xx<W && yy>=0 && yy<H ) {
	float kv = k[index];
	n += kv;
	v += kv * s[ yy * W + xx ];
      }

      index++;
    }
  }

  if ( n>0.0f ) {
    v /= n;
  }

  d[o] = v;
}


// Slow nonseparable version
static bool cuda_convolution_nonsep( const coco::cuda_kernel *kernel, 
				     size_t W, size_t H,
				     const float* in, float *out )
{
  // Matrix size has to be multiple of block size.
  dim3 dimGrid, dimBlock;
  coco::cuda_default_grid( W,H, dimGrid, dimBlock );
  //coco::write_test_image_unsigned( W,H, (float*)in, "./out/conv_in.png", 0 );

  // Compute divergence step
  cuda_convolution_nonsep_device<<< dimGrid, dimBlock >>>
    ( W, H,
      kernel->_data,
      kernel->_w, kernel->_h,
      ( kernel->_w -1 ) / 2, ( kernel->_h - 1 ) / 2, 
      in, out );

  //coco::write_test_image_unsigned( W,H, out, "./out/conv_out.png", 0 );
  //assert( false );
  return true;
}




// Convolve array with kernel
bool coco::cuda_convolution( const cuda_kernel *kernel, 
			     size_t W, size_t H,
			     const float* in, float* out )
{
  if ( !kernel->_separable ) {
    return cuda_convolution_nonsep( kernel, W,H, in, out );
  }
  // Needs a temp array
  float *tmp = NULL;
  CUDA_SAFE_CALL( cudaMalloc( &tmp, W*H*sizeof(float) ));

  // Compute radius
  const int KERNEL_RADIUS_X = kernel->_w / 2;
  assert( kernel->_w == size_t(KERNEL_RADIUS_X*2 + 1) );
  const int KERNEL_RADIUS_Y = kernel->_h / 2;
  assert( kernel->_h == size_t(KERNEL_RADIUS_Y*2 + 1) );
  // Compute alignment radius: must be multiple of 16 (half warp size)
  // for maximum performance.
  const int KERNEL_RADIUS_ALIGNED = ((KERNEL_RADIUS_X-1) / 16 + 1) * 16;

  // Call CUDA kernels
  dim3 blockGridRows(iDivUp(W, ROW_TILE_W), H);
  dim3 blockGridColumns(iDivUp(W, COLUMN_TILE_W), iDivUp(H, COLUMN_TILE_H));
  dim3 threadBlockRows(KERNEL_RADIUS_ALIGNED + ROW_TILE_W + KERNEL_RADIUS_X);
  dim3 threadBlockColumns(COLUMN_TILE_W, 8);

  CUDA_SAFE_CALL( cudaMemset( out, 0, sizeof(float)*W*H ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  size_t memsize_row = sizeof(float) * (KERNEL_RADIUS_X + ROW_TILE_W + KERNEL_RADIUS_X);
  convolutionRowGPU<<<blockGridRows, threadBlockRows, memsize_row>>>
    (tmp, in,
     kernel->_data_x,
     KERNEL_RADIUS_X,
     KERNEL_RADIUS_ALIGNED,
     W,H );
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  size_t memsize_column = sizeof(float) * COLUMN_TILE_W * (KERNEL_RADIUS_Y + COLUMN_TILE_H + KERNEL_RADIUS_Y);
  convolutionColumnGPU<<<blockGridColumns, threadBlockColumns, memsize_column>>>
    ( out, tmp,
      kernel->_data_y, KERNEL_RADIUS_Y,
      W,H,
      COLUMN_TILE_W * threadBlockColumns.y,
      W * threadBlockColumns.y );

  CUDA_SAFE_CALL( cudaFree( tmp ));

  return false;
}



static __global__ void convolution_row3_device( int W, int H,
						float k0, float k1, float k2,
						const float *in, float *out )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
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
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
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
bool coco::cuda_convolution_row( float k0, float k1, float k2,
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
bool coco::cuda_convolution_column( float k0, float k1, float k2,
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


