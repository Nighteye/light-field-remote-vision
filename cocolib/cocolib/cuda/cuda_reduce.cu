/* -*-c++-*- */
/** \file cuda_reduce.cu

    CUDA reduce operation implementation.
    Uses cudpp library.

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
#include "cuda_reduce.h"
#include "cuda_helper.h"
#include "../defs.h"

using namespace std;

/** KERNEL FROM TALK "Optimizing Parallel Reduction in CUDA"
    by Mark Harris, NVIDIA Developer Technology 
*/
template <class T, unsigned int blockSize>
__global__ void sum_reduce( T *g_idata, T *g_odata, int size )
{
  extern __shared__ T sdata[];

  int tid = threadIdx.x;
  int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
  if ( i>size ) {
    sdata[tid] = 0;
  }
  else {
    sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
  }
  __syncthreads();

  if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
  if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
  if (tid < 32) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
  }
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


// template param gives errors for some reason
template <unsigned int blockSize>
__global__ void sum_reduce_int( int *g_idata, int *g_odata, int size )
{
  extern __shared__ int sidata[];

  int tid = threadIdx.x;
  int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
  if ( i>size ) {
    sidata[tid] = 0;
  }
  else {
    sidata[tid] = g_idata[i] + g_idata[i+blockDim.x];
  }
  __syncthreads();

  if (blockSize >= 512) { if (tid < 256) { sidata[tid] += sidata[tid + 256]; } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { sidata[tid] += sidata[tid + 128]; } __syncthreads(); }
  if (blockSize >= 128) { if (tid < 64) { sidata[tid] += sidata[tid + 64]; } __syncthreads(); }
  if (tid < 32) {
    if (blockSize >= 64) sidata[tid] += sidata[tid + 32];
    if (blockSize >= 32) sidata[tid] += sidata[tid + 16];
    if (blockSize >= 16) sidata[tid] += sidata[tid + 8];
    if (blockSize >= 8) sidata[tid] += sidata[tid + 4];
    if (blockSize >= 4) sidata[tid] += sidata[tid + 2];
    if (blockSize >= 2) sidata[tid] += sidata[tid + 1];
  }
  if (tid == 0) g_odata[blockIdx.x] = sidata[0];
}



/** KERNEL FROM TALK "Optimizing Parallel Reduction in CUDA"
    by Mark Harris, NVIDIA Developer Technology 
*/
template <class T, unsigned int blockSize>
__global__ void max_reduce( T *g_idata, T *g_odata, int size )
{
  extern __shared__ T sdata[];

  int tid = threadIdx.x;
  int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
  if ( i>size ) {
    sdata[tid] = 0;
  }
  else {
    sdata[tid] = max( g_idata[i], g_idata[i+blockDim.x] );
  }
  __syncthreads();

  if (blockSize >= 512) { if (tid < 256) { sdata[tid] = max( sdata[tid], sdata[tid + 256] ); } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { sdata[tid] = max( sdata[tid], sdata[tid + 128] ); } __syncthreads(); }
  if (blockSize >= 128) { if (tid < 64) { sdata[tid] = max( sdata[tid], sdata[tid + 64] ); } __syncthreads(); }
  if (tid < 32) {
    if (blockSize >= 64) sdata[tid] = max( sdata[tid], sdata[tid + 32] );
    if (blockSize >= 32) sdata[tid] = max( sdata[tid], sdata[tid + 16] );
    if (blockSize >= 16) sdata[tid] = max( sdata[tid], sdata[tid + 8] );
    if (blockSize >= 8) sdata[tid] = max( sdata[tid], sdata[tid + 4] );
    if (blockSize >= 4) sdata[tid] = max( sdata[tid], sdata[tid + 2] );
    if (blockSize >= 2) sdata[tid] = max( sdata[tid], sdata[tid + 1] );
  }
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}



/** KERNEL FROM TALK "Optimizing Parallel Reduction in CUDA"
    by Mark Harris, NVIDIA Developer Technology 
*/
template <class T, unsigned int blockSize>
__global__ void min_reduce( T *g_idata, T *g_odata, int size )
{
  extern __shared__ T sdata[];

  int tid = threadIdx.x;
  int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
  if ( i>size ) {
    sdata[tid] = 0;
  }
  else {
    sdata[tid] = min( g_idata[i], g_idata[i+blockDim.x] );
  }
  __syncthreads();

  if (blockSize >= 512) { if (tid < 256) { sdata[tid] = min( sdata[tid], sdata[tid + 256] ); } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { sdata[tid] = min( sdata[tid], sdata[tid + 128] ); } __syncthreads(); }
  if (blockSize >= 128) { if (tid < 64) { sdata[tid] = min( sdata[tid], sdata[tid + 64] ); } __syncthreads(); }
  if (tid < 32) {
    if (blockSize >= 64) sdata[tid] = min( sdata[tid], sdata[tid + 32] );
    if (blockSize >= 32) sdata[tid] = min( sdata[tid], sdata[tid + 16] );
    if (blockSize >= 16) sdata[tid] = min( sdata[tid], sdata[tid + 8] );
    if (blockSize >= 8) sdata[tid] = min( sdata[tid], sdata[tid + 4] );
    if (blockSize >= 4) sdata[tid] = min( sdata[tid], sdata[tid + 2] );
    if (blockSize >= 2) sdata[tid] = min( sdata[tid], sdata[tid + 1] );
  }
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}





/********************************************************
  Reduction functions
*********************************************************/

// Reduce an array to a single float using addition.
// First element of second array contains result, one element is sufficient
bool coco::cuda_sum_reduce( size_t W, size_t H, cuflt *in, cuflt *out, float *cpu_result )
{
  int N           = W*H;
  int block_size  = 128;
  int shared_size = block_size * sizeof(float);
  int blocks      = N / block_size + (N%block_size==0 ? 0 : 1);
  dim3 dimGrid( blocks,1 );
  dim3 dimBlock( block_size,1 );
  TRACE9( "sum reduce " << N << " elements, " << blocks << " blocks " );

  // call kernel
  sum_reduce<float, 128><<< dimGrid, dimBlock, shared_size >>>
    ( in, out, N );

  // end of recursion?
  if ( blocks==1 ) {
    if ( cpu_result != NULL ) {
      CUDA_SAFE_CALL( cudaMemcpy( cpu_result, out, sizeof(float), cudaMemcpyDeviceToHost ));
      CUDA_SAFE_CALL( cudaThreadSynchronize() );
      TRACE9( "result " << *cpu_result << endl );
    }
    return true;
  }

  // enter recursion
  int offset = (blocks / 512 + 1) * 512;
  TRACE9( "next recursion level offset " << offset << endl );
  return cuda_sum_reduce( blocks,1, out, out + offset, cpu_result );
}



// Reduce an array to a single float using addition.
// First element of second array contains result, one element is sufficient
bool coco::cuda_max_reduce( size_t W, size_t H, cuflt *in, cuflt *out, float *cpu_result )
{
  int N           = W*H;
  int block_size  = 128;
  int shared_size = block_size * sizeof(float);
  int blocks      = N / block_size + (N%block_size==0 ? 0 : 1);
  dim3 dimGrid( blocks,1 );
  dim3 dimBlock( block_size,1 );
  TRACE9( "sum reduce " << N << " elements, " << blocks << " blocks " );

  // call kernel
  max_reduce<float, 128><<< dimGrid, dimBlock, shared_size >>>
    ( in, out, N );

  // end of recursion?
  if ( blocks==1 ) {
    if ( cpu_result != NULL ) {
      CUDA_SAFE_CALL( cudaMemcpy( cpu_result, out, sizeof(float), cudaMemcpyDeviceToHost ));
      CUDA_SAFE_CALL( cudaThreadSynchronize() );
      TRACE9( "result " << *cpu_result << endl );
    }
    return true;
  }

  // enter recursion
  int offset = (blocks / 512 + 1) * 512;
  TRACE9( "next recursion level offset " << offset << endl );
  return cuda_max_reduce( blocks,1, out, out + offset, cpu_result );
}



// Reduce an array to a single float using addition.
// First element of second array contains result, one element is sufficient
bool coco::cuda_min_reduce( size_t W, size_t H, cuflt *in, cuflt *out, float *cpu_result )
{
  int N           = W*H;
  int block_size  = 128;
  int shared_size = block_size * sizeof(float);
  int blocks      = N / block_size + (N%block_size==0 ? 0 : 1);
  dim3 dimGrid( blocks,1 );
  dim3 dimBlock( block_size,1 );
  TRACE9( "sum reduce " << N << " elements, " << blocks << " blocks " );

  // call kernel
  min_reduce<float, 128><<< dimGrid, dimBlock, shared_size >>>
    ( in, out, N );

  // end of recursion?
  if ( blocks==1 ) {
    if ( cpu_result != NULL ) {
      CUDA_SAFE_CALL( cudaMemcpy( cpu_result, out, sizeof(float), cudaMemcpyDeviceToHost ));
      CUDA_SAFE_CALL( cudaThreadSynchronize() );
      TRACE9( "result " << *cpu_result << endl );
    }
    return true;
  }

  // enter recursion
  int offset = (blocks / 512 + 1) * 512;
  TRACE9( "next recursion level offset " << offset << endl );
  return cuda_min_reduce( blocks,1, out, out + offset, cpu_result );
}



// Reduce an array to a single float using addition.
// First element of second array contains result, one element is sufficient
bool coco::cuda_sum_reduce( size_t W, size_t H, int *in, int *out, int *cpu_result )
{
  int N           = W*H;
  int block_size  = 128;
  int shared_size = block_size * sizeof(float);
  int blocks      = N / block_size + (N%block_size==0 ? 0 : 1);
  dim3 dimGrid( blocks,1 );
  dim3 dimBlock( block_size,1 );
  TRACE9( "sum reduce " << N << " elements, " << blocks << " blocks " );

  // call kernel
  sum_reduce_int<128><<< dimGrid, dimBlock, shared_size >>>
    ( in, out, N );

  // end of recursion?
  if ( blocks==1 ) {
    if ( cpu_result != NULL ) {
      CUDA_SAFE_CALL( cudaMemcpy( cpu_result, out, sizeof(float), cudaMemcpyDeviceToHost ));
      CUDA_SAFE_CALL( cudaThreadSynchronize() );
      TRACE9( "result " << *cpu_result << endl );
    }
    return true;
  }

  // enter recursion
  int offset = (blocks / 512 + 1) * 512;
  TRACE9( "next recursion level offset " << offset << endl );
  return cuda_sum_reduce( blocks,1, out, out + offset, cpu_result );
}
