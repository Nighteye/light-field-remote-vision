/* -*-c++-*- */
/** \file volume_4d.cpp
   data structure for 3d/4d-volumes

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

#include "cuda_volume_4d.h"
#include "cuda_volume_4d.cuh"
#include "cuda_helper.h"


/*****************************************************************************
       4D volume creation/access
*****************************************************************************/

// Create empty 4D volume
coco::cuda_volume_4d *coco::cuda_volume_4d_alloc( size_t X, size_t Y, size_t S, size_t T )
{
  cuda_volume_4d *V = new cuda_volume_4d;
  memset( V, 0, sizeof( cuda_volume_4d ));
  V->_w = new cuda_volume_4d_workspace;
  memset( V->_w, 0, sizeof( cuda_volume_4d_workspace ));

  assert( S*T*X*Y > 0 );
  V->_S = S;
  V->_T = T;
  V->_X = X;
  V->_Y = Y;

  V->_w->_nbytes = S*T*X*Y*sizeof( float );
  TRACE2( "Allocating 4D volume " << S << "x" << T << " layers size " << X << " x " << Y << std::endl );
  TRACE2( "  total memory: " << V->_w->_nbytes / 1048576 << " MiB." << std::endl );
  CUDA_SAFE_CALL( cudaMalloc( &V->_w->_data, V->_w->_nbytes ));
  V->_w->_nbytes_layer = X*Y*sizeof(float);

  // Block sizes
  cuda_default_grid( X,Y, V->_w->_dimGrid, V->_w->_dimBlock );
  return V;
}

// Destroy 4D volume
bool coco::cuda_volume_4d_free( cuda_volume_4d *V )
{
  if ( V == NULL ) {
    return false;
  }
  CUDA_SAFE_CALL( cudaFree( V->_w->_data ));
  delete V->_w;
  delete V;
  return true;
}

// Return offset of element at specific position
size_t coco::cuda_volume_4d_offset( cuda_volume_4d *V, size_t x, size_t y, size_t s, size_t t )
{
  assert( x<V->_X );
  assert( y<V->_Y );
  assert( s<V->_S );
  assert( t<V->_T );
  size_t N = V->_X * V->_Y;
  size_t layer = s + t*V->_S;
  return layer * N + x + y*V->_X;
}


// Set one layer of the volume
bool coco::cuda_volume_4d_set_layer( cuda_volume_4d *V, size_t s, size_t t, float *data )
{
  cuda_volume_4d_workspace *w = V->_w;
  assert( w != NULL );

  CUDA_SAFE_CALL( cudaMemcpy( data,
			      w->_data + cuda_volume_4d_offset( V, 0,0,s,t ),
			      w->_nbytes_layer,
			      cudaMemcpyDeviceToHost ));
  return true;
}

// Get one layer of the volume
bool coco::cuda_volume_4d_get_layer( cuda_volume_4d *V, size_t s, size_t t, float *data )
{
  cuda_volume_4d_workspace *w = V->_w;
  assert( w != NULL );

  CUDA_SAFE_CALL( cudaMemcpy( w->_data + cuda_volume_4d_offset( V, 0,0,s,t ),
			      data,
			      w->_nbytes_layer,
			      cudaMemcpyHostToDevice ));
  return true;
}




/*****************************************************************************
       Layer extraction / write back
       Used in light field analysis (yields epipolar plane images)
*****************************************************************************/

static __global__ void extract_yt_slice_device( int W, int H, int XY,
						float *v, float *s )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int off_w = ox + oy*W;
  const int off_r = ox + oy*XY;
  s[off_w] = v[off_r];
}

// Extraction for constant y/t (horizontal slice)
bool coco::cuda_volume_4d_extract_yt_slice( cuda_volume_4d *V, size_t y, size_t t, float *slice )
{
  size_t W = V->_X;
  size_t H = V->_S;
  dim3 dimGrid;
  dim3 dimBlock;
  cuda_default_grid( W,H, dimGrid, dimBlock );

  extract_yt_slice_device<<< dimGrid, dimBlock >>>
    ( W,H, V->_X * V->_Y,
      V->_w->_data + y*V->_X + t*V->_X*V->_Y*V->_S,
      slice );

  return true;
}



static __global__ void write_back_yt_slice_device( int W, int H, int XY,
						   float *s, float *v )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int off_r = ox + oy*W;
  const int off_w = ox + oy*XY;
  v[off_w] = s[off_r];
}

// Write back for constant y/t (horizontal slice)
bool coco::cuda_volume_4d_write_back_yt_slice( cuda_volume_4d *V, size_t y, size_t t, float *slice )
{
  size_t W = V->_X;
  size_t H = V->_S;
  dim3 dimGrid;
  dim3 dimBlock;
  cuda_default_grid( W,H, dimGrid, dimBlock );

  write_back_yt_slice_device<<< dimGrid, dimBlock >>>
    ( W,H, V->_X * V->_Y,
      slice,
      V->_w->_data + y*V->_X + t*V->_X*V->_Y*V->_S );

  return true;
}



static __global__ void extract_xs_slice_device( int W, int H, int X, int XYS,
						float *v, float *s )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int off_w = ox + oy*W;
  const int off_r = ox*X + oy*XYS;
  s[off_w] = v[off_r];
}

// Extraction for constant x/s (vertical slice)
bool coco::cuda_volume_4d_extract_xs_slice( cuda_volume_4d *V, size_t x, size_t s, float *slice )
{
  size_t W = V->_Y;
  size_t H = V->_T;
  dim3 dimGrid;
  dim3 dimBlock;
  cuda_default_grid( W,H, dimGrid, dimBlock );

  extract_xs_slice_device<<< dimGrid, dimBlock >>>
    ( W,H, V->_X, V->_X * V->_Y * V->_S,
      V->_w->_data + x + s*V->_X*V->_Y,
      slice );

  return true;
}


static __global__ void write_back_xs_slice_device( int W, int H, int X, int XYS,
						   float *s, float *v )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int off_r = ox + oy*W;
  const int off_w = ox*X + oy*XYS;
  v[off_w] = s[off_r];
}

// Write back for constant x/s (vertical slice)
bool coco::cuda_volume_4d_write_back_xs_slice( cuda_volume_4d *V, size_t x, size_t s, float *slice )
{
  size_t W = V->_Y;
  size_t H = V->_T;
  dim3 dimGrid;
  dim3 dimBlock;
  cuda_default_grid( W,H, dimGrid, dimBlock );

  write_back_xs_slice_device<<< dimGrid, dimBlock >>>
    ( W,H, V->_X, V->_X * V->_Y * V->_S,
      slice,
      V->_w->_data + x + s*V->_X*V->_Y );

  return true;
}


// Return image buffer at (s,t) location
float* coco::cuda_volume_4d_image_buffer( cuda_volume_4d *V, size_t s, size_t t )
{
  assert( s < V->_S );
  assert( t < V->_T );
  size_t offset = (s + t*V->_S) * V->_X * V->_Y;
  return V->_w->_data + offset;
}





/*****************************************************************************
       4D volume derivative filters, computed on a single (S,T) layer.
*****************************************************************************/

// central derivative X direction
bool coco::cuda_volume_4d_derivative_central_x( cuda_volume_4d *V, size_t s, size_t t, float *target_layer )
{
  return false;
}

// central derivative Y direction
bool coco::cuda_volume_4d_derivative_central_y( cuda_volume_4d *V, size_t s, size_t t, float *target_layer )
{
  return false;
}

// central derivative S direction
bool coco::cuda_volume_4d_derivative_central_s( cuda_volume_4d *V, size_t s, size_t t, float *target_layer )
{
  return false;
}

// central derivative T direction
bool coco::cuda_volume_4d_derivative_central_t( cuda_volume_4d *V, size_t s, size_t t, float *target_layer )
{
  return false;
}





/*****************************************************************************
       4D convolutions with 1D kernels, computed on a single (S,T) layer.
*****************************************************************************/

// convolution X direction
bool coco::cuda_volume_4d_convolution_x( cuda_volume_4d *V, cuda_kernel *kernel, size_t s, size_t t, float *target_layer )
{
  return false;
}

// convolution Y direction
bool coco::cuda_volume_4d_convolution_y( cuda_volume_4d *V, cuda_kernel *kernel, size_t s, size_t t, float *target_layer )
{
  return false;
}

// convolution S direction
bool coco::cuda_volume_4d_convolution_s( cuda_volume_4d *V, cuda_kernel *kernel, size_t s, size_t t, float *target_layer )
{
  return false;
}

// convolution T direction
bool coco::cuda_volume_4d_convolution_t( cuda_volume_4d *V, cuda_kernel *kernel, size_t s, size_t t, float *target_layer )
{
  return false;
}




/*****************************************************************************
       Full 4D convolution with separable kernel,
       computed on a vector of (S,T) layers
       Number of target layers must be equal to (smax-smin+1) * (tmax-tmin+1)
       Kernel components must be 1D (full 4d kernel is built by convolution)
*****************************************************************************/

// Gaussian convolution
/*
bool coco::cuda_volume_4d_separable_convolution( cuda_volume_4d *src,
						 const vector<cuda_kernel*> &kernels,
						 size_t smin, size_t tmin,
						 cuda_volume_4d *dest )
{
  size_t S = dest->_S;
  size_t T = dest->_T;
  size_t nlayers = S*T;
  assert( nlayers > 0 );

  // temp buffers.
  size_t N = V->_W * V->_H;
  for ( size_t i=0; i<nlayers; i++ ) {
    float *tmp = NULL;
    CUDA_SAFE_CALL( cudaMalloc( &tmp, sizeof(float) * N ));
    assert( tmp != NULL );
    temp.push_back( tmp );
  }

  // simple part: separable convolutions in X and Y
  for ( size_t t=tmin; t<=tmax; t++ ) {
    for ( size_t s=smin; s<=smax; s++ ) {
      size_t layer_offset = N*(t*S + s);

      // convolve V to tmp in X
      cuda_volume_4d_convolution_x( V, kernels[0], s,t, tmp + layer_offset );
      // convolve tmp to target in Y
      cuda_volume_4d_convolution_y( V, tmp + layer_offset, s,t, target_layers + layer_offset );

  }

  // harder part: convolution in S directions
  for ( size_t t=tmin; t<=tmax; t++ ) {
    for ( size_t s=smin; s<=smax; s++ ) {
      // convolve target to tmp in S
    }
  }

  // hardest part: convolution in T direction
  for ( size_t t=tmin; t<=tmax; t++ ) {
    for ( size_t s=smin; s<=smax; s++ ) {
      // convolve tmp to target in T
    }
  }

}


*/
