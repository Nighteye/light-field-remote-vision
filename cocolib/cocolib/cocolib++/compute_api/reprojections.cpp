/* -*-c++-*- */
/** \file reprojections.cu

    Reprojection functions for compute arrays. To be moved to more high level API,
    abstract kernels.

    Copyright (C) 2014 Bastian Goldluecke,
    
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

#include "../compute_api/reprojections.h"
#include "../compute_api/kernels_reprojections.h"

#include "../../defs.h"

using namespace coco;
using namespace std;



// Array arithmetics

// L1-Norm
/*
float coco::l1_norm( vector_valued_function_2D &V )
{
  // needs a buffer
  int N = V.N();
  int W = V.W();
  int H = V.H();
  float *tmp = cuda_alloc_floats( W*H*2 );

  dim3 dimGrid, dimBlock;
  cuda_default_grid( W,H, dimGrid, dimBlock );

  // For efficiency, treat cases of dim 1-3 with special kernels
  if ( N == 1 ) {
    CUDA_SAFE_CALL( cudaMemcpy( tmp, V.channel(0), sizeof(float)*W*H, cudaMemcpyDeviceToDevice ));
    cuda_abs_device<<< dimGrid, dimBlock >>>
      ( W,H, tmp );
  }
  else if ( N == 2 ) {
    cuda_compute_norm_device<<< dimGrid, dimBlock >>>
      ( W,H, V.channel(0), V.channel(1), tmp );
  }
  else if ( N == 3 ) {
    cuda_compute_norm_device<<< dimGrid, dimBlock >>>
      ( W,H, V.channel(0), V.channel(1), V.channel(2), tmp );
  }
  else {
    // Generic case
    cuda_set_all_device<<< dimGrid, dimBlock >>>
      ( W, H, tmp, 0.0f );
    // TODO
    assert( false );
  }

  // Reduce to single value
  float result = 0.0;
  cuda_sum_reduce( W, H, tmp, tmp+W*H, &result );
  cuda_free( tmp );
  return result;
}


// L2-Norm
float coco::l2_norm( vector_valued_function_2D &V )
{
  // TODO
  assert( false );
  return 0.0f;
}
*/



// Array reprojections
// The reprojection functions project a consecutive subset of
// 2D vector fields within a vector-valued function
// to a circle with given radius according to various norms.
// For nfields==1, all projections are the same

// Reprojection according to maximum norm
// (in effect, each vector field projected separately)
bool coco::reprojection_max_norm( vector_valued_function_2D *U, float radius, int start_index, int nfields )
{
  assert( U != NULL );
  assert( start_index >= 0 );
  assert( nfields >= 0 );
  assert( start_index + nfields*2 <= U->N() );

  // Reprojection
  for ( int i=0; i<nfields; i++ ) {
    kernel_reproject_euclidean_2D
      ( U->grid(),
	radius,
	U->channel(start_index + 2*i + 0),
	U->channel(start_index + 2*i + 1) );
  }
  return true;
}

bool coco::reprojection_weighted_max_norm( vector_valued_function_2D *U, const compute_buffer &weight, int start_index, int nfields )
{
  return false;
}

// Reprojection to Euclidean ball
bool coco::reprojection_frobenius_norm( vector_valued_function_2D *U, float radius, int start_index, int nfields )
{
  assert( U != NULL );
  assert( start_index >= 0 );
  assert( nfields >= 0 );
  assert( start_index + nfields*2 <= U->N() );

  // Reprojection
  if ( nfields == 0 ) {
    // nothing to do
    return true;
  }
  else if ( nfields == 1 ) {
    /*
    if ( w->_g == NULL ) {
      cuda_reproject_to_unit_ball_2d<<< w->_dimGrid, w->_dimBlock >>>
        ( data->W(), data->H(), w->_X1[0], w->_X2[0] );
    }
    else {
    */
    kernel_reproject_euclidean_2D
      ( U->grid(),
	radius,
	U->channel(start_index+0), U->channel(start_index+1) );
  }
  else if ( nfields == 2 ) {
    /*
    if ( w->_g != NULL ) {
      static bool msg = false;
      if ( !msg ) {
	msg = true;
            ERROR( "weighted TV not supported for TV_J" << endl );
      }
    }
    */
    kernel_reproject_euclidean_4D
      ( U->grid(),
	radius,
	U->channel(start_index+0), U->channel(start_index+1),
	U->channel(start_index+2), U->channel(start_index+3) );
  }
  else if ( nfields == 3 ) {
    /*
    if ( w->_g != NULL ) {
      static bool msg = false;
      if ( !msg ) {
	msg = true;
            ERROR( "weighted TV not supported for TV_J" << endl );
      }
    }
    */
    kernel_reproject_euclidean_6D
      ( U->grid(),
	radius,
	U->channel(start_index+0), U->channel(start_index+1),
	U->channel(start_index+2), U->channel(start_index+3),
	U->channel(start_index+4), U->channel(start_index+5) );
  }
  else {
    ERROR( "reprojection Frobenius norm: unsupported number of channels (must be <=3)." << endl );
    assert( false );
  }

  return true;
}

bool coco::reprojection_weighted_frobenius_norm( vector_valued_function_2D *U, const compute_buffer &weight, int start_index, int nfields )
{
  return false;
}



// Reprojection according to Nuclear norm
// Only supports nfields==1 or nfields==3
bool coco::reprojection_nuclear_norm( vector_valued_function_2D *U, float radius, int start_index, int nfields )
{
  assert( U != NULL );
  assert( start_index >= 0 );
  assert( nfields >= 0 );
  assert( start_index + nfields*2 <= U->N() );

  // Reprojection
  if ( nfields == 0 ) {
    // nothing to do
    return true;
  }
  else if ( nfields == 1 ) {
    /*
    if ( w->_g == NULL ) {
      cuda_reproject_to_unit_ball_2d<<< w->_dimGrid, w->_dimBlock >>>
        ( data->W(), data->H(), w->_X1[0], w->_X2[0] );
    }
    else {
    */
    kernel_reproject_euclidean_2D
      ( U->grid(),
	radius,
	U->channel(start_index+0), U->channel(start_index+1) );
  }
  else if ( nfields == 3 ) {
    /*
    if ( w->_g != NULL ) {
      static bool msg = false;
      if ( !msg ) {
	msg = true;
            ERROR( "weighted TV not supported for TV_J" << endl );
      }
    }
    */
    kernel_reproject_nuclear_6D
      ( U->grid(),
	radius,
	U->channel(start_index+0), U->channel(start_index+1),
	U->channel(start_index+2), U->channel(start_index+3),
	U->channel(start_index+4), U->channel(start_index+5) );
  }
  else {
    ERROR( "reprojection Nuclear norm: unsupported number of channels (must be <=3)." << endl );
    assert( false );
  }

  return true;
}

bool coco::reprojection_weighted_nuclear_norm( vector_valued_function_2D *U, const compute_buffer &weight, int start_index, int nfields )
{
  return false;
}
