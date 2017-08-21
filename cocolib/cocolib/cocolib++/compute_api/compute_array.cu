/* -*-c++-*- */
/** \file compute_array.cu

    Structure for (vector) arrays allocated on GPU, many helper functions
    Simplifies standard allocation and data transfer operations

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

#include "../../cuda/cuda_interface.h"
#include "../../cuda/cuda_kernels.cuh"
#include "../../cuda/cuda_reduce.h"

#include "compute_array.h"
#include "../vtv_kernels.cuh"

#include "../../defs.h"

using namespace coco;




/***************************************************************
  CUDA vector of 2D float arrays
****************************************************************/

// Construction and destruction
vector_valued_function_2D::vector_valued_function_2D()
{
  // Public members
  _N = 0;
  _G = NULL;
  _data = NULL;
  _w = NULL;
  _fbuffer = NULL;
}


vector_valued_function_2D::~vector_valued_function_2D()
{
  free();
}


bool vector_valued_function_2D::dump() const
{
  TRACE5( "  vector_valued_function_2D:: " << this << endl );
  TRACE5( "    W " << _G->W() << "  H " << _G->H() << "  N " << _N << endl );
  TRACE5( "    data " << _data << "  nf " << _G->nbytes() << endl );
  return true;
}



// Simple queries
int vector_valued_function_2D::N() const
{return _N;}
int vector_valued_function_2D::W() const
{return _G->W();}
int vector_valued_function_2D::H() const
{return _G->H();}

compute_grid* vector_valued_function_2D::grid() const
{return _G;}

int vector_valued_function_2D::bytes_per_dim() const
{return W()*H()*sizeof(float);}

int vector_valued_function_2D::total_bytes() const
{return _G->nbytes();}

// Return CUDA workspace
vector_valued_function_2D_workspace *vector_valued_function_2D::workspace() const
{return _w;}

// Access to pointer buffer
float* vector_valued_function_2D::mem()
{return _data;}

const float* vector_valued_function_2D::mem() const
{return _data;}

vector_valued_function_2D::operator float*()
{return _data;}

vector_valued_function_2D::operator const float*() const
{return _data;}

// Access to individual channel
float *vector_valued_function_2D::channel( int dim )
{
  assert( dim>=0 && dim<_N );
  return _data + dim*W()*H();
}

const float *vector_valued_function_2D::channel( int dim ) const
{
  assert( dim>=0 && dim<_N );
  return _data + dim*W()*H();
}


// Copying and assigning forbidden.
vector_valued_function_2D &vector_valued_function_2D::operator= ( const vector_valued_function_2D & )
{
  assert( false );
  return *this;
}

vector_valued_function_2D::vector_valued_function_2D( const vector_valued_function_2D &V )
{
  assert( false );
}




// Alloc memory on GPU
// given number of channels N, width W and height H
bool vector_valued_function_2D::alloc( compute_grid *G, int N )
{
  free();
  assert( G != NULL && N>0 );
  _G = G;
  _N = N;
  _data = _G->alloc_layers( N );
  assert( _data != NULL );
  TRACE5( "  allocated " << W() << " x " << H() << " x " << _N << "  " << _G->nbytes() << " bytes, mem " << _data << endl );
  _w = new vector_valued_function_2D_workspace;
  cuda_default_grid( W(),H(), _w->_dimGrid, _w->_dimBlock );
  _fbuffer = new float[ W() * H() ];
  return true;
}

// Release memory
bool vector_valued_function_2D::free()
{
  if ( _data != NULL ) {
    CUDA_SAFE_CALL( cudaFree( _data ));
  }
  _data = NULL;
  delete _w;
  _w = NULL;
  delete[] _fbuffer;
  _fbuffer = NULL;
  return true;
}

// Check for compatibility (same dimension)
bool vector_valued_function_2D::equal_dim( const vector_valued_function_2D *U ) const
{
  assert( U != NULL );
  return (_N == U->_N && W() == U->W() && H() == U->H() );
}



// Set all values in array to zero
bool vector_valued_function_2D::set_zero()
{
  for ( int i=0; i<_N; i++ ) {
    float *buffer = channel(i);
    assert( buffer != NULL );
    CUDA_SAFE_CALL( cudaMemset( buffer, 0, _G->nbytes() ));
  }
  return true;
}



    
// Copy from and to various sources (for convenience)
bool vector_valued_function_2D::copy_from_cpu( const gsl_matrix* M )
{
  assert( _N == 1 );
  assert( M != NULL );
  assert( (int)M->size2 == W() );
  assert( (int)M->size1 == H() );
  copy_from_cpu( 0, M->data );
  return true;
}

bool vector_valued_function_2D::copy_from_cpu( const vector<gsl_matrix*> &V )
{
  assert( (int)V.size() >= _N );
  for ( int i=0; i<_N; i++ ) {
    gsl_matrix *M = V[i];
    assert( M != NULL );
    assert( (int)M->size2 == W() );
    assert( (int)M->size1 == H() );
    copy_from_cpu( i, M->data );
  }
  return true;
}

bool vector_valued_function_2D::copy_from_cpu( const gsl_image *I )
{
  vector<gsl_matrix*> V = gsl_image_get_channels( const_cast<gsl_image*>(I) );
  return copy_from_cpu( V );
}

bool vector_valued_function_2D::copy_from_cpu( int n, const float *src )
{
  float *dest = channel(n);
  CUDA_SAFE_CALL( cudaMemcpy( dest, src, bytes_per_dim(), cudaMemcpyHostToDevice ));
  return true;
}


bool vector_valued_function_2D::copy_from_cpu( const vector<float*> &V )
{
  assert( (int)V.size() >= _N );
  for ( int i=0; i<_N; i++ ) {
    copy_from_cpu( i, V[i] );
  }
  return true;
}

bool vector_valued_function_2D::copy_from_cpu( const float **mem_ptrs )
{
  for ( int i=0; i<_N; i++ ) {
    copy_from_cpu( i, mem_ptrs[i] );
  }
  return true;
}

bool vector_valued_function_2D::copy_from_cpu( const float *mem )
{
  for ( int i=0; i<_N; i++ ) {
    copy_from_cpu( i, mem + W()*H()*i );
  }
  return true;
}

bool vector_valued_function_2D::copy_from_cpu( int n, const double *src )
{
  int N = W() * H();
  for ( int i=0; i<N; i++ ) {
    _fbuffer[i] = src[i];
  }
  float *dest = channel(n);
  CUDA_SAFE_CALL( cudaMemcpy( dest, _fbuffer, bytes_per_dim(), cudaMemcpyHostToDevice ));
  return true;
}


bool vector_valued_function_2D::copy_from_cpu( const vector<double*> &V )
{
  assert( (int)V.size() >= _N );
  for ( int i=0; i<_N; i++ ) {
    copy_from_cpu( i, V[i] );
  }
  return true;
}

bool vector_valued_function_2D::copy_from_cpu( const double **mem_ptrs )
{
  for ( int i=0; i<_N; i++ ) {
    copy_from_cpu( i, mem_ptrs[i] );
  }
  return true;
}

bool vector_valued_function_2D::copy_from_cpu( const double *mem )
{
  for ( int i=0; i<_N; i++ ) {
    copy_from_cpu( i, mem + i * W() * H() );
  }
  return true;
}

bool vector_valued_function_2D::copy_from_gpu( const vector_valued_function_2D *G )
{
  assert( equal_dim( G ));
  CUDA_SAFE_CALL( cudaMemcpy( mem(), G->mem(), _N*_G->nbytes(), cudaMemcpyDeviceToDevice ));
  return true;
}

bool vector_valued_function_2D::copy_from_gpu( const vector<float*> &V )
{
  assert( (int)V.size() >= _N );
  for ( int i=0; i<_N; i++ ) {
    float *dest_buffer = channel(i);
    float *source_buffer = V[i];
    assert( dest_buffer != NULL ); 
    assert( source_buffer != NULL ); 
    CUDA_SAFE_CALL( cudaMemcpy( dest_buffer, source_buffer, W()*H()*sizeof(float), cudaMemcpyDeviceToDevice ));
  }
  return true;
}


bool vector_valued_function_2D::copy_to_cpu( vector<gsl_matrix*> &V ) const
{
  assert( (int)V.size() >= _N );
  for ( int i=0; i<_N; i++ ) {
    const float *buffer = channel(i);
    assert( buffer != NULL );
    gsl_matrix *M = V[i];
    assert( M != NULL );
    assert( (int)M->size2 == W() );
    assert( (int)M->size1 == H() );
    cuda_memcpy( M, buffer );
  }
  return true;
}



bool vector_valued_function_2D::copy_to_cpu( gsl_image *I ) const
{
  vector<gsl_matrix*> V = gsl_image_get_channels( I );
  return copy_to_cpu( V );
}



bool vector_valued_function_2D::copy_to_cpu( vector<float*> &V ) const
{
  assert( (int)V.size() >= _N );
  for ( int i=0; i<_N; i++ ) {
    copy_to_cpu( i, V[i] );
  }
  return true;
}

bool vector_valued_function_2D::copy_to_cpu( float **mem_ptrs ) const
{
  for ( int i=0; i<_N; i++ ) {
    copy_to_cpu( i, mem_ptrs[i] );
  }
  return true;
}

bool vector_valued_function_2D::copy_to_cpu( float *mem ) const
{
  for ( int i=0; i<_N; i++ ) {
    copy_to_cpu( i, mem + W()*H()*i );
  }
  return true;
}

bool vector_valued_function_2D::copy_to_cpu( int n, float *dest ) const
{
  const float *src = channel(n);
  assert( src != NULL );
  CUDA_SAFE_CALL( cudaMemcpy( dest, src, bytes_per_dim(), cudaMemcpyDeviceToHost ));
  return true;
}



bool vector_valued_function_2D::copy_to_cpu( vector<double*> &V ) const
{
  assert( (int)V.size() >= _N );
  for ( int i=0; i<_N; i++ ) {
    copy_to_cpu( i, V[i] );
  }
  return true;
}

bool vector_valued_function_2D::copy_to_cpu( double **mem_ptrs ) const
{
  for ( int i=0; i<_N; i++ ) {
    copy_to_cpu( i, mem_ptrs[i] );
  }
  return true;
}

bool vector_valued_function_2D::copy_to_cpu( double *mem ) const
{
  for ( int i=0; i<_N; i++ ) {
    copy_to_cpu( i, mem + W()*H()*i );
  }
  return true;
}

bool vector_valued_function_2D::copy_to_cpu( int n, double *dest ) const
{
  const float *src = channel(n);
  assert( src != NULL );
  CUDA_SAFE_CALL( cudaMemcpy( _fbuffer, src, bytes_per_dim(), cudaMemcpyDeviceToHost ));
  for ( int i=0; i< W()*H(); i++ ) {
    dest[i] = _fbuffer[i];
  }
  return true;
}



// Array arithmetics

// L1-Norm
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
  vector_valued_function_2D_workspace *w = U->workspace();
  assert( w != NULL );

  // Reprojection
  for ( int i=0; i<nfields; i++ ) {
    cuda_reproject_to_ball_2d<<< w->_dimGrid, w->_dimBlock >>>
      ( U->W(), U->H(), radius, U->channel(start_index + 2*i + 0), U->channel(start_index + 2*i + 1) );
  }
  return true;
}

bool coco::reprojection_weighted_max_norm( vector_valued_function_2D *U, const float *weight, int start_index, int nfields )
{
  return false;
}

// Reprojection according to Frobenius norm
// (in effect, reprojected as if one single long vector)
bool coco::reprojection_frobenius_norm( vector_valued_function_2D *U, float radius, int start_index, int nfields )
{
  assert( U != NULL );
  assert( start_index >= 0 );
  assert( nfields >= 0 );
  assert( start_index + nfields*2 <= U->N() );
  vector_valued_function_2D_workspace *w = U->workspace();
  assert( w != NULL );

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
    cuda_reproject_to_ball_2d<<< w->_dimGrid, w->_dimBlock >>>
      ( U->W(), U->H(), radius, U->channel(start_index + 0), U->channel(start_index + 1) );
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
    kernel_vtvf_reproject_3D<<< w->_dimGrid, w->_dimBlock >>>
      ( U->W(), U->H(),
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
    kernel_vtvf_reproject_3D<<< w->_dimGrid, w->_dimBlock >>>
      ( U->W(), U->H(),
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

bool coco::reprojection_weighted_frobenius_norm( vector_valued_function_2D *U, const float *weight, int start_index, int nfields )
{
  return false;
}



// Reprojection according to Nuclear norm
// Only supports nfields==1 or nfields==3
bool coco::reprojection_nuclear_norm( vector_valued_function_2D *U, float radius, int start_index, int nfields )
{
  return false;
}

bool coco::reprojection_weighted_nuclear_norm( vector_valued_function_2D *U, const float *weight, int start_index, int nfields )
{
  return false;
}
