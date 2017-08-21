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
#include "compute_array.h"

using namespace coco;
using namespace std;



/***************************************************************
  CUDA vector of 2D float arrays
****************************************************************/

// Construction and destruction
vector_valued_function_2D::vector_valued_function_2D()
{
  // Public members
  _N = 0;
  _G = NULL;
  _CE = NULL;
  _buffer = NULL;
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
  TRACE5( "    data " << _buffer << "  nf " << _G->nbytes() << endl );
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

// Access to pointer buffer
compute_buffer& vector_valued_function_2D::buffer()
{return *_buffer;}

const compute_buffer& vector_valued_function_2D::buffer() const
{return *_buffer;}

// Access to individual channel
compute_buffer &vector_valued_function_2D::channel( int dim )
{
  assert( dim>=0 && dim<_N );
  return *(_channels[dim]);
}

const compute_buffer &vector_valued_function_2D::channel( int dim ) const
{
  assert( dim>=0 && dim<_N );
  return *(_channels[dim]);
}


// Copying and assigning forbidden.
vector_valued_function_2D &vector_valued_function_2D::operator= ( const vector_valued_function_2D & )
{
  assert( false );
  return *this;
}

vector_valued_function_2D::vector_valued_function_2D( const vector_valued_function_2D & )
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
  _CE = G->engine();
  _N = N;

  _buffer = _G->alloc_layers( N );
  assert( _buffer != NULL );

  int layer_size = bytes_per_dim();
  for ( int i=0; i<_N; i++ ) {
    int start_mem = layer_size * i;
    compute_buffer *channel = new compute_buffer( _buffer, start_mem, layer_size );
    assert( channel != NULL );
    _channels.push_back( channel );
  }

  TRACE5( "  allocated " << W() << " x " << H() << " x " << _N << "  " << _G->nbytes() << " bytes, mem " << _buffer << endl );
  _fbuffer = new float[ W() * H() ];
  return true;
}

// Release memory
bool vector_valued_function_2D::free()
{
  for ( size_t i=0; i<_channels.size(); i++ ) {
    delete _channels[i];
  }
  _channels.clear();
  delete _buffer;
  _buffer = NULL;
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
    channel(i).memset( 0 );
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
  return channel(n).memcpy_from_cpu( src );
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
  return channel(n).memcpy_from_cpu( _fbuffer );
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
  assert( _buffer != NULL );
  assert( equal_dim( G ));
  return _buffer->memcpy_from_engine( &G->buffer() );
}


bool vector_valued_function_2D::copy_from_gpu( const vector<compute_buffer*> &V )
{
  assert( (int)V.size() >= _N );
  for ( int i=0; i<_N; i++ ) {
    compute_buffer *source_buffer = V[i];
    assert( source_buffer != NULL ); 
    channel(i).memcpy_from_engine( source_buffer );
  }
  return true;
}


bool vector_valued_function_2D::copy_to_cpu( vector<gsl_matrix*> &V ) const
{
  assert( (int)V.size() >= _N );
  for ( int i=0; i<_N; i++ ) {
    gsl_matrix *M = V[i];
    assert( M != NULL );
    assert( (int)M->size2 == W() );
    assert( (int)M->size1 == H() );
    copy_to_cpu( i, M->data );
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
  return channel(n).memcpy_to_cpu( dest );
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
  channel(n).memcpy_to_cpu( _fbuffer );
  for ( int i=0; i< W()*H(); i++ ) {
    dest[i] = _fbuffer[i];
  }
  return true;
}



bool vector_valued_function_2D::trace_pixel( int x, int y ) const
{
  TRACE( "    ca@(" << x << " " << y << ") " );
  for ( int i=0; i<_N; i++ ) {
    copy_to_cpu( i, _fbuffer );
    TRACE( _fbuffer[ x + y*W() ] << " " ); 
  }
  TRACE( endl );
  return true;
}
