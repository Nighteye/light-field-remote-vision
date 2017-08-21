/* -*-c++-*- */
/** \file data_term_deconvolution.cpp

    Data term of the L^p deconvolution model.

    Copyright (C) 2014 Bastian Goldluecke.

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

#include "deconvolution.h"

#include "../compute_api/kernels_vtv.h"
#include "../compute_api/kernels_algebra.h"

using namespace coco;



// Construction and destruction
data_term_deconvolution::data_term_deconvolution()
{
  // empty blur kernel
  _b = NULL;
  _bq = NULL;
  _tmp = new vector_valued_function_2D;
}

data_term_deconvolution::~data_term_deconvolution()
{
  // destroy blur kernel
  delete _b;
  delete _bq;
  delete _tmp;
}


// Resize problem
bool data_term_deconvolution::resize( compute_grid *G, int N )
{
  data_term_denoising::resize( G,N );

  // One dual variable required per solution dimension
  _K = N;

  // Temporary space
  _tmp->alloc( G, N );
  return true;
}


// Init kernel
bool data_term_deconvolution::set_separable_kernel( const gsl_vector *kernel_x, const gsl_vector *kernel_y )
{
  if ( _b != NULL ) {
    delete _b;
    _b = NULL;
  }
  if ( _bq != NULL ) {
    delete _bq;
    _bq = NULL;
  }

  // Reflect kernel components
  int W = kernel_x->size;
  gsl_vector *kernel_xr = gsl_vector_alloc( W );
  for ( int x=0; x<W; x++ ) {
    kernel_xr->data[x] = kernel_x->data[ W-1-x ];
  }
  
  int H = kernel_y->size;
  gsl_vector *kernel_yr = gsl_vector_alloc( H );
  for ( int y=0; y<H; y++ ) {
    kernel_yr->data[y] = kernel_y->data[ H-1-y ];
  }
  
  // Alloc kernels and clean up
  _b = new convolution_kernel( _G->engine(), kernel_x, kernel_y );
  _bq = new convolution_kernel( _G->engine(), kernel_xr, kernel_yr );

  gsl_vector_free( kernel_xr );
  gsl_vector_free( kernel_yr );
  return true;
}



// Init kernel
bool data_term_deconvolution::set_kernel( const gsl_matrix *kernel )
{
  if ( _b != NULL ) {
    delete _b;
    _b = NULL;
  }
  if ( _bq != NULL ) {
    delete _bq;
    _bq = NULL;
  }

  int W = kernel->size1;
  int H = kernel->size2;
  gsl_matrix *kq = gsl_matrix_alloc( W,H );
  for ( int x=0; x<W; x++ ) {
    for ( int y=0; y<H; y++ ) {
      kq->data[ x + y*W ] = kernel->data[ W-1-x + W*(H-1-y) ];
    }
  }


  // Alloc kernels and clean up
  _b = new coco::convolution_kernel( _G->engine(), kernel );
  _bq = new coco::convolution_kernel( _G->engine(), kq );

  gsl_matrix_free( kq );
  return true;
}


// Algorithm implementation
// Perform primal update, i.e. gradient step + reprojection
bool data_term_deconvolution::primal_update( vector_valued_function_2D *U,
				   float *step_U,
				   vector_valued_function_2D *Q )
{
  if ( _b==NULL || _bq==NULL ) {
    ERROR( "deconvolution: kernel not initialized." << std::endl );
    return data_term_denoising::primal_update( U,step_U, Q );
  } 

  // Default is one gradient operator step
  assert( U != NULL );
  assert( Q != NULL );
  assert_valid_dims( U,Q );

  // Assumed memory layout is consecutive storage of dual variables
  // for each dimension, starting at index 0

  // Kernel call for each channel
  for ( int i=0; i<U->N(); i++ ) {
    // b^T * q
    convolution( Q->grid(), _bq,
		 Q->channel(i), _tmp->channel(i) );
    // Descent step is equal to -bq*q times step size
    kernel_linear_combination
      ( U->grid(), -step_U[i], _tmp->channel(i), 1.0f, U->channel(i), U->channel(i) );
  }

  return true;
}



// Perform dual update, i.e. gradient step + reprojection
bool data_term_deconvolution::dual_update( const vector_valued_function_2D *U,
				       vector_valued_function_2D *Q,
				       float *step_Q )
{
  if ( _b==NULL || _bq==NULL ) {
    ERROR( "deconvolution: kernel not initialized." << std::endl );
    return data_term_denoising::dual_update( U,Q, step_Q );
  } 

  assert( U != NULL );
  assert( Q != NULL );
  assert_valid_dims( U,Q );
  assert( U->equal_dim( _F ));

  // Kernel call for each channel
  for ( int i=0; i<U->N(); i++ ) {
    // Ascend step is equal to b*u-f times step size
    // b*u
    convolution( U->grid(), _b,
		 U->channel(i), _tmp->channel(i) );
    // b*u-f
    kernel_subtract_from
      ( _F->grid(), _F->channel(i), _tmp->channel(i) );
    // Update step
    kernel_linear_combination
      ( Q->grid(), step_Q[i], _tmp->channel(i), 1.0f, Q->channel(i), Q->channel(i) );
  }

  dual_reprojection( Q, step_Q );
  return true;
}
