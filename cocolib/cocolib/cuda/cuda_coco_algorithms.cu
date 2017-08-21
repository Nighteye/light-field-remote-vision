/* -*-c++-*- */
/** \file cuda_coco_algorithms.cu

    High level entry functions for cocolib
Some helper functions for CUDA texture bindings and memcopy operations.

    Copyright (C) 2013 Bastian Goldluecke,
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

#include "cuda_helper.h"
#include "cuda_interface.h"
#include "cuda_coco_algorithms.h"

#include "../common/gsl_matrix_helper.h"
#include "../defs.h"

#include "../vtv/vtv.h"
#include "../tv/tv_linear.h"
#include "../multilabel/tv_multilabel.h"
#include "../multilabel/multilabel.cuh"

using namespace coco;
using namespace std;

// TV-L^p denoising of a raw float array
// supports p=1 or p=2
bool coco::coco_tv_Lp_denoising( float *data, size_t W, size_t H, float lambda, int p, size_t iter, float *weight )
{
  // Just a wrapper for matrix version
  gsl_matrix *M = gsl_matrix_alloc( H,W );
  for ( size_t i=0; i<W*H; i++ ) {
    M->data[i] = data[i];
  }

  coco_tv_Lp_denoising( M, lambda, p, iter, weight );

  for ( size_t i=0; i<W*H; i++ ) {
    data[i] = M->data[i];
  }
  gsl_matrix_free( M );
  return true;
}


// TV-L^p denoising of a matrix
// supports p=1 or p=2
bool coco::coco_tv_Lp_denoising( gsl_matrix *M, float lambda, int p, size_t iter, float *weight )
{
  size_t W = M->size2;
  size_t H = M->size1;
 
  // Zero initialization for testing
  gsl_matrix *zero = gsl_matrix_alloc( H,W );
  memset( zero->data, 0, sizeof(double)*W*H );
  vector<gsl_matrix*> ZERO;
  ZERO.push_back( zero );
  vector<gsl_matrix*> INPUT;
  INPUT.push_back( M );

  // Create solver workspaces
  coco_vtv_data* mtv = coco_vtv_alloc( INPUT );
  mtv->_lambda = lambda;
  mtv->_regularizer = 0;
  mtv->_data_term_p = p;
  assert( p==1 || p==2 );
  if ( weight != NULL ) {
    // set local regularizer weight
    gsl_matrix *M = gsl_matrix_from_buffer( W,H, weight );
    coco_vtv_set_regularizer_weight( mtv, M );
    gsl_matrix_free( M );
  }

  // hack p=1 only implemented in deblurring
  gsl_vector *id = gsl_vector_alloc(3);
  id->data[0] = 0.0;
  id->data[1] = 1.0;
  id->data[2] = 0.0;
  if ( p==1 ) {
    coco_vtv_set_separable_kernel( mtv, id, id );
  }

  coco_vtv_initialize( mtv, INPUT );
  for ( size_t k=0; k<iter; k++ ) {
    // hack p=1 only implemented in deblurring
    if ( p==1 ) {
      coco_vtv_deblurring_iteration_fista( mtv );
    }
    else {
      coco_vtv_rof_iteration_chambolle_pock_1( mtv );
    }
  }
  
  // Get result
  coco_vtv_get_solution( mtv, INPUT );

  // Cleanup
  coco_vtv_free( mtv );
  gsl_matrix_free( zero );
  gsl_vector_free( id );
  return true;
}


// VTV-L^p denoising of a matrix
// supports p=1 or p=2
bool coco::coco_vtv_Lp_denoising( gsl_image *I, float lambda, int p, size_t iter, float *weight )
{
  size_t W = I->_w;
  size_t H = I->_h;
 
  // Zero initialization for testing
  gsl_matrix *zero = gsl_matrix_alloc( H,W );
  memset( zero->data, 0, sizeof(double)*W*H );
  vector<gsl_matrix*> ZERO;
  ZERO.push_back( zero );
  ZERO.push_back( zero );
  ZERO.push_back( zero );
  vector<gsl_matrix*> INPUT;
  INPUT.push_back( I->_r );
  INPUT.push_back( I->_g );
  INPUT.push_back( I->_b );

  // Create solver workspaces
  coco_vtv_data* mtv = coco_vtv_alloc( INPUT );
  mtv->_lambda = lambda;
  mtv->_regularizer = 0;
  mtv->_data_term_p = p;
  assert( p==1 || p==2 );
  if ( weight != NULL ) {
    // set local regularizer weight
    gsl_matrix *M = gsl_matrix_from_buffer( W,H, weight );
    coco_vtv_set_regularizer_weight( mtv, M );
    gsl_matrix_free( M );
  }

  // hack p=1 only implemented in deblurring
  gsl_vector *id = gsl_vector_alloc(3);
  id->data[0] = 0.0;
  id->data[1] = 1.0;
  id->data[2] = 0.0;
  if ( p==1 ) {
    coco_vtv_set_separable_kernel( mtv, id, id );
  }

  coco_vtv_initialize( mtv, INPUT );
  for ( size_t k=0; k<iter; k++ ) {
    // hack p=1 only implemented in deblurring
    if ( p==1 ) {
      coco_vtv_deblurring_iteration_fista( mtv );
    }
    else {
      coco_vtv_rof_iteration_chambolle_pock_1( mtv );
    }
  }
  
  // Get result
  coco_vtv_get_solution( mtv, INPUT );

  // Cleanup
  coco_vtv_free( mtv );
  gsl_vector_free( id );
  gsl_matrix_free( zero );
  return true;
}



// Multilabel algorithm (TV regularity - lifting method)
//
// L         : number of labels
// lmin, lmax: label range
// lambda    : global regularizer weight (larger = more smoothing)
// rho       : data term
//             layout: offset for rho( x,y,l ) is l * W*H + y*W + x
// niter     : number of iterations (depends on problem size and data term precision)
// result    : pointer to W*H float array, layout y*W + x
// reg_weight: point-wise weight for regularizer (overrides lambda)
//
bool coco::coco_tv_multilabel( size_t W, size_t H, size_t N,
			       float lmin, float lmax,
			       float lambda, float *rho,
			       size_t niter,
			       float *result,
			       float *reg_weight )
{
  // Structure for multilabel integration
  tv_multilabel_data *tvm = tv_multilabel_data_alloc( W,H,N );
  tvm->_lambda = lambda;
  // Set label range for multilabel integration
  multilabel_set_label_range( tvm, lmin, lmax );

  // Compute pointwise optimum of data term
  TRACE1( "Computing pointwise optimum of data term ..." );
  for ( size_t y=0; y<H; y++ ) {
    for ( size_t x=0; x<W; x++ ) {
      double vopt = 0.0;
      double eopt = 1e10;
      for ( size_t i=0; i<N; i++ ) {

	size_t index = x + y*W + i*W*H;
	float err = rho[index];
	if ( err<eopt ) {
	  eopt = err;
	  vopt = tvm->_labels[i];
	}

      }

      result[ x + y*W ] = vopt;
    }
  }
  TRACE( " done." << endl );


  // Perform global integration
  // Set data term
  multilabel_set_data_term( tvm, rho );
  if ( reg_weight != NULL ) {
    CUDA_SAFE_CALL( cudaMalloc( &tvm->_w->_g, sizeof( float ) * W*H ));
    CUDA_SAFE_CALL( cudaMemcpy( tvm->_w->_g, reg_weight, sizeof( float ) * W*H, cudaMemcpyHostToDevice ));
  }
  // Set initial solution
  tv_multilabel_set_solution( tvm, result );

  // Init and iterate
  TRACE3( "TV multilabel " << W << " x " << H << " x " << N << ", lambda = " << lambda << endl );
  TRACE3( "  " << niter << " iterations [" );
  clock_t c0 = clock();
  tv_multilabel_init( tvm );
  for ( size_t i=0; i<niter; i++ ) {
    if ( (i%(niter/10+1)) == 0 ) {
      TRACE3( "." ); cout.flush();
    }
    tv_multilabel_iteration( tvm );
  }
  clock_t c1 = clock();
  TRACE3( "] done, " << (c1-c0) / float(CLOCKS_PER_SEC) << "s." << endl );
  
  // Copy result
  tv_multilabel_get_solution( tvm, result );
  tv_multilabel_data_free( tvm );
  return true;
}




// TV-segmentation using input matrix as data term
// result not thresholded yet
bool coco::coco_tv_segmentation( size_t W, size_t H, float *a, float lambda, size_t iter, float *weight )
{
  gsl_matrix *M = gsl_matrix_alloc( H,W );
  size_t N = W*H;
  for ( size_t i=0; i<N; i++ ) {
    M->data[i] = a[i];
  }
  coco_tv_segmentation( M,lambda, iter, weight );
  for ( size_t i=0; i<N; i++ ) {
    a[i] = M->data[i];
  }
  return true;
}


// TV-segmentation using input matrix as data term
// result not thresholded yet
bool coco::coco_tv_segmentation( gsl_matrix *M, float lambda, size_t iter, float *weight )
{
  // Create workspace
  tv_linear_data *data = tv_linear_data_alloc( M );

  // Write initial field as image
  int W = M->size2;
  int H = M->size1;
  gsl_image *I = gsl_image_alloc( W,H );
  gsl_matrix *f = gsl_matrix_alloc( H,W );
  gsl_image_from_signed_matrix( I, M );
  gsl_image_normalize( I );
  gsl_image_save( "./out/dataterm_segmentation.png", I );
  gsl_image_from_matrix( I,f );
  gsl_image_normalize( I );
  gsl_image_save( "./out/init_segmentation.png", I );

  gsl_matrix_copy_to( M,f );
  gsl_matrix_scale( f, -1.0 );
  gsl_matrix_normalize( f );
  gsl_matrix_scale( M, 1.0 / (2.0 * lambda));

  // Perform 200 ROF iterations, fgp algorithm
  TRACE( "Test TV-linear FISTA " << W << "x" << H << endl );
  TRACE( "  [" );
  tv_linear_initialize( data, f );
  clock_t t0 = clock();
  for ( size_t k=0; k<iter; k++ ) {
    if ( k%iter == 0 ) {
      TRACE( "." );
    }
    tv_linear_iteration_fista( data );
  }
  TRACE( "] done." << endl );
  clock_t t1 = clock();
  double secs = double(t1 - t0) / double(CLOCKS_PER_SEC);
  TRACE( "total runtime : " << secs << "s." << endl );
  TRACE( "per iteration : " << secs / double(iter)  << "s." << endl );
  TRACE( "iter / s      : " << double(iter) / secs  << endl );

  // Copy result to buffer
  tv_linear_get_solution( data, M );
  gsl_image_from_signed_matrix( I, M );
  gsl_image_normalize( I );
  gsl_image_save( "./out/result_segmentation.png", I );


  // Cleanup
  gsl_image_free( I );
  gsl_matrix_free( f );
  tv_linear_data_free( data );
  return true;
 }

