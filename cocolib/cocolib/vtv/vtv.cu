/* -*-c++-*- */

#include "../defs.h"
#include "../cuda/cuda_helper.h"

#include "vtv.h"
#include "vtv.cuh"

#include "../common/gsl_matrix_derivatives.h"

using namespace std;

// Alloc PDE data with sensible defaults
coco::coco_vtv_data* coco::coco_vtv_alloc( vector<gsl_matrix*> F )
{
  size_t nchannels = F.size();
  assert( nchannels > 0 );
  coco_vtv_data *data = new coco_vtv_data;

  // Texture sizes
  data->_nchannels = nchannels;
  data->_W = F[0]->size2;
  data->_H = F[0]->size1;
  data->_N = data->_W * data->_H;
  // Smoothness parameter
  data->_lambda = 1.0f;
  // Regularizer
  data->_regularizer = 2;
  data->_data_term_p = 2;

  // Chosen according to heuristics in paper
  // Primal step size
  data->_tau = 1.0f/sqrtf(8.0f);
  // Dual step size
  data->_sigma = 1.0f/sqrtf(8.0f);
  // Number of inner iterations (fista)
  data->_inner_iterations = 10;
  data->_disp_threshold = 1.;

  // Workspace
  data->_workspace = new coco_vtv_workspace;
  memset( data->_workspace, 0, sizeof( coco_vtv_workspace ));

  // Size of fields over Image space
  data->_nfbytes = data->_N * sizeof(cuflt);
  // Alloc fields
  coco_vtv_workspace *w = data->_workspace;
  w->_U.resize( nchannels );
  w->_Uq.resize( nchannels );
  w->_F.resize( nchannels );
  w->_G.resize( nchannels );
  w->_g.resize( nchannels , NULL );
  w->_X1.resize( nchannels );
  w->_X2.resize( nchannels );
  w->_X1q.resize( nchannels );
  w->_X2q.resize( nchannels );
  w->_temp.resize( nchannels );

  for ( size_t i=0; i<nchannels; i++ ) {

    // Primal variable components
    CUDA_SAFE_CALL( cudaMalloc( (void**)&(w->_U[i]), data->_nfbytes ));
    CUDA_SAFE_CALL( cudaMemset( w->_U[i], 0, data->_nfbytes ));

    CUDA_SAFE_CALL( cudaMalloc( (void**)&(w->_Uq[i]), data->_nfbytes ));
    CUDA_SAFE_CALL( cudaMemset( w->_Uq[i], 0, data->_nfbytes ));

    CUDA_SAFE_CALL( cudaMalloc( (void**)&(w->_G[i]), data->_nfbytes ));
    CUDA_SAFE_CALL( cudaMemset( w->_G[i], 0, data->_nfbytes ));

    // Regularizer weight, if defined
    w->_g[i] = NULL;

    // Dual variable XI
    CUDA_SAFE_CALL( cudaMalloc( (void**)&(w->_X1[i]), data->_nfbytes ));
    CUDA_SAFE_CALL( cudaMalloc( (void**)&(w->_X2[i]), data->_nfbytes ));

    CUDA_SAFE_CALL( cudaMemset( w->_X1[i], 0, data->_nfbytes ));
    CUDA_SAFE_CALL( cudaMemset( w->_X2[i], 0, data->_nfbytes ));

    // Lead dual variable XI
    CUDA_SAFE_CALL( cudaMalloc( (void**)&(w->_X1q[i]), data->_nfbytes ));
    CUDA_SAFE_CALL( cudaMalloc( (void**)&(w->_X2q[i]), data->_nfbytes ));

    CUDA_SAFE_CALL( cudaMemset( w->_X1q[i], 0, data->_nfbytes ));
    CUDA_SAFE_CALL( cudaMemset( w->_X2q[i], 0, data->_nfbytes ));

    CUDA_SAFE_CALL( cudaMalloc( (void**)&(w->_temp[i]), data->_nfbytes ));
    CUDA_SAFE_CALL( cudaMemset( w->_temp[i], 0, data->_nfbytes ));

    // RHS fields
    cuflt *tmp;
    CUDA_SAFE_CALL( cudaMalloc( (void**)&tmp, data->_nfbytes ));

    // Copy RHS to GPU
    gsl_matrix *f = F[i];
    assert( f != NULL );
    assert( f->size2 == data->_W );
    assert( f->size1 == data->_H );
    cuda_memcpy( tmp, f );
    w->_F[i] = tmp;
  }

  // Stencil, if defined
  w->_stencil = NULL;

  // ROF sum data
  w->_delete_rof_sum_data = true;

  // Init rest of fields
  w->_b = NULL;
  w->_bq = NULL;
  w->_nfbytes = data->_nfbytes;
  w->_iteration = 0;

  // CUDA Block dimensions
  size_t W = data->_W;
  size_t H = data->_H;
  w->_dimBlock = dim3( cuda_default_block_size_x(),
		       cuda_default_block_size_y() );
  size_t blocks_w = W / w->_dimBlock.x;
  if ( W % w->_dimBlock.x != 0 ) {
    blocks_w += 1;
  }
  size_t blocks_h = H / w->_dimBlock.y;
  if ( H % w->_dimBlock.y != 0 ) {
    blocks_h += 1;
  }
  w->_dimGrid = dim3(blocks_w, blocks_h);

  data->_basedir = "./out/";
  data->_lambda_max_factor = 1.0f;
  data->_sr_data = NULL;
  data->_sr_data_unstructured = NULL;
  return data;
}


// alloc rarely needed variables
bool coco::coco_vtv_alloc_aux_fields( coco_vtv_data *data )
{
  coco_vtv_workspace *w = data->_workspace;
  if ( w->_E1.size() != 0 ) {
    // already done.
    return false;
  }

  size_t nchannels = data->_nchannels;
  w->_E1.resize( nchannels );
  w->_E2.resize( nchannels );
  w->_X1t.resize( nchannels );
  w->_X2t.resize( nchannels );
  w->_E1q.resize( nchannels );
  w->_E2q.resize( nchannels );

  for ( size_t i=0; i<nchannels; i++ ) {
    // Dual variable ETA
    CUDA_SAFE_CALL( cudaMalloc( (void**)&(w->_E1[i]), data->_nfbytes ));
    CUDA_SAFE_CALL( cudaMalloc( (void**)&(w->_E2[i]), data->_nfbytes ));

    CUDA_SAFE_CALL( cudaMemset( w->_E1[i], 0, data->_nfbytes ));
    CUDA_SAFE_CALL( cudaMemset( w->_E2[i], 0, data->_nfbytes ));

    // Lead dual variable ETA
    CUDA_SAFE_CALL( cudaMalloc( (void**)&(w->_E1q[i]), data->_nfbytes ));
    CUDA_SAFE_CALL( cudaMalloc( (void**)&(w->_E2q[i]), data->_nfbytes ));

    CUDA_SAFE_CALL( cudaMemset( w->_E1q[i], 0, data->_nfbytes ));
    CUDA_SAFE_CALL( cudaMemset( w->_E2q[i], 0, data->_nfbytes ));

    // Temp dual variable XI
    CUDA_SAFE_CALL( cudaMalloc( (void**)&(w->_X1t[i]), data->_nfbytes ));
    CUDA_SAFE_CALL( cudaMalloc( (void**)&(w->_X2t[i]), data->_nfbytes ));

    CUDA_SAFE_CALL( cudaMemset( w->_X1t[i], 0, data->_nfbytes ));
    CUDA_SAFE_CALL( cudaMemset( w->_X2t[i], 0, data->_nfbytes ));
  }

  return true;
}



// Free up PDE data
bool coco::coco_vtv_free( coco_vtv_data *data )
{
  // superresolution data needs to be deallocated separately
  if ( data->_sr_data != NULL ) {
    coco_vtv_sr_free( data );
  }
  coco_vtv_rof_sum_data_free( data );

  // Free GPU fields
  coco_vtv_workspace *w = data->_workspace;
  for ( size_t i=0; i<data->_nchannels; i++ ) {
    CUDA_SAFE_CALL( cudaFree( w->_U[i] ));
    CUDA_SAFE_CALL( cudaFree( w->_Uq[i] ));
    CUDA_SAFE_CALL( cudaFree( w->_F[i] ));
    CUDA_SAFE_CALL( cudaFree( w->_G[i] ));

    CUDA_SAFE_CALL( cudaFree( w->_g[i] ));

    CUDA_SAFE_CALL( cudaFree( w->_X1[i] ));
    CUDA_SAFE_CALL( cudaFree( w->_X2[i] ));

    CUDA_SAFE_CALL( cudaFree( w->_X1q[i] ));
    CUDA_SAFE_CALL( cudaFree( w->_X2q[i] ));
  }

  if ( w->_E1.size() != 0 ) {
    for ( size_t i=0; i<data->_nchannels; i++ ) {
      CUDA_SAFE_CALL( cudaFree( w->_E1[i] ));
      CUDA_SAFE_CALL( cudaFree( w->_E2[i] ));
      
      CUDA_SAFE_CALL( cudaFree( w->_E1q[i] ));
      CUDA_SAFE_CALL( cudaFree( w->_E2q[i] ));
      
      CUDA_SAFE_CALL( cudaFree( w->_X1t[i] ));
      CUDA_SAFE_CALL( cudaFree( w->_X2t[i] ));
    }
  }

  // Temp buffers might be more
  for ( size_t i=0; i<w->_temp.size(); i++ ) {
    CUDA_SAFE_CALL( cudaFree( w->_temp[i] ));
  }
  CUDA_SAFE_CALL( cudaFree( w->_stencil ));

  if ( w->_b != NULL ) {
    cuda_kernel_free( w->_b );
  }

  delete data->_workspace;
  delete data;
  return true;
}



// Initialize workspace with current solution
bool coco::coco_vtv_initialize( coco_vtv_data *data,
				vector<gsl_matrix*> &U )
{
  coco_vtv_workspace *w = data->_workspace;
  assert( U.size() == data->_nchannels );

  for ( size_t i=0; i<data->_nchannels; i++ ) {
    gsl_matrix *u = U[i];
    assert( u->size2 == data->_W );
    assert( u->size1 == data->_H );
    cuda_memcpy( w->_U[i], u );
    cuda_memcpy( w->_Uq[i], u );
  }
  
  // Cleanup
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );

  // Init step sizes
  data->_tau = 1.0 / sqrt( 8.0 );
  data->_sigma = 1.0 / sqrt( 8.0 );
  data->_gamma = 1.0 / data->_lambda;
  data->_alpha = 1.0;

  return true;
}

// Get current solution
bool coco::coco_vtv_get_solution( coco_vtv_data *data,
				  vector<gsl_matrix*> &U )
{
  coco_vtv_workspace *w = data->_workspace;
  assert( U.size() >= data->_nchannels );
  for ( size_t i=0; i<data->_nchannels; i++ ) {
    gsl_matrix *u = U[i];
    assert( u->size2 == data->_W );
    assert( u->size1 == data->_H );
    cuda_memcpy( u, w->_U[i] );
  }
  for ( size_t i=data->_nchannels; i<U.size(); i++ ) {
    gsl_matrix *u = U[i];
    assert( u->size2 == data->_W );
    assert( u->size1 == data->_H );
    cuda_memcpy( u, w->_U[0] );
  }
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
  return true;
}


// Get dual variable XI (vector of dimension 2)
bool coco::coco_vtv_get_dual_xi( coco_vtv_data *data,
				 vector<gsl_matrix*> &XI,
				 size_t channel )
{
  assert( channel < 3 );
  coco_vtv_workspace *w = data->_workspace;
  assert( XI.size() == 2 );
  for ( size_t i=0; i<2; i++ ) {
    gsl_matrix *xi = XI[i];
    assert( xi->size2 == data->_W );
    assert( xi->size1 == data->_H );
  }
  cuda_memcpy( XI[0], w->_X1[channel] );
  cuda_memcpy( XI[1], w->_X2[channel] );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
  return true;
}

// Get dual variable ETA (vector of dimension equal to channel number)
bool coco::coco_vtv_get_dual_eta( coco_vtv_data *data,
				  vector<gsl_matrix*> &ETA )
{
  coco_vtv_workspace *w = data->_workspace;
  assert( ETA.size() == data->_nchannels );
  for ( size_t i=0; i<data->_nchannels; i++ ) {
    gsl_matrix *eta = ETA[i];
    assert( eta->size2 == data->_W );
    assert( eta->size1 == data->_H );
    cuda_memcpy( eta, w->_E1[i] );
  }
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );
  return true;
}



// Init kernel
bool coco::coco_vtv_set_separable_kernel( coco_vtv_data *data, gsl_vector *kernel_x, gsl_vector *kernel_y )
{
  coco_vtv_workspace *w = data->_workspace;
  assert( w->_b == NULL );
  assert( w->_bq == NULL );
  w->_b = coco::cuda_kernel_alloc_separable( kernel_x, kernel_y );

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
  
  w->_bq = coco::cuda_kernel_alloc_separable( kernel_xr, kernel_yr );
  gsl_vector_free( kernel_xr );
  gsl_vector_free( kernel_yr );
  return true;
}



// Init kernel
bool coco::coco_vtv_set_kernel( coco_vtv_data *data, gsl_matrix *kernel )
{
  coco_vtv_workspace *w = data->_workspace;
  assert( w->_b == NULL );
  assert( w->_bq == NULL );

  int W = kernel->size1;
  int H = kernel->size2;
  gsl_matrix *kq = gsl_matrix_alloc( W,H );
  for ( int x=0; x<W; x++ ) {
    for ( int y=0; y<H; y++ ) {
      kq->data[ x + y*W ] = kernel->data[ W-1-x + W*(H-1-y) ];
    }
  }

  w->_b = coco::cuda_kernel_alloc( kernel );
  w->_bq = coco::cuda_kernel_alloc( kq );

  gsl_matrix_free( kq );
  return true;
}


// Init stencil
bool coco::coco_vtv_set_stencil( coco_vtv_data *data, gsl_matrix *stencil )
{
  coco_vtv_workspace *w = data->_workspace;
  assert( w->_stencil == NULL );
  assert( stencil != NULL );
  CUDA_SAFE_CALL( cudaMalloc( (void**)&(w->_stencil), data->_nfbytes ));
  cuda_memcpy( w->_stencil, stencil );
  return true;
}


// Init local regularizer weights: vector version
bool coco::coco_vtv_set_regularizer_weight( coco_vtv_data *data, vector<gsl_matrix*> g )
{
  assert( g.size() != data->_nchannels );
  coco_vtv_workspace *w = data->_workspace;
  for (size_t i=0; i< g.size(); ++i) {
    if ( w->_g[i] == NULL ) {
      CUDA_SAFE_CALL( cudaMalloc( (void**)&(w->_g[i]), data->_nfbytes ));
    }
    cuda_memcpy( w->_g[i], g[i] );
  }
  return true;
}

// Init local regularizer weights: single channel version
bool coco::coco_vtv_set_regularizer_weight( coco_vtv_data *data, gsl_matrix *g )
{
  assert( g );
  coco_vtv_workspace *w = data->_workspace;

  if ( w->_g[0] == NULL ) {
    CUDA_SAFE_CALL( cudaMalloc( (void**)&(w->_g[0]), data->_nfbytes ));
  }
  cuda_memcpy( w->_g[0], g );

  return true;
}

