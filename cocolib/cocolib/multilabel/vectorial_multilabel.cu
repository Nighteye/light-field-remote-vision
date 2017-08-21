/* -*-c++-*- */
/** \file vectorial_multilabel.cu

   Vectorial multilabel solvers
   Experimental code for kD label space

   Copyright (C) 2012 Bastian Goldluecke,
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
#include <float.h>

#include "vectorial_multilabel.h"
#include "vectorial_multilabel.cuh"

#include "../cuda/cuda_helper.h"
#include "../cuda/cuda_reduce.h"
#include "../cuda/simplex_reprojection.h"

#include "../common/gsl_matrix_helper.h"
#include "../common/profiler.h"

#include "vectorial_multilabel_primal_prox_kernels.cu"
#include "vectorial_multilabel_dual_prox_kernels.cu"
#include "../cuda/cuda_kernels.cuh"

using namespace std;

/*
enum vml_regularizer {
  VML_POTTS,
  VML_LINEAR,
  VML_CYCLIC,
  VML_TRUNCATED_LINEAR,
  VML_COST_MATRIX,
  VML_MUMFORD_SHAH,
  VML_HUBER,
  VML_REG_N,
};
*/
// Regularizer names
const char* coco::vml_regularizer_name[] = {
  "POTTS",
  "LINEAR",
  "CYCLIC",
  "TRUNCATED_LINEAR",
  "COST_MATRIX",
  "MUMFORD_SHAH",
  "HUBER"
};

// Get regularizer ID from name
coco::vml_regularizer coco::vml_regularizer_id( const char *reg_name )
{
  for ( size_t i=0; i<VML_REG_N; i++ ) {
    if ( strcmp( vml_regularizer_name[i], reg_name ) == 0 ) {
      return (vml_regularizer)i;
    }
  }
  return VML_REG_N;
}

/*****************************************************************************
       Dimension data creation / access
*****************************************************************************/

// Alloc data for different types of regularizers
coco::vml_dimension_data *coco::vml_dimension_data_alloc_potts( size_t G, double lambda )
{
  vml_dimension_data *ddata = new vml_dimension_data;
  memset( ddata, 0, sizeof( vml_dimension_data ));

  // number of labels
  assert( G>0 );
  ddata->_G = G;

  // regularizer type
  ddata->_type = VML_POTTS;
  // regularizer weight
  ddata->_lambda = lambda;

  // workspace structure initially empty
  vml_dimension_data_set_label_range( ddata, 0.0f, 1.0f );
  return ddata;
}


// Alloc data for different types of regularizers
coco::vml_dimension_data *coco::vml_dimension_data_alloc_linear( size_t G, double lambda,
								 double range_min, double range_max,
								 bool order_invert )
{
  vml_dimension_data *ddata = new vml_dimension_data;
  memset( ddata, 0, sizeof( vml_dimension_data ));

  // number of labels
  assert( G>0 );
  ddata->_G = G;

  // regularizer type
  ddata->_type = VML_LINEAR;
  // regularizer weight
  ddata->_lambda = lambda;

  // normalize to range
  vml_dimension_data_set_label_range( ddata, range_min, range_max );
  ddata->_order_invert = order_invert;

  // cost matrix
  ddata->_cost = gsl_matrix_alloc( G,G );
  for ( size_t g1=0; g1<G; g1++ ) {
    float v1 = ddata->_values[g1];
    for ( size_t g2=0; g2<G; g2++ ) {
      float v2 = ddata->_values[g2];
      float cost = lambda * fabs( v1-v2 );
      gsl_matrix_set( ddata->_cost, g1,g2, cost );
    }
  }

  // workspace structure initially empty
  return ddata;
}



// Alloc data for different types of regularizers
coco::vml_dimension_data *coco::vml_dimension_data_alloc_cyclic( size_t G, double lambda,
								 double range_min, double range_max )
{
  vml_dimension_data *ddata = new vml_dimension_data;
  memset( ddata, 0, sizeof( vml_dimension_data ));

  // number of labels
  assert( G>0 );
  ddata->_G = G;

  // regularizer type
  ddata->_type = VML_CYCLIC;
  // regularizer weight
  ddata->_lambda = lambda;

  // normalize to range
  vml_dimension_data_set_label_range( ddata, range_min, range_max );

  // cost matrix
  ddata->_cost = gsl_matrix_alloc( G,G );
  for ( size_t g1=0; g1<G; g1++ ) {
    float v1 = ddata->_values[g1];
    for ( size_t g2=0; g2<G; g2++ ) {
      float v2 = ddata->_values[g2];
      if ( (g1==0 && g2+1==G) ) {
	// wrap around at g2
	v2 = ddata->_values[1];
      }
      else if ( (g2==0 && g1+1==G) ) {
	// wrap around at g1
	v1 = ddata->_values[1];
      }
      float cost = lambda * fabs( v1-v2 );
      gsl_matrix_set( ddata->_cost, g1,g2, cost );
    }
  }

  // workspace structure initially empty
  return ddata;
}



// Alloc data for different types of regularizers
coco::vml_dimension_data *coco::vml_dimension_data_alloc_truncated_linear( size_t G, double lambda, double t,
									   double range_min, double range_max )
{
  vml_dimension_data *ddata = new vml_dimension_data;
  memset( ddata, 0, sizeof( vml_dimension_data ));

  // number of labels
  assert( G>0 );
  ddata->_G = G;

  // regularizer type
  ddata->_type = VML_TRUNCATED_LINEAR;
  // regularizer weight
  ddata->_lambda = lambda;

  // normalize to range
  vml_dimension_data_set_label_range( ddata, range_min, range_max );

  // cost matrix
  ddata->_cost = gsl_matrix_alloc( G,G );
  float cutoff = t * (range_max - range_min);
  for ( size_t g1=0; g1<G; g1++ ) {
    float v1 = ddata->_values[g1];
    for ( size_t g2=0; g2<G; g2++ ) {
      float v2 = ddata->_values[g2];
      float cost = lambda * min( cutoff, fabs( v1-v2 ));
      gsl_matrix_set( ddata->_cost, g1,g2, cost );
    }
  }

  // workspace structure initially empty
  return ddata;
}


// Alloc data for different types of regularizers
bool coco::vml_dimension_data_alloc_workspace( vml_data *data, vml_dimension_data *ddata )
{
  vml_dim_workspace *w = new vml_dim_workspace;
  memset( w, 0, sizeof( vml_dim_workspace ));
  ddata->_w = w;
  ddata->_W = data->_W;
  ddata->_H = data->_H;
  ddata->_N = data->_W * data->_H;
  size_t N = ddata->_N;
  size_t G = ddata->_G;

  // Current integer solution
  w->_urbytes = N * sizeof( int );
  CUDA_SAFE_CALL( cudaMalloc( &w->_ur, w->_urbytes ));

  // Size of 3D fields over Image x Label space
  w->_nfbytes = N * G * sizeof(float);

  // Primal variable
  CUDA_SAFE_CALL( cudaMalloc( &w->_u, w->_nfbytes ));
  CUDA_SAFE_CALL( cudaMalloc( &w->_uq, w->_nfbytes ));
  // Dual variables
  CUDA_SAFE_CALL( cudaMalloc( &w->_px, w->_nfbytes ));
  CUDA_SAFE_CALL( cudaMalloc( &w->_py, w->_nfbytes ));
  CUDA_SAFE_CALL( cudaMalloc( &w->_q, w->_nfbytes ));

  w->_nfbytes_sigma = N * sizeof( float );
  CUDA_SAFE_CALL( cudaMalloc( &w->_sigma, w->_nfbytes_sigma ));

  // Count
  w->_total_mem = w->_nfbytes * 5 + w->_urbytes * 1 + w->_nfbytes_sigma;

  // Extra regularizer variables
  w->_nfbytes_eta = 0;
  w->_eta_x = NULL;
  w->_eta_y = NULL;
  // depends on regularizer
  w->_sigma_p = 1.0f / 4.0f;
  w->_tau_eta = 0.5f;

  if ( ddata->_type == VML_LINEAR ) {
    w->_nfbytes_eta = N * (G-1) * sizeof(float);
    CUDA_SAFE_CALL( cudaMalloc( &w->_eta_x, w->_nfbytes_eta ));    
    CUDA_SAFE_CALL( cudaMalloc( &w->_eta_y, w->_nfbytes_eta ));    
    w->_total_mem += w->_nfbytes_eta * 2;
    // 2 p's per eta, 2 eta per p in addition to normal variables
    w->_sigma_p = 1.0f / 6.0f;
  }
  else if ( ddata->_type == VML_CYCLIC ) {
    w->_nfbytes_eta = N * G * sizeof(float);
    CUDA_SAFE_CALL( cudaMalloc( &w->_eta_x, w->_nfbytes_eta ));    
    CUDA_SAFE_CALL( cudaMalloc( &w->_eta_y, w->_nfbytes_eta ));    
    w->_total_mem += w->_nfbytes_eta * 2;
    // 2 p's per eta, 2 eta per p in addition to normal variables
    w->_sigma_p = 1.0f / 6.0f;
  }
  else if ( ddata->_type == VML_TRUNCATED_LINEAR ) {
    w->_nfbytes_eta = N * G * (G-1) * sizeof(float) / 2;
    CUDA_SAFE_CALL( cudaMalloc( &w->_eta_x, w->_nfbytes_eta ));    
    CUDA_SAFE_CALL( cudaMalloc( &w->_eta_y, w->_nfbytes_eta ));    
    w->_total_mem += w->_nfbytes_eta * 2;
    // 2 p's per eta, G eta per p in addition to normal variables
    w->_sigma_p = 1.0f / (4.0f + ddata->_G);
  }

  // Each q is responsible in every mu it appears,
  // amounts to the product of all other dimensions
  size_t sum_K_q = data->_G / G;
  assert( sum_K_q * G == data->_G );
  // q also appears once in the scalar product with the corresponding u
  w->_sigma_q = 1.0f / (1.0f + sum_K_q);
  // Each sigma is influenced by local dimension us
  w->_sigma_s = 1.0f / float(ddata->_G);
  // u gets 4 from gradient, 1 from sigma, 1 from data term
  w->_tau_u = 1.0f / 6.0f;

  return true;
}


// Get total mem usage
size_t coco::vml_total_mem( vml_data *data )
{
  assert( data->_w != NULL );
  return data->_w->_total_mem;
}


// Free data
bool coco::vml_dimension_data_free( vml_dimension_data *ddata )
{
  if ( ddata->_cost != NULL ) {
    gsl_matrix_free( ddata->_cost );
  }

  // Clean up dimension workspace
  vml_dim_workspace *w = ddata->_w;
  assert( w != NULL );

  CUDA_SAFE_CALL( cudaFree( w->_ur ));
  CUDA_SAFE_CALL( cudaFree( w->_u ));
  CUDA_SAFE_CALL( cudaFree( w->_uq ));
  CUDA_SAFE_CALL( cudaFree( w->_px ));
  CUDA_SAFE_CALL( cudaFree( w->_py ));
  CUDA_SAFE_CALL( cudaFree( w->_sigma ));
  CUDA_SAFE_CALL( cudaFree( w->_q ));
  if ( w->_g != NULL ) {
    CUDA_SAFE_CALL( cudaFree( w->_g ));
  }
  if ( w->_eta_x != NULL ) {
    CUDA_SAFE_CALL( cudaFree( w->_eta_x ));
  }
  if ( w->_eta_y != NULL ) {
    CUDA_SAFE_CALL( cudaFree( w->_eta_y ));
  }

  delete ddata;
  return true;
}

// Set label range
bool coco::vml_dimension_data_set_label_range( vml_dimension_data *ddata, double gmin, double gmax )
{
  assert( ddata != NULL );
  ddata->_values.clear();

  float gwidth = (gmax - gmin) / (ddata->_G-1);
  float s = gmin;
  TRACE6( "Initializing label range from " << gmin << " to " << gmax << " steps " << ddata->_G-1 << endl );
  for ( size_t g=1; g<=ddata->_G; g++ ) {
    ddata->_values.push_back( s );
    s += gwidth;
  }

  return true;
}





/*****************************************************************************
       VML data creation / access
*****************************************************************************/

// Alloc multilabel problem structure for vectorial multilabel model
coco::vml_data* coco::vml_data_alloc( size_t W, size_t H, size_t K,
				      bool cpu_data_term,
				      size_t chunk_width_warps )
{
  assert( K >= 1 );
  vml_data *data = new vml_data;
  
  // Image dimension
  data->_W = W;
  data->_H = H;
  // Label space dimension
  data->_K = K;
  
  // Data for each dimension
  data->_dim = new vml_dimension_data* [ K ];
  memset( data->_dim, 0, K * sizeof( vml_dimension_data* ));

  // CPU data term?
  data->_cpu_data_term = cpu_data_term;
  // Multiple chunks?
  data->_chunk_width_warps = chunk_width_warps;

  // Workspace
  data->_w = NULL;
  TRACE( "Allocating vectorial multilabel structure, " << W << " x " << H << endl );
  TRACE( "Label space dimensions: " << K << endl );
  return data;
}


// Set data for a single label space dimension (can be done only once)
bool coco::vml_init_dimension_data( vml_data *data,
				    size_t k,
				    vml_dimension_data *ddata )
{
  // Can only be assigned once
  assert( k<data->_K );
  assert( data->_dim[k] == NULL );
  data->_dim[k] = ddata;

  // Alloc workspace structure
  //vml_dimension_data_alloc_workspace( data, ddata );
  //TRACE( "  dimension " << k << " : " << ddata->_w->_total_mem / 1048576 << " MB." << endl );
  return true;
}


bool coco::vml_index_to_label( vml_data *data, size_t idx, int* L )
{
  assert( L != NULL );

  int tmp = idx;
  for ( size_t k=0; k<data->_K; k++ ) {
    assert( data->_dim[k] != NULL );
    int G = data->_dim[k]->_G;
    L[k] = tmp % G;
    tmp = tmp / G;
  }
  assert( tmp == 0 );
  return true;
}

size_t coco::vml_label_index( vml_data *data, int *L )
{
  int mul = 1;
  int idx = 0;
  for ( size_t k=0; k<data->_K; k++ ) {
    assert( data->_dim[k] != NULL );
    int G = data->_dim[k]->_G;
    idx += L[k] * mul;
    mul *= G;
  }
  return idx;
}



// Finalize initialization after all dimensions have been defined
bool coco::vml_alloc_finalize( vml_data *data )
{
  // Workspace
  data->_w = new vml_workspace;
  vml_workspace *w = data->_w;

  // List of all labels (as index array)
  size_t G = vml_total_label_count( data );
  for ( size_t i=0; i<G; i++ ) {
    int *L = new int[ data->_K ];
    vml_index_to_label( data, i, L );
    data->_labels.push_back( L );
    // sanity check of numbering scheme
    size_t idx = vml_label_index( data, L );
    assert( idx == i );
  }
  // sanity check
  data->_G = G;
  assert( G == data->_labels.size() );

  // Size of 3D fields over Image x Label space
  size_t N = data->_W * data->_H;
  w->_nfbytes = N * G * sizeof(float);
  w->_rho = NULL;

  if ( data->_chunk_width_warps == 0 ) {
    CUDA_SAFE_CALL( cudaMalloc( &w->_mu, w->_nfbytes ));
    CUDA_SAFE_CALL( cudaMemset( w->_mu, 0, w->_nfbytes ));
  }
  else {
    TRACE( "RE-PROJECTION FOR Q PERFORMED IN EACH ITERATION." << endl );
    TRACE( "  quite slow, but saves " << w->_nfbytes / 1048576 << " MiB." << endl );
    w->_mu = NULL;
  }

  // On-the-fly computation for dataterm initially off
  w->_dataterm_on_the_fly = false;
  w->_dataterm_on_the_fly_segmentation = false;
  w->_dataterm_segmentation_r = NULL;
  w->_dataterm_segmentation_g = NULL;
  w->_dataterm_segmentation_b = NULL;
    
  // Overrelaxation factor
  w->_theta = 1.0f;

  // Step sizes
  w->_tau_mu = 1.0f / float( data->_K );

  // Block sizes
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

  // Chunk layout
  w->_nchunks = 0;
  w->_nfbytes_chunk = 0;
  w->_chunk_width = 0;
  if ( data->_chunk_width_warps != 0 ) {
    w->_chunk_width = data->_chunk_width_warps * 32;
    w->_nchunks = W / w->_chunk_width;
    if ( w->_nchunks * w->_chunk_width < W ) {
      w->_nchunks += 1;
    }
    w->_nfbytes_chunk = w->_chunk_width * H * data->_G * sizeof(float);
    w->_dimGridChunk = dim3( w->_chunk_width / w->_dimBlock.x, w->_dimGrid.y );
    assert( w->_dimGridChunk.x * w->_dimBlock.x == w->_chunk_width );
    TRACE( "  using " << w->_nchunks << " chunks, width " << w->_chunk_width << endl );
  }

  // Alloc workspace structures for dimensions
  for ( size_t k=0; k<data->_K; k++ ) {
    vml_dimension_data_alloc_workspace( data, data->_dim[k] );
    TRACE( "  dimension " << k << " : " << data->_dim[k]->_w->_total_mem / 1048576 << " MB." << endl );
  }

  w->_total_mem = 0;
  if ( data->_chunk_width_warps == 0 ) {
    w->_total_mem += w->_nfbytes;
  }
  if ( !data->_cpu_data_term ) {
    w->_total_mem += w->_nfbytes;
  }

  TRACE( "  global mem  : " << w->_total_mem / 1048576 << " MB." << endl );
  for ( size_t k=0; k<data->_K; k++ ) {
    w->_total_mem += data->_dim[k]->_w->_total_mem;
  }
  TRACE( "  total mem   : " << w->_total_mem / 1048576 << " MB." << endl );

  return true;
}


// Free multilabel problem structure
bool coco::vml_data_free( vml_data* data )
{
  for ( size_t k=0; k<data->_K; k++ ) {
    vml_dimension_data_free( data->_dim[k] );
  }
  delete[] data->_dim;

  // clear labels
  for ( size_t g=0; g<data->_G; g++ ) {
    delete data->_labels[g];
  }
  data->_labels.clear();

  // clear workspace
  // Clean up dimension workspace
  vml_workspace *w = data->_w;
  assert( w != NULL );

  for ( size_t j=0; j<w->_chunk_rho.size(); j++ ) {
    delete[] w->_chunk_rho[j];
  }
  w->_chunk_rho.clear();

  if ( data->_cpu_data_term ) {
    w->_rho = NULL;
  }
  else if ( w->_rho != NULL ) {
    CUDA_SAFE_CALL( cudaFree( w->_rho ));
  }

  if ( w->_dataterm_on_the_fly_segmentation ) {
    CUDA_SAFE_CALL( cudaFree( w->_dataterm_segmentation_r ));
    CUDA_SAFE_CALL( cudaFree( w->_dataterm_segmentation_g ));
    CUDA_SAFE_CALL( cudaFree( w->_dataterm_segmentation_b ));
  }

  if ( w->_mu != NULL ) {
    CUDA_SAFE_CALL( cudaFree( w->_mu ));
  }

  delete data->_w;
  delete data;
  return true;
}


// Total label count
size_t coco::vml_total_label_count( vml_data *data )
{
  size_t N = 1;
  for ( size_t k=0; k<data->_K; k++ ) {
    assert( data->_dim[k] != NULL );
    N *= data->_dim[k]->_G;
  }
  return N;
}



// Offset into label array
size_t coco::vml_dim_label_offset( vml_dimension_data *ddata, size_t g, size_t x, size_t y )
{
  return ddata->_N * g + y * ddata->_W + x;
}


// Offset into eta array
size_t coco::vml_dim_regularizer_offset( vml_dimension_data *ddata, size_t g1, size_t g2, size_t x, size_t y )
{
  switch ( ddata->_type ) {
  case VML_LINEAR:
    {
      assert( g1+1 == g2 );
      assert( g2 < ddata->_G );
      return ddata->_N * g1 + y * ddata->_W + x;
    }
  case VML_CYCLIC:
    {
      assert( g1 < ddata->_G );
      if ( g1 == ddata->_G-1 ) {
	assert( g2 == 0 );
      }
      else {
	assert( g2 == g1+1 );
      }
      return ddata->_N * g1 + y * ddata->_W + x;
    }
  case VML_TRUNCATED_LINEAR:
    {
      assert( g1<g2 );
      assert( g2<ddata->_G );
      int offset = 0;
      for ( size_t gg1=0; gg1+1<ddata->_G; gg1++ ) {
	for ( size_t gg2=gg1+1; gg2<ddata->_G; gg2++ ) {
	  if ( gg1 == g1 && gg2 == g2 ) {
	    return offset * ddata->_N;
	  }
	  offset++;
	}
      }
      // should have returned.
      assert( false );
      return 0;
    }
  default:
    // the other regularizers are either not implemented or do not require the eta variables
    assert( false );
    return 0;
  }
}



// Set current solution data
bool coco::vml_set_solution( vml_data* D,
			     size_t k, int *ur )
{
  assert( k<D->_K );
  vml_dimension_data *ddata = D->_dim[k];
  assert( ddata != NULL );
  vml_dim_workspace *w = ddata->_w;
  assert( w != NULL );

  // Set up 4D structure for data term
  size_t W = ddata->_W;
  size_t H = ddata->_H;
  size_t G = ddata->_G;
  float *u = new float[W*H*G];
  memset( u, 0, w->_nfbytes );

  size_t index = 0;
  for ( size_t y=0; y<ddata->_H; y++ ) {
    for ( size_t x=0; x<ddata->_W; x++ ) {
      size_t v = ur[index];
      for ( size_t g=0; g<ddata->_G; g++ ) {
	if ( v == g ) {
	  size_t offset = vml_dim_label_offset( ddata, g, x,y );
	  u[ offset ] = 1.0f;
	  break;
	}
      }
      index++;
    }
  }

  // Copy to GPU
  CUDA_SAFE_CALL( cudaMemcpy( w->_u, u, w->_nfbytes,
			      cudaMemcpyHostToDevice ));
  CUDA_SAFE_CALL( cudaMemcpy( w->_ur, ur, w->_urbytes,
			      cudaMemcpyHostToDevice ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // Cleanup
  delete[] u;
  return true;
}


// Reinit algorithm with current solution
bool coco::vml_init( vml_data* D )
{
  // init main workspace
  vml_workspace *Dw = D->_w;
  assert( Dw != NULL );
  Dw->_iteration = 0;
  if ( Dw->_mu != NULL ) {
    CUDA_SAFE_CALL( cudaMemset( Dw->_mu, 0, Dw->_nfbytes ));
  }

  // init dimensions
  for ( size_t k=0; k<D->_K; k++ ) {
    assert( k<D->_K );
    vml_dimension_data *ddata = D->_dim[k];
    assert( ddata != NULL );
    vml_dim_workspace *w = ddata->_w;
    assert( w != NULL );

    // Init primal variables
    CUDA_SAFE_CALL( cudaMemcpy( w->_uq, w->_u, w->_nfbytes, cudaMemcpyDeviceToDevice ));
    // Init dual variables
    CUDA_SAFE_CALL( cudaMemset( w->_px, 0, w->_nfbytes ));
    CUDA_SAFE_CALL( cudaMemset( w->_py, 0, w->_nfbytes ));
    // Init Lagrange multipliers
    CUDA_SAFE_CALL( cudaMemset( w->_sigma, 0, w->_nfbytes_sigma ));
    // Init data term dual variable
    CUDA_SAFE_CALL( cudaMemset( w->_q, 0, w->_nfbytes ));
    // Regularizer variables
    if ( w->_eta_x != NULL ) {
      CUDA_SAFE_CALL( cudaMemset( w->_eta_x, 0, w->_nfbytes_eta ));
      CUDA_SAFE_CALL( cudaMemset( w->_eta_y, 0, w->_nfbytes_eta ));
    }
  }

  // Sync
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}


static __global__ void vml_project_solution_device( int W, int H, int N, int G,
						    float *u, int *ur, float *sum )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = ox + oy*W;
  float umax = u[o];
  int uopt = 0;
  u[o] = 0.0f;
  float s = 0.0f;
  for ( int g=1; g<G; g++ ) {
    float uv = u[o+g*N];
    s += uv;
    u[o+g*N] = 0.0f;
    if ( uv > umax ) {
      umax = uv;
      uopt = g;
    }
  }

  sum[o] = s;
  ur[o] = uopt;
  u[o + uopt*N] = 1.0f;
}


static __global__ void vml_compute_best_label_device( int W, int H, int N, int G,
						      float *u, int *ur )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = ox + oy*W;
  float umax = u[o];
  int uopt = 0;
  u[o] = 0.0f;
  for ( int g=1; g<G; g++ ) {
    float uv = u[o+g*N];
    if ( uv > umax ) {
      umax = uv;
      uopt = g;
    }
  }

  ur[o] = uopt;
}



static __global__ void vml_label_cost_device( int W, int H, int N,
					      float *rho, int *labels, float *cost )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = ox + oy*W;
  cost[o] = rho[ o + labels[o] * N ];
}



static __global__ void vml_project_q_device( int W, int H,
					     float K,
					     float *cost, float *q )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = ox + oy*W;
  q[o] = cost[o] / K;
}


static __global__ void vml_project_p_potts_device( int W, int H,
						   float lambda,
						   float *u,
						   float *px, float *py )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = ox + oy*W;

  // Step for each p equals gradient component of phi
  // Forward differences, Neumann
  // X
  float uv = u[o];
  float new_px = 0.0f;
  if ( ox < W-1 ) {
    new_px = uv - u[o+1];
  }
  // Y
  float new_py = 0.0f;
  if ( oy < H-1 ) {
    new_py = uv - u[o+W];
  }

  // Reprojection is combined for all channels
  float L = hypotf( new_px, new_py );
  if ( L>lambda ) {
    new_px = lambda * new_px / L;
    new_py = lambda * new_py / L;
  }
  px[o] = new_px;
  py[o] = new_py;
}



static __global__ void vml_project_p_linear_device( int W, int H,
						    float cost,
						    float *u,
						    float *eta_x1, float *eta_y1,
						    float *eta_x2, float *eta_y2,
						    float *px, float *py )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = ox + oy*W;

  // Step for each p equals gradient component of phi
  // Forward differences, Neumann
  // X
  float uv = u[o];
  float new_px = 0.0f;
  if ( ox < W-1 ) {
    new_px = uv - u[o+1];
  }
  // Y
  float new_py = 0.0f;
  if ( oy < H-1 ) {
    new_py = uv - u[o+W];
  }

  if ( eta_x1 != NULL ) {
    new_px += eta_x1[o];
    new_py += eta_y1[o];
  }
  /*
  if ( eta_x2 != NULL ) {
    new_px -= eta_x2[o];
    new_py -= eta_y2[o];
  }
  */

  // Reprojection is combined for all channels
  float L = hypotf( new_px, new_py );
  if ( L > cost ) {
    new_px = cost * new_px / L;
    new_py = cost * new_py / L;
  }

  px[o] = new_px;
  py[o] = new_py;
}





// Project current relaxed solution onto integer values
// This version projects both q as well as u, and seems to be the correct one
bool coco::vml_project_solution( vml_data *D )
{
  size_t W = D->_W;
  size_t H = D->_H;
  size_t N = W*H;
  vml_workspace *Dw = D->_w;

  // Project primal solution (u)
  float *sum = NULL;
  CUDA_SAFE_CALL( cudaMalloc( &sum, N*sizeof(float)) );
  int *temp = NULL;
  CUDA_SAFE_CALL( cudaMalloc( &temp, N*sizeof(int)) );
  for ( size_t k=0; k<D->_K; k++ ) {
    vml_dimension_data *ddata = D->_dim[k];
    assert( ddata != NULL );
    vml_dim_workspace *w = ddata->_w;
    assert( w != NULL );
    
    vml_project_solution_device<<< Dw->_dimGrid, Dw->_dimBlock >>>
      ( W,H, W*H, ddata->_G, w->_u, w->_ur, sum );

    // constraint check
    float r = 0.0f;
    cuda_sum_reduce( W,H, sum, (float*)temp, &r );
    TRACE( "average variable sum dimension " << k << " is " << r / float(N) << endl );
  }
  CUDA_SAFE_CALL( cudaFree( sum ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // Write cost of winning label divided by number of dimensions
  // into each variable q to make energy cost computation at the optimum
  // correct.
  int *labels[VML_MAX_DIM];
  for ( size_t k=0; k<D->_K; k++ ) {
    labels[k] = new int[W*H];
    vml_get_solution( D, k, labels[k] );
  }
  int *winning_label = new int[W*H];
  for ( size_t n=0; n<N; n++ ) {
    int label[VML_MAX_DIM];
    for ( size_t k=0; k<D->_K; k++ ) {
      label[k] = labels[k][n];
    }
    winning_label[n] = vml_label_index( D, label );
  }

  // Cost of winning label
  float *atemp = NULL;
  CUDA_SAFE_CALL( cudaMalloc( &atemp, N*sizeof(float)) );

  if ( D->_cpu_data_term ) {
    // Compute on CPU
    float *cost = new float[ N ];
    for ( size_t i=0; i<N; i++ ) {
      cost[ i ] = Dw->_rho[ winning_label[i]*N + i ];
    }
    CUDA_SAFE_CALL( cudaMemcpy( atemp, cost, N*sizeof(float), cudaMemcpyHostToDevice ));
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    delete[] cost;
  }
  else {
    CUDA_SAFE_CALL( cudaMemcpy( temp, winning_label, N*sizeof(int), cudaMemcpyHostToDevice ));
    // Compute cost of winning label array
    vml_label_cost_device<<< Dw->_dimGrid, Dw->_dimBlock >>>
      ( W,H, N, Dw->_rho, temp, atemp );
  }

  // Update q arrays
  for ( size_t k=0; k<D->_K; k++ ) {
    vml_dimension_data *ddata = D->_dim[k];
    assert( ddata != NULL );
    vml_dim_workspace *w = ddata->_w;
    assert( w != NULL );

    for ( size_t g=0; g<ddata->_G; g++ ) {
      size_t offset = vml_dim_label_offset( ddata, g );
      vml_project_q_device<<< Dw->_dimGrid, Dw->_dimBlock >>>
	( W,H, D->_K, atemp, w->_q + offset );
    }

    delete[] labels[k];
  }

  delete[] winning_label;

  // sanity check
  float r = 0.0f;
  cuda_sum_reduce( W,H, atemp, (float*)temp, &r );
  TRACE( "Projection total data cost: " << r << endl );

  CUDA_SAFE_CALL( cudaFree( temp ));
  CUDA_SAFE_CALL( cudaFree( atemp ));
  return true;
}


/* Separate solution and q projection.
   Required for the computation of accurate bounds in some cases.
 */

// Project current relaxed solution onto integer values
bool coco::vml_project_q( vml_data *D )
{
  size_t W = D->_W;
  size_t H = D->_H;
  size_t N = W*H;
  vml_workspace *Dw = D->_w;

  // Write cost of winning label divided by number of dimensions
  // into each variable q to make energy cost computation at the optimum
  // correct.
  int *labels[VML_MAX_DIM];
  for ( size_t k=0; k<D->_K; k++ ) {
    labels[k] = new int[W*H];
    vml_get_solution_relaxation( D, k, labels[k] );
  }
  int *winning_label = new int[W*H];
  for ( size_t n=0; n<N; n++ ) {
    int label[VML_MAX_DIM];
    for ( size_t k=0; k<D->_K; k++ ) {
      label[k] = labels[k][n];
    }
    winning_label[n] = vml_label_index( D, label );
  }

  // Cost of winning label
  float *atemp = NULL;
  CUDA_SAFE_CALL( cudaMalloc( &atemp, N*sizeof(float)) );
  int *temp = NULL;
  CUDA_SAFE_CALL( cudaMalloc( &temp, N*sizeof(int)) );

  if ( D->_cpu_data_term ) {
    // Compute on CPU
    float *cost = new float[ N ];
    for ( size_t i=0; i<N; i++ ) {
      cost[ i ] = Dw->_rho[ winning_label[i]*N + i ];
    }
    CUDA_SAFE_CALL( cudaMemcpy( atemp, cost, N*sizeof(float), cudaMemcpyHostToDevice ));
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    delete[] cost;
  }
  else {
    // Copy to CUDA
    CUDA_SAFE_CALL( cudaMemcpy( temp, winning_label, N*sizeof(int), cudaMemcpyHostToDevice ));
    // Compute cost of winning label array
    vml_label_cost_device<<< Dw->_dimGrid, Dw->_dimBlock >>>
      ( W,H, N, Dw->_rho, temp, atemp );
  }

  // Update q arrays
  for ( size_t k=0; k<D->_K; k++ ) {
    vml_dimension_data *ddata = D->_dim[k];
    assert( ddata != NULL );
    vml_dim_workspace *w = ddata->_w;
    assert( w != NULL );

    for ( size_t g=0; g<ddata->_G; g++ ) {
      size_t offset = vml_dim_label_offset( ddata, g );
      vml_project_q_device<<< Dw->_dimGrid, Dw->_dimBlock >>>
	( W,H, D->_K, atemp, w->_q + offset );
    }

    delete[] labels[k];
  }

  delete[] winning_label;

  // sanity check
  float r = 0.0f;
  cuda_sum_reduce( W,H, atemp, (float*)temp, &r );
  TRACE( "Projection total data cost: " << r << endl );

  CUDA_SAFE_CALL( cudaFree( temp ));
  CUDA_SAFE_CALL( cudaFree( atemp ));
  return true;
}


// Project current relaxed solution onto integer values
bool coco::vml_project_p( vml_data *D )
{
  size_t W = D->_W;
  size_t H = D->_H;
  vml_workspace *Dw = D->_w;

  // Update p arrays (using eta and u)
  for ( size_t k=0; k<D->_K; k++ ) {
    vml_dimension_data *ddata = D->_dim[k];
    assert( ddata != NULL );
    vml_dim_workspace *w = ddata->_w;
    assert( w != NULL );

    for ( size_t g=0; g<ddata->_G; g++ ) {
      size_t offset = vml_dim_label_offset( ddata, g );
      if ( ddata->_type == VML_POTTS ) {
	vml_project_p_potts_device<<< Dw->_dimGrid, Dw->_dimBlock >>>
	  ( W,H,
	    ddata->_lambda,
	    w->_u + offset,
	    w->_px + offset,
	    w->_py + offset );
      }
      else if ( ddata->_type == VML_LINEAR ||
		ddata->_type == VML_CYCLIC ) {

	float *eta_x1 = NULL;
	float *eta_y1 = NULL;
	float *eta_x2 = NULL;
	float *eta_y2 = NULL;
	float cost = 0.0f;
	if ( g>0 ) {
	  size_t offset2 = vml_dim_label_offset( ddata, g-1 );
	  eta_x2 = w->_eta_x + offset2;
	  eta_y2 = w->_eta_y + offset2;
	  cost = gsl_matrix_get( ddata->_cost, g, g-1 );
	}
	else if ( ddata->_type == VML_CYCLIC ) {
	  size_t offset2 = vml_dim_label_offset( ddata, ddata->_G-1 );
	  eta_x2 = w->_eta_x + offset2;
	  eta_y2 = w->_eta_y + offset2;
	  cost = gsl_matrix_get( ddata->_cost, 0, ddata->_G-1 );
	}

	if ( g+1<ddata->_G || ddata->_type == VML_CYCLIC ) {
	  eta_x1 = w->_eta_x + offset;
	  eta_y1 = w->_eta_y + offset;
	}

	// Inaccurate (too conservative, TODO: implement correct one)
	cost *= 6.0f / sqrtf( ddata->_G );
	vml_project_p_linear_device<<< Dw->_dimGrid, Dw->_dimBlock >>>
	  ( W,H, 
	    cost,
	    w->_u + offset,
	    eta_x1, eta_y1,
	    eta_x2, eta_y2,
	    w->_px + offset,
	    w->_py + offset );

	CUDA_SAFE_CALL( cudaThreadSynchronize() );
      }
      else if ( ddata->_type == VML_TRUNCATED_LINEAR ) {

	// Inaccurate (too conservative, TODO: implement correct one)
	vml_project_p_potts_device<<< Dw->_dimGrid, Dw->_dimBlock >>>
	  ( W,H,
	    ddata->_lambda / sqrtf(ddata->_G),
	    w->_u + offset,
	    w->_px + offset,
	    w->_py + offset );
	/*
	// Update for each pair of labels once, ignoring ordering
	for ( size_t g2=g+1; g2<ddata->_G; g2++ ) {
	  size_t offset2 = vml_dim_label_offset( ddata, g2 );
	  float *eta_x2 = w->_eta_x + offset2;
	  float *eta_y2 = w->_eta_y + offset2;
	  float cost = gsl_matrix_get( ddata->_cost, g,g2 );

	  // Seems to work, but only a heuristic to approximate the true projection
	  vml_project_p_linear_device<<< Dw->_dimGrid, Dw->_dimBlock >>>
	    ( W,H, 
	      cost,
	      w->_u + offset,
	      w->_eta_x + offset,
	      w->_eta_y + offset,
	      eta_x2, eta_y2,
	      w->_px + offset,
	      w->_py + offset );
	}
	*/
      }
      else {
	// currently unsupported regularizer
	assert( false );
      }
    }
  }

  return true;
}


// Project current relaxed solution onto integer values
bool coco::vml_project_solution_separate_q( vml_data *D )
{
  size_t W = D->_W;
  size_t H = D->_H;
  size_t N = W*H;
  vml_workspace *Dw = D->_w;

  // Project primal solution (u)
  float *sum = NULL;
  CUDA_SAFE_CALL( cudaMalloc( &sum, N*sizeof(float)) );
  float *temp = NULL;
  CUDA_SAFE_CALL( cudaMalloc( &temp, 1*sizeof(float)) );

  for ( size_t k=0; k<D->_K; k++ ) {
    vml_dimension_data *ddata = D->_dim[k];
    assert( ddata != NULL );
    vml_dim_workspace *w = ddata->_w;
    assert( w != NULL );
    
    vml_project_solution_device<<< Dw->_dimGrid, Dw->_dimBlock >>>
      ( W,H, W*H, ddata->_G, w->_u, w->_ur, sum );

    // Write test image for sum
    //write_test_image_unsigned( W,H, sum, "out/sum_%i.png", k, false );

    // constraint check
    float r = 0.0f;
    cuda_sum_reduce( W,H, sum, temp, &r );
    TRACE( "average variable sum dimension " << k << " is " << r / float(N) << endl );
  }
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  CUDA_SAFE_CALL( cudaFree( sum ));
  CUDA_SAFE_CALL( cudaFree( temp ));

  return true;
}


// Get current solution
bool coco::vml_get_solution( vml_data *D,
			     size_t k, int *ur )
{
  assert( k<D->_K );
  vml_dimension_data *ddata = D->_dim[k];
  assert( ddata != NULL );
  vml_dim_workspace *w = ddata->_w;
  assert( w != NULL );
  CUDA_SAFE_CALL( cudaMemcpy( ur, w->_ur, w->_urbytes, cudaMemcpyDeviceToHost ));
  return true;
}


// Get current solution of the relaxation
bool coco::vml_get_solution_relaxation( vml_data *D,
					size_t k, int *ur )
{
  size_t W = D->_W;
  size_t H = D->_H;
  vml_workspace *Dw = D->_w;

  assert( k<D->_K );
  vml_dimension_data *ddata = D->_dim[k];
  assert( ddata != NULL );
  vml_dim_workspace *w = ddata->_w;
  assert( w != NULL );

  vml_compute_best_label_device<<< Dw->_dimGrid, Dw->_dimBlock >>>
    ( W,H, W*H, ddata->_G, w->_u, w->_ur );
  CUDA_SAFE_CALL( cudaMemcpy( ur, w->_ur, w->_urbytes, cudaMemcpyDeviceToHost ));

  return true;
}




// Get current solution
bool coco::vml_set_data_term_on_the_fly_segmentation( vml_data *D, 
						      gsl_image *I )
{
  vml_workspace *w = D->_w;
  assert( w != NULL );
  size_t W = D->_W;
  size_t H = D->_H;
  w->_dataterm_on_the_fly = true;
  w->_dataterm_on_the_fly_segmentation = true;
  size_t nfbytes = W*H*sizeof(float);
  CUDA_SAFE_CALL( cudaMalloc( &w->_dataterm_segmentation_r, nfbytes ));
  CUDA_SAFE_CALL( cudaMalloc( &w->_dataterm_segmentation_g, nfbytes ));
  CUDA_SAFE_CALL( cudaMalloc( &w->_dataterm_segmentation_b, nfbytes ));

  // Copy image to GPU
  cuda_memcpy( w->_dataterm_segmentation_r, I->_r );
  cuda_memcpy( w->_dataterm_segmentation_g, I->_g );
  cuda_memcpy( w->_dataterm_segmentation_b, I->_b );
  return true;
}


// Get current solution
bool coco::vml_set_data_term( vml_data *D, float *rho )
{
  vml_workspace *w = D->_w;
  assert( w != NULL );
  size_t W = D->_W;
  size_t H = D->_H;
  size_t G = D->_G;

  if ( D->_cpu_data_term ) {
    TRACE( "USING CPU DATA TERM: Much slower, but saves " << w->_nfbytes / 1048576 << " MiB of memory." << endl )
    w->_rho = rho;

    if ( w->_nchunks > 0 && !w->_dataterm_on_the_fly) {

      for ( size_t j=0; j<w->_nchunks; j++ ) {
	float *chunk_rho = new float[ w->_chunk_width * H * G ];
	// Copy part of rho
	for ( size_t y=0; y<H; y++ ) {
	  for ( size_t x=0; x<w->_chunk_width; x++ ) {
	    for ( size_t g=0; g<G; g++ ) {
	      size_t x_rho = w->_chunk_width * j + x;
	      if ( x_rho < W ) {
		chunk_rho[ x + ( y + g*H ) * w->_chunk_width ] = rho[ x_rho + y * W + g * W * H ];
	      }
	      else {
		chunk_rho[ x + ( y + g*H ) * w->_chunk_width ] = 0.0f;
	      }
	    }
	  }
	}
	w->_chunk_rho.push_back( chunk_rho );
      }
    }
  }
  else {
    CUDA_SAFE_CALL( cudaMalloc( &w->_rho, w->_nfbytes ));
    CUDA_SAFE_CALL( cudaMemcpy( w->_rho, rho, w->_nfbytes, cudaMemcpyHostToDevice ));
  }
  return true;
}





/************************************************
    EXPERIMENTAL VECTORIAL MULTILABEL ALGORITHM
*************************************************/

// Update one layer of the fields for fgp relaxation
__global__ void update_overrelaxation_device( int W, int H,
					      float theta,
					      float *u_prox, float *u )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;
  float uq = u_prox[o];
  u[o] = uq + theta * ( uq - u[o] );
}

// Update overrelaxation
bool coco::vml_update_overrelaxation( vml_data *D )
{
  size_t W = D->_W;
  size_t H = D->_H;
  vml_workspace *Dw = D->_w;

  // Kernel call for each layer in each dimension
  // Update for primal solution (u)
  for ( size_t k=0; k<D->_K; k++ ) {
    vml_dimension_data *ddata = D->_dim[k];
    assert( ddata != NULL );
    vml_dim_workspace *w = ddata->_w;
    assert( w != NULL );
    
    for ( size_t g=0; g<ddata->_G; g++ ) {
      size_t offset = vml_dim_label_offset( ddata, g );
      update_overrelaxation_device<<< Dw->_dimGrid, Dw->_dimBlock >>>
	( W, H,
	  Dw->_theta,
	  w->_uq + offset,
	  w->_u + offset );
    }
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    // Switch fields
    swap( w->_u, w->_uq );
  }

  return true;
}



/******************************************
    MAIN MINIMIZATION LOOP
*******************************************/

// Perform one iteration
bool coco::vml_iteration( vml_data *data )
{
  // Compute dual prox operator
  profiler()->beginTask( "dual prox vml" );
  vml_dual_prox( data );
  profiler()->endTask( "dual prox vml" );

  // Compute primal prox operator
  profiler()->beginTask( "primal prox vml" );
  vml_primal_prox( data, false );
  profiler()->endTask( "primal prox vml" );

  // Update overrelaxation scheme
  vml_update_overrelaxation( data );

  // Finalize
  data->_w->_iteration ++;
  return true;
}


// Perform the final iteration
// (different prox operator, see header)
bool coco::vml_final_iteration( vml_data *data )
{
  // Compute dual prox operator
  profiler()->beginTask( "dual prox vml" );
  vml_dual_prox( data );
  profiler()->endTask( "dual prox vml" );

  // Compute primal prox operator
  profiler()->beginTask( "primal prox vml" );
  vml_primal_prox( data, true );
  profiler()->endTask( "primal prox vml" );

  // Update overrelaxation scheme
  vml_update_overrelaxation( data );

  // Finalize
  data->_w->_iteration ++;
  return true;
}


// Perform one primal step
bool coco::vml_primal_prox( vml_data *D, bool final_iteration )
{
  size_t W = D->_W;
  size_t H = D->_H;
  vml_workspace *Dw = D->_w;
  assert( Dw != NULL );

  // Kernel call for each layer in each dimension
  // Update for primal solution (u)
  for ( size_t k=0; k<D->_K; k++ ) {
    vml_dimension_data *ddata = D->_dim[k];
    assert( ddata != NULL );
    vml_dim_workspace *w = ddata->_w;
    assert( w != NULL );
    
    for ( size_t g=0; g<ddata->_G; g++ ) {
      size_t offset = vml_dim_label_offset( ddata, g );
      compute_primal_prox_vml_device<<< Dw->_dimGrid, Dw->_dimBlock >>>
	( W, H,
	  w->_tau_u,
	  w->_u + offset,
	  w->_q + offset,
	  w->_sigma,
	  w->_px + offset, w->_py + offset,
	  w->_uq + offset );
    }

    // Update regularizer relaxation
    vml_update_primal_regularizer( D, ddata, final_iteration );
  }

  // Data term relaxation
  if ( D->_chunk_width_warps == 0 ) {
    if ( Dw->_dataterm_on_the_fly_segmentation ) {
      vml_update_primal_relaxation_on_the_fly_segmentation( D, final_iteration );
    }
    else {
      vml_update_primal_relaxation( D, final_iteration );
    }
  }
  else {
    if ( Dw->_dataterm_on_the_fly_segmentation ) {
      //vml_primal_prox_projection( D, final_iteration );
      vml_primal_prox_projection_on_the_fly_segmentation( D, final_iteration );
    }
    else {
      vml_primal_prox_projection( D, final_iteration );
    }
  }

  // Kernel call for each layer in each dimension
  // Update for primal solution (u)
  for ( size_t k=0; k<D->_K; k++ ) {
    vml_dimension_data *ddata = D->_dim[k];
    assert( ddata != NULL );
    vml_dim_workspace *w = ddata->_w;
    assert( w != NULL );
    
    for ( size_t g=0; g<ddata->_G; g++ ) {
      size_t offset = vml_dim_label_offset( ddata, g );
      primal_prox_project_vml_device<<< Dw->_dimGrid, Dw->_dimBlock >>>
	( W, H,
	  w->_uq + offset );
    }
  }

  return true;
}



// Primal projection, global Lagrange multipliers
bool coco::vml_update_primal_relaxation( vml_data *D, bool final_iteration )
{
  size_t W = D->_W;
  size_t H = D->_H;
  vml_workspace *Dw = D->_w;
  assert( Dw != NULL );

  // Allocate temporary mem for q updates
  vector<float*> tmp;
  for ( size_t k=0; k<D->_K; k++ ) {
    vml_dimension_data *ddata = D->_dim[k];
    assert( ddata != NULL );
    vml_dim_workspace *w = ddata->_w;
    assert( w != NULL );
    float *tmpv = NULL;
    CUDA_SAFE_CALL( cudaMalloc( &tmpv, w->_nfbytes ));
    CUDA_SAFE_CALL( cudaMemcpy( tmpv, w->_q, w->_nfbytes, cudaMemcpyDeviceToDevice ));
    tmp.push_back( tmpv );
  }
  // Allocate temporary mem for data term
  float *rho = NULL;
  size_t nfbytes_dataterm_layer =  W*H*sizeof(float);
  if ( D->_cpu_data_term ) {
    CUDA_SAFE_CALL( cudaMalloc( &rho, nfbytes_dataterm_layer ));
  }

  // Update for primal Lagrange multiplier (mu) and dual qs
  for ( size_t g=0; g<D->_labels.size(); g++ ) {
    int *label = D->_labels[g];

    // STANDARD RELAXATION (ICCV 2011)
    // Collect qs (one per dimension)
    float *qs[ VML_MAX_DIM ];
    float *qs_new[ VML_MAX_DIM ];
    
    for ( size_t k=0; k<D->_K; k++ ) {
      vml_dimension_data *ddata = D->_dim[k];
      assert( ddata != NULL );
      vml_dim_workspace *w = ddata->_w;
      assert( w != NULL );
      
      size_t offset = vml_dim_label_offset( ddata, label[k] );
      qs[k] = w->_q + offset;
      qs_new[k] = tmp[k] + offset;
    }
 
    // Copy data term layer, if stored on CPU
    size_t offset = g * W * H;
    if ( D->_cpu_data_term ) {
      CUDA_SAFE_CALL( cudaMemcpy( rho, Dw->_rho + offset, nfbytes_dataterm_layer, cudaMemcpyHostToDevice ));
      CUDA_SAFE_CALL( cudaThreadSynchronize() );
    }
    else {
      rho = Dw->_rho + offset;
    }
   
    // Store qs on GPU?
    // For now, temp solution (hard code dim 2 and 3)
    if ( D->_K == 2 ) {
      update_mu_q_dim2_vml_device<<< Dw->_dimGrid, Dw->_dimBlock >>>
	( W, H,
	  Dw->_tau_mu, 
	  D->_dim[0]->_w->_sigma_q,
	  D->_dim[1]->_w->_sigma_q,
	  Dw->_theta,
	  qs[0], qs[1],
	  rho,
	  Dw->_mu + offset,
	  qs_new[0], qs_new[1] );
    }
    else if ( D->_K == 3 ) {
      update_mu_q_dim3_vml_device<<< Dw->_dimGrid, Dw->_dimBlock >>>
	( W, H,
	  Dw->_tau_mu,
	  D->_dim[0]->_w->_sigma_q,
	  D->_dim[1]->_w->_sigma_q,
	  D->_dim[2]->_w->_sigma_q,
	  Dw->_theta,
	  qs[0], qs[1], qs[2],
	  rho,
	  Dw->_mu + offset,
	  qs_new[0], qs_new[1], qs_new[2] );
    }
    else {
      ERROR( "Currently VML is only implemented for label space dimension 2 and 3." << endl );
      assert( false );
    }
  }

  // Store q updates
  for ( size_t k=0; k<D->_K; k++ ) {
    vml_dimension_data *ddata = D->_dim[k];
    assert( ddata != NULL );
    vml_dim_workspace *w = ddata->_w;
    assert( w != NULL );
    float *tmpv = tmp[k];
    if ( final_iteration ) {
      CUDA_SAFE_CALL( cudaFree( tmpv ));
    }
    else {
      CUDA_SAFE_CALL( cudaFree( w->_q ));
      w->_q = tmpv;
    }
  }
  tmp.clear();
  if ( D->_cpu_data_term ) {
    CUDA_SAFE_CALL( cudaFree( rho ));
  }
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}



// Primal projection, global Lagrange multipliers
bool coco::vml_update_primal_relaxation_on_the_fly_segmentation( vml_data *D, bool final_iteration )
{
  size_t W = D->_W;
  size_t H = D->_H;
  vml_workspace *Dw = D->_w;
  assert( Dw != NULL );

  // Allocate temporary mem for q updates
  vector<float*> tmp;
  for ( size_t k=0; k<D->_K; k++ ) {
    vml_dimension_data *ddata = D->_dim[k];
    assert( ddata != NULL );
    vml_dim_workspace *w = ddata->_w;
    assert( w != NULL );
    float *tmpv = NULL;
    CUDA_SAFE_CALL( cudaMalloc( &tmpv, w->_nfbytes ));
    CUDA_SAFE_CALL( cudaMemcpy( tmpv, w->_q, w->_nfbytes, cudaMemcpyDeviceToDevice ));
    tmp.push_back( tmpv );
  }

  // Only supported for certain layouts
  assert( D->_K == 3 );
  assert( Dw->_dataterm_on_the_fly_segmentation );

  // Update for primal Lagrange multiplier (mu) and dual qs
  for ( size_t g=0; g<D->_labels.size(); g++ ) {
    int *label = D->_labels[g];

    // STANDARD RELAXATION (ICCV 2011)
    // Collect qs (one per dimension)
    float *qs[ VML_MAX_DIM ];
    float *qs_new[ VML_MAX_DIM ];
    
    float vlabel[3];
    for ( size_t k=0; k<D->_K; k++ ) {
      vml_dimension_data *ddata = D->_dim[k];
      assert( ddata != NULL );
      vml_dim_workspace *w = ddata->_w;
      assert( w != NULL );

      vlabel[k] = ddata->_values[ label[k] ];
      size_t offset = vml_dim_label_offset( ddata, label[k] );
      qs[k] = w->_q + offset;
      qs_new[k] = tmp[k] + offset;
    }
 
    // Copy data term layer, if stored on CPU
    size_t offset = g * W * H;
    update_mu_q_dim3_vml_segmentation_device<<< Dw->_dimGrid, Dw->_dimBlock >>>
      ( W, H,
	Dw->_tau_mu,
	D->_dim[0]->_w->_sigma_q,
	D->_dim[1]->_w->_sigma_q,
	D->_dim[2]->_w->_sigma_q,
	Dw->_theta,
	qs[0], qs[1], qs[2],
	vlabel[0], vlabel[1], vlabel[2],
	Dw->_dataterm_segmentation_r,
	Dw->_dataterm_segmentation_g,
	Dw->_dataterm_segmentation_b,
	Dw->_mu + offset,
	qs_new[0], qs_new[1], qs_new[2] );
  }

  // Store q updates
  for ( size_t k=0; k<D->_K; k++ ) {
    vml_dimension_data *ddata = D->_dim[k];
    assert( ddata != NULL );
    vml_dim_workspace *w = ddata->_w;
    assert( w != NULL );
    float *tmpv = tmp[k];
    if ( final_iteration ) {
      CUDA_SAFE_CALL( cudaFree( tmpv ));
    }
    else {
      CUDA_SAFE_CALL( cudaFree( w->_q ));
      w->_q = tmpv;
    }
  }
  tmp.clear();
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}



// Primal projection, direct projection chunk-wise
bool coco::vml_primal_prox_projection( vml_data *D, bool final_iteration )
{
  size_t W = D->_W;
  size_t H = D->_H;
  vml_workspace *Dw = D->_w;
  assert( Dw != NULL );

  // Allocate temporary mem for q updates
  vector<float*> tmp;
  for ( size_t k=0; k<D->_K; k++ ) {
    vml_dimension_data *ddata = D->_dim[k];
    assert( ddata != NULL );
    vml_dim_workspace *w = ddata->_w;
    assert( w != NULL );
    float *tmpv = NULL;
    CUDA_SAFE_CALL( cudaMalloc( &tmpv, w->_nfbytes ));
    CUDA_SAFE_CALL( cudaMemcpy( tmpv, w->_q, w->_nfbytes, cudaMemcpyDeviceToDevice ));
    tmp.push_back( tmpv );
  }

  // Allocate temporary mem for mu
  float *mu = NULL;
  CUDA_SAFE_CALL( cudaMalloc( &mu, Dw->_nfbytes_chunk ));
  size_t W_mu = Dw->_chunk_width;

  // Allocate temporary mem for data term
  float *rho = NULL;
  size_t W_rho = W;
  if ( D->_cpu_data_term ) {
    CUDA_SAFE_CALL( cudaMalloc( &rho, Dw->_nfbytes_chunk ));
    W_rho = W_mu;
  }

  // For each chunk ...
  for ( size_t j=0; j<Dw->_nchunks; j++ ) {

    // Initialize rho for this chunk
    size_t chunk_offset = Dw->_chunk_width * j;
    //TRACE( "CHUNK " << j << "  offset " << chunk_offset <<
    //	   "  blocksize " << Dw->_dimBlock.x << "  gridsize " << Dw->_dimGridChunk.x << endl ); 

    float *current_rho;
    if ( D->_cpu_data_term ) {
      assert( Dw->_chunk_rho.size() > j );
      CUDA_SAFE_CALL( cudaMemcpy( rho, Dw->_chunk_rho[j], Dw->_nfbytes_chunk, cudaMemcpyHostToDevice ));
      current_rho = rho;
    }
    else {
      current_rho = Dw->_rho + chunk_offset;
    }

    // Clear mu
    CUDA_SAFE_CALL( cudaMemset( mu, 0, Dw->_nfbytes_chunk ));

    // For each inner iteration ...
    for ( size_t n=0; n<D->_inner_iterations; n++ ) {

      // Update for primal Lagrange multiplier (mu) and dual qs
      for ( size_t g=0; g<D->_labels.size(); g++ ) {
	int *label = D->_labels[g];
	
	// STANDARD RELAXATION (ICCV 2011)
	// Collect qs (one per dimension)
	float *qs[ VML_MAX_DIM ];
	float *qs_new[ VML_MAX_DIM ];
	
	for ( size_t k=0; k<D->_K; k++ ) {
	  vml_dimension_data *ddata = D->_dim[k];
	  assert( ddata != NULL );
	  vml_dim_workspace *w = ddata->_w;
	  assert( w != NULL );
	  
	  size_t offset = vml_dim_label_offset( ddata, label[k] );
	  qs[k] = w->_q + offset;
	  qs_new[k] = tmp[k] + offset;
	}
      
	// For now, temp solution (hard code dim 2 and 3)
	size_t mu_offset = g * W_mu * H;
	size_t rho_offset = g * W_rho * H;
	if ( D->_K == 2 ) {
	  update_mu_q_dim2_vml_chunk_device<<< Dw->_dimGridChunk, Dw->_dimBlock >>>
	    ( W, W_mu, W_rho, H,
	      chunk_offset,
	      Dw->_tau_mu,
	      D->_dim[0]->_w->_sigma_q,
	      D->_dim[1]->_w->_sigma_q,
	      Dw->_theta,
	      qs[0], qs[1],
	      current_rho + rho_offset,
	      mu + mu_offset,
	      qs_new[0], qs_new[1] );
	}
	else if ( D->_K == 3 ) {
	  update_mu_q_dim3_vml_chunk_device<<< Dw->_dimGridChunk, Dw->_dimBlock >>>
	    ( W, W_mu, W_rho, H,
	      chunk_offset,
	      Dw->_tau_mu,
	      D->_dim[0]->_w->_sigma_q,
	      D->_dim[1]->_w->_sigma_q,
	      D->_dim[2]->_w->_sigma_q,
	      Dw->_theta,
	      qs[0], qs[1], qs[2],
	      current_rho + rho_offset,
	      mu + mu_offset,
	      qs_new[0], qs_new[1], qs_new[2] );
	}
	else {
	  ERROR( "Currently VML is only implemented for label space dimension 2 and 3." << endl );
	  assert( false );
	}
      } // label loop

      // Store q updates
      for ( size_t k=0; k<D->_K; k++ ) {
	vml_dimension_data *ddata = D->_dim[k];
	assert( ddata != NULL );
	vml_dim_workspace *w = ddata->_w;
	assert( w != NULL );
	swap( w->_q, tmp[k] );
      }

      // Project qs
      for ( size_t k=0; k<D->_K; k++ ) {
	vml_dimension_data *ddata = D->_dim[k];
	assert( ddata != NULL );
	vml_dim_workspace *w = ddata->_w;
	assert( w != NULL );
	for ( size_t g=0; g<ddata->_G; g++ ) {
	  size_t offset = vml_dim_label_offset( ddata, g );
	  float *qs = w->_q + offset;
	  cuda_max_truncate_device<<< Dw->_dimGrid, Dw->_dimBlock >>>
	    ( W, H, qs, 0.0f );
	}
      }

      // Update step sizes (Chambolle/Pock algorithm)

    } // inner iteration loop
  } // chunk loop
  CUDA_SAFE_CALL( cudaThreadSynchronize() );


  // Cleanup
  for ( size_t k=0; k<D->_K; k++ ) {
    float *tmpv = tmp[k];
    CUDA_SAFE_CALL( cudaFree( tmpv ));
  }
  tmp.clear();
  if ( D->_cpu_data_term ) {
    CUDA_SAFE_CALL( cudaFree( rho ));
  }
  CUDA_SAFE_CALL( cudaFree( mu ));

  return true;
}


// Primal projection, direct projection chunk-wise
bool coco::vml_primal_prox_projection_on_the_fly_segmentation( vml_data *D, bool final_iteration )
{
  size_t W = D->_W;
  size_t H = D->_H;
  vml_workspace *Dw = D->_w;
  assert( Dw != NULL );

  // Allocate temporary mem for q updates
  vector<float*> tmp;
  for ( size_t k=0; k<D->_K; k++ ) {
    vml_dimension_data *ddata = D->_dim[k];
    assert( ddata != NULL );
    vml_dim_workspace *w = ddata->_w;
    assert( w != NULL );
    float *tmpv = NULL;
    CUDA_SAFE_CALL( cudaMalloc( &tmpv, w->_nfbytes ));
    CUDA_SAFE_CALL( cudaMemcpy( tmpv, w->_q, w->_nfbytes, cudaMemcpyDeviceToDevice ));
    tmp.push_back( tmpv );
  }

  // Allocate temporary mem for mu
  float *mu = NULL;
  CUDA_SAFE_CALL( cudaMalloc( &mu, Dw->_nfbytes_chunk ));
  size_t W_mu = Dw->_chunk_width;

  // For each chunk ...
  for ( size_t j=0; j<Dw->_nchunks; j++ ) {

    // Initialize rho for this chunk
    size_t chunk_offset = Dw->_chunk_width * j;

    // Clear mu
    CUDA_SAFE_CALL( cudaMemset( mu, 0, Dw->_nfbytes_chunk ));

    // For each inner iteration ...
    for ( size_t n=0; n<D->_inner_iterations; n++ ) {

      // Update for primal Lagrange multiplier (mu) and dual qs
      for ( size_t g=0; g<D->_labels.size(); g++ ) {
	int *label = D->_labels[g];
	
	// STANDARD RELAXATION (ICCV 2011)
	// Collect qs (one per dimension)
	float *qs[ VML_MAX_DIM ];
	float *qs_new[ VML_MAX_DIM ];
	float vlabel[3];

	for ( size_t k=0; k<D->_K; k++ ) {
	  vml_dimension_data *ddata = D->_dim[k];
	  assert( ddata != NULL );
	  vml_dim_workspace *w = ddata->_w;
	  assert( w != NULL );
	  
	  vlabel[k] = ddata->_values[ label[k] ];
	  size_t offset = vml_dim_label_offset( ddata, label[k] );
	  qs[k] = w->_q + offset;
	  qs_new[k] = tmp[k] + offset;
	}
      
	// For now, temp solution (hard code dim 2 and 3)
	size_t mu_offset = g * W_mu * H;
	update_mu_q_dim3_vml_chunk_segmentation_device<<< Dw->_dimGridChunk, Dw->_dimBlock >>>
	  ( W, W_mu, H,
	    chunk_offset,
	    Dw->_tau_mu,
	    D->_dim[0]->_w->_sigma_q,
	    D->_dim[1]->_w->_sigma_q,
	    D->_dim[2]->_w->_sigma_q,
	    Dw->_theta,
	    qs[0], qs[1], qs[2],
	    vlabel[0], vlabel[1], vlabel[2],
	    Dw->_dataterm_segmentation_r,
	    Dw->_dataterm_segmentation_g,
	    Dw->_dataterm_segmentation_b,
	    mu + mu_offset,
	    qs_new[0], qs_new[1], qs_new[2] );
      } // label loop

      // Store q updates
      for ( size_t k=0; k<D->_K; k++ ) {
	vml_dimension_data *ddata = D->_dim[k];
	assert( ddata != NULL );
	vml_dim_workspace *w = ddata->_w;
	assert( w != NULL );
	swap( w->_q, tmp[k] );
      }

      // Project qs
      for ( size_t k=0; k<D->_K; k++ ) {
	vml_dimension_data *ddata = D->_dim[k];
	assert( ddata != NULL );
	vml_dim_workspace *w = ddata->_w;
	assert( w != NULL );
	for ( size_t g=0; g<ddata->_G; g++ ) {
	  size_t offset = vml_dim_label_offset( ddata, g );
	  float *qs = w->_q + offset;
	  cuda_max_truncate_device<<< Dw->_dimGrid, Dw->_dimBlock >>>
	    ( W, H, qs, 0.0f );
	}
      }

      // Update step sizes (Chambolle/Pock algorithm)

    } // inner iteration loop
  } // chunk loop
  CUDA_SAFE_CALL( cudaThreadSynchronize() );


  // Cleanup
  for ( size_t k=0; k<D->_K; k++ ) {
    float *tmpv = tmp[k];
    CUDA_SAFE_CALL( cudaFree( tmpv ));
  }
  tmp.clear();
  CUDA_SAFE_CALL( cudaFree( mu ));

  return true;
}


/*
// Primal projection, direct projection chunk-wise
bool coco::vml_primal_prox_projection_on_the_fly_segmentation( vml_data *D, bool final_iteration )
{
  size_t W = D->_W;
  size_t H = D->_H;
  vml_workspace *Dw = D->_w;
  assert( Dw != NULL );

  // Allocate temporary mem for q updates
  vector<float*> tmp;
  for ( size_t k=0; k<D->_K; k++ ) {
    vml_dimension_data *ddata = D->_dim[k];
    assert( ddata != NULL );
    vml_dim_workspace *w = ddata->_w;
    assert( w != NULL );
    float *tmpv = NULL;
    CUDA_SAFE_CALL( cudaMalloc( &tmpv, w->_nfbytes ));
    CUDA_SAFE_CALL( cudaMemcpy( tmpv, w->_q, w->_nfbytes, cudaMemcpyDeviceToDevice ));
    tmp.push_back( tmpv );
  }

  // Allocate temporary mem for mu
  float *mu = NULL;
  CUDA_SAFE_CALL( cudaMalloc( &mu, Dw->_nfbytes_chunk ));
  size_t W_mu = Dw->_chunk_width;

  // Only special layouts supported
  assert( D->_K == 3 );
  assert( Dw->_dataterm_on_the_fly_segmentation );

  // For each chunk ...
  for ( size_t j=0; j<Dw->_nchunks; j++ ) {

    // Initialize rho for this chunk
    size_t chunk_offset = Dw->_chunk_width * j;

    // Clear mu
    CUDA_SAFE_CALL( cudaMemset( mu, 0, Dw->_nfbytes_chunk ));

    // For each inner iteration ...
    for ( size_t n=0; n<D->_inner_iterations; n++ ) {

      // Update for primal Lagrange multiplier (mu) and dual qs
      for ( size_t g=0; g<D->_labels.size(); g++ ) {
	int *label = D->_labels[g];
	
	// STANDARD RELAXATION (ICCV 2011)
	// Collect qs (one per dimension)
	float *qs[ VML_MAX_DIM ];
	float *qs_new[ VML_MAX_DIM ];
	float vlabel[3];

	for ( size_t k=0; k<D->_K; k++ ) {
	  vml_dimension_data *ddata = D->_dim[k];
	  assert( ddata != NULL );
	  vml_dim_workspace *w = ddata->_w;
	  assert( w != NULL );
	  
	  vlabel[k] = ddata->_values[ label[k] ];
	  size_t offset = vml_dim_label_offset( ddata, label[k] );
	  qs[k] = w->_q + offset;
	  qs_new[k] = tmp[k] + offset;
	}
      
	// For now, temp solution (hard code dim 2 and 3)
	size_t mu_offset = g * W_mu * H;
	update_mu_q_dim3_vml_chunk_segmentation_device<<< Dw->_dimGridChunk, Dw->_dimBlock >>>
	  ( W, W_mu, H,
	    chunk_offset,
	    Dw->_tau_mu,
	    D->_dim[0]->_w->_sigma_q,
	    D->_dim[1]->_w->_sigma_q,
	    D->_dim[2]->_w->_sigma_q,
	    Dw->_theta,
	    qs[0], qs[1], qs[2],
	    vlabel[0], vlabel[1], vlabel[2],
	    Dw->_dataterm_segmentation_r,
	    Dw->_dataterm_segmentation_g,
	    Dw->_dataterm_segmentation_b,
	    mu + mu_offset,
	    qs_new[0], qs_new[1], qs_new[2] );
      } // label loop

      // Store q updates
      for ( size_t k=0; k<D->_K; k++ ) {
	vml_dimension_data *ddata = D->_dim[k];
	assert( ddata != NULL );
	vml_dim_workspace *w = ddata->_w;
	assert( w != NULL );
	swap( w->_q, tmp[k] );
      }

      // Update step sizes (Chambolle/Pock algorithm)

    } // inner iteration loop
  } // chunk loop
  CUDA_SAFE_CALL( cudaThreadSynchronize() );


  // Cleanup
  for ( size_t k=0; k<D->_K; k++ ) {
    float *tmpv = tmp[k];
    CUDA_SAFE_CALL( cudaFree( tmpv ));
  }
  tmp.clear();
  CUDA_SAFE_CALL( cudaFree( mu ));

  return true;
}
*/



// Perform one primal step
bool coco::vml_update_primal_regularizer( vml_data *D, vml_dimension_data *ddata,
					  bool final_iteration )
{
  assert( D != NULL );
  assert( ddata != NULL );

  size_t W = D->_W;
  size_t H = D->_H;
  vml_workspace *Dw = D->_w;
  assert( Dw != NULL );
  vml_dim_workspace *w = ddata->_w;
  assert( w != NULL );
  size_t G = ddata->_G;
  assert( G>1 );

  // Second step: Kernel call for each label
  // Update for primal Lagrange multiplier (mu)
  // Temporary layer to store mu
  size_t tmpsize = 0;
  if ( ddata->_type == VML_LINEAR ||
       ddata->_type == VML_CYCLIC ||
       ddata->_type == VML_TRUNCATED_LINEAR ) {
    // ps need to be saved
    tmpsize = w->_nfbytes;
  }

  float *tmp_x = NULL;
  float *tmp_y = NULL;
  if ( tmpsize != 0 ) {
    CUDA_SAFE_CALL( cudaMalloc( &tmp_x, tmpsize ));
    CUDA_SAFE_CALL( cudaMemcpy( tmp_x, w->_px, w->_nfbytes, cudaMemcpyDeviceToDevice ));
    CUDA_SAFE_CALL( cudaMalloc( &tmp_y, tmpsize ));
    CUDA_SAFE_CALL( cudaMemcpy( tmp_y, w->_py, w->_nfbytes, cudaMemcpyDeviceToDevice ));
  }

  if ( ddata->_type == VML_LINEAR || ddata->_type == VML_CYCLIC ) {

    for ( size_t g=0; g+1<G; g++ ) {

      float cost = gsl_matrix_get( ddata->_cost, g, g+1 );
      size_t offset1 = vml_dim_label_offset( ddata, g );
      size_t offset2 = vml_dim_label_offset( ddata, g+1 );
      // sanity check
      size_t eta_offset = vml_dim_regularizer_offset( ddata, g, g+1 );
      assert( eta_offset == offset1 );

      update_eta_vml_device<<< Dw->_dimGrid, Dw->_dimBlock >>>
	( W, H,
	  cost,
	  w->_tau_eta,
	  w->_sigma_p,
	  Dw->_theta,
	  w->_eta_x + eta_offset, w->_eta_y + eta_offset,
	  w->_px + offset1, w->_py + offset1,
	  w->_px + offset2, w->_py + offset2,
	  tmp_x + offset1, tmp_y + offset1,
	  tmp_x + offset2, tmp_y + offset2 );
    }

    if ( ddata->_type == VML_CYCLIC ) {
      // wraparound on top
      float cost = gsl_matrix_get( ddata->_cost, G-1, 0 );
      size_t offset1 = vml_dim_label_offset( ddata, G-1 );
      size_t offset2 = vml_dim_label_offset( ddata, 0 );
      // sanity check
      size_t eta_offset = vml_dim_regularizer_offset( ddata, G-1, 0 );
      assert( eta_offset == offset1 );
      
      update_eta_vml_device<<< Dw->_dimGrid, Dw->_dimBlock >>>
	( W, H,
	  cost,
	  w->_tau_eta,
	  w->_sigma_p,
	  Dw->_theta,
	  w->_eta_x + eta_offset, w->_eta_y + eta_offset,
	  w->_px + offset1, w->_py + offset1,
	  w->_px + offset2, w->_py + offset2,
	  tmp_x + offset1, tmp_y + offset1,
	  tmp_x + offset2, tmp_y + offset2 );
    }
  }
  else if ( ddata->_type == VML_TRUNCATED_LINEAR ) {

    for ( size_t g1=0; g1<G; g1++ ) {
      for ( size_t g2=g1+1; g2<G; g2++ ) {

	float cost = gsl_matrix_get( ddata->_cost, g1, g2 );
	size_t offset1 = vml_dim_label_offset( ddata, g1 );
	size_t offset2 = vml_dim_label_offset( ddata, g2 );
	size_t eta_offset = vml_dim_regularizer_offset( ddata, g1, g2 );
	
	update_eta_vml_device<<< Dw->_dimGrid, Dw->_dimBlock >>>
	  ( W, H,
	    cost,
	    w->_tau_eta,
	    w->_sigma_p,
	    Dw->_theta,
	    w->_eta_x + eta_offset, w->_eta_y + eta_offset,
	    w->_px + offset1, w->_py + offset1,
	    w->_px + offset2, w->_py + offset2,
	    tmp_x + offset1, tmp_y + offset1,
	    tmp_x + offset2, tmp_y + offset2 );
      }
    }
  }


  if ( tmpsize != 0 ) {
    // Copy updated ps
    CUDA_SAFE_CALL( cudaFree( w->_px ));
    CUDA_SAFE_CALL( cudaFree( w->_py ));
    w->_px = tmp_x;
    w->_py = tmp_y;
  }

  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}


// Perform one dual step
bool coco::vml_dual_prox( vml_data *D )
{
  size_t W = D->_W;
  size_t H = D->_H;
  vml_workspace *Dw = D->_w;

  // Kernel call for each layer in each dimension
  // Update for primal solution (u)
  for ( size_t k=0; k<D->_K; k++ ) {
    vml_dimension_data *ddata = D->_dim[k];
    assert( ddata != NULL );
    vml_dim_workspace *w = ddata->_w;
    assert( w != NULL );
    
    for ( size_t g=0; g<ddata->_G; g++ ) {
      size_t offset = vml_dim_label_offset( ddata, g );

      if ( ddata->_type == VML_POTTS ) {
	// lambda multiplied with 0.5 to compensate for duplicate edge count
	// in potts model
	compute_dual_prox_vml_potts_device<<< Dw->_dimGrid, Dw->_dimBlock >>>
	  ( W, H, 
	    0.5f * ddata->_lambda,
	    w->_sigma_p, w->_sigma_q,
	    w->_uq + offset,
	    w->_px + offset, w->_py + offset,
	    w->_q + offset );
      }
      else {
	compute_dual_prox_vml_device<<< Dw->_dimGrid, Dw->_dimBlock >>>
	  ( W, H, 
	    w->_sigma_p, w->_sigma_q,
	    w->_uq + offset,
	    w->_px + offset, w->_py + offset,
	    w->_q + offset );
      }
    }

    update_sigma_vml_device<<< Dw->_dimGrid, Dw->_dimBlock >>>
      ( W, H, W*H, ddata->_G,
	w->_sigma_s,
	w->_uq,
	w->_sigma );
  }
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}




/******************************************
    RELAXATION ENERGIES FOR POTTS MODEL
*******************************************/

static __global__ void vml_accumulate_dataterm_energy_device( int W, int H,
							      float *q,
							      float *u,
							      float *e )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;
  e[o] += u[o] * q[o];
}



/* MERGED WITH GENERAL METHOD
static __global__ void potts_multilabel_accumulate_regularizer_energy_device( int W, int H,
									      float lambda,
									      float *u,
									      float *e )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = ox + oy*W;
  float ex = 0.0f;
  float uv = u[o];
  if ( ox<W-1 ) {
    ex = u[o+1] - uv;
  }
  float ey = 0.0f;
  if ( oy<H-1 ) {
    ey = u[o+W] - uv;
  }
  e[o] += 0.5f * lambda * hypotf( ex, ey );
}
*/


static __global__ void multilabel_accumulate_regularizer_energy_device( int W, int H,
									float lambda,
									float *u,
									float *px,
									float *py,
									float *e )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = ox + oy*W;
  // Divergence of p, backward differences, dirichlet
  float div = px[o] + py[o];
  if ( ox>0 ) {
    div -= px[o-1];
  }
  if ( oy>0 ) {
    div -= py[o-W];
  }
  e[o] += u[o] * div;
}


double coco::vml_energy( vml_data *D, bool data_q )
{
  size_t W = D->_W;
  size_t H = D->_H;
  vml_workspace *Dw = D->_w;

  // One helper array for accumulation
  float *e = NULL;
  CUDA_SAFE_CALL( cudaMalloc( &e, W*H*sizeof(float)) );
  CUDA_SAFE_CALL( cudaMemset( e, 0, W*H*sizeof(float)) );
  // One helper array for reduction
  float *r = NULL;
  CUDA_SAFE_CALL( cudaMalloc( &r, W*H*sizeof(float) ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  if ( data_q ) {

    // Accumulate dataterm energy (variant using qs)
    for ( size_t k=0; k<D->_K; k++ ) {
      vml_dimension_data *ddata = D->_dim[k];
      assert( ddata != NULL );
      vml_dim_workspace *w = ddata->_w;
      assert( w != NULL );
      
      for ( size_t g=0; g<ddata->_G; g++ ) {
	size_t offset = vml_dim_label_offset( ddata, g );
	float *q = w->_q + offset;
	float *u = w->_u + offset;
	vml_accumulate_dataterm_energy_device<<< Dw->_dimGrid, Dw->_dimBlock >>>
	  ( W,H, q,u, e );
      }
    }
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
  }
  else {

    // Accumulate dataterm energy, separate for each label
    // Update for primal Lagrange multiplier (mu) and dual qs
    // Second helper array for data term layer
    float *rho = NULL;
    size_t nfbytes_layer = W*H*sizeof(float);
    CUDA_SAFE_CALL( cudaMalloc( &rho, nfbytes_layer ));
    
    for ( size_t g=0; g<D->_labels.size(); g++ ) {
      int *label = D->_labels[g];
      
      // Copy data term layer, if stored on CPU
      size_t offset = g * W * H;
      if ( D->_cpu_data_term ) {
	CUDA_SAFE_CALL( cudaMemcpy( rho, Dw->_rho + offset, nfbytes_layer, cudaMemcpyHostToDevice ));
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
      }
      else {
	CUDA_SAFE_CALL( cudaMemcpy( rho, Dw->_rho + offset, nfbytes_layer, cudaMemcpyDeviceToDevice ));
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
      }
      
      // Multiply all solution functions with data term layer
      for ( size_t k=0; k<D->_K; k++ ) {
	vml_dimension_data *ddata = D->_dim[k];
	assert( ddata != NULL );
	vml_dim_workspace *w = ddata->_w;
	assert( w != NULL );
	size_t offset = vml_dim_label_offset( ddata, label[k] );
	cuda_multiply_with_device<<< Dw->_dimGrid, Dw->_dimBlock >>>
	  ( W,H, rho, w->_u + offset );
      }
      // add to energy
      cuda_add_to_device<<< Dw->_dimGrid, Dw->_dimBlock >>>
	( W,H, rho, e );
    }
    // Kill temp memory
    CUDA_SAFE_CALL( cudaFree( rho ));
  }

  // Reduce
  float result_e_data;
  cuda_sum_reduce( W,H, e, r, &result_e_data );

  // Second step: regularizer energy
  CUDA_SAFE_CALL( cudaMemset( e, 0, W*H*sizeof(float) ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // Accumulate regularizer energy
  for ( size_t k=0; k<D->_K; k++ ) {
    vml_dimension_data *ddata = D->_dim[k];
    assert( ddata != NULL );
    vml_dim_workspace *w = ddata->_w;
    assert( w != NULL );
    
    for ( size_t g=0; g<ddata->_G; g++ ) {
      size_t offset = vml_dim_label_offset( ddata, g );
      float *u = w->_u + offset;
      float *px = w->_px + offset;
      float *py = w->_py + offset;
      /*
      if ( ddata->_type == VML_POTTS ) {
	potts_multilabel_accumulate_regularizer_energy_device<<< Dw->_dimGrid, Dw->_dimBlock >>>
	  ( W,H, ddata->_lambda, u,e );
      }
      else {
      */
	multilabel_accumulate_regularizer_energy_device<<< Dw->_dimGrid, Dw->_dimBlock >>>
	  ( W,H, ddata->_lambda, u, px, py, e );
	//}
    }
  }
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // Reduce and half (edges counted twice)
  float result_e_reg;
  cuda_sum_reduce( W,H, e, r, &result_e_reg );

  // Cleanup
  TRACE( "VML relaxation energy: " << endl );
  TRACE( "  regularizer: " << result_e_reg << endl );
  TRACE( "  data term  : " << result_e_data << endl );
  TRACE( "  total      : " << result_e_data + result_e_reg << endl );

  CUDA_SAFE_CALL( cudaFree( e ));
  CUDA_SAFE_CALL( cudaFree( r ));
  return result_e_data + result_e_reg;
}


