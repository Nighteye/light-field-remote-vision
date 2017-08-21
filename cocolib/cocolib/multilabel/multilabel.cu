/* -*-c++-*- */
#include "multilabel.h"
#include "multilabel.cuh"

#include "cuda.h"

#include "../defs.h"
#include "../cuda/cuda_helper.h"
//#include "../cuda/cuda_kernels.cu"

using namespace std;

// Alloc multilabel problem structure
bool coco::multilabel_data_init( multilabel_data *data, size_t W, size_t H, size_t L )
{
  assert( L > 1 );

  // Image sizes
  data->_W = W;
  data->_H = H;
  data->_N = data->_W * data->_H;
  // Label space discretization
  data->_G = L;

  // Regularizer
  data->_lambda = 1.0f;
  multilabel_set_label_range( data, 0.0f, 1.0f );
  return true;
}


// Mem requirements
size_t coco::multilabel_workspace_size( multilabel_data *data )
{
  size_t mb = 1048576;
  size_t total = data->_w->_nfbytes * 3 + data->_w->_urbytes;
  TRACE3( "allocating multilabel optimization structure " << data->_W << " x " << data->_H << " x " << data->_G << endl );
  TRACE3( "  total mem: " << total / mb << " Mb." << endl );
  return total;
}

// Init/free
bool coco::multilabel_workspace_init( multilabel_data *data, multilabel_workspace *w )
{
  // Current integer solution
  w->_urbytes = data->_N * sizeof( int );
  CUDA_SAFE_CALL( cudaMalloc( &w->_ur, w->_urbytes ));

  // Size of 3D fields over Image x Label space
  w->_nfbytes = data->_N * data->_G * sizeof(float);

  // Primal variable
  CUDA_SAFE_CALL( cudaMalloc( &w->_u, w->_nfbytes ));
  CUDA_SAFE_CALL( cudaMalloc( &w->_uq, w->_nfbytes ));
  // Precomputed data term
  CUDA_SAFE_CALL( cudaMalloc( &w->_rho, w->_nfbytes ));
  CUDA_SAFE_CALL( cudaMemset( w->_rho, 0, w->_nfbytes ));

  // Simplex constraint dual variable
  w->_nfbytes_sigma = data->_N * sizeof( float );
  CUDA_SAFE_CALL( cudaMalloc( &w->_sigma, w->_nfbytes_sigma ));

  // Step sizes
  w->_sigma_p = 1.0f / 4.0f;
  w->_tau_u = 1.0f / 6.0f;
  w->_sigma_s = 1.0f / data->_G;

  // Finalize
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
  w->_g = NULL;

  data->_w = w;
  return true;
}



// Set equidistant labels
bool coco::multilabel_set_label_range( multilabel_data *data,
				       float gmin, float gmax )
{
  float gwidth = (gmax - gmin) / (data->_G-1);
  data->_labels.clear();
  float s = gmin;
  TRACE6( "Initializing label range from " << gmin << " to " << gmax << " steps " << data->_G-1 << endl );
  for ( size_t g=1; g<=data->_G; g++ ) {
    data->_labels.push_back( s );
    s += gwidth;
  }
  return true;
}


// Free multilabel problem structure
bool coco::multilabel_workspace_free( multilabel_data *data, multilabel_workspace *w )
{
  CUDA_SAFE_CALL( cudaFree( w->_ur ));
  CUDA_SAFE_CALL( cudaFree( w->_u ));
  CUDA_SAFE_CALL( cudaFree( w->_uq ));
  CUDA_SAFE_CALL( cudaFree( w->_sigma ));
  CUDA_SAFE_CALL( cudaFree( w->_rho ));
  if ( w->_g != NULL ) {
    CUDA_SAFE_CALL( cudaFree( w->_g ));
  }
  return true;
}

bool coco::multilabel_data_free( multilabel_data* data )
{
  return true;
}


// Set local spatial smoothness weight
bool coco::multilabel_set_spatial_smoothness_weight( multilabel_data* data,
						     const gsl_matrix *g )
{
  CUDA_SAFE_CALL( cudaMalloc( &data->_w->_g, sizeof( float ) * data->_N ));
  cuda_memcpy( data->_w->_g, const_cast<gsl_matrix*> (g) );
  return true;
}



// Set current solution data
bool coco::multilabel_set_solution_lifting( multilabel_data* data,
					    const gsl_matrix *u )
{
  // Set up 3D structure
  multilabel_workspace *w = data->_w;
  float *phi = new float[ w->_nfbytes ];
  size_t index = 0;
  for ( size_t y=0; y<data->_H; y++ ) {
    for ( size_t x=0; x<data->_W; x++ ) {
      float v = u->data[index];
      for ( size_t g=0; g<data->_G; g++ ) {
	size_t p = index + g*data->_N;
	float l = data->_labels[g];
	phi[ p ] = (v >= l) ? 1.0f : 0.0f;
      }
      index++;
    }
  }

  // Copy to GPU
  CUDA_SAFE_CALL( cudaMemcpy( w->_u, phi, w->_nfbytes, cudaMemcpyHostToDevice ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // Cleanup
  delete[] phi;
  return true;
}


// Set current solution data
bool coco::multilabel_set_solution_lifting( multilabel_data* data,
					    const float *u )
{
  // Set up 3D structure
  multilabel_workspace *w = data->_w;
  float *phi = new float[ w->_nfbytes ];
  size_t index = 0;
  for ( size_t y=0; y<data->_H; y++ ) {
    for ( size_t x=0; x<data->_W; x++ ) {
      float v = u[index];
      for ( size_t g=0; g<data->_G; g++ ) {
	size_t p = index + g*data->_N;
	float l = data->_labels[g];
	phi[ p ] = (v >= l) ? 1.0f : 0.0f;
      }
      index++;
    }
  }

  // Copy to GPU
  CUDA_SAFE_CALL( cudaMemcpy( w->_u, phi, w->_nfbytes, cudaMemcpyHostToDevice ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // Cleanup
  delete[] phi;
  return true;
}



// Set precomputed data term as W x H x L 3D array
bool coco::multilabel_set_data_term( multilabel_data* data,
				     const float *rho )
{
  multilabel_workspace *w = data->_w;
  CUDA_SAFE_CALL( cudaMemcpy( w->_rho, rho, w->_nfbytes, cudaMemcpyHostToDevice ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}




// Get current solution
bool coco::multilabel_get_solution_lifting( multilabel_data *data,
					    gsl_matrix *u,
					    const float threshold )
{
  // Set up 3D structure
  multilabel_workspace *w = data->_w;
  float *phi = new float[ w->_nfbytes ];
  // Copy from GPU
  CUDA_SAFE_CALL( cudaMemcpy( phi, w->_u, w->_nfbytes, cudaMemcpyDeviceToHost ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // Reconstruct u
  size_t index = 0;
  for ( size_t y=0; y<data->_H; y++ ) {
    for ( size_t x=0; x<data->_W; x++ ) {
      int v = -1;

      if ( y == data->_H / 2 ) {
	TRACE6( endl << "Solution values " << x << " " << y << ":   " );
      }

      for ( size_t g=0; g<data->_G; g++ ) {
	size_t p = index + g*data->_N;

	if ( y == data->_H / 2 ) {
	  TRACE6( phi[p] << "   " );
	}

	if ( phi[ p ] >= threshold ) {
	  v++;
	}
      }

      u->data[index] = data->_labels[v];
      index++;
    }
  }


  // Cleanup
  delete[] phi;
  return true;
}



// Get current solution
bool coco::multilabel_get_solution_lifting( multilabel_data *data,
					    float *u,
					    const float threshold )
{
  // Set up 3D structure
  multilabel_workspace *w = data->_w;
  float *phi = new float[ w->_nfbytes ];
  // Copy from GPU
  CUDA_SAFE_CALL( cudaMemcpy( phi, w->_u, w->_nfbytes, cudaMemcpyDeviceToHost ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // Reconstruct u
  size_t index = 0;
  for ( size_t y=0; y<data->_H; y++ ) {
    for ( size_t x=0; x<data->_W; x++ ) {
      int v = -1;

      if ( y == data->_H / 2 ) {
	TRACE6( endl << "Solution values " << x << " " << y << ":   " );
      }

      for ( size_t g=0; g<data->_G; g++ ) {
	size_t p = index + g*data->_N;

	if ( y == data->_H / 2 ) {
	  TRACE6( phi[p] << "   " );
	}

	if ( phi[ p ] >= threshold ) {
	  v++;
	}
      }

      u[index] = data->_labels[v];
      index++;
    }
  }


  // Cleanup
  delete[] phi;
  return true;
}




// Set current solution data
bool coco::multilabel_set_solution_indicator( multilabel_data* data,
					      const gsl_matrix *um )
{
  // Set up 4D structure for data term
  size_t W = data->_W;
  size_t H = data->_H;
  size_t G = data->_G;
  float *u = new float[W*H*G];
  memset( u, 0, data->_w->_nfbytes );

  size_t index = 0;
  for ( size_t y=0; y<data->_H; y++ ) {
    for ( size_t x=0; x<data->_W; x++ ) {
      float v = um->data[index];
      for ( size_t g=0; g<data->_G; g++ ) {
	if ( v <= data->_labels[g] ) {
	  u[ multilabel_solution_index( data, x,y, g ) ] = 1.0f;
	  break;
	}
      }
      index++;
    }
  }

  // Copy to GPU
  CUDA_SAFE_CALL( cudaMemcpy( data->_w->_u, u, data->_w->_nfbytes,
			      cudaMemcpyHostToDevice ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // Cleanup
  delete[] u;
  return true;
}




static __global__ void multilabel_project_solution_indicator_device( int W, int H, int N, int G,
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
    u[o+g*N] = 0.0f;
    if ( uv > umax ) {
      umax = uv;
      uopt = g;
    }
  }

  ur[o] = uopt;
  u[o + uopt*N] = 1.0f;
}


// Project current relaxed solution onto integer values
bool coco::multilabel_project_solution_indicator( multilabel_data *data )
{
  size_t W = data->_W;
  size_t H = data->_H;
  size_t G = data->_G;
  multilabel_workspace *w = data->_w;
  multilabel_project_solution_indicator_device<<< w->_dimGrid, w->_dimBlock >>>
    ( W,H, W*H, G, w->_u, w->_ur );
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}


// Get current solution (integer labeling after projection)
bool coco::multilabel_get_solution( multilabel_data *data,
				    int *ur )
{
  multilabel_workspace *w = data->_w;
  CUDA_SAFE_CALL( cudaMemcpy( ur, w->_ur, w->_urbytes, cudaMemcpyDeviceToHost ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}


// Set current solution (integer labeling after projection)
bool coco::multilabel_set_solution( multilabel_data *data,
				    int *ur )
{
  multilabel_workspace *w = data->_w;
  CUDA_SAFE_CALL( cudaMemcpy( w->_ur, ur, w->_urbytes, cudaMemcpyHostToDevice ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}



static __global__ void multilabel_set_solution_pointwise_optimum_lifting_device( int W, int H, int N, int G,
										 float *rho, float *u )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = ox + oy*W;
  float rho_min = rho[o];
  int uopt = 0;
  u[o] = 0.0f;
  for ( int g=1; g<G; g++ ) {
    float rhov = rho[o+g*N];
    u[o+g*N] = 0.0f;
    if ( rhov < rho_min ) {
      rho_min = rhov;
      uopt = g;
    }
  }

  for ( int g=1; g<uopt+1; g++ ) {
    u[o+g*N] = 1.0f;
  }
}


// Set current solution data to point-wise optimum of data term
bool coco::multilabel_set_solution_pointwise_optimum_lifting( multilabel_data* data )
{
  size_t W = data->_W;
  size_t H = data->_H;
  size_t G = data->_G;
  multilabel_workspace *w = data->_w;
  multilabel_set_solution_pointwise_optimum_lifting_device<<< w->_dimGrid, w->_dimBlock >>>
    ( W,H, W*H, G, w->_rho, w->_u );
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}


static __global__ void multilabel_set_solution_pointwise_optimum_indicator_device( int W, int H, int N, int G,
										   float *rho, float *u )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = ox + oy*W;
  float rho_min = rho[o];
  int uopt = 0;
  u[o] = 0.0f;
  for ( int g=1; g<G; g++ ) {
    float rhov = rho[o+g*N];
    u[o+g*N] = 0.0f;
    if ( rhov < rho_min ) {
      rho_min = rhov;
      uopt = g;
    }
  }

  u[o + uopt*N] = 1.0f;
}


// Set current solution data to point-wise optimum of data term
bool coco::multilabel_set_solution_pointwise_optimum_indicator( multilabel_data* data )
{
  size_t W = data->_W;
  size_t H = data->_H;
  size_t G = data->_G;
  multilabel_workspace *w = data->_w;
  multilabel_set_solution_pointwise_optimum_indicator_device<<< w->_dimGrid, w->_dimBlock >>>
    ( W,H, W*H, G, w->_rho, w->_u );
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}



// Get current solution
bool coco::multilabel_get_solution_indicator( multilabel_data *data,
					      gsl_matrix *um )
{
  // Set up 4D structure
  size_t W = data->_W;
  size_t H = data->_H;
  size_t G = data->_G;
  float *u = new float[ W*H*G ];

  // Copy from GPU
  multilabel_workspace *w = data->_w;
  CUDA_SAFE_CALL( cudaMemcpy( u, w->_u, w->_nfbytes, cudaMemcpyDeviceToHost ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // Reconstruct u by finding the maximum label variable
  // TODO: Implement better scheme from Lelmann et al., TR
  size_t index = 0;
  for ( size_t y=0; y<data->_H; y++ ) {
    for ( size_t x=0; x<data->_W; x++ ) {
      TRACE6( "Solution @" << x << ", " << y << ": " << endl );

      // u: argmax over G
      int v = -1; float umax = -1.0;
      float sum = 0.0;
      TRACE6( "  dim1: " );
      for ( size_t g=0; g<data->_G; g++ ) {
	float uv = u[ multilabel_solution_index( data, x,y, g ) ];
	sum += uv;
	TRACE6( uv << " " );
	if ( uv > umax ) {
	  v = g; umax = uv;
	}
      }
      um->data[index] = data->_labels[v];
      TRACE6( "  " << v << " = " << data->_labels[v] << "  sum=" << sum << endl );

      // Next pixel
      index++;
    }
  }

  // Cleanup
  delete[] u;
  return true;
}


// Compute current energy
double coco::multilabel_energy( multilabel_data *data )
{
  // TODO
  return 0.0;
}


// Helper functions to compute indices
// Memory layout for solution fields:
// One (W*H) slice per label, first a stack of G1 slices, the a stack of G2 slices
size_t coco::multilabel_solution_index( multilabel_data* data,
					size_t x, size_t y, size_t g )
{
  size_t slice_size = data->_W*data->_H;
  return (x + y*data->_W) + slice_size * g;
}

// Memory layout for data term:
// One (W*H) slice per label, (G1*G2) stacks
size_t coco::multilabel_dataterm_index( multilabel_data* data,
					size_t x, size_t y, size_t g )
{
  size_t slice_size = data->_W*data->_H;
  return (x + y*data->_W) + slice_size * g;
}



