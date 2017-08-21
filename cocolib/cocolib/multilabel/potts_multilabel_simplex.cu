/* -*-c++-*- */
#include "potts_multilabel_simplex.h"
#include "potts_multilabel_simplex.cuh"

#include "../cuda/cuda_helper.h"
#include "../cuda/cuda_reduce.h"
#include "../cuda/simplex_reprojection.h"

#include "../common/gsl_matrix_helper.h"
#include "../common/profiler.h"

using namespace std;

// Alloc multilabel_2d problem structure
coco::potts_multilabel_simplex_data* coco::potts_multilabel_simplex_data_alloc( size_t W, size_t H, size_t G )
{
  assert( G > 1 );
  potts_multilabel_simplex_data *data = new potts_multilabel_simplex_data;
  multilabel_data_init( data, W,H,G );

  // Workspace
  data->_potts_w = new potts_multilabel_simplex_workspace;
  potts_multilabel_simplex_workspace *w = data->_potts_w;
  potts_multilabel_workspace_init( data, w );
  return data;
}



bool coco::potts_multilabel_workspace_init( potts_multilabel_simplex_data* data,
					    potts_multilabel_simplex_workspace *w )
{
  multilabel_workspace_init( data, w );

  // Clear primal
  CUDA_SAFE_CALL( cudaMemset( w->_u, 0, w->_nfbytes ));
  CUDA_SAFE_CALL( cudaMemset( w->_uq, 0, w->_nfbytes ));
  // Init Lagrange multipliers
  CUDA_SAFE_CALL( cudaMemset( w->_sigma, 0, w->_nfbytes_sigma ));
  // Dual variables
  CUDA_SAFE_CALL( cudaMalloc( &w->_x1, w->_nfbytes ));
  CUDA_SAFE_CALL( cudaMalloc( &w->_x2, w->_nfbytes ));
  return data;
}


// Mem requirements
size_t coco::potts_multilabel_simplex_workspace_size( potts_multilabel_simplex_data *data )
{
  size_t mb = 1048576;
  size_t total = data->_w->_nfbytes * 2;
  size_t base = multilabel_workspace_size( data );
  TRACE3( "  Potts regularizer: " << total / mb << " Mb." << endl );
  TRACE3( "  Potts total      : " << (total+base) / mb << " Mb." << endl );
  return total + base;
}



bool coco::potts_multilabel_simplex_data_free( potts_multilabel_simplex_data* data )
{
  multilabel_workspace_free( data, data->_w );
  CUDA_SAFE_CALL( cudaFree( data->_potts_w->_x1 ));
  CUDA_SAFE_CALL( cudaFree( data->_potts_w->_x2 ));
  multilabel_data_free( data );
  delete data->_w;
  delete data;
  return true;
}



// Shortcuts
size_t multilabel_index( coco::multilabel_data *data, size_t x, size_t y, size_t g )
{
  size_t N = data->_W * data->_H;
  return N * g + y * data->_W + x;
}



// Set current solution data
bool coco::potts_multilabel_simplex_set_solution( potts_multilabel_simplex_data* data,
						  const gsl_matrix *um )
{
  return multilabel_set_solution_indicator( data, um );
}


// Reinit algorithm with current solution
bool coco::potts_multilabel_simplex_init( potts_multilabel_simplex_data* data )
{
  potts_multilabel_simplex_workspace *w = data->_potts_w;
  w->_simplex_iter = 0;

  // Init primal variables
  CUDA_SAFE_CALL( cudaMemcpy( w->_uq, w->_u, w->_nfbytes, cudaMemcpyDeviceToDevice ));
  CUDA_SAFE_CALL( cudaMemset( w->_sigma, 0, w->_nfbytes_sigma ));
  CUDA_SAFE_CALL( cudaMemset( w->_x1, 0, w->_nfbytes ));
  CUDA_SAFE_CALL( cudaMemset( w->_x2, 0, w->_nfbytes ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}


// Get current solution
bool coco::potts_multilabel_simplex_get_solution( potts_multilabel_simplex_data *data,
						  gsl_matrix *um )
{
  return multilabel_get_solution_indicator( data, um );
}



/******************************************
    EXPERIMENTAL MULTILABEL ALGORITHM
*******************************************/


// Reprojection of dual vector, for now to norm 1 (TODO: correct metrication)
__global__ void compute_simplex_fgp_dual_reprojection( int W, int H, int G, int N,
						       float *px, float *py )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;
  // Compute norm
  float n = 0.0;
  for ( int g=0; g<G; g++ ) {
    n += px[o]*px[o] + py[o]*py[o];
    o += N;
  }
  n = max( 1.0f, sqrtf(n) );
  // Reproject
  o = oy*W + ox;
  for ( int g=0; g<G; g++ ) {
    px[o] /= n;
    py[o] /= n;
    o += N;
  }
}


// Update one layer of the fields for fgp relaxation
__global__ void update_simplex_relaxation_device( int W, int H,
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
bool coco::potts_multilabel_simplex_update_overrelaxation( potts_multilabel_simplex_data *data )
{
  size_t W = data->_W;
  size_t H = data->_H;
  potts_multilabel_simplex_workspace *w = data->_potts_w;

  // Kernel call for each layer
  for ( size_t g=0; g<data->_G; g++ ) {
    size_t offset_primal = multilabel_index( data, 0,0,g );
    update_simplex_relaxation_device<<< w->_dimGrid, w->_dimBlock >>>
      ( W, H,
	1.0f,
	w->_uq + offset_primal,
	w->_u + offset_primal );
  }
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // Switch fields
  swap( w->_u, w->_uq );
  return true;
}





/******************************************
    RELAXATION ENERGIES FOR POTTS MODEL
*******************************************/

static __global__ void potts_multilabel_accumulate_dataterm_energy_device( int W, int H,
									   float *a,
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
  e[o] += u[o] * a[o];
}



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
  e[o] += lambda * hypotf( ex, ey );
}


double coco::potts_multilabel_simplex_energy( potts_multilabel_simplex_data *data )
{
  size_t W = data->_W;
  size_t H = data->_H;
  size_t G = data->_G;
  potts_multilabel_simplex_workspace *w = data->_potts_w;

  // One helper array for reduce result
  float *e = NULL;
  CUDA_SAFE_CALL( cudaMalloc( &e, w->_nfbytes ));
  CUDA_SAFE_CALL( cudaMemset( e, 0, w->_nfbytes ));
  float *r = NULL;
  CUDA_SAFE_CALL( cudaMalloc( &r, w->_nfbytes ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // Accumulate dataterm energy
  for ( size_t g=0; g<G; g++ ) {
    float *a = w->_rho + multilabel_index( data, 0,0,g );
    float *u = w->_u + multilabel_index( data, 0,0,g );
    potts_multilabel_accumulate_dataterm_energy_device<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H, 
	a,u, e );
  }
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // Reduce
  float result_e_data;
  cuda_sum_reduce( W,H, e, r, &result_e_data );

  // Second step: regularizerenergy
  CUDA_SAFE_CALL( cudaMemset( e, 0, w->_nfbytes ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // Accumulate regularizer energy
  for ( size_t g=0; g<G; g++ ) {
    float *u = w->_u + multilabel_index( data, 0,0,g );
    potts_multilabel_accumulate_regularizer_energy_device<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H, 
	data->_lambda,
	u, e );
  }
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // Reduce and half (edges counted twice)
  float result_e_reg;
  cuda_sum_reduce( W,H, e, r, &result_e_reg );
  result_e_reg /= 2.0f;

  // Cleanup
  TRACE( "Potts relaxation energy: " << endl );
  TRACE( "  regularizer: " << result_e_reg << endl );
  TRACE( "  data term  : " << result_e_data << endl );
  TRACE( "  total      : " << result_e_data + result_e_reg << endl );

  CUDA_SAFE_CALL( cudaFree( e ));
  CUDA_SAFE_CALL( cudaFree( r ));
  return result_e_data + result_e_reg;
}


/******************************************
    SPECIAL CASE POTTS MODEL
*******************************************/

// Perform one iteration of the FISTA scheme
bool coco::potts_multilabel_simplex_iteration( potts_multilabel_simplex_data *data )
{
  // Compute dual prox operator
  profiler()->beginTask( "dual prox potts" );
  potts_multilabel_simplex_dual_prox( data );
  profiler()->endTask( "dual prox potts" );

  // Compute primal prox operator
  profiler()->beginTask( "primal prox potts" );
  potts_multilabel_simplex_primal_prox( data );
  profiler()->endTask( "primal prox potts" );

  // Update overrelaxation scheme
  potts_multilabel_simplex_update_overrelaxation( data );

  // Finalize
  data->_potts_w->_simplex_iter ++;
  return true;
}


// Perform dual ascent primal step with metrication matrix A
// Called for each label layer
__global__ void compute_simplex_primal_prox_potts_device( int W, int H,
						 	  float tau_p,
						 	  float *u,
							  float *a,
							  float *sigma,
							  float *px, float *py,
							  float *u_prox )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;

  // Step equals divergence of p, backward differences, dirichlet
  float div = px[o] + py[o];
  if ( ox>0 ) {
    div -= px[o-1];
  }
  if ( oy>0 ) {
    div -= py[o-W];
  }    

  // Multiplied with 2 to compensate for duplicate edge count
  div -= 2.0f * a[o];
  div += sigma[o];

  // Projection onto allowed range
  u_prox[o] = max( 0.0f, min( 1.0f, u[o] + tau_p * div ));
  //u_prox[o] = u[o] + tau_p * div;
}


// Perform one primal step
bool coco::potts_multilabel_simplex_primal_prox( potts_multilabel_simplex_data *data )
{
  size_t W = data->_W;
  size_t H = data->_H;
  potts_multilabel_simplex_workspace *w = data->_potts_w;

  // Kernel call for each layer
  for ( size_t g=0; g<data->_G; g++ ) {
    size_t offset_primal = multilabel_index( data, 0,0, g );
    size_t offset_dual = g*W*H;
    compute_simplex_primal_prox_potts_device<<< w->_dimGrid, w->_dimBlock >>>
      ( W, H,
	w->_tau_u,
	w->_u + offset_primal,
	w->_rho + offset_primal,
	w->_sigma,
	w->_x1 + offset_dual, w->_x2 + offset_dual,
	w->_uq + offset_primal );
  }
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  //simplex_reprojection( W,H, data->_G, w->_uq );
  return true;
}


__global__ void compute_simplex_dual_prox_potts_device( int W, int H,
							float lambda,
							float tau_d,
							float *u,
							float *px, float *py )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>= H ) {
    return;
  }
  int o = oy*W + ox;

  // Step for each p equals gradient component of phi
  // Forward differences, Neumann
  // X
  float gradX = 0.0f;
  if ( ox < W-1 ) {
    gradX = u[o+1] - u[o];
  }
  // Y
  float gradY = 0.0f;
  if ( oy < H-1 ) {
    gradY = u[o+W] - u[o];
  }
  
  // Ascent step
  float new_px = px[o] + tau_d * gradX;
  float new_py = py[o] + tau_d * gradY;

  // Reprojection is combined for all channels
  float L = hypotf( new_px, new_py );
  if ( L>lambda ) {
    new_px = lambda * new_px / L;
    new_py = lambda * new_py / L;
  }
  px[o] = new_px;
  py[o] = new_py;
}

__global__ void update_simplex_multilabel_sigma_device( int W, int H, int N, int G,
							float sigma_s,
							float *u,
							float *sigma )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>= H ) {
    return;
  }
  int o = ox + oy*W;

  float sum = -1.0f;
  for ( int g=0; g<G; g++ ) {
    sum += u[o + g*N];
  }
  sigma[o] -= sigma_s * sum;
}



// Perform one dual step
bool coco::potts_multilabel_simplex_dual_prox( potts_multilabel_simplex_data *data )
{
  size_t W = data->_W;
  size_t H = data->_H;
  potts_multilabel_simplex_workspace *w = data->_potts_w;

  // Kernel call for each layer
  for ( size_t r=0; r<data->_G; r++ ) {
    size_t offset_primal = multilabel_index( data, 0,0,r );
    size_t offset_dual = r * W * H;
    compute_simplex_dual_prox_potts_device<<< w->_dimGrid, w->_dimBlock >>>
      ( W, H, 
	data->_lambda,
	w->_sigma_p,
	w->_uq + offset_primal,
	w->_x1 + offset_dual, w->_x2 + offset_dual );
  }


  // Update Lagrange multipliers for Simplex constraint
  update_simplex_multilabel_sigma_device<<< w->_dimGrid, w->_dimBlock >>>
    ( W, H, W*H, data->_G,
      w->_sigma_s,
      w->_uq,
      w->_sigma );

  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}
