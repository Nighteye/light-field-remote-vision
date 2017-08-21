/* -*-c++-*- */
#include "tv_multilabel.h"
#include "tv_multilabel.cuh"

#include "cuda.h"

#include "../defs.h"
#include "../cuda/cuda_helper.h"
#include "../cuda/cuda_kernels.cuh"



// Alloc multilabel problem structure
coco::tv_multilabel_data* coco::tv_multilabel_data_alloc( size_t W, size_t H, size_t L )
{
  assert( L > 1 );
  tv_multilabel_data *data = new tv_multilabel_data;
  multilabel_data_init( data, W,H,L );

  // workspace
  tv_multilabel_workspace *w = new tv_multilabel_workspace;
  tv_multilabel_workspace_init( data, w );
  
  TRACE( "  lambda (coco) = " << data->_lambda << std::endl );
  return data;
}


// Init/free
bool coco::tv_multilabel_workspace_init( tv_multilabel_data *data, tv_multilabel_workspace *w )
{
  multilabel_workspace_init( data, w );

  // Overrelaxation factor
  w->_theta = 1.0f;

  // Chosen according to heuristics in paper
  // Primal step size
  w->_tau_p = 1.0 / sqrt(12.0);
  // Dual step size
  w->_tau_d = 1.0 / sqrt(12.0);

  // Size of 3D fields over Image x Label space
  w->_nfbytes = data->_N * data->_G * sizeof(float);

  // Dual variables
  CUDA_SAFE_CALL( cudaMalloc( &w->_p1, w->_nfbytes ));
  CUDA_SAFE_CALL( cudaMalloc( &w->_p2, w->_nfbytes ));
  CUDA_SAFE_CALL( cudaMalloc( &w->_p3, w->_nfbytes ));

  // Setup
  data->_tv_w = w;
  return true;
}



// Free multilabel problem structure
bool coco::tv_multilabel_data_free( tv_multilabel_data* data )
{
  tv_multilabel_workspace_free( data, data->_tv_w );
  multilabel_data_free( data );
  delete data->_tv_w;
  delete data;
  return true;
}


// Free multilabel problem structure
bool coco::tv_multilabel_workspace_free( tv_multilabel_data* data, tv_multilabel_workspace *w )
{
  multilabel_workspace_free( data, w );
  CUDA_SAFE_CALL( cudaFree( w->_p1 ));
  CUDA_SAFE_CALL( cudaFree( w->_p2 ));
  CUDA_SAFE_CALL( cudaFree( w->_p3 ));
  return true;
}



// Set current solution data
bool coco::tv_multilabel_set_solution( tv_multilabel_data* data,
				       const gsl_matrix *u )
{
  return multilabel_set_solution_lifting( data, u );
}


// Get current solution
bool coco::tv_multilabel_get_solution( tv_multilabel_data *data,
				       gsl_matrix *u,
				       const float threshold )
{
  return multilabel_get_solution_lifting( data, u, threshold );
}


// Set current solution data
bool coco::tv_multilabel_set_solution( tv_multilabel_data* data,
				       const float *u )
{
  return multilabel_set_solution_lifting( data, u );
}


// Get current solution
bool coco::tv_multilabel_get_solution( tv_multilabel_data *data,
				       float *u,
				       const float threshold )
{
  return multilabel_get_solution_lifting( data, u, threshold );
}



// Compute current energy
double coco::tv_multilabel_energy( tv_multilabel_data *data )
{
  // TODO
  return 0.0;
}



// Set dataterm weight
/*
bool coco::tv_multilabel_set_dataterm_weight( tv_multilabel_data *data,
					      gsl_matrix *g )
{
  tv_multilabel_workspace *w = data->_tv_w;
  if ( w->_g != NULL ) {
    CUDA_SAFE_CALL( cudaFree( w->_g ));
    w->_g = NULL;
  }
  if ( g == NULL ) {
    return true;
  }

  int W = data->_W;
  int H = data->_H;
  CUDA_SAFE_CALL( cudaMalloc( &w->_g, W*H*sizeof(float) ));
  assert( (int)g->size2 == W );
  assert( (int)g->size1 == H );
  cuda_memcpy( w->_g, g );
  return true;
}
*/




/************************************************************
  SOLVER
*************************************************************/


// Init step sizes
bool coco::tv_multilabel_init( tv_multilabel_data* data )
{
  tv_multilabel_workspace *w = data->_tv_w;
  CUDA_SAFE_CALL( cudaMemcpy( w->_uq, w->_u, w->_nfbytes, cudaMemcpyDeviceToDevice ));
  CUDA_SAFE_CALL( cudaMemset( w->_p1, 0, w->_nfbytes ));
  CUDA_SAFE_CALL( cudaMemset( w->_p2, 0, w->_nfbytes ));
  CUDA_SAFE_CALL( cudaMemset( w->_p3, 0, w->_nfbytes ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}




__global__ void tv_multilabel_compute_lead_device( int W, int H, int N, int g,
						   float theta,
						   float *phi,
						   float *phi_q )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;
  o += g*N;
  float pq = phi_q[o];
  phi[o] = pq + theta * (pq - phi[o]);
}

// Compute leading (overshooting) term
bool coco::tv_multilabel_compute_lead( tv_multilabel_data *data )
{
  // Kernel call
  tv_multilabel_workspace *w = data->_tv_w;
  for ( size_t g=0; g<data->_G; g++ ) {
    tv_multilabel_compute_lead_device<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H, data->_N, g, 
	w->_theta,
	w->_u,
	w->_uq );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
  }

  std::swap( w->_u, w->_uq );
  return true;
}




__global__ void tv_multilabel_primal_step_device( int W, int H, int N, int g, int G,
						  float tau_p,
						  float *phi,
						  float *phi_q,
						  float *px, float *py, float *pz )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;
  o += g*N;

  // Step equals divergence of p, backward differences, dirichlet
  float step = px[o] + py[o] + pz[o];
  if ( ox>0 ) {
    step -= px[o-1];
  }
  if ( oy>0 ) {
    step -= py[o-W];
  }
  if ( g>0 ) {
    step -= pz[o-N];
  }
  // Projection onto allowed range
  float new_phi = phi[o] + tau_p * step;
  if ( new_phi > 1.0f ) {
    new_phi = 1.0f;
  }
  else if ( new_phi < 0.0f ) {
    new_phi = 0.0f;
  }
  // TODO: Optimize
  if ( g==0 ) {
    new_phi = 1;
  }
  phi_q[o] = new_phi;
}


// Perform one primal step
bool coco::tv_multilabel_primal_step( tv_multilabel_data *data )
{
  // Kernel call
  tv_multilabel_workspace *w = data->_tv_w;
  for ( size_t g=0; g<data->_G; g++ ) {
    tv_multilabel_primal_step_device<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H, data->_N, g, data->_G, w->_tau_p,
	w->_u, w->_uq,
	w->_p1,	w->_p2,	w->_p3 );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
  }
  return true;
}


__global__ void tv_multilabel_dual_step_device( int W, int H, int N, int g, int G,
						float tau_d,
						float rho_scale,
						float *rho,
						float *phi,
						float *px, float *py, float *pz )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;

  // Step for each p equals gradient component of phi
  // Forward differences, Neumann
  // X
  float grad = 0.0f;
  if ( ox < W-1 ) {
    grad = phi[o+1] - phi[o];
  }
  px[o] += tau_d * grad;
  // Y
  grad = 0.0f;
  if ( oy < H-1 ) {
    grad = phi[o+W] - phi[o];
  }
  py[o] += tau_d * grad;

  // Z
  grad = 0.0f;
  if ( g==G-1 ) {
    // Last layer is zero
    grad = - phi[o];
  }
  else {
    grad = phi[o+N] - phi[o];
  }

  float pz_new = pz[o] + tau_d * grad;
  // Reproject Z (according to new journal paper)
  float r = rho_scale * rho[o];
  if ( pz_new < -r ) {
    pz_new = -r;
  }
  pz[o] = pz_new;
}



// Perform one dual step
bool coco::tv_multilabel_dual_step( tv_multilabel_data *data )
{
  // Kernel call
  tv_multilabel_workspace *w = data->_tv_w;
  size_t W = data->_W;
  size_t H = data->_H;
  size_t N = data->_N;
  size_t G = data->_G;
  for ( size_t g=0; g<data->_G; g++ ) {
    size_t offset = g * data->_N;
    tv_multilabel_dual_step_device<<< w->_dimGrid, w->_dimBlock >>>
      ( W,H,N, g, G, w->_tau_d,
	0.5f, // / max( data->_lambda, 1e-6 ),
	w->_rho + offset,
	w->_uq + offset,
	w->_p1 + offset, w->_p2 + offset, w->_p3 + offset );

    // Reprojection of px, py
    if ( data->_nx.size() != 0 ) {
      assert( data->_nx.size() == data->_G );
      assert( data->_ny.size() == data->_G );

      // TEST ANISOTROPY
      if ( w->_g == NULL ) {
	cuda_reproject_to_ellipse<<< w->_dimGrid, w->_dimBlock >>>
	  ( W,H, data->_nx[g], data->_ny[g], 0.05, w->_p1 + offset, w->_p2 + offset );
      }
      else {
	cuda_reproject_to_ellipse<<< w->_dimGrid, w->_dimBlock >>>
	  ( W,H, data->_nx[g], data->_ny[g], 0.05, w->_g, w->_p1 + offset, w->_p2 + offset );
      }
    }
    else  if ( w->_g == NULL ) {
      /*
      float r = 1.0f;
      if ( data->_lambda == 0.0 ) {
	r = 0.0f;
      }
      */
      float r = data->_lambda;
      cuda_reproject_to_ball_2d<<< w->_dimGrid, w->_dimBlock >>>
	( W,H, r, w->_p1 + offset, w->_p2 + offset );
    }
    else {
      cuda_reproject_to_ball_2d<<< w->_dimGrid, w->_dimBlock >>>
	( W,H, w->_g, w->_p1 + offset, w->_p2 + offset );
    }
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
  }
  return true;
}


// Init dual variables (corresponds to a dual step with size 1)
bool coco::tv_multilabel_dual_init( tv_multilabel_data *data )
{
  // Clear dual variables
  tv_multilabel_workspace *w = data->_tv_w;
  CUDA_SAFE_CALL( cudaMemset( w->_p1, 0, w->_nfbytes ));
  CUDA_SAFE_CALL( cudaMemset( w->_p2, 0, w->_nfbytes ));
  CUDA_SAFE_CALL( cudaMemset( w->_p3, 0, w->_nfbytes ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // Kernel call
  for ( size_t g=0; g<data->_G; g++ ) {
    tv_multilabel_dual_step_device<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H, data->_N, g, data->_G,
	1.0, // step size 1 to initialize with derivative values directly
	1.0f / max( data->_lambda, 1e-6 ),
	w->_rho,
	w->_uq,
	w->_p1,
	w->_p2,
	w->_p3 );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
  }
  return true;
}


// Perform one iteration
bool coco::tv_multilabel_iteration( tv_multilabel_data *data )
{
  bool ok2 = tv_multilabel_dual_step( data );
  bool ok1 = tv_multilabel_primal_step( data );
  tv_multilabel_compute_lead( data );
  return ok1 && ok2;
}
