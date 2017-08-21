/* -*-c++-*- */
#include <iostream>

#include "vtv.h"
#include "vtv.cuh"

#include "../common/gsl_matrix_helper.h"
#include "../cuda/cuda_helper.h"
#include "../cuda/cuda_kernels.cuh"
#include "../defs.h"

// Perform one primal step
bool coco::coco_vtv_rof_primal_step( coco_vtv_data *data )
{
  coco_vtv_workspace *w = data->_workspace;

  // Kernel call for each channel
  for ( size_t i=0; i<data->_nchannels; i++ ) {
    cuda_rof_primal_prox_step_device<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H,
	data->_tau, data->_lambda,
	w->_U[i], w->_Uq[i], w->_F[i], w->_X1[i], w->_X2[i] );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
  }
  return true;
}

__global__ void coco_vtv_rof_overrelaxation_device( int W, 
						    int H,
						    cuflt theta,
						    cuflt* uq,
						    cuflt* u )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;
  cuflt uv = u[o];
  cuflt uqv = uq[o];
  uq[o] = uv + theta * ( uqv - uv );
  u[o] = uqv;
}

bool coco::coco_vtv_rof_overrelaxation( coco_vtv_data *data, cuflt theta )
{
  coco_vtv_workspace *w = data->_workspace;
  for ( size_t i=0; i<data->_nchannels; i++ ) {
    coco_vtv_rof_overrelaxation_device<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H, theta, w->_Uq[i], w->_U[i] );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
  }

  return true;
}


bool coco::coco_vtv_rof_fgp_overrelaxation( coco_vtv_data *data, cuflt theta )
{
  coco_vtv_workspace *w = data->_workspace;
  for ( size_t i=0; i<data->_nchannels; i++ ) {
    coco_vtv_rof_overrelaxation_device<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H, theta, w->_X1[i], w->_X1q[i] );
    coco_vtv_rof_overrelaxation_device<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H, theta, w->_X2[i], w->_X2q[i] );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
  }

  return true;
}



__global__ void coco_vtv_rof_primal_infinite_device( int W, int H,
						     cuflt lambda,
						     cuflt *u,
						     cuflt *f,
						     cuflt *px, cuflt *py )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;

  // Step equals divergence of p, backward differences, dirichlet
  cuflt step = px[o] + py[o];
  if ( ox>0 ) {
    step -= px[o-1];
  }
  if ( oy>0 ) {
    step -= py[o-W];
  }
  // Projecton onto allowed range
  u[o] = f[o] + lambda * step;
}

// Perform one primal step
bool coco::coco_vtv_rof_primal_infinite( coco_vtv_data *data )
{
  coco_vtv_workspace *w = data->_workspace;

  // Kernel call for each channel
  for ( size_t i=0; i<data->_nchannels; i++ ) {
    coco_vtv_rof_primal_infinite_device<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H, data->_lambda,
	w->_Uq[i], w->_F[i], w->_X1[i], w->_X2[i] );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
  }
  return true;
}


__global__ void coco_vtv_rof_dual_step_device( int W, int H, cuflt tstep,
					       cuflt *u,
					       cuflt *px, cuflt *py )
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
  cuflt grad = 0.0;
  if ( ox < W-1 ) {
    grad = u[o+1] - u[o];
  }
  cuflt px_new = px[o] + tstep * grad;
  // Y
  grad = 0.0;
  if ( oy < H-1 ) {
    grad = u[o+W] - u[o];
  }
  cuflt py_new = py[o] + tstep * grad;
  // Reprojection is combined for all channels
  px[o] = px_new;
  py[o] = py_new;
}


// Reprojection for RGB, TV_J
__global__ void coco_vtv_rof_reproject_3D_tvj_device( int W, int H,
						      //  cuflt tau, cuflt sigma,
						      // int C, cuflt *c,
						      cuflt *px1, cuflt *py1,
						      cuflt *px2, cuflt *py2,
						      cuflt *px3, cuflt *py3 )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>= H ) {
    return;
  }
  int o = oy*W + ox;

  // Local vars
  cuflt a11 = px1[o];
  cuflt a21 = px2[o];
  cuflt a31 = px3[o];
  cuflt a12 = py1[o];
  cuflt a22 = py2[o];
  cuflt a32 = py3[o];

  // Compute A^T A
  cuflt d11 = a11*a11 + a21*a21 + a31*a31;
  cuflt d12 = a12*a11 + a22*a21 + a32*a31;
  cuflt d22 = a12*a12 + a22*a22 + a32*a32;

  // Compute larger Eigenvalue (= square of largest singular value)
  cuflt trace = d11 + d22;
  cuflt det = d11*d22 - d12*d12;
  cuflt d = sqrt( 0.25*trace*trace - det );
  cuflt lmax = max( 0.0, 0.5 * trace + d );
  cuflt lmin = max( 0.0, 0.5 * trace - d );
  cuflt smax = sqrt( lmax );
  cuflt smin = sqrt( lmin );

  // If smax + smin > 1:
  // Project (smax,smin) to line (0,1) + tau * (1,-1), 0<=tau<=1.
  if ( smax + smin > 1.0 ) {

    cuflt v11, v12, v21, v22;
    if ( d12 == 0.0 ) {
      if ( d11 >= d22 ) {
	v11 = 1.0; v21 = 0.0; v12 = 0.0; v22 = 1.0;
      }
      else {
	v11 = 0.0; v21 = 1.0; v12 = 1.0; v22 = 0.0;
      }
    }
    else {
      v11 = lmax - d22; v21 = d12;
      cuflt l1 = hypotf( v11, v21 );
      v11 /= l1; v21 /= l1;
      v12 = lmin - d22; v22 = d12;
      cuflt l2 = hypot( v12, v22 );
      v12 /= l2; v22 /= l2;
    }

    // Compute projection of Eigenvalues
    cuflt tau = 0.5 * (smax - smin + 1.0);
    cuflt s1 = min( 1.0, tau );
    cuflt s2 = 1.0 - s1;
    // Compute \Sigma^{-1} * \Sigma_{new}
    s1 /= smax;
    s2 = (smin > 0.0) ? s2 / smin : 0.0;

    // A_P = A * \Sigma^{-1} * \Sigma_{new}
    cuflt t11 = s1*v11*v11 + s2*v12*v12;
    cuflt t12 = s1*v11*v21 + s2*v12*v22;
    cuflt t21 = s1*v21*v11 + s2*v22*v12;
    cuflt t22 = s1*v21*v21 + s2*v22*v22;

    // Result
    px1[o] = a11 * t11 + a12 * t21;
    px2[o] = a21 * t11 + a22 * t21;
    px3[o] = a31 * t11 + a32 * t21;
    
    py1[o] = a11 * t12 + a12 * t22;
    py2[o] = a21 * t12 + a22 * t22;
    py3[o] = a31 * t12 + a32 * t22;
  }
}


// Reprojection for RGB, TV_F
__global__ void coco_vtv_rof_reproject_3D_tvf_device( int W, int H,
						      //  cuflt tau, cuflt sigma,
						      // int C, cuflt *c,
						      cuflt *px1, cuflt *py1,
						      cuflt *px2, cuflt *py2,
						      cuflt *px3, cuflt *py3 )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>= H ) {
    return;
  }
  int o = oy*W + ox;

  // Local vars
  cuflt a11 = px1[o];
  cuflt a21 = px2[o];
  cuflt a31 = px3[o];
  cuflt a12 = py1[o];
  cuflt a22 = py2[o];
  cuflt a32 = py3[o];

  // Total norm
  cuflt n = 0.0;
  n += pow( a11, 2.0f ) + pow( a21, 2.0f ) + pow( a31, 2.0f );
  n += pow( a12, 2.0f ) + pow( a22, 2.0f ) + pow( a32, 2.0f );
  n = sqrt( n );

  // Project
  if ( n > 1.0 ) {
    px1[o] = a11 / n;
    py1[o] = a12 / n;
    px2[o] = a21 / n;
    py2[o] = a22 / n;
    px3[o] = a31 / n;
    py3[o] = a32 / n;
  }
}



// Reprojection for RGB, TV_F
__global__ void coco_vtv_rof_reproject_3D_tvf_weighted_device( int W, int H,
							       float *g,
							       //  cuflt tau, cuflt sigma,
							       // int C, cuflt *c,
							       cuflt *px1, cuflt *py1,
							       cuflt *px2, cuflt *py2,
							       cuflt *px3, cuflt *py3 )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>= H ) {
    return;
  }
  int o = oy*W + ox;

  // Local vars
  cuflt a11 = px1[o];
  cuflt a21 = px2[o];
  cuflt a31 = px3[o];
  cuflt a12 = py1[o];
  cuflt a22 = py2[o];
  cuflt a32 = py3[o];

  // Total norm
  cuflt n = 0.0;
  n += pow( a11, 2.0f ) + pow( a21, 2.0f ) + pow( a31, 2.0f );
  n += pow( a12, 2.0f ) + pow( a22, 2.0f ) + pow( a32, 2.0f );
  n = sqrt( n );

  // Project
  float r = g[o];
  if ( n > r ) {
    n = r/n;
    px1[o] = a11 * n;
    py1[o] = a12 * n;
    px2[o] = a21 * n;
    py2[o] = a22 * n;
    px3[o] = a31 * n;
    py3[o] = a32 * n;
  }
}



// Reprojection for RGB, TV_S
__global__ void coco_vtv_rof_reproject_3D_cbc_device( int W, int H,
								  //  cuflt tau, cuflt sigma,
								  // int C, cuflt *c,
								  cuflt *px1, cuflt *py1,
								  cuflt *px2, cuflt *py2,
								  cuflt *px3, cuflt *py3 )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>= H ) {
    return;
  }
  int o = oy*W + ox;

  // Local vars
  cuflt a11 = px1[o];
  cuflt a21 = px2[o];
  cuflt a31 = px3[o];
  cuflt a12 = py1[o];
  cuflt a22 = py2[o];
  cuflt a32 = py3[o];

  // Separate norms in each channel
  cuflt n1 = max( 1.0, hypot( a11, a12 ));
  cuflt n2 = max( 1.0, hypot( a21, a22 ));
  cuflt n3 = max( 1.0, hypot( a31, a32 ));

  // Project
  px1[o] = a11 / n1;
  py1[o] = a12 / n1;
  px2[o] = a21 / n2;
  py2[o] = a22 / n2;
  px3[o] = a31 / n3;
  py3[o] = a32 / n3;
}



// Reprojection for RGB, TV_S
__global__ void coco_vtv_rof_reproject_3D_cbc_weighted_device( int W, int H,
							       float *g1, float *g2, float *g3,
							       //  cuflt tau, cuflt sigma,
							       // int C, cuflt *c,
							       cuflt *px1, cuflt *py1,
							       cuflt *px2, cuflt *py2,
							       cuflt *px3, cuflt *py3 )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>= H ) {
    return;
  }
  int o = oy*W + ox;

  // Local vars
  cuflt a11 = px1[o];
  cuflt a21 = px2[o];
  cuflt a31 = px3[o];
  cuflt a12 = py1[o];
  cuflt a22 = py2[o];
  cuflt a32 = py3[o];

  // Separate norms in each channel
  cuflt n1 = max( 1.0, hypot( a11, a12 ));
  cuflt n2 = max( 1.0, hypot( a21, a22 ));
  cuflt n3 = max( 1.0, hypot( a31, a32 ));

  // Project
  float r1 = 1., r2 = 1., r3 = 1.;
  if (g1 != NULL) {
    r1 = g1[o];
    if (g2 == NULL || g3 == NULL ) {
      r2 = r1;
      r3 = r1;
    } else {
      r2 = g2[o];
      r3 = g3[o];
    }
  }

  if ( n1>r1 ) {
    n1 = r1/n1;
    px1[o] = a11 * n1;
    py1[o] = a12 * n1;
  }
  if ( n2>r2 ) {
    n2 = r2/n2;
    px2[o] = a21 * n2;
    py2[o] = a22 * n2;
  }
  if ( n3>r3 ) {
    n3 = r3/n3;
    px3[o] = a31 * n3;
    py3[o] = a32 * n3;
  }
}




// Perform one dual step
bool coco::coco_vtv_rof_dual_step( coco_vtv_data *data )
{
  coco_vtv_workspace *w = data->_workspace;

  for ( size_t i=0; i<data->_nchannels; i++ ) {
    coco_vtv_rof_dual_step_device<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H, data->_sigma,
      w->_Uq[i],
      w->_X1[i], w->_X2[i] );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
  }

  // Reprojection
  if ( data->_nchannels == 1 ) {
    if ( w->_g[0] == NULL ) {
      cuda_reproject_to_unit_ball_2d<<< w->_dimGrid, w->_dimBlock >>>
        ( data->_W, data->_H, w->_X1[0], w->_X2[0] );
    }
    else {
      cuda_reproject_to_ball_2d<<< w->_dimGrid, w->_dimBlock >>>
        ( data->_W, data->_H, w->_g[0], w->_X1[0], w->_X2[0] );
    }
  }
  else if ( data->_nchannels == 3 ) {
    switch ( data->_regularizer ) {
    case 0:
      {
        if ( w->_g[0] != NULL ) {
          coco_vtv_rof_reproject_3D_cbc_weighted_device<<< w->_dimGrid, w->_dimBlock >>>
	        ( data->_W, data->_H,
            w->_g[0], w->_g[1], w->_g[2],
            w->_X1[0], w->_X2[0],
            w->_X1[1], w->_X2[1],
            w->_X1[2], w->_X2[2] );
        }
        else {
          coco_vtv_rof_reproject_3D_cbc_device<<< w->_dimGrid, w->_dimBlock >>>
            ( data->_W, data->_H,
            w->_X1[0], w->_X2[0],
            w->_X1[1], w->_X2[1],
            w->_X1[2], w->_X2[2] );
        }
        CUDA_SAFE_CALL( cudaThreadSynchronize() );  
      }
      break;

    case 1:
      {
        if ( w->_g[0] != NULL ) {
          coco_vtv_rof_reproject_3D_tvf_weighted_device<<< w->_dimGrid, w->_dimBlock >>>
            ( data->_W, data->_H,
            w->_g[0],
            w->_X1[0], w->_X2[0],
            w->_X1[1], w->_X2[1],
            w->_X1[2], w->_X2[2] );
        }
        else {
          coco_vtv_rof_reproject_3D_tvf_device<<< w->_dimGrid, w->_dimBlock >>>
            ( data->_W, data->_H,
            w->_X1[0], w->_X2[0],
            w->_X1[1], w->_X2[1],
            w->_X1[2], w->_X2[2] );
        }
        CUDA_SAFE_CALL( cudaThreadSynchronize() );  
      }
      break;
      
    case 2:
      {
        if ( w->_g[0] != NULL ) {
          static bool msg = false;
          if ( !msg ) {
            msg = true;
            ERROR( "weighted TV not supported for TV_J" << std::endl );
          }
        }
        coco_vtv_rof_reproject_3D_tvj_device<<< w->_dimGrid, w->_dimBlock >>>
          ( data->_W, data->_H,
          w->_X1[0], w->_X2[0],
          w->_X1[1], w->_X2[1],
          w->_X1[2], w->_X2[2] );
        CUDA_SAFE_CALL( cudaThreadSynchronize() );  
      }
      break;
      
    default:
      {
        ERROR( "Unknown regularizer." << std::endl );
        assert( false );
      }
    }
  }
  else {
    ERROR( "Unsupported number of channels (only grayscale or RGB implemented at the moment)." << std::endl );
    assert( false );
  }

  return true;
}

bool coco::coco_vtv_rof_ista_step( coco_vtv_data *data )
{
  // Compute update for U: infinite step size
  coco_vtv_rof_primal_infinite( data );
  // Then dual step if step size 1/(8\lambda)
  data->_sigma = 1.0 / (data->_lambda * 8.0);
  coco_vtv_rof_dual_step( data );

  // New U is stored in Uq
  // Dual variables updated.

  // Copy result to U (lead computed only in Uq)
  coco_vtv_workspace *w = data->_workspace;
  for ( size_t i=0; i<data->_nchannels; i++ ) {
    CUDA_SAFE_CALL( cudaMemcpy( w->_U[i], w->_Uq[i], w->_nfbytes, cudaMemcpyDeviceToDevice ));
  }
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}



// Compute primal energy
double coco::coco_vtv_rof_primal_energy( coco_vtv_data *data )
{
  size_t W = data->_W;
  size_t H = data->_H;
  coco_vtv_workspace *w = data->_workspace;

  // Compute gradient of current solution
  for ( size_t i=0; i<data->_nchannels; i++ ) {
    cuda_compute_gradient_device<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H, w->_U[i], w->_X1t[i], w->_X2t[i] );
  }
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // Compute largest singular value of gradient matrix
  cuda_compute_largest_singular_value_device<<< w->_dimGrid, w->_dimBlock >>>
    ( data->_W, data->_H,
      w->_X1t[0], w->_X2t[0], w->_X1t[1], w->_X2t[1], w->_X1t[2], w->_X2t[2],
      w->_G[0] );

  // Compute gradient of data term
  for ( size_t i=0; i<data->_nchannels; i++ ) {
    // u
    CUDA_SAFE_CALL( cudaMemcpy( w->_temp[i], w->_Uq[i], w->_nfbytes, cudaMemcpyDeviceToDevice ));
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    // u-f
    cuda_subtract_from_device<<< w->_dimGrid, w->_dimBlock >>>
      ( W,H, w->_F[i], w->_temp[i] );
    // square
    cuda_square_device<<< w->_dimGrid, w->_dimBlock >>>
      ( W,H, w->_temp[i] );
    // 1/(2 lambda) * ...
    cuda_scale_device<<< w->_dimGrid, w->_dimBlock >>>
      ( W,H, w->_temp[i], 1.0 / ( 2.0 * data->_lambda ));

    // Add to smoothness term
    cuda_add_to_device<<< w->_dimGrid, w->_dimBlock >>>
      ( W,H, w->_temp[i], w->_G[0] );
  }
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // Copy temp array to CPU
  cuflt *E = new cuflt[ W*H ];
  CUDA_SAFE_CALL( cudaMemcpy( E, w->_G[0], w->_nfbytes, cudaMemcpyDeviceToHost ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  // Compute total energy
  double e = 0.0;
  for ( size_t i=0; i<W*H; i++ ) {
    e += E[i];
  }
  delete[] E;
  return e / double(W*H);
}


// Perform one iteration of Bermudez-Morena scheme
bool coco::coco_vtv_rof_iteration_bermudez_morena( coco_vtv_data *data )
{
  data->_sigma = 1.0 / (4.0 * data->_lambda);
  bool ok2 = coco_vtv_rof_dual_step( data );
  bool ok1 = coco_vtv_rof_primal_infinite( data );
  bool ok3 = coco_vtv_rof_overrelaxation( data, 0.0 );
  return ok1 && ok2 && ok3;
}

// Perform one iteration of Arrow-Hurwicz scheme
bool coco::coco_vtv_rof_iteration_arrow_hurwicz( coco_vtv_data *data )
{
  bool ok2 = coco_vtv_rof_dual_step( data );
  bool ok1 = coco_vtv_rof_primal_step( data );
  bool ok3 = coco_vtv_rof_overrelaxation( data, 0.0 );
  return ok1 && ok2 && ok3;
}

// Perform one iteration of Algorithm 1, Chambolle-Pock
bool coco::coco_vtv_rof_iteration_chambolle_pock_1( coco_vtv_data *data )
{
  bool ok2 = coco_vtv_rof_dual_step( data );
  bool ok1 = coco_vtv_rof_primal_step( data );
  bool ok3 = coco_vtv_rof_overrelaxation( data, 1.0 );
  return ok1 && ok2 && ok3;
}

// Perform one iteration of Algorithm 2, Chambolle-Pock
bool coco::coco_vtv_rof_iteration_chambolle_pock_2( coco_vtv_data *data )
{
  bool ok2 = coco_vtv_rof_dual_step( data );
  bool ok1 = coco_vtv_rof_primal_step( data );

  data->_gamma = 1.0 / data->_lambda;
  cuflt theta = 1.0 / sqrt( 1.0 + 2.0 * data->_gamma * data->_tau );
  data->_tau = data->_tau * theta;
  data->_sigma = data->_sigma / theta;

  bool ok3 = coco_vtv_rof_overrelaxation( data, theta );
  return ok1 && ok2 && ok3;
}


// Perform one iteration of Algorithm 1, Chambolle-Pock
bool coco::coco_vtv_rof_iteration_fista( coco_vtv_data *data )
{
  // Todo: verify correct maximum step sizes.
  data->_tau = 0.3 / sqrt( 8.0 );
  data->_sigma = 0.3 / sqrt( 8.0 );
  data->_L = 1.0 / data->_lambda;
  bool ok2 = coco_vtv_rof_ista_step( data );
  cuflt alpha_new = 0.5 * ( 1.0 + sqrt( 1.0 + 4.0 * pow( data->_alpha, 2.0 ) ));
  bool ok3 = coco_vtv_rof_fgp_overrelaxation( data, ( data->_alpha - 1.0 ) / alpha_new );
  data->_alpha = alpha_new;
  return ok2 && ok3;
}



