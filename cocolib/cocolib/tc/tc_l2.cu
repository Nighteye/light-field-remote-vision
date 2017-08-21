/* -*-c++-*- */
/** \file tc_l2.cu
   Algorithms to solve the TC model with L2 data term.

   Copyright (C) 2010 Bastian Goldluecke,
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

#include <iostream>

#include "tc.h"
#include "tc.cuh"
#include "tc_arrays.cuh"
#include "tc_energies.cuh"
#include "tc_l2.h"

#include "../defs.h"
#include "../cuda/cuda_helper.h"
#include "../cuda/cuda_kernels.cuh"
#include "../common/gsl_image.h"
#include "../common/gsl_matrix_derivatives.h"
#include "../common/menger_curvature.h"
#include "../common/profiler.h"


/******************************************
    EXPERIMENTAL MINIMIZATION ALGORITHM
    see initial version paper ICCV 2011
*******************************************/

/*************************************************
    MAIN ITERATION: Chambolle Pock
    based on sum of dual variables formulation
**************************************************/

// Initialize TC-ROF model
bool coco::tc_l2_init( tc_data *data )
{
  tc_workspace *w = data->_workspace;

  w->_lambda = data->_lambda;
  cv_init_rof_stepsize( data, data->_lambda );
 
  return true;
}


// Initialize TC-ROF model
bool coco::cv_init_rof_stepsize( tc_data *data, double lambda )
{
  tc_workspace *w = data->_workspace;
  w->_rof_lambda = lambda;
  w->_tv_lambda = data->_tv_lambda;

  double Ks_norm = 3.0 * pow( w->_N, 2.0 ) * sqrt( 8.0 );
  w->_sigma = 2.0 / (w->_lambda * Ks_norm * Ks_norm);
  w->_tv_sigma = ( w->_tv_lambda != 0.0 ) ? 1.0 / (4.0 * w->_tv_lambda) : 0.0;

  return true;
}


// Perform one ROF iteration
bool coco::cv_rof_iteration( tc_data *data )
{
  return tc_l2_iteration( data );
}




// Perform one iteration (outer loop)
bool coco::tc_l2_iteration( tc_data *data )
{
  if (!tc_l2_dual_prox( data )) {
    ERROR( "Dual step failure." << std::endl );
    return false;
  }

  if (!tc_l2_primal_prox( data )) {
    ERROR( "Primal step failure." << std::endl );
    return false;
  }

  if (!tc_l2_overrelaxation( data )) {
    ERROR( "Overrelaxation step failure." << std::endl );
    return false;
  }

  return true;
}

bool coco::tc_l2_dual_prox( tc_data *data )
{
  profiler()->beginTask( "tcrof_dual_prox" );

  // Compute scaled gradient field
  tc_workspace *w = data->_workspace;
  cuda_compute_gradient_device<<< w->_dimGrid, w->_dimBlock >>>
    ( data->_W, data->_H, w->_uq, w->_px, w->_py );

  if ( w->_tv_sigma != 0.0 ) {
    cuda_add_scaled_to_device<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H, w->_px, w->_tv_sigma, w->_pxq );
    cuda_add_scaled_to_device<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H, w->_py, w->_tv_sigma, w->_pyq );
    cuda_reproject_to_unit_ball_2d<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H, w->_pxq, w->_pyq );
  }

  cuda_scale_device<<< w->_dimGrid, w->_dimBlock >>>
    ( data->_W, data->_H, w->_px, w->_sigma );
  cuda_scale_device<<< w->_dimGrid, w->_dimBlock >>>
    ( data->_W, data->_H, w->_py, w->_sigma );


  // Test flag to toggle curvature computations off
  if ( w->_curvature ) {

    // VX
    int N2 = w->_N2;
    for ( int y0=-N2; y0<=N2; y0++ ) {
      for ( int y1=-N2; y1<=N2; y1++ ) {
	for ( int z0=-N2; z0<=N2; z0++ ) {
	  for ( int z1=-N2; z1<=N2; z1++ ) {
	    stcflt *px1 = NULL;
	    stcflt *px2 = NULL;
	    tc_get_dual_x( data, px1, px2, y0,y1, z0,z1 );
	    cuda_add_to_device<<< w->_dimGrid, w->_dimBlock >>>
	      ( data->_W, data->_H, w->_px, px1 );
	    cuda_add_to_device<<< w->_dimGrid, w->_dimBlock >>>
	      ( data->_W, data->_H, w->_py, px2 );
	  }
	}
      }
    }
    // VY
    for ( int x0=-N2; x0<=N2; x0++ ) {
      for ( int x1=-N2; x1<=N2; x1++ ) {
	for ( int z0=-N2; z0<=N2; z0++ ) {
	  for ( int z1=-N2; z1<=N2; z1++ ) {
	    stcflt *py1 = NULL;
	    stcflt *py2 = NULL;
	    tc_get_dual_y( data, py1, py2, x0,x1, z0,z1 );
	    cuda_add_to_device<<< w->_dimGrid, w->_dimBlock >>>
	      ( data->_W, data->_H, w->_px, py1 );
	    cuda_add_to_device<<< w->_dimGrid, w->_dimBlock >>>
	      ( data->_W, data->_H, w->_py, py2 );
	  }
	}
      }
    }
    // VZ
    for ( int x0=-N2; x0<=N2; x0++ ) {
      for ( int x1=-N2; x1<=N2; x1++ ) {
	for ( int y0=-N2; y0<=N2; y0++ ) {
	  for ( int y1=-N2; y1<=N2; y1++ ) {
	    stcflt *pz1 = NULL;
	    stcflt *pz2 = NULL;
	    tc_get_dual_z( data, pz1, pz2, x0,x1, y0,y1 );
	    cuda_add_to_device<<< w->_dimGrid, w->_dimBlock >>>
	      ( data->_W, data->_H, w->_px, pz1 );
	    cuda_add_to_device<<< w->_dimGrid, w->_dimBlock >>>
	      ( data->_W, data->_H, w->_py, pz2 );
	  }
	}
      }
    }
    
    // Lambdas
    int N = w->_N;
    stcflt lambda_step = -data->_alpha * w->_sigma;
    for ( int x0=-N2; x0<=N2; x0++ ) {
      for ( int x1=-N2; x1<=N2; x1++ ) {
	for ( int y0=-N2; y0<=N2; y0++ ) {
	  for ( int y1=-N2; y1<=N2; y1++ ) {
	    stcflt *lambda = NULL;
	    stcflt cp = w->_cp_cpu[ AIND( x0,x1, y0,y1 ) ];
	    lambda = cvl_get_3d_variable( w, w->_p[LAMBDA_X], x0,x1,y0,y1 );
	    cuda_add_scalar_to_device<<< w->_dimGrid, w->_dimBlock >>>
	      ( data->_W, data->_H, lambda_step * cp, lambda );
	    lambda = cvl_get_3d_variable( w, w->_p[LAMBDA_Y], x0,x1,y0,y1 );
	    cuda_add_scalar_to_device<<< w->_dimGrid, w->_dimBlock >>>
	      ( data->_W, data->_H, lambda_step * cp, lambda );
	    lambda = cvl_get_3d_variable( w, w->_p[LAMBDA_Z], x0,x1,y0,y1 );
	    cuda_add_scalar_to_device<<< w->_dimGrid, w->_dimBlock >>>
	      ( data->_W, data->_H, lambda_step * cp, lambda );
	  }
	}
      }
    }
  }

  profiler()->endTask( "tcrof_dual_prox" );
  return tc_l2_dual_reprojection( data );
}



__global__ void tc_l2_dual_reprojection_lambda_device( int W, int H, int N, int N2,
							int y0o, int y1o, int z0o, int z1o,
							stcflt *cp,
							stcflt *px1b, stcflt *px2b,
							stcflt *py1b, stcflt *py2b,
							stcflt *pz1b, stcflt *pz2b,
							stcflt *lx, stcflt *ly, stcflt *lz )
{
  // Global thread index
  int x0 = blockDim.x * blockIdx.x + threadIdx.x - 2*N2;
  int x1 = blockDim.y * blockIdx.y + threadIdx.y - 2*N2;
  if ( x0>=W || x1>=H ) {
    return;
  }
  // Get correct offsets
  bool xvalid = true;
  bool yvalid = true;
  bool zvalid = true;
  // Base array index
  int y0 = x0 + y0o;
  int z0 = x0 + z0o;
  int y1 = x1 + y1o;
  int z1 = x1 + z1o;
  if ( x0<0 || x0>=W ) xvalid = false;
  if ( x1<0 || x1>=H ) xvalid = false;
  if ( y0<0 || y0>=W ) yvalid = false;
  if ( y1<0 || y1>=H ) yvalid = false;
  if ( z0<0 || z0>=W ) zvalid = false;
  if ( z1<0 || z1>=H ) zvalid = false;
  // Offsets
  int xo = x0 + W*x1;
  int yo = y0 + W*y1;
  int zo = z0 + W*z1;
  // Offset index X array
  int xy0 = y0-x0 + N2; int xy1 = y1-x1 + N2;
  int xz0 = z0-x0 + N2; int xz1 = z1-x1 + N2;
  if ( xy0<0 || xy0>2*N2 ) xvalid = false;
  if ( xy1<0 || xy1>2*N2 ) xvalid = false;
  if ( xz0<0 || xz0>2*N2 ) xvalid = false;
  if ( xz1<0 || xz1>2*N2 ) xvalid = false;
  // Offset index Y array
  int yx0 = x0-y0 + N2; int yx1 = x1-y1 + N2;
  int yz0 = z0-y0 + N2; int yz1 = z1-y1 + N2;
  if ( yx0<0 || yx0>2*N2 ) yvalid = false;
  if ( yx1<0 || yx1>2*N2 ) yvalid = false;
  if ( yz0<0 || yz0>2*N2 ) yvalid = false;
  if ( yz1<0 || yz1>2*N2 ) yvalid = false;
  // Offset index Z array
  int zx0 = x0-z0; int zx1 = x1-z1;
  int zy0 = y0-z0; int zy1 = y1-z1;
  if ( zx0<0 || zx0>2*N2 ) zvalid = false;
  if ( zx1<0 || zx1>2*N2 ) zvalid = false;
  if ( zy0<0 || zy0>2*N2 ) zvalid = false;
  if ( zy1<0 || zy1>2*N2 ) zvalid = false;

  // Get vector values
  stcflt *px1a = NULL;
  stcflt *px2a = NULL;
  stcflt px1 = 0.0;
  stcflt px2 = 0.0;
  if ( xvalid ) {
    px1a = px1b + AOFF( xy0,xy1, xz0,xz1 );
    px2a = px2b + AOFF( xy0,xy1, xz0,xz1 );
    px1 = px1a[ xo ];
    px2 = px2a[ xo ];
  }

  stcflt *py1a = NULL;
  stcflt *py2a = NULL;
  stcflt py1 = 0.0;
  stcflt py2 = 0.0;
  if ( yvalid ) {
    py1a = py1b + AOFF( yx0,yx1, yz0,yz1 );
    py2a = py2b + AOFF( yx0,yx1, yz0,yz1 );
    py1 = py1a[ yo ];
    py2 = py2a[ yo ];
  }

  stcflt *pz1a = NULL;
  stcflt *pz2a = NULL;
  stcflt pz1 = 0.0;
  stcflt pz2 = 0.0;
  if ( zvalid ) {
    pz1a = pz1b + AOFF( zx0,zx1, zy0,zy1 );
    pz2a = pz2b + AOFF( zx0,zx1, zy0,zy1 );
    pz1 = pz1a[ zo ];
    pz2 = pz2a[ zo ];
  }

  // Finally: compute projection
  stcflt nx = hypot( px1, px2 );
  stcflt ny = hypot( py1, py2 );
  stcflt nz = hypot( pz1, pz2 );
  stcflt c = 0.0;
  stcflt *lp = NULL;
  stcflt lambda = 0.0;
  if ( xvalid ) {
    c = cp[ AIND( xy0, xy1, xz0, xz1 ) ];
    lp = lx + AOFF( xy0,xy1, xz0,xz1 );
    lambda = lp[ xo ];
  }
  else if (yvalid) {
    c = cp[ AIND( yx0, yx1, yz0, yz1 ) ];
    lp = ly + AOFF( yx0,yx1, yz0,yz1 );
    lambda = lp[ yo ];
  }
  else if (zvalid) {
    c = cp[ AIND( zx0, zx1, zy0, zy1 ) ];
    lp = lz + AOFF( zx0,zx1, zy0,zy1 );
    lambda = lp[ zo ];
  }
  c = c / 3.0;

  // Perform backprojection
  stcflt m = (nx + ny + nz) / 3.0;
  if ( lambda < m ) lambda = (lambda + m) / 2.0;
  lambda = max( 0.0, min( c, lambda ));
  if ( nx > lambda ) {
    px1 *= lambda / nx;
    px2 *= lambda / nx;
  }
  if ( ny > lambda ) {
    py1 *= lambda / ny;
    py2 *= lambda / ny;
  }
  if ( nz > lambda ) {
    pz1 *= lambda / nz;
    pz2 *= lambda / nz;
  }

  // Write back projected vector values
  if ( xvalid ) {
    px1a[ xo ] = px1;
    px2a[ xo ] = px2;
  }
  if ( yvalid ) {
    py1a[ yo ] = py1;
    py2a[ yo ] = py2;
  }
  if ( zvalid ) {
    pz1a[ zo ] = pz1;
    pz2a[ zo ] = pz2;
  }
  if ( xvalid ) {
    lp[xo] = lambda;
  }
  else if ( yvalid ) {
    lp[yo] = lambda;
  }
  else if ( zvalid ) {
    lp[zo] = lambda;
  }
}



bool coco::tc_l2_dual_reprojection( tc_data *data )
{
  tc_workspace *w = data->_workspace;
  if ( !w->_curvature ) {
    return true;
  }

  profiler()->beginTask( "tcrof_dual_reprojection" );

  int W = w->_W;
  int H = w->_H;
  int N2 = w->_N2;

  dim3 dimBlock( cuda_default_block_size_x(),
		 cuda_default_block_size_y() );
  dim3 dimGrid( (W+4*N2) / dimBlock.x + 1, (H+4*N2) / w->_dimBlock.y + 1);

  // Iterate over Omega \times W
  for ( int y0o=-2*N2; y0o<=2*N2; y0o++ ) {
    for ( int y1o=-2*N2; y1o<=2*N2; y1o++ ) {
      for ( int z0o=-2*N2; z0o<=2*N2; z0o++ ) {
	for ( int z1o=-2*N2; z1o<=2*N2; z1o++ ) {

	  tc_l2_dual_reprojection_lambda_device<<< dimGrid, dimBlock >>>
	    ( W, H, w->_N, N2,
	      y0o, y1o, z0o, z1o,
	      w->_cp,
	      w->_p[PX1][0], w->_p[PX2][0],
	      w->_p[PY1][0], w->_p[PY2][0],
	      w->_p[PZ1][0], w->_p[PZ2][0],
	      w->_p[LAMBDA_X][0], w->_p[LAMBDA_Y][0], w->_p[LAMBDA_Z][0] );

	}
      }
    }
  }

  profiler()->endTask( "tcrof_dual_reprojection" );
  return true;
}




__global__ void tc_l2_primal_descent_device( int W, int H,
					     stcflt *u,
					     stcflt *px, stcflt *py )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  int o = oy*W + ox;

  // Step equals divergence of p, backward differences, dirichlet
  stcflt step = px[o] + py[o];
  if ( ox>0 ) {
    step -= px[o-1];
  }
  if ( oy>0 ) {
    step -= py[o-W];
  }
  // Projection onto allowed range
  u[o] += step;
}

static __global__ void tc_l2_primal_prox_device( int W, int H,
						  stcflt lambda,
						  stcflt lambda_tv, stcflt *px, stcflt *py,
						  stcflt *f,
						  stcflt *u,
						  stcflt *D,
						  stcflt *uq )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;

  // TV step equals divergence of p, backward differences, dirichlet
  stcflt step = px[o] + py[o];
  if ( ox>0 ) {
    step -= px[o-1];
  }
  if ( oy>0 ) {
    step -= py[o-W];
  }

  // Projection onto allowed range
  /* BM */
  stcflt unew = f[o] + lambda * D[o] + lambda_tv * step;
  uq[o] = max( 0.0, min( 1.0, unew ));
}


bool coco::tc_get_dual_x( tc_data *data,
			  stcflt* &p1, stcflt* &p2,
			  int y0, int y1, int z0, int z1 )
{
  tc_workspace *w = data->_workspace;
  p1 = cvl_get_3d_variable( w, w->_p[PX1], y0,y1, z0,z1 );
  p2 = cvl_get_3d_variable( w, w->_p[PX2], y0,y1, z0,z1 );
  return true;
}

bool coco::tc_get_dual_y( tc_data *data,
			  stcflt* &p1, stcflt* &p2,
			  int x0, int x1, int z0, int z1 )
{
  tc_workspace *w = data->_workspace;
  p1 = cvl_get_3d_variable( w, w->_p[PY1], x0,x1, z0,z1 );
  p2 = cvl_get_3d_variable( w, w->_p[PY2], x0,x1, z0,z1 );
  return true;
}

bool coco::tc_get_dual_z( tc_data *data,
			  stcflt* &p1, stcflt* &p2,
			  int x0, int x1, int y0, int y1 )
{
  tc_workspace *w = data->_workspace;
  p1 = cvl_get_3d_variable( w, w->_p[PZ1], x0,x1, y0,y1 );
  p2 = cvl_get_3d_variable( w, w->_p[PZ2], x0,x1, y0,y1 );
  return true;
}


bool coco::tc_l2_primal_prox( tc_data *data )
{
  profiler()->beginTask( "tcrof_primal_prox" );

  // Clear uq for accumulation of derivative
  tc_workspace *w = data->_workspace;
  CUDA_SAFE_CALL( cudaMemset( w->_D, 0, w->_Nf ));
  int N2 = w->_N2;

  if ( w->_curvature ) {
    // Primal step for dual prox
    // VX
    for ( int y0=-N2; y0<=N2; y0++ ) {
      for ( int y1=-N2; y1<=N2; y1++ ) {
	for ( int z0=-N2; z0<=N2; z0++ ) {
	  for ( int z1=-N2; z1<=N2; z1++ ) {
	    stcflt *px1 = NULL;
	    stcflt *px2 = NULL;
	    tc_get_dual_x( data, px1, px2, y0,y1, z0,z1 );
	    tc_l2_primal_descent_device<<< w->_dimGrid, w->_dimBlock >>>
	      ( data->_W, data->_H,
		w->_D, px1, px2 );
	  }
	}
      }
    }
    // VY
    for ( int x0=-N2; x0<=N2; x0++ ) {
      for ( int x1=-N2; x1<=N2; x1++ ) {
	for ( int z0=-N2; z0<=N2; z0++ ) {
	  for ( int z1=-N2; z1<=N2; z1++ ) {
	    stcflt *py1 = NULL;
	    stcflt *py2 = NULL;
	    tc_get_dual_y( data, py1, py2, x0,x1, z0,z1 );
	    tc_l2_primal_descent_device<<< w->_dimGrid, w->_dimBlock >>>
	      ( data->_W, data->_H,
		w->_D, py1, py2 );
	  }
	}
      }
    }
    // VZ
    for ( int x0=-N2; x0<=N2; x0++ ) {
      for ( int x1=-N2; x1<=N2; x1++ ) {
	for ( int y0=-N2; y0<=N2; y0++ ) {
	  for ( int y1=-N2; y1<=N2; y1++ ) {
	    stcflt *pz1 = NULL;
	    stcflt *pz2 = NULL;
	    tc_get_dual_z( data, pz1, pz2, x0,x1, y0,y1 );
	    tc_l2_primal_descent_device<<< w->_dimGrid, w->_dimBlock >>>
	      ( data->_W, data->_H,
		w->_D, pz1, pz2 );
	  }
	}
      }
    }
  }

  // ROF proximal step to update u
   tc_l2_primal_prox_device<<< w->_dimGrid, w->_dimBlock >>>
     ( data->_W, data->_H,
      w->_rof_lambda,
      w->_tv_lambda, w->_pxq, w->_pyq,
      w->_f, w->_u, w->_D, w->_uq );

  profiler()->endTask( "tcrof_primal_prox" );
  return true;
}


bool coco::tc_l2_overrelaxation( tc_data *data )
{
  tc_workspace *w = data->_workspace;
  CUDA_SAFE_CALL( cudaMemcpy( w->_u, w->_uq, w->_Nf, cudaMemcpyDeviceToDevice ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}
