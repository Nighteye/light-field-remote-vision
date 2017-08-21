/* -*-c++-*- */
/** \file curvature_linear_energies.cu
   Algorithms to solve the curvature model with linear data term.

   Energy computations

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

#include "../defs.h"
#include "../cuda/cuda_helper.h"
#include "../common/gsl_image.h"
#include "../common/gsl_matrix_derivatives.h"
#include "../common/menger_curvature.h"
#include "../common/profiler.h"


/******************************************************
ENERGY COMPUTATIONS
******************************************************/

// Energy of the curvature term with dual variables
// Uses given 6D function
double coco::compute_primal_rof_energy( coco::tc_data *data )
{
  int W = (int)data->_W;
  int H = (int)data->_H;
  coco::tc_workspace *w = data->_workspace;

  // Other vars
  stcflt *v = new stcflt[ W*H ];
  stcflt *f = new stcflt[ W*H ];
  CUDA_SAFE_CALL( cudaMemcpy( f, w->_f, w->_Nf, cudaMemcpyDeviceToHost ));

  // Energy
  double energy = 0.0; //compute_primal_curvature_energy_v( data, w->_vq );
  energy /= w->_energy_scale_v;

  // Loop over all y,z
  /*
  int N2 = w->_N2;
  for ( int y0=-N2; y0<=N2; y0++ ) {
    for ( int y1=-N2; y1<=N2; y1++ ) {
      for ( int z0=-N2; z0<=N2; z0++ ) {
	for ( int z1=-N2; z1<=N2; z1++ ) {

	  // Copy final derivative back to CPU
	  stcflt *v_yz = cvl_get_3d_variable( w, w->_vq, y0,y1, z0,z1 );
	  CUDA_SAFE_CALL( cudaMemcpy( v, v_yz, w->_Nf, cudaMemcpyDeviceToHost ));
	  CUDA_SAFE_CALL( cudaThreadSynchronize() );

	  // Accumulate energy
	  size_t i=0;
	  double energy_xy = 0.0;
	  for ( int x1=0; x1<H; x1++ ) {
	    for ( int x0=0; x0<W; x0++ ) {

	      stcflt yf = 0.0;
	      int y0c = y0 + x0;
	      int y1c = y1 + x1;
	      if ( y0c>=0 && y0c<W && y1c>=0 && y1c<H ) {
		yf = f[ y1c*W + y0c ];
	      }
	      stcflt zf = 0.0;
	      int z0c = z0 + x0;
	      int z1c = z1 + x1;
	      if ( z0c>=0 && z0c<W && z1c>=0 && z1c<H ) {
		zf = f[ z1c*W + z0c ];
	      }
	      
	      stcflt vf = f[ x1*W + x0 ] * yf * zf;
	      stcflt p = fabs( v[i] - vf );
	      energy_xy += pow( p, 2.0 ) / ( 2.0 * w->_rof_lambda );
	      i++;
	    }
	  }	  

	  // Weight is included in dual variables.
	  energy += energy_xy;
	  
	}
      }
    }
  }
  */

  // Cleanup
  delete[] v;
  delete[] f;

  // Show totals
  energy *= w->_energy_scale_v;
  return energy;
}



// Energy of the curvature term with dual variables
// Uses given 6D function
double coco::compute_primal_inpainting_energy( coco::tc_data *data )
{
  int W = (int)data->_W;
  int H = (int)data->_H;
  coco::tc_workspace *w = data->_workspace;

  // Other vars
  stcflt *u = new stcflt[ W*H ];
  CUDA_SAFE_CALL( cudaMemcpy( u, w->_u, w->_Nf, cudaMemcpyDeviceToHost ));
  stcflt *f = new stcflt[ W*H ];
  CUDA_SAFE_CALL( cudaMemcpy( f, w->_a, w->_Nf, cudaMemcpyDeviceToHost ));
  stcflt *m = new stcflt[ W*H ];
  CUDA_SAFE_CALL( cudaMemcpy( m, w->_mask, w->_Nf, cudaMemcpyDeviceToHost ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // Energy
  double energy = 0.0; //compute_primal_curvature_energy_v( data, w->_vq );
  energy /= w->_energy_scale_v;
  TRACE( "Curvature energy: " << energy << std::endl );

  // Accumulate energy
  size_t i=0;
  double energy_xy = 0.0;
  for ( int x1=0; x1<H; x1++ ) {
    for ( int x0=0; x0<W; x0++ ) {
      
      stcflt vf = pow( (double)fabs( u[i] - f[i] ), 2.0 );
      stcflt alpha = 0.0;
      if ( m[i] != 0.0 ) {
	alpha = 1.0 / (2.0 * w->_lambda);
      }
      energy_xy += alpha * vf;
      i++;
    }
  }
  //energy += w->_energy_scale_u * energy_xy;
  energy += energy_xy;

  // Cleanup
  delete[] u;
  delete[] f;
  delete[] m;

  // Show totals
  return energy;
}



// Energy of the curvature term with dual variables
// Uses given 6D function

// Energy of the tv term, primal variable only
double coco::compute_primal_tv_energy_u( coco::tc_data *data, stcflt *gpu_u )
{
  int W = (int)data->_W;
  int H = (int)data->_H;
  coco::tc_workspace *w = data->_workspace;

  // Alloc local vars
  // Primal variables
  stcflt *u = new stcflt[ W*H ];
  CUDA_SAFE_CALL( cudaMemcpy( u, gpu_u, w->_Nf, cudaMemcpyDeviceToHost ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  // Copy to matrix
  bool threshold = false;
  double threshold_value = 0.5;
  gsl_matrix *U = gsl_matrix_alloc( H,W );
  for ( int i=0; i<H*W; i++ ) {
    double uv = u[i];
    if ( threshold ) {
      uv = (uv>threshold_value) ? 1.0 : 0.0;
    }
    U->data[i] = uv;
  }

  // Copy gradient of U
  gsl_matrix *UX = gsl_matrix_alloc( H,W );
  gsl_matrix_dx_forward( 1.0, U, UX );
  gsl_matrix *UY = gsl_matrix_alloc( H,W );
  gsl_matrix_dy_forward( 1.0, U, UY );

  // Compute TV term
  double energy_tv = 0.0;
  for ( int x0=0; x0<W; x0++ ) {
    for ( int x1=0; x1<H; x1++ ) {
      double tv = hypot( gsl_matrix_get( UX, x1,x0 ), gsl_matrix_get( UY, x1,x0 ));
      energy_tv += data->_tv_lambda * tv;
    }
  }
  return energy_tv / w->_energy_scale_u;
}




