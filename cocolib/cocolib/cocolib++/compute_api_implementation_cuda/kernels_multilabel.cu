/* -*-c++-*- */
/** \file kernels_multilabel.cu

    MULTILABEL kernels on grids

    Copyright (C) 2011-2014 Bastian Goldluecke,
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

#include "../compute_api/kernels_multilabel.h"
#include "compute_api_implementation_cuda.h"


static __global__ void kernel_assign_optimum_label( int W, int H, int N, int G,
						    const float *rho, float *u )
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

  u[o+uopt*N] = 1.0f;
}


void coco::kernel_assign_optimum_label( const compute_grid *G,
					int N,
					const compute_buffer &a,
					compute_buffer &u )
{
  dim3 dimGrid, dimBlock;
  kernel_configure( G, dimGrid, dimBlock );
  ::kernel_assign_optimum_label<<< dimGrid, dimBlock >>>
      ( G->W(), G->H(), G->W() * G->H(),
	N, a, u );
}


