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


__global__ void compute_dual_prox_vml_potts_device( int W, int H,
						    float lambda,
						    float sigma_p,
						    float sigma_q,
						    float *u,
						    float *px, float *py,
						    float *q )
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
  float uv = u[o];
  float gradX = 0.0f;
  if ( ox < W-1 ) {
    gradX = u[o+1] - uv;
  }
  // Y
  float gradY = 0.0f;
  if ( oy < H-1 ) {
    gradY = u[o+W] - uv;
  }
  
  // Ascent step
  float new_px = px[o] + sigma_p * gradX;
  float new_py = py[o] + sigma_p * gradY;

  // Reprojection is combined for all channels
  float L = hypotf( new_px, new_py );
  if ( L>lambda ) {
    new_px = lambda * new_px / L;
    new_py = lambda * new_py / L;
  }
  px[o] = new_px;
  py[o] = new_py;

  // Ascent for q
  q[o] += sigma_q * uv;
}



// Standard dual prox: no projection in p (taken care of by Lagrange multipliers eta)
__global__ void compute_dual_prox_vml_device( int W, int H,
					      float sigma_p,
					      float sigma_q,
					      float *u,
					      float *px, float *py,
					      float *q )
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
  float uv = u[o];
  float gradX = 0.0f;
  if ( ox < W-1 ) {
    gradX = u[o+1] - uv;
  }
  // Y
  float gradY = 0.0f;
  if ( oy < H-1 ) {
    gradY = u[o+W] - uv;
  }
  
  // Ascent step
  px[o] += sigma_p * gradX;
  py[o] += sigma_p * gradY;

  // Ascent for q
  q[o] += sigma_q * uv;
}





__global__ void update_sigma_vml_device( int W, int H, int N, int G,
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
