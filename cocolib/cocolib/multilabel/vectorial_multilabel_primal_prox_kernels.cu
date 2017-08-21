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


// Perform dual ascent primal step with metrication matrix A
// Called for each label layer
__global__ void compute_primal_prox_vml_device( int W, int H,
						float tau_u,
						float *u,
						float *q,
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

  // Add step from data term
  div -= q[o];
  div += sigma[o];

  // Projection is done later
  u_prox[o] = u[o] + tau_u * div;
}


// Perform dual ascent primal step with metrication matrix A
// Called for each label layer
__global__ void primal_prox_project_vml_device( int W, int H,
						float *u )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;
  u[o] = max( 0.0f, min( 1.0f, u[o] ));
}


// Perform primal ascent + dual descent for the data term relaxation
// Version for ICCV, two label space dimensions
// Called for each label layer
__global__ void update_mu_dim2_vml_device( int W, int H,
					   float tau_mu,
					   float *q1, float *q2, float *rho,
					   float *mu )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;

  // Perform update step and extragradient step for mu
  float qsum = q1[o] + q2[o] - rho[o];
  mu[o] += qsum * tau_mu;
}


// Perform primal ascent + dual descent for the data term relaxation
// Version for ICCV, three label space dimensions
// Called for each label layer
__global__ void update_mu_dim3_vml_device( int W, int H,
					   float tau_mu,
					   float *q1, float *q2, float *q3,
					   float *rho,
					   float *mu )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;

  // Perform update step and extragradient step for mu
  float qsum = q1[o] + q2[o] + q3[o] - rho[o];
  mu[o] += qsum * tau_mu;
}



// Update one layer of the fields for fgp relaxation (no swap required)
__global__ void update_overrelaxation_mu_device( int W, int H,
						 float theta,
						 float *mu_prox, float *mu )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;
  float mup = max( 0.0f, mu_prox[o] );
  float muv = mu[o];
  mu[o] = mup;
  mu_prox[o] = mup + theta * ( mup - muv );
}


// Perform primal ascent + dual descent for the data term relaxation
// Version for ICCV, two label space dimensions
// Called for each label layer
__global__ void update_mu_q_dim2_vml_device( int W, int H,
					     float tau_p,
					     float sigma_q1,
					     float sigma_q2,
					     float theta,
					     float *q1, float *q2,
					     float *rho,
					     float *mu,
					     float *q1_new, float *q2_new )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;

  // Perform update step and extragradient step for mu
  float qsum = q1[o] + q2[o] - rho[o];
  float mu_new = max( 0.0f, mu[o] + qsum * tau_p );
  float mu_extra = mu_new + theta * (mu_new - mu[o]);

  // Perform dual update step for q with extragradient
  q1_new[o] -= sigma_q1 * mu_extra;
  q2_new[o] -= sigma_q2 * mu_extra;

  // Save new mu
  mu[o] = mu_new;
}


// Perform primal ascent + dual descent for the data term relaxation
// Version for ICCV, three label space dimensions
// Called for each label layer
__global__ void update_mu_q_dim3_vml_device( int W, int H,
					     float tau_p, 
					     float sigma_q1,
					     float sigma_q2,
					     float sigma_q3,
					     float theta,
					     float *q1, float *q2, float *q3,
					     float *rho,
					     float *mu,
					     float *q1_new, float *q2_new, float *q3_new )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;

  // Perform update step and extragradient step for mu
  float qsum = q1[o] + q2[o] + q3[o] - rho[o];
  float mu_new = max( 0.0f, mu[o] + qsum * tau_p );
  float mu_extra = mu_new + theta * (mu_new - mu[o]);

  // Perform dual update step for q with extragradient
  q1_new[o] -= sigma_q1 * mu_extra;
  q2_new[o] -= sigma_q2 * mu_extra;
  q3_new[o] -= sigma_q3 * mu_extra;

  // Save new mu
  mu[o] = mu_new;
}


// Perform primal ascent + dual descent for the data term relaxation
// Version for ICCV, three label space dimensions
// Called for each label layer
__global__ void update_mu_q_dim3_vml_segmentation_device( int W, int H,
							  float tau_p, 
							  float sigma_q1,
							  float sigma_q2,
							  float sigma_q3,
							  float theta,
							  float *q1, float *q2, float *q3,
							  float label_r, float label_g, float label_b,
							  float *Ir, float *Ig, float *Ib,
							  float *mu,
							  float *q1_new, float *q2_new, float *q3_new )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;

  // Compute data term
  float rho_r = ( Ir[o] - label_r );
  float rho_g = ( Ig[o] - label_g );
  float rho_b = ( Ib[o] - label_b );
  // Perform update step and extragradient step for mu
  float qsum = q1[o] + q2[o] + q3[o] - (rho_r*rho_r + rho_g*rho_g + rho_b*rho_b );
  float mu_new = max( 0.0f, mu[o] + qsum * tau_p );
  float mu_extra = mu_new + theta * (mu_new - mu[o]);

  // Perform dual update step for q with extragradient
  q1_new[o] -= sigma_q1 * mu_extra;
  q2_new[o] -= sigma_q2 * mu_extra;
  q3_new[o] -= sigma_q3 * mu_extra;

  // Save new mu
  mu[o] = mu_new;
}



// Perform primal ascent + dual descent for the data term relaxation
// Version for ICCV, two label space dimensions
// Called for each label layer
__global__ void update_mu_q_dim2_vml_chunk_device( int W, int W_mu, int W_rho,
						   int H,
						   int chunk_offset,
						   float tau_p,
						   float sigma_q1,
						   float sigma_q2,
						   float theta,
						   float *q1, float *q2,
						   float *rho,
						   float *mu,
						   float *q1_new, float *q2_new )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox + chunk_offset>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox + chunk_offset;

  // Perform update step and extragradient step for mu
  float qsum = q1[o] + q2[o] - rho[oy*W_rho + ox];
  float muv = mu[oy*W_mu + ox];
  float mu_new = max( 0.0f, muv + qsum * tau_p );
  float mu_extra = mu_new + theta * (mu_new - muv);

  // Perform dual update step for q with extragradient
  q1_new[o] -= sigma_q1 * mu_extra;
  q2_new[o] -= sigma_q2 * mu_extra;

  // Save new mu
  mu[oy*W_mu+ox] = mu_new;
}


// Perform primal ascent + dual descent for the data term relaxation
// Version for ICCV, three label space dimensions
// Called for each label layer
__global__ void update_mu_q_dim3_vml_chunk_device( int W, int W_mu, int W_rho,
						   int H,
						   int chunk_offset,
						   float tau_p, 
						   float sigma_q1,
						   float sigma_q2,
						   float sigma_q3,
						   float theta,
						   float *q1, float *q2, float *q3,
						   float *rho,
						   float *mu,
						   float *q1_new, float *q2_new, float *q3_new )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox + chunk_offset>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox + chunk_offset;

  // Perform update step and extragradient step for mu
  float qsum = q1[o] + q2[o] + q3[o] - rho[oy*W_rho + ox];
  float muv = mu[oy*W_mu + ox];
  float mu_new = max( 0.0f, muv + qsum * tau_p );
  float mu_extra = mu_new + theta * (mu_new - muv);

  // Perform dual update step for q with extragradient
  q1_new[o] -= sigma_q1 * mu_extra;
  q2_new[o] -= sigma_q2 * mu_extra;
  q3_new[o] -= sigma_q3 * mu_extra;

  // Save new mu
  mu[oy*W_mu+ox] = mu_new;
}



// Perform primal ascent + dual descent for the data term relaxation
// Version for ICCV, three label space dimensions
// Called for each label layer
__global__ void update_mu_q_dim3_vml_chunk_segmentation_device( int W, int W_mu, 
								int H,
								int chunk_offset,
								float tau_p, 
								float sigma_q1,
								float sigma_q2,
								float sigma_q3,
								float theta,
								float *q1, float *q2, float *q3,
								float label_r, float label_g, float label_b,
								float *Ir, float *Ig, float *Ib,
								float *mu,
								float *q1_new, float *q2_new, float *q3_new )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox + chunk_offset>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox + chunk_offset;

  // Compute data term
  float rho_r = Ir[o] - label_r;
  float rho_g = Ig[o] - label_g;
  float rho_b = Ib[o] - label_b;
  // Perform update step and extragradient step for mu
  float qsum = q1[o] + q2[o] + q3[o] - (rho_r*rho_r + rho_g*rho_g + rho_b*rho_b );
  float muv = mu[oy*W_mu + ox];
  float mu_new = max( 0.0f, muv + qsum * tau_p );
  float mu_extra = mu_new + theta * (mu_new - muv);

  // Perform dual update step for q with extragradient
  q1_new[o] -= sigma_q1 * mu_extra;
  q2_new[o] -= sigma_q2 * mu_extra;
  q3_new[o] -= sigma_q3 * mu_extra;

  // Save new mu
  mu[oy*W_mu+ox] = mu_new;
}




// Perform primal ascent + dual descent for the data term relaxation
// Version for ICCV, three label space dimensions
// Called for each label layer
__global__ void update_eta_vml_device( int W, int H,
				       float cost,
				       float tau_eta, 
				       float sigma_p,
				       float theta,
				       float *eta_x, float *eta_y,
				       float *px1, float *py1,
				       float *px2, float *py2,
				       float *px1_new, float *py1_new,
				       float *px2_new, float *py2_new )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;

  // Perform update step for eta
  float eta_xo = eta_x[o];
  float eta_yo = eta_y[o];
  float eta_xv = eta_xo - tau_eta * (px1[o] - px2[o]);
  float eta_yv = eta_yo - tau_eta * (py1[o] - py2[o]);

  // Prox operator
  float n = hypotf( eta_xv, eta_yv );
  float alpha = n - cost * tau_eta;
  if ( alpha <= 0.0f ) {
    eta_xv = 0.0f;
    eta_yv = 0.0f;
  }
  else {
    eta_xv = alpha * eta_xv / n;
    eta_yv = alpha * eta_yv / n;
  }

  // Overrelaxation
  float eta_xq = eta_xv + theta * (eta_xv - eta_xo);
  float eta_yq = eta_yv + theta * (eta_yv - eta_yo);

  // Perform dual update step for p with extragradient
  px1_new[o] += sigma_p * eta_xq;
  py1_new[o] += sigma_p * eta_yq;

  px2_new[o] -= sigma_p * eta_xq;
  py2_new[o] -= sigma_p * eta_yq;

  // Save new eta
  eta_x[o] = eta_xv;
  eta_y[o] = eta_yv;
}
