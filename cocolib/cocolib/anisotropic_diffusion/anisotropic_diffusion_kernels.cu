/* -*-c++-*- */
/** \file anisotropic_diffusion_kernels.cu
    Perona-Malik isotropic and Weickert's coherence-enhancing diffusion,
    different discretizations,
    inpainting models.

    CUDA kernels.

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

/*********************************************************************
 ** DIFFUSION TENSORS
 *********************************************************************/

static __global__ void diffusion_tensor_coherence_enhancing( int W, int H,
							     float mu1, float c2,
							     float *a, float *b, float *c )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;

  // Get structure tensor matrix
  cuflt d11 = a[o];
  cuflt d12 = b[o];
  cuflt d22 = c[o];

  // Compute Eigenvalues
  cuflt trace = d11 + d22;
  cuflt det = d11*d22 - d12*d12;
  cuflt d = sqrtf( 0.25f*trace*trace - det );
  cuflt lmax = max( 0.0f, 0.5f * trace + d );
  cuflt lmin = max( 0.0f, 0.5f * trace - d );
  // Compute system of Eigenvectors
  cuflt v11, v12, v21, v22;
  if ( d12 == 0.0f ) {
    if ( d11 >= d22 ) {
      v11 = 1.0f; v21 = 0.0f; v12 = 0.0f; v22 = 1.0f;
      }
    else {
      v11 = 0.0f; v21 = 1.0f; v12 = 1.0f; v22 = 0.0f;
    }
  }
  else {
    v11 = lmax - d22; v21 = d12;
    cuflt l1 = hypotf( v11, v21 );
    v11 /= l1; v21 /= l1;
    v12 = lmin - d22; v22 = d12;
    cuflt l2 = hypotf( v12, v22 );
    v12 /= l2; v22 /= l2;
  }

  // Compute new Eigenvalues of diffusion tensor
  float mu2 = mu1;
  if ( lmax != lmin ) {
    float z = c2 / (lmax - lmin);
    mu2 += ( 1.0f - mu1 ) * expf( -z*z );
  }

  // Assemble diffusion tensor
  // TEST: Diagonal isotropic, should be equivalent to identity matrix
  /*
  mu1 = 1.0f;
  mu2 = 1.0f;
  v11 = 1.0f / sqrtf( 2.0f );
  v21 = 1.0f / sqrtf( 2.0f );
  v12= -1.0f / sqrtf( 2.0f );
  v22=  1.0f / sqrtf( 2.0f );
  */
  a[o] = mu1 * v11*v11 + mu2 * v21*v21;
  b[o] = (mu1-mu2) * v11*v21;
  c[o] = mu1 * v21*v21 + mu2 * v11*v11;

  // TEST: Isotropic, identity
  /*
  a[o] = 1.0f;
  b[o] = 0.0f;
  c[o] = 1.0f;
  */
}




static __global__ void diffusion_tensor_perona_malik( int W, int H,
						      float K_sq,
						      float *a, float *b, float *c )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;

  // Get structure tensor matrix
  float diff_c = exp( - ( a[o] + c[o] ) / K_sq ); 

  a[o] = diff_c;
  b[o] = 0.0f;
  c[o] = diff_c;
}




/*********************************************************************
 ** ANISOTROPIC DIFFUSION KERNELS
 *********************************************************************/

static __global__ void compute_anisotropic_diffusion_flux_field( int W, int H,
								 float *a, float *b, float *c,
								 float *ux, float *uy,
								 float *jx, float *jy )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;

  // Get structure tensor matrix
  float uxv = ux[o];
  float uyv = uy[o];
  float bv = b[o];
  jx[o] = a[o] * uxv + bv * uyv;
  jy[o] = bv * uxv + c[o] * uyv;
}


static __global__ void compute_anisotropic_diffusion_update( int W, int H,
							     float tau,
							     float *ux, float *uy,
							     float xi,
							     float *filter_x, float *filter_y,
							     float *u )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;

  // Update u
  u[o] += tau * ( ux[o] + uy[o] ) + xi * ( filter_x[o] + filter_y[o] );
}


static __global__ void identity_filter_x_5( int W, int H,
					    float *u, float *r )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;

  // Get values
  float u3 = u[o];
  float u1, u2, u4, u5;
  if ( ox>0 )
    u2 = u[o-1];
  else 
    u2 = u3;
  if ( ox>1 )
    u1 = u[o-2];
  else
    u1 = u2;
  
  if ( ox<W-1 )
    u4 = u[o+1];
  else
    u4 = u3;
  if ( ox<W-2 )
    u5 = u[o+2];
  else
    u5 = u4;
  
  r[o] = (-u1 + 4.0f * u2 + 10.0f * u3 + 4.0f * u4 - u5) / 16.0f - u3;
}


static __global__ void identity_filter_y_5( int W, int H,
					    float *u, float *r )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;

  // Get values
  float u3 = u[o];
  float u1, u2, u4, u5;
  if ( oy>0 )
    u2 = u[o-W];
  else 
    u2 = u3;
  if ( oy>1 )
    u1 = u[o-2*W];
  else
    u1 = u2;
  
  if ( oy<H-1 )
    u4 = u[o+W];
  else
    u4 = u3;
  if ( oy<H-2 )
    u5 = u[o+2*W];
  else
    u5 = u4;
  
  r[o] = (-u1 + 4.0f * u2 + 10.0f * u3 + 4.0f * u4 - u5) / 16.0f - u3;
}





// Load values for a 3x3 kernel from an array, Neumann boundary conditions
inline __device__ void load_values( int W, int H, float *a, int o, int ox, int oy,
				    float &u11, float &u12, float &u13,
				    float &u21, float &u22, float &u23,
				    float &u31, float &u32, float &u33 )
{
  u22 = a[o];
  if ( ox>0 ) {
    u21 = a[o-1];
    if ( oy>0 ) {
      u11 = a[o-1-W];
      u12 = a[o-W];
    }
    else {
      u11 = u21;
      u12 = u22;
    }
    if ( oy<H-1 ) {
      u31 = a[o-1+W];
      u32 = a[o+W];
    }
    else {
      u31 = u21;
      u32 = u22;
    }
  }
  else {
    u21 = u22;
    if ( oy>0 ) {
      u12 = a[o-W];
    }
    else {
      u12 = u22;
    }
    u11 = u12;

    if ( oy<H-1 ) {
      u32 = a[o+W];
    }
    else {
      u32 = u22;
    }
    u31 = u32;
  }
  if ( ox<W-1 ) {
    u23 = a[o+1];
    if ( oy>0 )
      u13 = a[o-W+1];
    else
      u13 = u23;
    if ( oy<H-1 )
      u33 = a[o+W+1];
    else 
      u33 = u23;
  }
  else {
    u23 = u22;
    u13 = u12;
    u33 = u32;
  }
}




static __global__ void compute_anisotropic_diffusion_nonneg_update( int W, int H,
								    float tau,
								    float *a, float *b, float *c,
								    float *u )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;

  // Compute kernel values for non-negativity scheme
  float u11, u12, u13, u21, u22, u23, u31, u32, u33;
  // Load all "a" values
  load_values( W,H, a, o, ox, oy,
	       u11, u12, u13, u21, u22, u23, u31, u32, u33 );
  // all kernel elements which include a
  float k21 = 0.5f * (u21 + u22);
  float k23 = 0.5f * (u23 + u22);
  float k22 = -0.5f * (u21 + u22 + u22 + u23);

  // Load all "b" values
  load_values( W,H, b, o, ox, oy,
	       u11, u12, u13, u21, u22, u23, u31, u32, u33 );
  // process everything with b
  // corners
  float k11 = 0.25f * ( fabsf(u22) + u22 );
  float k33 = k11;
  k11 += 0.25f * ( fabsf(u11) + u11 );
  k33 += 0.25f * ( fabsf(u33) + u33 );
  // Paper (probably wrong)
  float k31 = 0.25f * ( fabsf(u22) - u22 );
  float k13 = k31;
  k31 += 0.25f * ( fabsf(u31) - u31 );
  k13 += 0.25f * ( fabsf(u13) - u13 );

  // edges
  float k12 = -0.5f * ( fabsf(u12) + fabs(u22) );
  float k32 = -0.5f * ( fabsf(u32) + fabs(u22) );
  k21 -= 0.5f * ( fabsf(u21) + fabs(u22) );
  k23 -= 0.5f * ( fabsf(u23) + fabs(u22) );
  // center
  k22 -= 0.25f * ( fabsf( u31 ) - u31 + fabsf( u33 ) + u33 );
  k22 -= 0.25f * ( fabsf( u11 ) + u11 + fabsf( u13 ) - u13 );
  k22 += 0.25f * ( fabsf( u21 ) + fabsf( u23 ) + fabsf( u12 ) + fabsf( u32 ) + 2.0f * fabsf( u22 ));

  // Load all "c" values
  load_values( W,H, c, o, ox, oy,
	       u11, u12, u13, u21, u22, u23, u31, u32, u33 );
  // process everything with "c"
  k12 += 0.5f * (u12 + u22);
  k32 += 0.5f * (u32 + u22);
  k22 -= 0.5f * (u12 + u22 + u22 + u32);

  // Load all "u" values
  load_values( W,H, u, o, ox, oy,
	       u11, u12, u13, u21, u22, u23, u31, u32, u33 );

  // 3x3 Convolution with Neumann boundary conditions
  u[o] = u22 + tau * ( k11 * u11 + k12 * u12 + k13 * u13 +
		       k21 * u21 + k22 * u22 + k23 * u23 +
		       k31 * u31 + k32 * u32 + k33 * u33 );
}



static __global__ void compute_anisotropic_diffusion_simple_update( int W, int H,
								    float tau,
								    float *a, float *b, float *c,
								    float *u )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;

  // Compute kernel values for non-negativity scheme
  float u11, u12, u13, u21, u22, u23, u31, u32, u33;
  // Load all "a" values
  load_values( W,H, a, o, ox, oy,
	       u11, u12, u13, u21, u22, u23, u31, u32, u33 );
  // all kernel elements which include a
  float k21 = 0.5f * (u21 + u22);
  float k23 = 0.5f * (u23 + u22);
  float k22 = -0.5f * (u21 + u22 + u22 + u23);

  // Load all "b" values
  load_values( W,H, b, o, ox, oy,
	       u11, u12, u13, u21, u22, u23, u31, u32, u33 );
  // process everything with b
  // corners
  float k11 = 0.25f * ( u21 + u12 );
  float k33 = 0.25f * ( u23 + u32 );
  float k31 = -0.25f * ( u21 + u32 );
  float k13 = -0.25f * ( u23 + u12 );
  
  // Load all "c" values
  load_values( W,H, c, o, ox, oy,
	       u11, u12, u13, u21, u22, u23, u31, u32, u33 );
  // process everything with "c"
  float k12 = 0.5f * (u12 + u22);
  float k32 = 0.5f * (u32 + u22);
  k22 -= 0.5f * (u12 + u22 + u22 + u32);

  // Load all "u" values
  load_values( W,H, u, o, ox, oy,
	       u11, u12, u13, u21, u22, u23, u31, u32, u33 );

  // 3x3 Convolution with Neumann boundary conditions
  u[o] = u22 + tau * ( k11 * u11 + k12 * u12 + k13 * u13 +
		       k21 * u21 + k22 * u22 + k23 * u23 +
		       k31 * u31 + k32 * u32 + k33 * u33 );
}

