/* -*-c++-*- */
/** \of_tv_l1_helpers.cu

    Example code for COCOLIB: call solvers for the optical flow models
    with TV, VTV_J or VTV_F regularizer term.

    Copyright (C) 2012 Ole Johannsen,
    <first name>.<last name>ATberlin.de

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
#include <math.h>
#include <float.h>

#include "vtv_l1_optical_flow.h"
#include "vtv_l1_optical_flow.cuh"
#include "vtv_l1_optical_flow_helpers.cuh"
#include "vtv_l1_optical_flow_image.cuh"

#include "../defs.h"
#include "../cuda/cuda_helper.h"

#define BOUNDARY_CONDITION = 0;

// value of a pixel without an flow value in an .flo file
const int gt_invalid = 1666666752;



__global__ void coco::of_tv_l1_multiply_by_float( float *I,
						  int nx,
						  int ny,
						  float factor )
{
  int ox = threadIdx.x + blockIdx.x*blockDim.x; //output x
  int oy = threadIdx.y + blockIdx.y*blockDim.y; //output y
  int ooffset = ox+ oy * nx;  //output offset
  
  if(ox<nx && oy<ny){
    I[ooffset]=I[ooffset]*factor;
  }
}


__global__ void coco::of_tv_l1_zoom_out( float *I, //input image small
					 float *Iout,    //output image bin
					 int nx,         //width of the original image small
					 int ny,         //height of the original image
					 int nxx,        //width of the zoomed image big
					 int nyy         //height of the zoomed image
					 )
{
  float factorx = ((float)nxx / nx);
  float factory = ((float)nyy / ny);
  
  int ox = threadIdx.x + blockIdx.x*blockDim.x; //output x
  int oy = threadIdx.y + blockIdx.y*blockDim.y; //output y

  if(ox<nxx && oy<nyy) {
    int y =  (int) ((float)oy/factory);
    int x =  (int) ((float)ox/factorx);
    
    if(y >= ny) y=ny;
    if(x >= nx) x=nx;
    Iout[oy * nxx + ox] = I[x + nx * y];
  }
}

__global__ void coco::of_tv_l1_calculate_dual( float *u1x,
					       float *u1y,
					       float *u2x,
					       float *u2y,
					       float *p11,
					       float *p12,
					       float *p21,
					       float *p22,
					       int nx,
					       int ny,
					       float taut
					       )
{
  int ox = threadIdx.x + blockIdx.x*blockDim.x; //output x
  int oy = threadIdx.y + blockIdx.y*blockDim.y; //output y
  int ooffset = ox+ oy * nx;  //output offset
  
  if(ox<nx && oy<ny) {
    float g1   = sqrt(u1x[ooffset] * u1x[ooffset] + u1y[ooffset] * u1y[ooffset]);
    float g2   = sqrt(u2x[ooffset] * u2x[ooffset] + u2y[ooffset] * u2y[ooffset]);
    float ng1  = 1.0 + taut * g1;
    float ng2  = 1.0 + taut * g2;
    
    p11[ooffset] = (p11[ooffset] + taut * u1x[ooffset]) / ng1;
    p12[ooffset] = (p12[ooffset] + taut * u1y[ooffset]) / ng1;
    p21[ooffset] = (p21[ooffset] + taut * u2x[ooffset]) / ng2;
    p22[ooffset] = (p22[ooffset] + taut * u2y[ooffset]) / ng2;
  }
}


__global__ void coco::of_tv_l1_calculate_error( float *u1,
						float *u2,
						float *v1,
						float *v2,
						float *div_p1,
						float *div_p2,
						int nx,
						int ny,
						float theta,
						float *error)
{
  int ox = threadIdx.x + blockIdx.x*blockDim.x; //output x
  int oy = threadIdx.y + blockIdx.y*blockDim.y; //output y
  int ooffset = ox+ oy * nx;  //output offset
  
  if(ox<nx && oy<ny){
    float u1k = u1[ooffset];
    float u2k = u2[ooffset];
    
    u1[ooffset] = v1[ooffset] + theta * div_p1[ooffset];
    u2[ooffset] = v2[ooffset] + theta * div_p2[ooffset];

    error[ooffset] = (u1[ooffset] - u1k) * (u1[ooffset] - u1k) +
      (u2[ooffset] - u2k) * (u2[ooffset] - u2k);
  }
}


__global__ void coco::of_tv_l1_calculate_TH( float *I1wx,
					     float *I1wy,
					     float *grad,
					     float *rho_c,
					     float *u1,
					     float *u2,
					     float *v1,
					     float *v2,
					     float *mask,
					     int nx,
					     int ny,
					     float l_t )
{
  int ox = threadIdx.x + blockIdx.x*blockDim.x; //output x
  int oy = threadIdx.y + blockIdx.y*blockDim.y; //output y
  int ooffset = ox+ oy * nx;  //output offset
  
  if(ox<nx && oy<ny){
    float rho = rho_c[ooffset] + (I1wx[ooffset] * u1[ooffset] + I1wy[ooffset] * u2[ooffset]);
    
    float d1, d2;
    
    if(rho < - l_t * grad[ooffset]) {
      d1 = l_t * I1wx[ooffset];
      d2 = l_t * I1wy[ooffset];
    }
    else {
      if(rho > l_t * grad[ooffset]) {
	d1 = -l_t * I1wx[ooffset];
	d2 = -l_t * I1wy[ooffset];
      }
      else {
	if(grad[ooffset] < 1E-10){
	  d1 = 0;
	  d2 = 0;
	}
	else{
	  d1 = - rho * I1wx[ooffset] / grad[ooffset];
	  d2 = - rho * I1wy[ooffset] / grad[ooffset];
	}
      }
    }
    
    v1[ooffset] = u1[ooffset] + mask[ooffset] * d1;
    v2[ooffset] = u2[ooffset] + mask[ooffset] * d2;
  }
}


__global__ void coco::of_tv_l1_calculate_rho_const(float* I1w, float* I0, float* I1wx,float* I1wy,
						   float* grad,float* rho_c,float* u1,float* u2,
						   int nx,int ny)
{
  int ox = threadIdx.x + blockIdx.x*blockDim.x; //output x
  int oy = threadIdx.y + blockIdx.y*blockDim.y; //output y
  int ooffset = ox+ oy * nx;  //output offset
  
  if(ox<nx && oy<ny){
    float Ix2 = I1wx[ooffset] * I1wx[ooffset];
    float Iy2 = I1wy[ooffset] * I1wy[ooffset];
    
    //store the |Grad(I1)|^2
    grad[ooffset] = (Ix2 + Iy2);
    
    //compute the ant part of the rho function
    rho_c[ooffset] = (I1w[ooffset] - I1wx[ooffset] * u1[ooffset] - I1wy[ooffset] * u2[ooffset] - I0[ooffset]);
  }
}

__device__ int of_tv_l1_neumann_bc(int x, int nx, float *mask)
{
  if(x < 0) {
    x = 0;
    *mask = 0;
  }
  else if (x >= nx){
    x = nx - 1;
    *mask = 0;
  }
  return x;
}

__global__ void of_tv_l1_bilinear_interpolation_device( float *input, //image to be warped
							float *u,     //x component of the vector field
							float *v,     //y component of the vector field
							float *output,      //warped output image
							int nx,       //width of the image
							int ny,       //height of the image
							float *mask
							)
{
  int ox = threadIdx.x + blockIdx.x*blockDim.x; //output x
  int oy = threadIdx.y + blockIdx.y*blockDim.y; //output y
  int ooffset = ox+ oy * nx;  //output offset
  
  if(ox<nx && oy<ny){
    mask[ooffset]=1;
	  
    float uu = (float) (ox  + u[ooffset]); //warped x coordinate
    float vv = (float) (oy + v[ooffset]); //warped y coordinate
    int sx = (uu < 0)? -1: 1;
    int sy = (vv < 0)? -1: 1;
    
    int x, y, dx, dy;
    
    //TODO: add further boundary conditions
    
    x  = of_tv_l1_neumann_bc((int) uu, nx, &mask[ooffset]);
    y  = of_tv_l1_neumann_bc((int) vv, ny, &mask[ooffset]);
    dx = of_tv_l1_neumann_bc(x + sx, nx,   &mask[ooffset]);
    dy = of_tv_l1_neumann_bc(y + sy, ny,   &mask[ooffset]);
    
    float p1 = input[x  + nx * y];
    float p2 = input[dx + nx * y];
    float p3 = input[x  + nx * dy];
    float p4 = input[dx + nx * dy];
    
    float e1 = ((float) sx * (uu - x));
    float E1 = ((float) 1.0 - e1);
    float e2 = ((float) sy * (vv - y));
    float E2 = ((float) 1.0 - e2);
    
    float w1 = E1 * p1 + e1 * p2;
    float w2 = E1 * p3 + e1 * p4;
    
    output[ooffset] = E2 * w1 + e2 * w2;
  }
}

void coco::of_tv_l1_bilinear_interpolation( float *input, //image to be warped
					    float *u,     //x component of the vector field
					    float *v,     //y component of the vector field
					    float *output,      //warped output image
					    int nx,       //width of the image
					    int ny,       //height of the image
					    float *mask         //mask to detect the motions outside the image
					    )
{
  dim3 dimBlock( cuda_default_block_size_x(),
		 cuda_default_block_size_y() );
  dim3 dimGrid((int)(nx / dimBlock.x+1),(int)(ny / dimBlock.y+1));
  of_tv_l1_bilinear_interpolation_device<<< dimGrid, dimBlock>>>
    ( input,u,v,output,nx,ny,mask);
}

void coco::of_tv_l1_zoom_size( int nx,             //width of the orignal image
			       int ny,             //height of the orignal image
			       int *nxx,           //width of the zoomed image
			       int *nyy,           //height of the zoomed image
			       float factor  //zoom factor between 0 and 1
			       )
{
  *nxx = (int)((float) nx * factor + 0.5);
  *nyy = (int)((float) ny * factor + 0.5);
}

__global__ void coco::of_tv_l1_centered_gradient( float *input,  //input image
						  float *dx,           //computed x derivative
						  float *dy,           //computed y derivative
						  int nx,        //image width
						  int ny         //image height
						  )
{
  int ox = threadIdx.x + blockIdx.x*blockDim.x; //output x
  int oy = threadIdx.y + blockIdx.y*blockDim.y; //output y
  int ooffset = ox+ oy * nx;  //output offset
  
  if(ox<nx && oy<ny){
    if(ox==0){ //first column
      if(oy==0){//corner 0,0 (ooffset==0)
	dx[0] = 0.5*(input[ooffset+1] - input[ooffset]);
	dy[0] = 0.5*(input[ooffset+nx] - input[ooffset]);
      }else if (oy==ny-1){ //corner 0,ny-1
	dx[(ny-1)*nx] = 0.5*(input[ooffset + 1] - input[ooffset]);
	dy[(ny-1)*nx] = 0.5*(input[ooffset] - input[ooffset-nx]);
      }else{//inside fist column
	dx[ooffset] = 0.5*(input[ooffset+1] - input[ooffset]);
	dy[ooffset] = 0.5*(input[ooffset+nx] - input[ooffset-nx]);
      }
    }else if(ox==nx-1){ //last column
      if(oy==0){//corner nx-1,0
	dx[nx-1] = 0.5*(input[ooffset] - input[ooffset-1]);
	dy[nx-1] = 0.5*(input[ooffset+nx] - input[ooffset-1]);
      }else if (oy==ny-1){ //corner nx-1,ny-1
	dx[ny*nx-1] = 0.5*(input[ooffset] - input[ooffset-1]);
	dy[ny*nx-1] = 0.5*(input[ooffset] - input[ooffset-nx]);
      }else{//inside last column
	dx[ooffset] = 0.5*(input[ooffset] - input[ooffset-1]);
	dy[ooffset] = 0.5*(input[ooffset+nx] - input[ooffset-nx]);
      }
    }else if(oy==0){//first row
      dx[ooffset] = 0.5*(input[ooffset+1] - input[ooffset-1]);
      dy[ooffset] = 0.5*(input[ooffset+nx] - input[ooffset]);
    }else if(oy==ny-1){//last row
      dx[ooffset] = 0.5*(input[ooffset+1] - input[ooffset-1]);
      dy[ooffset] = 0.5*(input[ooffset] - input[ooffset-nx]);
    }else{//inside
      dx[ooffset] = 0.5*(input[ooffset+1] - input[ooffset-1]);
      dy[ooffset] = 0.5*(input[ooffset+nx] - input[ooffset-nx]);
    }
  }
}

__global__ void of_tv_l1_zoom_device(float* I, float* Iout, float factor, int nxx, int nyy, int nx, int ny)
{
  int ox = threadIdx.x + blockIdx.x*blockDim.x; //output x
  int oy = threadIdx.y + blockIdx.y*blockDim.y; //output y
  int ooffset = ox+ oy * nxx;  //output offset
  
  if(ox<nxx && oy<nyy){
    int x =(int)((float)ox/factor); //x in (big) input image
    int y =(int)((float)oy/factor); //y in (big) input image
    
    int mx = x - 1;
    int dx = x + 1;
    int my = y - 1;
    int dy = y + 1;

    if(mx < 0) mx = 0;
    if(dx >= nx) dx = nx-1;
    if(my < 0) my = 0;
    if(dy >= ny) dy = ny-1;
    
    //smooth with the neighbors
    float value =
      0.07842776544 * (I[mx + nx * my]  + I[mx + nx * dy]  + I[dx + nx * my]  + I[dx + nx * dy]) +
      0.1231940459  * (I[x  + nx * my]  + I[mx + nx *  y]  + I[dx + nx *  y]  + I[x  + nx * dy])  +
      0.1935127547  *  I[x  + nx *  y];
    
    Iout[ooffset] = value;
  }
}


bool coco::of_tv_l1_zoom( float *I,    //input image
			  float *Iout,       //output image
			  int nx,            //image width
			  int ny,            //image height
			  float factor //zoom factor between 0 and 1
			  )
{
  int nxx, nyy;
  of_tv_l1_zoom_size(nx, ny, &nxx, &nyy, factor);
  
  dim3 dimBlock( cuda_default_block_size_x(),
		 cuda_default_block_size_y() );
  dim3 dimGrid((int)(ceil((float)nxx / dimBlock.x)),(int)(ceil((float)nyy / dimBlock.y)));
  
  of_tv_l1_zoom_device<<< dimGrid, dimBlock >>>
    ( I,Iout,factor,nxx,nyy,nx,ny);
  return true;
}


__global__ void coco::of_vtv_l1_dual_step( int W, int H,
					   float tstep,
					   float *u,
					   float *px,
					   float *py
					   )
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

__global__ void coco::of_vtv_l1_calculate_dual_TVJ(
		float *u1x,
		float *u1y,
		float *u2x,
		float *u2y,
		float *p11,
		float *p12,
		float *p21,
		float *p22,
		int nx,
		int ny,
		float tau
		)
{
  // Global thread index
  int ox = threadIdx.x + blockIdx.x*blockDim.x; //output x
  int oy = threadIdx.y + blockIdx.y*blockDim.y; //output y
  int ooffset = ox+ oy * nx;  //output offset
  
  if(ox<nx && oy<ny){
    p11[ooffset] = (p11[ooffset] + tau * u1x[ooffset]);
    p12[ooffset] = (p12[ooffset] + tau * u1y[ooffset]);
    p21[ooffset] = (p21[ooffset] + tau * u2x[ooffset]);
    p22[ooffset] = (p22[ooffset] + tau * u2y[ooffset]);
    
    cuflt a11 = p11[ooffset];
    cuflt a21 = p12[ooffset];
    cuflt a12 = p21[ooffset];
    cuflt a22 = p22[ooffset];
    
    // Compute A^T A
    cuflt d11 = a11*a11 + a21*a21;
    cuflt d12 = a12*a11 + a22*a21;
    cuflt d22 = a12*a12 + a22*a22;
    
    // Compute larger Eigenvalue (= square of largest singular value)
    cuflt trace = d11 + d22;
    cuflt det = d11*d22 - d12*d12;
    cuflt d = __powf( 0.25*trace*trace - det ,0.5);
    cuflt lmax = max( 0.0, 0.5 * trace + d );
    cuflt lmin = max( 0.0, 0.5 * trace - d );
    cuflt smax = __powf( lmax ,0.5);
    cuflt smin = __powf( lmin ,0.5);
    
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
	cuflt l1 = hypot( v11, v21 );
	v11 /= l1; v21 /= l1;
	v12 = lmin - d22; v22 = d12;
	cuflt l2 = hypot( v12, v22 );
	v12 /= l2; v22 /= l2;
      }
      
      // Compute projection of Eigenvalues
      cuflt gamma = 0.5 * (smax - smin + 1.0);
      cuflt s1 = min( 1.0, gamma );
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
      p11[ooffset] = a11 * t11 + a12 * t21;
      p12[ooffset] = a21 * t11 + a22 * t21;
      
      p21[ooffset] = a11 * t12 + a12 * t22;
      p22[ooffset] = a21 * t12 + a22 * t22;
    }
  }
}


__global__ void coco::of_vtv_l1_calculate_dual_TVF( float *u1x,
						    float *u1y,
						    float *u2x,
						    float *u2y,
						    float *p11,
						    float *p12,
						    float *p21,
						    float *p22,
						    int nx,
						    int ny,
						    float tau
						    )
{
  // Global thread index
  int ox = threadIdx.x + blockIdx.x*blockDim.x; //output x
  int oy = threadIdx.y + blockIdx.y*blockDim.y; //output y
  int ooffset = ox+ oy * nx;  //output offset

  if(ox<nx && oy<ny){
    p11[ooffset] = (p11[ooffset] + tau * u1x[ooffset]);
    p12[ooffset] = (p12[ooffset] + tau * u1y[ooffset]);
    p21[ooffset] = (p21[ooffset] + tau * u2x[ooffset]);
    p22[ooffset] = (p22[ooffset] + tau * u2y[ooffset]);
    
    float p = sqrt(p11[ooffset] * p11[ooffset] + p12[ooffset] * p12[ooffset] + p21[ooffset] * p21[ooffset] + p22[ooffset] * p22[ooffset]);

    if ( p > 1.0f ) {
      p11[ooffset] = (p11[ooffset]) / p;
      p12[ooffset] = (p12[ooffset]) / p;
      p21[ooffset] = (p21[ooffset]) / p;
      p22[ooffset] = (p22[ooffset]) / p;
    }
  }  // Local vars
}



// Compare flow field to ground truth
bool coco::optic_flow_compare_gt( gsl_matrix *fx, gsl_matrix *fy,
				  gsl_matrix *gtx, gsl_matrix *gty,
				  double &average_endpoint_error,
				  double &average_angular_error )
{
  size_t W = fx->size2;
  size_t H = fx->size1;
  assert( W == fy->size2 );
  assert( H == fy->size1 );
  assert( W == gtx->size2 );
  assert( H == gtx->size1 );
  assert( W == gty->size2 );
  assert( H == gty->size1 );
  size_t N = W*H;

  // ground truth comparison
  double epe = 0.0; // average endpoint error
  double aae = 0.0; // average angular error
  double count_valid = 0.0;

  for (size_t i=0; i<N; i++) {
    float vfx = fx->data[i];
    float vfy = fy->data[i];
    float vgtx = gtx->data[i];
    float vgty = gty->data[i];

    if ( vgtx == gt_invalid ) {
      continue;
    }

    count_valid += 1.0;
    epe += hypot( vfx - vgtx, vfy - vgty );
    
    float Z = vfx * vgtx + vfy * vgty + 1.0f;
    float N1 = sqrt( vfx*vfx + vfy*vfy + 1.0f );
    float N2 = sqrt( vgtx*vgtx + vgty*vgty + 1.0f );
    float cos_aae = max( -1.0f, min( 1.0f, Z / (N1 * N2) ));
    aae += acos( cos_aae ) * 180.0f / M_PI;
  }
    
  if ( count_valid == 0.0 ) {
    ERROR( "no pixel in optic flow ground truth is valid." << std::endl );
    count_valid = 1.0;
  }

  average_endpoint_error = epe / count_valid;
  average_angular_error = aae / count_valid;
  return true;
}

