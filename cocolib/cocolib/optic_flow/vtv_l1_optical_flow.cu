/* -*-c++-*- */
/** \of_tv_l1.cu

	optical flow detection
	Implemented from Zach/Pock,Bischof 2007,
	"A Duality Based Approach for Realtime TV-L1 Optical Flow"
	combined with vectorial total variation

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
#include "../defs.h"
#include "../cuda/cuda_helper.h"
#include "../cuda/cuda_kernels.cuh"
#include "../cuda/cuda_reduce.h"
#include "../common/gsl_matrix_helper.h"
#include "../common/gsl_image.h"

#include <iostream>
#include <math.h>
#include <cuda.h>

#include "vtv_l1_optical_flow.h"
#include "vtv_l1_optical_flow.cuh"
#include "vtv_l1_optical_flow_helpers.cuh"
#include "vtv_l1_optical_flow_image.cuh"
 
//debug function for printing device matrices
void print_matrix(float* dev_matrix, int w, int h, char str[])
{ 
  coco::gsl_matrix *matrix;
  matrix = coco::gsl_matrix_alloc(h,w);
  coco::cuda_memcpy( matrix,dev_matrix );
  double *m = matrix->data;
  
  printf("%10s\n",str);
  for(int x=0; x<w;x++){
    for(int y=0; y<h;y++){
      printf("%4.0f\t",m[x+y*w]);
		}
    printf("\n");
  }
}

bool coco::of_tv_l1_step(of_tv_l1_data *data, int cs) //workspace + current scale
{
  of_tv_l1_workspace *w = data->_workspace;

  int   size = w->Ws[cs]*w->Hs[cs];
  float l_t = data->_lambda * data->_theta;
  float taut = data->_tau / data->_theta;
  
  float *I1x;
  CUDA_SAFE_CALL( cudaMalloc( (void**)&I1x, size*sizeof(float) )); //grad(I1) in x direction
  
  float *I1y;
  CUDA_SAFE_CALL( cudaMalloc( (void**)&I1y, size*sizeof(float) )); //grad(I1) in y direction
  
  float *I1w;
  CUDA_SAFE_CALL( cudaMalloc( (void**)&I1w, size*sizeof(float) )); //warped Image I1
	
  float *I1wx;
  CUDA_SAFE_CALL( cudaMalloc( (void**)&I1wx, size*sizeof(float) )); //grad(I1w) in x direction
  
  float *I1wy;
  CUDA_SAFE_CALL( cudaMalloc( (void**)&I1wy, size*sizeof(float) )); //grad(I1w) in y direction

  float *rho_c;
  CUDA_SAFE_CALL( cudaMalloc( (void**)&rho_c, size*sizeof(float) )); //constant part of rho for TH

  float *v1;
  CUDA_SAFE_CALL( cudaMalloc( (void**)&v1, size*sizeof(float) )); //x component of v

  float *v2;
  CUDA_SAFE_CALL( cudaMalloc( (void**)&v2, size*sizeof(float) )); //y component of v

  float *p11;
  CUDA_SAFE_CALL( cudaMalloc( (void**)&p11, size*sizeof(float) )); //x component of dual variable p1

  float *p12;
  CUDA_SAFE_CALL( cudaMalloc( (void**)&p12, size*sizeof(float) )); //y component of dual variable p1

  float *p21;
  CUDA_SAFE_CALL( cudaMalloc( (void**)&p21, size*sizeof(float) )); //x component of dual variable p2

  float *p22;
  CUDA_SAFE_CALL( cudaMalloc( (void**)&p22, size*sizeof(float) )); //y component of dual variable p2

  float *div;
  CUDA_SAFE_CALL( cudaMalloc( (void**)&div, size*sizeof(float) )); //divergence of p1,2

  float *grad;
  CUDA_SAFE_CALL( cudaMalloc( (void**)&grad, size*sizeof(float) )); //gradient!?

  float *div_p1;
  CUDA_SAFE_CALL( cudaMalloc( (void**)&div_p1, size*sizeof(float) )); //divergence of p1

  float *div_p2;
  CUDA_SAFE_CALL( cudaMalloc( (void**)&div_p2, size*sizeof(float) )); //divergence of p2
  
  float *u1x;
  CUDA_SAFE_CALL( cudaMalloc( (void**)&u1x, size*sizeof(float) )); //grad(u1) in x direction

  float *u1y;
  CUDA_SAFE_CALL( cudaMalloc( (void**)&u1y, size*sizeof(float) )); //grad(u1) in y direction

  float *u2x;
  CUDA_SAFE_CALL( cudaMalloc( (void**)&u2x, size*sizeof(float) )); //grad(u2) in x direction
  
  float *u2y;
  CUDA_SAFE_CALL( cudaMalloc( (void**)&u2y, size*sizeof(float) )); //grad(u2) in y direction

  float *mask;
  CUDA_SAFE_CALL( cudaMalloc( (void**)&mask, size*sizeof(float) )); //mask for thresholding
  
  float *dev_error;
  CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_error, size*sizeof(float) )); //error_array
  
  float *dev_error_res;
  CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_error_res, size*sizeof(float) )); //error reduction
  
  dim3 dimBlock( cuda_default_block_size_x(),
		 cuda_default_block_size_y() );
  dim3 dimGrid((int)(w->Ws[cs] / dimBlock.x+1),(int)(w->Hs[cs] / dimBlock.y+1));
	
  //on older graphic cards shared memory will increase performance, while on more recent ones calculation might be slower
  
  /*
    size_t dimShared2 = (dimBlock.x+2) * (dimBlock.y+2) * sizeof(float);
    of_tv_l1_centered_gradient_shared<<< dimGrid, dimBlock,dimShared2 >>>(w->I1s[cs], I1x, I1y,  w->Ws[cs],w->Hs[cs]);
  */
  of_tv_l1_centered_gradient<<< dimGrid, dimBlock>>>(w->I1s[cs], I1x, I1y,  w->Ws[cs],w->Hs[cs]);
  
  CUDA_SAFE_CALL(cudaMemset(p11,0,size*sizeof(float)));
  CUDA_SAFE_CALL(cudaMemset(p12,0,size*sizeof(float)));
  CUDA_SAFE_CALL(cudaMemset(p21,0,size*sizeof(float)));
  CUDA_SAFE_CALL(cudaMemset(p22,0,size*sizeof(float)));
  
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  for(int warpings = 0; warpings < data->_warps; warpings++) {
    coco::of_tv_l1_bilinear_interpolation(w->I1s[cs],w->U1s[cs],w->U2s[cs], I1w,  w->Ws[cs],w->Hs[cs], mask);
    coco::of_tv_l1_bilinear_interpolation(I1x,w->U1s[cs],w->U2s[cs], I1wx, w->Ws[cs],w->Hs[cs], mask);
    coco::of_tv_l1_bilinear_interpolation(I1y,w->U1s[cs],w->U2s[cs], I1wy, w->Ws[cs],w->Hs[cs], mask);
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    
    of_tv_l1_calculate_rho_const<<< dimGrid, dimBlock >>>
      (I1w,
       w->I0s[cs],
       I1wx, I1wy,
       grad, rho_c,
       w->U1s[cs],w->U2s[cs],
       w->Ws[cs],w->Hs[cs]);

    int n = 0;
    float error = FLOAT_MAX;
    while(error > data->_stopping_threshold && n < data->_iterations) {
      n++;
      of_tv_l1_calculate_TH<<< dimGrid, dimBlock >>>
	(I1wx,I1wy,grad,rho_c,w->U1s[cs],w->U2s[cs],v1,v2,mask, w->Ws[cs],w->Hs[cs],l_t);
      
      cuda_compute_divergence_device<<<dimGrid,dimBlock>>>
	( w->Ws[cs],w->Hs[cs], p11, p12, div_p1 );
      cuda_compute_divergence_device<<<dimGrid,dimBlock>>>
	( w->Ws[cs],w->Hs[cs], p21, p22, div_p2 );
      
      of_tv_l1_calculate_error<<< dimGrid, dimBlock >>>
	(w->U1s[cs],w->U2s[cs],v1,v2,div_p1,div_p2,w->Ws[cs],w->Hs[cs],data->_theta,dev_error);

      if ( cuda_sum_reduce(w->Ws[cs],w->Hs[cs],dev_error,dev_error_res,&error)) {
	error /= size;
      }
      else {
	// reductions not available
	error = FLOAT_MAX;
      }

      //of_tv_l1_forward_gradient<<< dimGrid, dimBlock,dimShared1 >>>(w->U1s[cs],u1x,u1y,w->Ws[cs],w->Hs[cs]);
      //of_tv_l1_forward_gradient<<< dimGrid, dimBlock,dimShared1 >>>(w->U2s[cs],u2x,u2y,w->Ws[cs],w->Hs[cs]);
      cuda_compute_gradient_device<<< dimGrid, dimBlock >>>
	( w->Ws[cs],w->Hs[cs], w->U1s[cs],u1x,u1y );
      cuda_compute_gradient_device<<< dimGrid, dimBlock >>>
	( w->Ws[cs],w->Hs[cs], w->U2s[cs],u2x,u2y );
      
      switch(data->_regularizer){
      case 0: //TV
	of_tv_l1_calculate_dual<<< dimGrid, dimBlock >>>
	  (u1x,u1y,u2x,u2y,p11,p12,p21,p22,w->Ws[cs],w->Hs[cs],taut);
	break;
      case 1: //VTV_J
	of_vtv_l1_calculate_dual_TVJ<<< dimGrid, dimBlock >>>
	  (u1x,u1y,u2x,u2y,p11,p12,p21,p22,w->Ws[cs],w->Hs[cs],taut);
	break;
      case 2: //VTV_F
	of_vtv_l1_calculate_dual_TVF<<< dimGrid, dimBlock >>>
	  (u1x,u1y,u2x,u2y,p11,p12,p21,p22,w->Ws[cs],w->Hs[cs],taut);
	break;
      default:
	ERROR( "Unknown regularizer." << std::endl );
	assert( false );
      }
    }
  }
  
  CUDA_SAFE_CALL( cudaFree(I1x ));
  CUDA_SAFE_CALL( cudaFree(I1y ));
  CUDA_SAFE_CALL( cudaFree(I1w ));
  CUDA_SAFE_CALL( cudaFree(I1wx));
  CUDA_SAFE_CALL( cudaFree(I1wy ));
  CUDA_SAFE_CALL( cudaFree(rho_c));
  CUDA_SAFE_CALL( cudaFree(v1 ));
  CUDA_SAFE_CALL( cudaFree(v2 ));
  CUDA_SAFE_CALL( cudaFree(p11 ));
  CUDA_SAFE_CALL( cudaFree(p12 ));
  CUDA_SAFE_CALL( cudaFree(p21 ));
  CUDA_SAFE_CALL( cudaFree(p22 ));
  CUDA_SAFE_CALL( cudaFree(div));
  CUDA_SAFE_CALL( cudaFree(grad ));
  CUDA_SAFE_CALL( cudaFree(div_p1 ));
  CUDA_SAFE_CALL( cudaFree(div_p2 ));
  CUDA_SAFE_CALL( cudaFree(u1x ));
  CUDA_SAFE_CALL( cudaFree(u1y ));
  CUDA_SAFE_CALL( cudaFree( u2x));
  CUDA_SAFE_CALL( cudaFree(u2y ));
  CUDA_SAFE_CALL( cudaFree(mask ));
  CUDA_SAFE_CALL( cudaFree(dev_error));
  CUDA_SAFE_CALL( cudaFree(dev_error_res));
  
  return true;
}



coco::of_tv_l1_data* coco::of_tv_l1_data_alloc( gsl_matrix* im0, gsl_matrix* im1, size_t nscales, double zfactor )
{
  of_tv_l1_data *data = new of_tv_l1_data;
  
  data->_W = im0->size2;
  data->_H = im0->size1;
  assert(im0->size1==im1->size1);
  assert(im0->size2==im1->size2);
  data->_N = data->_W * data->_H;
  
  // Size of image matrices in bytes
  data->_nfbytes = data->_N * sizeof(float);
  
  data->_workspace = new of_tv_l1_workspace;
  memset( data->_workspace, 0, sizeof( of_tv_l1_workspace ));

  // Alloc fields
  of_tv_l1_workspace *w = data->_workspace;

  w->I0s = (float**)malloc(nscales * sizeof(float*));
  w->I1s = (float**)malloc(nscales * sizeof(float*));
  w->U1s = (float**)malloc(nscales * sizeof(float*));
  w->U2s = (float**)malloc(nscales * sizeof(float*));
  w->Ws = (int*)malloc(nscales * sizeof(int*));
  w->Hs = (int*)malloc(nscales * sizeof(int*));

  w->Ws[0]=im0->size2;
  w->Hs[0]=im0->size1;

  for(int s = 0; s < (int)nscales; s++) {
    if(s) of_tv_l1_zoom_size(w->Ws[s-1], w->Hs[s-1], &w->Ws[s], &w->Hs[s], zfactor);
    
    const int sizes = w->Ws[s] * w->Hs[s];
    CUDA_SAFE_CALL( cudaMalloc( &w->I0s[s], sizes*sizeof(float) ));
    CUDA_SAFE_CALL( cudaMalloc( &w->I1s[s], sizes*sizeof(float) ));
    CUDA_SAFE_CALL( cudaMalloc( &w->U1s[s], sizes*sizeof(float) ));
    CUDA_SAFE_CALL( cudaMalloc( &w->U2s[s], sizes*sizeof(float) ));
  }

  data->_im0 = gsl_matrix_alloc( data->_H, data->_W );
  data->_im1 = gsl_matrix_alloc( data->_H, data->_W );
  data->_u = gsl_matrix_alloc( data->_H, data->_W );
  gsl_matrix_copy_to(im0,data->_im0);
  gsl_matrix_copy_to(im1,data->_im1);

  // data attachment weight
  data->_lambda = 0.2;
  // algo data
  data->_tau = 0.25;
  data->_theta = 0.1;
  // stopping threshold
  data->_stopping_threshold = 0.0001;
  // number of warps
  data->_warps = 5;
  // number of scales
  data->_scales = 5;
  // scale factor
  data->_factor = 0.5;
  // number of inner iterations
  data->_iterations = 150;

  // Regularizer
  // 0: TV
  // 1: VTV_J
  data->_regularizer = 0;
  // Local CPU copy of ground Truth
  data->_gt = NULL;
  
  return data;
}

bool coco::of_tv_l1_data_free( of_tv_l1_data *data )
{
  // Free GPU fields
  of_tv_l1_workspace *w = data->_workspace;
  for(int s = 1; s < data->_scales; s++) {
    CUDA_SAFE_CALL( cudaFree( w->I0s[s] ));
    CUDA_SAFE_CALL( cudaFree( w->I1s[s] ));
    CUDA_SAFE_CALL( cudaFree( w->U1s[s] ));
    CUDA_SAFE_CALL( cudaFree( w->U2s[s] ));
  }
  
  gsl_matrix_free( data->_im0 );
  gsl_matrix_free( data->_im1 );
  gsl_matrix_free( data->_u );

  delete data->_workspace;
  delete data;
  return true;
}


// Initialize workspace with current solution
bool coco::of_tv_l1_initialize( of_tv_l1_data *data)
{
  of_tv_l1_workspace *w = data->_workspace;
  cuda_memcpy( w->I0s[0], data->_im0);
  cuda_memcpy( w->I1s[0], data->_im1);

  dim3 dimBlock( cuda_default_block_size_x(),
		 cuda_default_block_size_y() );
  dim3 dimGrid((int)(w->Ws[0] / dimBlock.x+1),(int)(w->Hs[0] / dimBlock.y+1));

  of_tv_l1_multiply_by_float<<< dimGrid, dimBlock >>>
    (w->I0s[0], w->Ws[0], w->Hs[0], 255);
  of_tv_l1_multiply_by_float<<< dimGrid, dimBlock >>>
    (w->I1s[0], w->Ws[0], w->Hs[0], 255);

  CUDA_SAFE_CALL( cudaMemset( w->U1s[0], 0, w->Hs[0]*w->Ws[0]*sizeof(float) ));
  CUDA_SAFE_CALL( cudaMemset( w->U2s[0], 0, w->Hs[0]*w->Ws[0]*sizeof(float) ));

  for(int s = 1; s < data->_scales; s++) {
    int H = w->Hs[s];
    int W = w->Ws[s];
    int ssize = W*H;
    
    CUDA_SAFE_CALL( cudaMemset( w->U1s[s], 0, ssize*sizeof(float) ));
    CUDA_SAFE_CALL( cudaMemset( w->U2s[s], 0, ssize*sizeof(float) ));
    of_tv_l1_zoom(w->I0s[s-1], w->I0s[s], w->Ws[s-1], w->Hs[s-1], data->_factor);
    of_tv_l1_zoom(w->I1s[s-1], w->I1s[s], w->Ws[s-1], w->Hs[s-1], data->_factor);
  }
  return true;
}


bool coco::of_tv_l1_calculate( of_tv_l1_data *data )
{
  of_tv_l1_workspace *w = data->_workspace;
  
  cudaEvent_t start, stop; 
  float elapsedTime;
  CUDA_SAFE_CALL(cudaEventCreate(&start));
  CUDA_SAFE_CALL(cudaEventCreate(&stop));
  //event record
  CUDA_SAFE_CALL(cudaEventRecord(start,0));
	
  for(int s = data->_scales-1; s >= 0; s--){
    of_tv_l1_step(data,s);
    if(s>0) {
      dim3 dimBlock( cuda_default_block_size_x(),
		     cuda_default_block_size_y() );
      dim3 dimGrid((int)(w->Ws[s-1] / dimBlock.x+1),(int)(w->Hs[s-1] / dimBlock.y+1));
      
      of_tv_l1_zoom_out<<< dimGrid, dimBlock >>>
	(w->U1s[s], w->U1s[s-1], w->Ws[s], w->Hs[s], w->Ws[s-1], w->Hs[s-1]);
      of_tv_l1_zoom_out<<< dimGrid, dimBlock >>>
	(w->U2s[s], w->U2s[s-1], w->Ws[s], w->Hs[s], w->Ws[s-1], w->Hs[s-1]);
      CUDA_SAFE_CALL( cudaThreadSynchronize() );
      
      of_tv_l1_multiply_by_float<<< dimGrid, dimBlock >>>
	(w->U1s[s-1],w->Ws[s-1], w->Hs[s-1],(float) 1.0 / data->_factor);
      of_tv_l1_multiply_by_float<<< dimGrid, dimBlock >>>
	(w->U2s[s-1],w->Ws[s-1], w->Hs[s-1],(float) 1.0 / data->_factor);
      
      CUDA_SAFE_CALL( cudaThreadSynchronize() );
    }
  }
  
  //stop timer
  CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
  CUDA_SAFE_CALL(cudaEventSynchronize(stop));
  CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));

  TRACE("runtime: " << elapsedTime << "ms" << std::endl);
  
  CUDA_SAFE_CALL(cudaEventDestroy(start));
  CUDA_SAFE_CALL(cudaEventDestroy(stop));

  return true;
}

// Get current solution (as raw flow field)
bool coco::of_tv_l1_get_solution( of_tv_l1_data *data,
				  gsl_matrix *fx, gsl_matrix *fy )
{
  assert( data != NULL );
  assert( fx != NULL );
  assert( fy != NULL );
  of_tv_l1_workspace *w = data->_workspace;
  cuda_memcpy( fx, w->U1s[0] );
  cuda_memcpy( fy, w->U2s[0] );
  return true;
}


// Get current solution
bool coco::of_tv_l1_get_solution( of_tv_l1_data *data,
				  gsl_image *u)
{
  of_tv_l1_workspace *w = data->_workspace;
  int H = w->Hs[0];
  int W = w->Ws[0];
  
  gsl_matrix *u1 = gsl_matrix_alloc( H,W );
  gsl_matrix *u2 = gsl_matrix_alloc( H,W );
  of_tv_l1_get_solution( data, u1, u2 );
	
  float epe = 0.0; //average endpoint error
  float aae = 0.0; //average angular error
  
  float maxx = -999, maxy = -999;
  float minx =  999, miny =  999;
  float maxrad=-1;

  for(int i = 0; i< w->Ws[0] * w->Hs[0]; i++){
    float fx = u1->data[i];
    float fy = u2->data[i];
    
    float rad = sqrt(fx * fx + fy * fy);
    if(fx>maxx)
      maxx=fx;
    if(fy>maxy)
			maxy=fy;
    if(fx<minx)
      minx=fx;
    if(fy<miny)
      miny=fy;
    if(maxrad < rad)
      maxrad = rad;
  }
  if(maxrad==0) maxrad=1;
  
  epe/=H*W;
  aae/=H*W;

  for(int i = 0; i< w->Ws[0] * w->Hs[0]; i++){
    float tmp[3];
    computeColor(-(u2->data[i])/maxrad, -(u1->data[i])/maxrad, tmp);
    u->_r->data[i]=tmp[0]/255;
    u->_g->data[i]=tmp[1]/255;
    u->_b->data[i]=tmp[2]/255;
  }

  return true;
}
