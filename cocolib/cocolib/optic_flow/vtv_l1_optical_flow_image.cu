/* -*-c++-*- */
/** \of_tv_l1_image.cu

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

#include <float.h>

#include "../common/gsl_image.h"
#include "../common/gsl_matrix_helper.h"
#include "../cuda/cuda_helper.h"

#include "vtv_l1_optical_flow.h"

int ncols = 0;
#define MAXCOLS 60
int colorwheel[MAXCOLS][3];
float PI = 3.14159265;

void setcols(int r, int g, int b, int k)
{
    colorwheel[k][0] = r;
    colorwheel[k][1] = g;
    colorwheel[k][2] = b;
}

void makecolorwheel()
{
    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow
    //  than between yellow and green)
    int RY = 15;
    int YG = 6;
    int GC = 4;
    int CB = 11;
    int BM = 13;
    int MR = 6;
    ncols = RY + YG + GC + CB + BM + MR;
    //printf("ncols = %d\n", ncols);
    if (ncols > MAXCOLS)
	exit(1);
    int i;
    int k = 0;
    for (i = 0; i < RY; i++) setcols(255,	   255*i/RY,	 0,	       k++);
    for (i = 0; i < YG; i++) setcols(255-255*i/YG, 255,		 0,	       k++);
    for (i = 0; i < GC; i++) setcols(0,		   255,		 255*i/GC,     k++);
    for (i = 0; i < CB; i++) setcols(0,		   255-255*i/CB, 255,	       k++);
    for (i = 0; i < BM; i++) setcols(255*i/BM,	   0,		 255,	       k++);
    for (i = 0; i < MR; i++) setcols(255,	   0,		 255-255*i/MR, k++);
}

void computeColor(float fx, float fy, float *pix)
{
    if (ncols == 0)
	makecolorwheel();

    float rad = sqrt(fx * fx + fy * fy);
    float a = atan2(-fy, -fx) /PI;
    float fk = (a + 1.0) / 2.0 * (ncols-1);
    int k0 = (int)fk;
    int k1 = (k0 + 1) % ncols;
    float f = fk - k0;
    //f = 0; // uncomment to see original color wheel
    for (int b = 0; b < 3; b++) {
		float col0 = colorwheel[k0][b] / 255.0;
		float col1 = colorwheel[k1][b] / 255.0;
		float col = (1 - f) * col0 + f * col1;
		if (rad <= 1)
			col = 1 - rad * (1 - col); // increase saturation with radius
		else
			col *= .75; // out of range*/
		pix[2 - b] = (int)(255.0 * col);
    }
}

__global__ void image_to_grayscale_device( size_t W, size_t H, // output width and height
					   float *r,
					   float *g,
					   float *b,
					   float *grayscale )
{
  // Compute global thread offset and check bounds.
  // These are usually always the first six code lines in a kernel.
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;

  // Actual computation: approximately luminance-preserving grayscale conversion
  grayscale[o] = 0.3f*r[o] + 0.59f*g[o] + 0.11f*b[o];
}

#define CUDA_BLOCK_SIZE 16
void coco::image_to_grayscale(size_t W, size_t H, coco::gsl_image *im, gsl_matrix *mim)
{
  dim3 dimBlock(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE);
  dim3 dimGrid((int)(W / dimBlock.x+1),(int)(H / dimBlock.y+1));
  
  // Alloc GPU arrays
  int N=W*H;
  float *grayscale = NULL;
  CUDA_SAFE_CALL( cudaMalloc( &grayscale, N*sizeof(float) ));
  
  float *r = NULL;
  CUDA_SAFE_CALL( cudaMalloc( &r, N*sizeof(float) ));
  float *g = NULL;
  CUDA_SAFE_CALL( cudaMalloc( &g, N*sizeof(float) ));
  float *b = NULL;
  CUDA_SAFE_CALL( cudaMalloc( &b, N*sizeof(float) ));
	  
  cuda_memcpy( r, im->_r );
  cuda_memcpy( g, im->_g );
  cuda_memcpy( b, im->_b );
  
  image_to_grayscale_device<<< dimGrid, dimBlock >>>
    ( W, H, r,g,b, grayscale );
  
  cuda_memcpy( mim, grayscale);

  CUDA_SAFE_CALL( cudaFree( r ));
  CUDA_SAFE_CALL( cudaFree( g ));
  CUDA_SAFE_CALL( cudaFree( b ));
}


// Convert flow field to color image using Middlebury color wheel
coco::gsl_image *coco::flow_field_to_image( gsl_matrix *u1, gsl_matrix *u2 )
{
  size_t W = u1->size2;
  size_t H = u1->size1;
  assert( W == u2->size2 );
  assert( H == u2->size1 );
  size_t N = W*H;

  gsl_image *I = gsl_image_alloc( W,H );
	
  gsl_matrix *r = I->_r;
  gsl_matrix *g = I->_g;
  gsl_matrix *b = I->_b;
  
  float maxx = -FLT_MAX, maxy = -FLT_MAX;
  float minx =  FLT_MAX, miny =  FLT_MAX;
  float maxrad = -1.0f;
  for( size_t i=0; i<N; i++ ) {
    float fx = u1->data[i];
    float fy = u2->data[i];
    
    float rad = hypotf( fx,fy );
    maxx = max( fx, maxx );
    maxy = max( fy, maxy );
    minx = min( fx, minx );
    miny = min( fy, miny );
    maxrad = max( maxrad, rad );
  }
  if ( maxrad <= 0.0f ) {
    maxrad = 1.0f;
  }
  
  for ( size_t i=0; i<N; i++) {
    float tmp[3];
    computeColor( -(u2->data[i])/maxrad, -(u1->data[i])/maxrad, tmp);
    r->data[i] = tmp[0] / 255.0f;
    g->data[i] = tmp[1] / 255.0f;
    b->data[i] = tmp[2] / 255.0f;
  }

  return I;
}
