/* -*-c++-*- */
#include "simplex_reprojection.h"

#include "../cuda/cuda_helper.h"
#include "../common/gsl_matrix_helper.h"
#include "../defs.h"

#define CBS_SMALL 4


/******************************************
Simplex reprojection algorithm
*******************************************/

// Perform FGP primal reprojection
// All values >= 0, sum of all values = 1
// Called once for all layers
// Implements Michelot, "A finite algorithm for finding the projection
// of a point onto the canonical simplex of R^n", 1986
__global__ void simplex_reprojection_device( int W, int H, int N, int G, float *u )
{
  // Global thread index
  int ox = CBS_SMALL * blockIdx.x + threadIdx.x;
  int oy = CBS_SMALL * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;

  // Temporary memory to store index map and projected vector
  extern __shared__ float shared[];
  float *x = shared + G * ( threadIdx.x + CBS_SMALL*threadIdx.y );
  int *I = (int*)(shared + G*CBS_SMALL*CBS_SMALL + G * ( threadIdx.x + CBS_SMALL*threadIdx.y ));
  float sum = 0.0f;
  float uv = 0.0f;
  for ( int g=0; g<G; g++ ) {
    uv = u[o+N*g];
    sum += uv;
    x[g] = uv;
    I[g] = g;
  }

  int M = G-1;
  int Mold;
  do {
    int idx = 0;
    Mold = M;
    float sub = (sum - 1.0f) / (M+1);
    sum = 0.0f;
    while ( idx <= M ) {
      int i = I[idx];
      uv = x[i] - sub;
      if ( uv >= 0.0f ) {
	x[i] = uv;
	sum += uv;
	idx++;
      }
      else {
	x[i] = 0.0f;
	I[idx] = I[M];
	M--;
      }
    }
  } while ( Mold != M );

  for ( int g=0; g<G; g++ ) {
    u[o+N*g] = x[g];
  }
}



// Reprojection onto allowed subset: \sum u = 1, \sum v = 1.
// W,H: Array size
// G: Vector length
bool coco::simplex_reprojection( size_t W, size_t H, size_t G, float *u )
{
  // Use smaller block size (else shared mem is exceeded)
  dim3 dimBlock(CBS_SMALL, CBS_SMALL);
  dim3 dimGrid(W / dimBlock.x + (W%CBS_SMALL)==0 ? 0 : 1,
	       H / dimBlock.y + (H%CBS_SMALL)==0 ? 0 : 1 );
  size_t dimShared = sizeof(float) * CBS_SMALL * CBS_SMALL * G * 2;
  if ( dimShared >= 16384 ) {
    ERROR( "Allowed size of shared memory exceeded (" << dimShared << ")" << std::endl );
    assert( false );
  }

  // Kernel call
  simplex_reprojection_device<<< dimGrid, dimBlock, dimShared >>>
    ( W, H, W*H, G, u );
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}
