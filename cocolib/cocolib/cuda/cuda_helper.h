/** \file cuda_helper.h

    Some helper functions for CUDA,
    texture binding and memory operations.

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

#ifndef __CUDA_HELPER_H
#define __CUDA_HELPER_H

#include "cuda_interface.h"
#include "cuda_functions.h"


/********************************************************
  Useful helper functions and macros for use with CUDA
*********************************************************/

/********************************************************
  Basics
*********************************************************/

#define CUDA_SAFE_CALL(c)\
  if ( c != cudaSuccess ) {\
    printf( "CUDA error: %s, line %d\n",\
            cudaGetErrorString( cudaGetLastError() ),\
            __LINE__);\
    assert( false );\
  }



// Return types for CUDA objects
typedef unsigned char* gpu_uchar_array;
typedef float*         gpu_float_array;


////////////////////////////////////////////////////////////////////////////////
// Common host and device functions (nVidia SDK)
////////////////////////////////////////////////////////////////////////////////
//Round a / b to nearest higher integer value
inline int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Round a / b to nearest lower integer value
inline int iDivDown(int a, int b){
    return a / b;
}

//Align a to nearest higher multiple of b
inline int iAlignUp(int a, int b){
    return (a % b != 0) ?  (a - a % b + b) : a;
}

//Align a to nearest lower multiple of b
inline int iAlignDown(int a, int b){
    return a - a % b;
}

//24-bit multiplication is faster on G80,
//but we must be sure to multiply integers
//only within [-8M, 8M - 1] range
#define IMUL(a, b) __mul24(a, b)




/********************************************************
  Handling of constants
*********************************************************/

// Set __constant__ or __global__ float variable
#define CUDA_SET_FLOAT(a,b) CUDA_SAFE_CALL( cudaMemcpyToSymbol( a, &b, sizeof(float), 0, cudaMemcpyHostToDevice ))
// Set __constant__ or __global__ float variable
#define CUDA_SET_FLOAT3(a,b) CUDA_SAFE_CALL( cudaMemcpyToSymbol( a, &b, sizeof(float3), 0, cudaMemcpyHostToDevice ))
// Set __constant__ or __global__ int variable
#define CUDA_SET_INT(a,b) CUDA_SAFE_CALL( cudaMemcpyToSymbol( a, &b, sizeof(int), 0, cudaMemcpyHostToDevice ))
// Set __constant__ or __global__ matrix variable
#define CUDA_SET_MATRIX(Z0,Z1,Z2,Z3, m)                                 \
  {                                                                     \
  float4 r;                                                             \
  r.x=m[0][0]; r.y=m[0][1]; r.z=m[0][2]; r.w=m[0][3];			\
  CUDA_SAFE_CALL( cudaMemcpyToSymbol( Z0,&r, sizeof(float4), 0, cudaMemcpyHostToDevice )); \
  r.x=m[1][0]; r.y=m[1][1]; r.z=m[1][2]; r.w=m[1][3];			\
  CUDA_SAFE_CALL( cudaMemcpyToSymbol( Z1,&r, sizeof(float4), 0, cudaMemcpyHostToDevice )); \
  r.x=m[2][0]; r.y=m[2][1]; r.z=m[2][2]; r.w=m[2][3];			\
  CUDA_SAFE_CALL( cudaMemcpyToSymbol( Z2,&r, sizeof(float4), 0, cudaMemcpyHostToDevice )); \
  r.x=m[3][0]; r.y=m[3][1]; r.z=m[3][2]; r.w=m[3][3];			\
  CUDA_SAFE_CALL( cudaMemcpyToSymbol( Z3,&r, sizeof(float4), 0, cudaMemcpyHostToDevice )); \
  }\



/********************************************************
  Handling of cameras
*********************************************************/

// Set __constant__ or __global__ matrix variables
// m0 ... m2 are float4 defining the three rows
// f is a float4 defining internal parameters
// c is the camera
#define CUDA_SET_CAMERA(m0,m1,m2,f, c)					\
  {                                                                     \
    float4 r;                                                           \
    Mat44f &b = c._matModelviewProjection;				\
    r.x=b[0][0]; r.y=b[0][1]; r.z=b[0][2]; r.w=b[0][3];			\
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( m0,&r, sizeof(float4), 0, cudaMemcpyHostToDevice )); \
    r.x=b[1][0]; r.y=b[1][1]; r.z=b[1][2]; r.w=b[1][3];			\
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( m1,&r, sizeof(float4), 0, cudaMemcpyHostToDevice )); \
    r.x=b[2][0]; r.y=b[2][1]; r.z=b[2][2]; r.w=b[2][3];			\
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( m2,&r, sizeof(float4), 0, cudaMemcpyHostToDevice )); \
    float4 a;								\
    a.x = c._fx; a.y = c._fy; a.z = c._cx; a.w = c._cy;			\
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( f,&a, sizeof(float4), 0, cudaMemcpyHostToDevice )); \
  }



/********************************************************
  Device macros
*********************************************************/

// Compute dot product between two float4
#define DOT4(a,b) (a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w)
#define DOT3(a,b) (a.x*b.x + a.y*b.y + a.z*b.z)

// Perform a camera projection in a kernel
// Camera parameters as above.
// v is the float4 input vertex
// p is a float3 for the output 2D point
#define CUDA_PROJECT(m0,m1,m2,f, v,p) \
  p.x = DOT4(m0, v);		      \
  p.y = DOT4(m1, v);		      \
  p.z = DOT4(m2, v);		      \
  if ( p.z != 0.0f ) {		      \
    p.x /= p.z;			      \
    p.y /= p.z;			      \
  }				      \
  p.x = p.x * f.x + f.z;	      \
  p.y = p.y * f.y + f.w;	      \


#define CUDA_TRANSFORM4(Z1,Z2,Z3,Z4, in,out)    \
  out.x = DOT4(Z1,in);                          \
  out.y = DOT4(Z2,in);                          \
  out.z = DOT4(Z3,in);                          \
  out.w = DOT4(Z4,in);                          \






namespace coco {

/********************************************************
  Memory allocation
*********************************************************/

// standard float array
float *cuda_alloc_floats( size_t nfloats );
// free memory
template<class T> bool cuda_free( T* &data ) {
  if ( data != NULL ) {
    CUDA_SAFE_CALL( cudaFree( data ));
    data = NULL;
    return true;
  }
  return false;
}


/********************************************************
  Texture handling
*********************************************************/

// Bind a matrix residing in standard allocation in system memory
// to a texture descriptor.
// New GPU memory is allocated if arr==NULL,
// otherwise existing GPU memory is overwritten.
bool cuda_bind_matrix_to_texture( int W, int H, void *m,
                                  cudaArray* &arr,
                                  textureReference &texRef,
                                  cudaChannelFormatDesc *desc=NULL, // default is 1 x float
                                  const cudaTextureFilterMode = cudaFilterModePoint,
                                  const cudaTextureAddressMode = cudaAddressModeClamp );

// Bind a 3D array residing in standard allocation in system memory
// to a texture descriptor.
// New GPU memory is allocated if arr==NULL,
// otherwise existing GPU memory is overwritten.
bool cuda_bind_array_to_texture( int X, int Y, int Z, void *m,
                                 cudaArray* &arr,
                                 textureReference &texRef,
                                 cudaChannelFormatDesc *desc=NULL, // default is 1 x float
                                 const cudaTextureFilterMode = cudaFilterModePoint,
                                 const cudaTextureAddressMode = cudaAddressModeClamp );

// Bind RGB image given as a tensor (standard format for Brox library)
// New GPU memory is allocated if arr==NULL,
// otherwise existing GPU memory is overwritten.
/*
bool cuda_bind_image_to_texture( const CTensor<float> &I, cudaArray* &arr,
                                 textureReference &texRef,
                                 const cudaTextureFilterMode = cudaFilterModeLinear,
                                 const cudaTextureAddressMode = cudaAddressModeClamp );
*/
bool cuda_bind_image_to_texture( const coco::gsl_image *I, cudaArray* &arr,
                                 textureReference &texRef,
                                 const cudaTextureFilterMode = cudaFilterModeLinear,
                                 const cudaTextureAddressMode = cudaAddressModeClamp );

bool cuda_bind_image_to_texture( const float *data, int W, int H,
                                 cudaArray* &arr,
                                 textureReference &texRef,
                                 const cudaTextureFilterMode = cudaFilterModeLinear,
                                 const cudaTextureAddressMode = cudaAddressModeClamp );



/********************************************************
  MemCpy wrappers (mostly 3D as this sucks)
*********************************************************/

// Transfer data to a 3D array
bool cuda_memcpy_to_array( int X, int Y, int Z,
                           cudaArray *dest, void *source,
                           cudaChannelFormatDesc *desc=NULL ); // default is 1x float

// Copy data back from a 3D array
bool cuda_memcpy_from_array( int X, int Y, int Z,
                             void *dest, cudaArray *src,
                             cudaChannelFormatDesc *desc=NULL ); // default is 1x float


// Returns default grid layout
bool cuda_default_grid( size_t W, size_t H, dim3 &dimGrid, dim3 &dimBlock );

} // namespace
#endif
