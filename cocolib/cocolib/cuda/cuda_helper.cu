/* -*-c++-*- */
/** \file cuda_helper.cu

    Some helper functions for CUDA texture bindings and memcopy operations.

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

#include <stdio.h>

#include "cuda_interface.h"
#include "cuda_helper.h"

#include "../defs.h"

using namespace std;

// Bind a matrix residing in standard allocation in system memory
// to a texture descriptor.
// New GPU memory is allocated if arr==NULL,
// otherwise existing GPU memory is overwritten.
bool coco::cuda_bind_matrix_to_texture( int W, int H, void *m,
					cudaArray* &arr,
					textureReference &texRef,
					cudaChannelFormatDesc *desc, // default is 1 x float
					const cudaTextureFilterMode fm,
					const cudaTextureAddressMode am )
{
  if ( W<=0 || H<=0 ) {
    return false;
  }

  if ( desc==NULL ) {
    static cudaChannelFormatDesc d_default = 
      cudaCreateChannelDesc(32,0,0,0, cudaChannelFormatKindFloat);
    desc = &d_default;
  }

  if ( arr==NULL ) {
    CUDA_SAFE_CALL( cudaMallocArray(&arr, desc, W,H ));
  }
  else {
    // TODO: Make sure array is of correct size.
  }

  int bpp = (desc->x + desc->y + desc->z + desc->w) / 8;
  CUDA_SAFE_CALL( cudaMemcpyToArray( arr,0,0, m, bpp*W*H, cudaMemcpyHostToDevice ));

  texRef.filterMode = fm;
  texRef.addressMode[0] = am;
  texRef.addressMode[1] = am;
  texRef.addressMode[2] = am;
  texRef.normalized = 0;
  CUDA_SAFE_CALL( cudaBindTextureToArray( &texRef, arr, desc ));

  return true;
}


// Bind a 3D array residing in standard allocation in system memory
// to a texture descriptor.
// New GPU memory is allocated if arr==NULL,
// otherwise existing GPU memory is overwritten.
bool coco::cuda_bind_array_to_texture( int X, int Y, int Z, void *data,
				       cudaArray* &arr,
				       textureReference &texRef,
				       cudaChannelFormatDesc *desc,
				       const cudaTextureFilterMode fm,
				       const cudaTextureAddressMode am )
{
  if ( X<=0 || Y<=0 || Z<=0 ) {
    return false;
  }
  cudaExtent e;
  e.width  = X;
  e.height = Y;
  e.depth  = Z;

  if ( desc==NULL ) {
    static cudaChannelFormatDesc d_default = 
      cudaCreateChannelDesc(32,0,0,0, cudaChannelFormatKindFloat);
    desc = &d_default;
  }

  if ( arr==NULL ) {
    CUDA_SAFE_CALL( cudaMalloc3DArray(&arr, desc, e ));
  }
  else {
    // TODO: Make sure array is of correct size.
  }

  // Copy data to array
  if (!cuda_memcpy_to_array( X,Y,Z, arr, data, desc )) {
    assert( false );
    return false;
  }

  // Bind texture
  texRef.filterMode = fm;
  texRef.addressMode[0] = am;
  texRef.addressMode[1] = am;
  texRef.addressMode[2] = am;
  texRef.normalized = 0;
  CUDA_SAFE_CALL( cudaBindTextureToArray( &texRef, arr, desc ));
  return true;
}


// Transfer data to a 3D array
bool coco::cuda_memcpy_to_array( int X, int Y, int Z,
				 cudaArray *dest, void *source,
				 cudaChannelFormatDesc *desc )
{
  if ( desc==NULL ) {
    static cudaChannelFormatDesc d_default = 
      cudaCreateChannelDesc(32,0,0,0, cudaChannelFormatKindFloat);
    desc = &d_default;
  }

  int bpp = (desc->x + desc->y + desc->z + desc->w) / 8;
  cudaMemcpy3DParms p3d;
  memset( &p3d, 0, sizeof(cudaMemcpy3DParms));
  p3d.srcPtr.ptr = source;
  p3d.srcPtr.pitch = bpp * X;
  p3d.srcPtr.xsize = bpp * X;
  p3d.srcPtr.ysize = Y;
  p3d.dstArray = dest;
  p3d.extent.width = X;
  p3d.extent.height = Y;
  p3d.extent.depth = Z;
  p3d.kind = cudaMemcpyHostToDevice;
  CUDA_SAFE_CALL( cudaMemcpy3D( &p3d ));
  return true;
}


// Copy data back from a 3D array
bool coco::cuda_memcpy_from_array( int X, int Y, int Z,
				   void *dest, cudaArray *src,
				   cudaChannelFormatDesc *desc )
{
  // Not yet implemented.
  assert( false );
  return false;
}




/// Bind RGB image given as a tensor (standard format for Brox library)
/// 1-4 input channels supported, but texture output always has 4.
/*
bool coco::cuda_bind_image_to_texture( const CTensor<float> &I, cudaArray* &arr, textureReference &texRef,
const cudaTextureFilterMode fm,
const cudaTextureAddressMode am )
{
  // Sizes must be positive
  int W = I.xSize();
  int H = I.ySize();
  if ( W<=0 || H<=0 ) {
    assert( false );
    return false;
  }
  // Up to four channels supported
  int depth = I.zSize();
  if ( depth<1 || depth>4 ) {
    assert( false );
    return false;
  }

  // Temporary copy to restructure crap mem layout
  float *temp = new float[ W*H*4 ];
  float *d = temp;
  for ( int y=0; y<H; y++ ) {
    for ( int x=0; x<W; x++ ) {
      *(d++) = I( x,y,0 );
      if ( depth>1 ) {
        *(d++) = I( x,y,1 );
      }
      else {
        *(d++) = 0.0f;
      }
      if ( depth>2 ) {
        *(d++) = I( x,y,2 );
      }
      else {
        *(d++) = 0.0f;
      }
      if ( depth>3 ) {
        *(d++) = I( x,y,3 );
      }
      else {
        *(d++) = 0.0f;
      }
    }
  }

  // Format descriptor checks how many channels the image has
  cudaChannelFormatDesc desc = cudaCreateChannelDesc(32,32,32,32, cudaChannelFormatKindFloat);
  cuda_bind_matrix_to_texture( W,H,temp, arr, texRef, &desc, fm, am );

  // Don't delete memory before copy is complete
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  delete[] temp;

  return true;
}
*/



/// Bind RGB image given as a tensor (standard format for Brox library)
/// 1-4 input channels supported, but texture output always has 4.
bool coco::cuda_bind_image_to_texture( const float *data, int W, int H, cudaArray* &arr, textureReference &texRef,
				       const cudaTextureFilterMode fm,
				       const cudaTextureAddressMode am )
{
  // Sizes must be positive
  if ( W<=0 || H<=0 ) {
    assert( false );
    return false;
  }

  // Format descriptor checks how many channels the image has
  cudaChannelFormatDesc desc = cudaCreateChannelDesc(32,32,32,32, cudaChannelFormatKindFloat);
  cuda_bind_matrix_to_texture( W,H, (void*)data, arr, texRef, &desc, fm, am );

  // Don't delete memory before copy is complete
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  return true;
}


// write cuda array to image file, unsigned version
bool coco::write_test_image_unsigned( size_t W, size_t H, float *data, const string &spattern, int hint, bool normalize )
{
  char str[500];
  sprintf( str, spattern.c_str(), hint );
  size_t N = W*H;
  assert( N>0 );
  float *cpu = new float[N];
  CUDA_SAFE_CALL( cudaMemcpy( cpu, data, sizeof(float) * N, cudaMemcpyDeviceToHost ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  gsl_matrix *M = gsl_matrix_alloc( H,W );
  for ( size_t i=0; i<N; i++ ) {
    M->data[i] = cpu[i];
  }
  delete[] cpu;
  gsl_image *I = gsl_image_alloc( W,H );
  gsl_image_from_matrix( I, M );
  if ( normalize ) {
    gsl_image_normalize( I );
  }
  gsl_image_save( str, I );
  gsl_image_free( I );
  return true;
}


// write cuda array to image file, signed version
bool coco::write_test_image_signed( size_t W, size_t H, float *data, const string &spattern, int hint, bool normalize )
{
  char str[500];
  sprintf( str, spattern.c_str(), hint );
  size_t N = W*H;
  assert( N>0 );
  float *cpu = new float[N];
  CUDA_SAFE_CALL( cudaMemcpy( cpu, data, sizeof(float) * N, cudaMemcpyDeviceToHost ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  gsl_matrix *M = gsl_matrix_alloc( H,W );
  for ( size_t i=0; i<N; i++ ) {
    M->data[i] = cpu[i];
  }
  delete[] cpu;
  coco::gsl_image *I = gsl_image_alloc( W,H );
  gsl_image_from_signed_matrix( I, M );
  if ( normalize ) {
    gsl_image_normalize( I );
  }
  gsl_image_save( str, I );
  gsl_image_free( I );
  return true;
}

inline bool endianness()
{
    const int x = 1;
    return ((unsigned char *)&x)[0] ? false : true;
}

void reverse_buffer(std::vector<float> &ptr, int w, int h, int depth) {
  for (int i=0; i< h/2; ++i) {
    std::swap_ranges(ptr.begin() + i*w*depth, ptr.begin()+ i*w*depth + w*depth, ptr.end() - w*depth - i*w*depth);
  }
}

// write cuda array to image file, signed version
bool coco::write_pfm_image_signed( size_t W, size_t H, float *cuda_data, const string &spattern, int hint)
{
  char str[500];
  sprintf( str, spattern.c_str(), hint );
  std::FILE *const nfile = std::fopen(str, "wb");
  
  size_t N = W*H;
  assert( N>0 );
  std::vector<float> cpu(N);
  CUDA_SAFE_CALL( cudaMemcpy( cpu.data(), cuda_data, sizeof(float) * N, cudaMemcpyDeviceToHost ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // reverse buffer:
  // cuda image 0,0 is at the TOP left corner of the image
  // pfm format has the 0,0 at the BOTTOM left corner of the image
  // lines must be swapped
  reverse_buffer(cpu, W,H,1);

  std::fprintf(nfile, "P%c\n%lu %lu\n%d.0\n", 'f', W, H, endianness() ? 1 : -1);
  std::fwrite(cpu.data(), sizeof(float), N, nfile);
  
  fclose(nfile);
  
  return true;
}

// write cuda array to image file, signed version
bool coco::write_pfm_image_signed( size_t W, size_t H, float *r, float *g, float *b, const string &spattern, int hint)
{
  char str[500];
  sprintf( str, spattern.c_str(), hint );
  std::FILE *const nfile = std::fopen(str, "wb");

  size_t N = W*H;
  assert( N>0 );
  std::vector<float> cpu_r(N);
  std::vector<float> cpu_g(N);
  std::vector<float> cpu_b(N);

  CUDA_SAFE_CALL( cudaMemcpy( cpu_r.data(), r, sizeof(float) * N, cudaMemcpyDeviceToHost ));
  CUDA_SAFE_CALL( cudaMemcpy( cpu_g.data(), g, sizeof(float) * N, cudaMemcpyDeviceToHost ));
  CUDA_SAFE_CALL( cudaMemcpy( cpu_b.data(), b, sizeof(float) * N, cudaMemcpyDeviceToHost ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // reverse buffer:
  // cuda image 0,0 is at the TOP left corner of the image
  // pfm format has the 0,0 at the BOTTOM left corner of the image
  // lines must be swapped
  reverse_buffer(cpu_r, W,H,1);
  reverse_buffer(cpu_g, W,H,1);
  reverse_buffer(cpu_b, W,H,1);

  std::fprintf(nfile, "P%c\n%lu %lu\n%d.0\n", 'F', W, H, endianness() ? 1 : -1);
  for(size_t i=0; i<N; ++i) {
    std::fwrite(&(cpu_r[i]), sizeof(float), 1, nfile);
    std::fwrite(&(cpu_g[i]), sizeof(float), 1, nfile);
    std::fwrite(&(cpu_b[i]), sizeof(float), 1, nfile);
  }

  fclose(nfile);

  return true;
}


// write cuda array to image file, rgb version
bool coco::write_test_image_rgb( size_t W, size_t H, const float *r, const float *g, const float *b,
				 const string &pattern, int hint, bool normalize )
{
  char str[500];
  sprintf( str, pattern.c_str(), hint );
  size_t N = W*H;
  assert( N>0 );

  gsl_image *I = gsl_image_alloc( W,H );
  cuda_memcpy( I->_r, r );
  cuda_memcpy( I->_g, g );
  cuda_memcpy( I->_b, b );
  if ( normalize ) {
    gsl_image_normalize( I );
  }
  gsl_image_save( str, I );
  gsl_image_free( I );
  return true;
}




// write cuda array to image file, unsigned version
bool coco::write_test_image_unsigned( size_t W, size_t H, int *data, const string &spattern, int hint, bool normalize )
{
  char str[500];
  sprintf( str, spattern.c_str(), hint );
  size_t N = W*H;
  assert( N>0 );
  int *cpu = new int[N];
  CUDA_SAFE_CALL( cudaMemcpy( cpu, data, sizeof(int) * N, cudaMemcpyDeviceToHost ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  gsl_matrix *M = gsl_matrix_alloc( H,W );
  for ( size_t i=0; i<N; i++ ) {
    M->data[i] = cpu[i];
  }
  delete[] cpu;
  gsl_image *I = gsl_image_alloc( W,H );
  gsl_image_from_matrix( I, M );
  if ( normalize ) {
    gsl_image_normalize( I );
  }
  gsl_image_save( str, I );
  gsl_image_free( I );
  return true;
}


// write cuda array to image file, signed version
bool coco::write_test_image_signed( size_t W, size_t H, int *data, const string &spattern, int hint, bool normalize )
{
  char str[500];
  sprintf( str, spattern.c_str(), hint );
  size_t N = W*H;
  assert( N>0 );
  int *cpu = new int[N];
  CUDA_SAFE_CALL( cudaMemcpy( cpu, data, sizeof(int) * N, cudaMemcpyDeviceToHost ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  gsl_matrix *M = gsl_matrix_alloc( H,W );
  for ( size_t i=0; i<N; i++ ) {
    M->data[i] = cpu[i];
  }
  delete[] cpu;
  coco::gsl_image *I = gsl_image_alloc( W,H );
  gsl_image_from_signed_matrix( I, M );
  if ( normalize ) {
    gsl_image_normalize( I );
  }
  gsl_image_save( str, I );
  gsl_image_free( I );
  return true;
}

// write cuda array to image file, bool version
bool coco::write_test_image_bool( size_t W, size_t H, bool *data, const string &spattern, int hint, bool normalize )
{
  char str[500];
  sprintf( str, spattern.c_str(), hint );
  size_t N = W*H;
  assert( N>0 );
  bool *cpu = new bool[N];
  CUDA_SAFE_CALL( cudaMemcpy( cpu, data, sizeof(bool) * N, cudaMemcpyDeviceToHost ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  gsl_matrix *M = gsl_matrix_alloc( H,W );
  for ( size_t i=0; i<N; i++ ) {
    M->data[i] = cpu[i];
  }
  delete[] cpu;
  coco::gsl_image *I = gsl_image_alloc( W,H );
  gsl_image_from_signed_matrix( I, M );
  if ( normalize ) {
    gsl_image_normalize( I );
  }
  gsl_image_save( str, I );
  gsl_image_free( I );
  return true;
}



// Copy data from gsl matrix to CUDA float array
bool coco::cuda_memcpy( float* gpu_target, const gsl_matrix *M )
{
  size_t N = M->size1 * M->size2;
  float *buf = make_float_buffer( M->data, N );
  CUDA_SAFE_CALL( cudaMemcpy( gpu_target, buf, N*sizeof(float), cudaMemcpyHostToDevice ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  delete[] buf;
  return true;
}

// Copy data from gsl matrix to CUDA float array
bool coco::cuda_memcpy( float* gpu_target, const double *d, size_t N )
{
  float *buf = make_float_buffer( d, N );
  CUDA_SAFE_CALL( cudaMemcpy( gpu_target, buf, N*sizeof(float), cudaMemcpyHostToDevice ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  delete[] buf;
  return true;
}



// Copy data from CUDA to gsl matrix
bool coco::cuda_memcpy( gsl_matrix *M, const float* gpu_source )
{
  size_t N = M->size1 * M->size2;
  float *buf = new float[N];
  CUDA_SAFE_CALL( cudaMemcpy( buf, gpu_source, N*sizeof(float), cudaMemcpyDeviceToHost ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  for ( size_t i=0; i<N; i++ ) {
    M->data[i] = buf[i];
  }
  delete[] buf;
  return true;
}

// Copy data from gsl vector to CUDA float array
bool coco::cuda_memcpy( float* gpu_target, const gsl_vector *v )
{
  size_t N = v->size;
  float *buf = make_float_buffer( v->data, N );
  CUDA_SAFE_CALL( cudaMemcpy( gpu_target, buf, N*sizeof(float), cudaMemcpyHostToDevice ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  delete[] buf;
  return true;
}

// Copy data from CUDA to gsl vector
bool coco::cuda_memcpy( gsl_vector *v, const float* gpu_source )
{
  size_t N = v->size;
  float *buf = new float[N];
  CUDA_SAFE_CALL( cudaMemcpy( buf, gpu_source, N*sizeof(float), cudaMemcpyDeviceToHost ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  for ( size_t i=0; i<N; i++ ) {
    v->data[i] = buf[i];
  }
  delete[] buf;
  return true;
}






// Float buffer from double array
float *coco::make_float_buffer( const double *data, size_t N )
{
  assert( data != NULL );
  assert( N>0 );
  float *buf = new float[N];
  for ( size_t i=0; i<N; i++ ) {
    buf[i] = (float)data[i];
  }
  return buf;
}

// Interleaved float buffer from double array
float *coco::make_interleaved_float_buffer( std::vector<double*> &data, size_t N )
{
  assert( N>0 );
  size_t K = data.size();
  float *buf = new float[N*K];
  size_t index = 0;
  for ( size_t i=0; i<N; i++ ) {
    for ( size_t k=0; k<K; k++ ) {
      buf[index++] = (float)data[k][i];
    }
  }
  return buf;
}



// Returns default grid layout
bool coco::cuda_default_grid( size_t W, size_t H, dim3 &dimGrid, dim3 &dimBlock )
{
  dimBlock = dim3( cuda_default_block_size_x(),
		   cuda_default_block_size_y() );

  size_t blocks_w = W / dimBlock.x;
  if ( W % dimBlock.x != 0 ) {
    blocks_w += 1;
  }
  size_t blocks_h = H / dimBlock.y;
  if ( H % dimBlock.y != 0 ) {
    blocks_h += 1;
  }
  dimGrid = dim3(blocks_w, blocks_h);

  if ( blocks_w==0 || blocks_h==0 ) {
    assert( false );
    return false;
  }

  return true;
}



/********************************************************
  Memory allocation
*********************************************************/

// standard float array
float *coco::cuda_alloc_floats( size_t nfloats )
{
  float *tmp = NULL;
  CUDA_SAFE_CALL( cudaMalloc( &tmp, nfloats * sizeof(float) ));
  assert( tmp != NULL );
  return tmp;
}

// free memory
/*
bool cuda_free( void* &data )
{
  if ( data != NULL ) {
    CUDA_SAFE_CALL( cudaFree( data ));
    data = NULL;
    return true;
  }
  return false;
}
*/






#ifdef CUDA_DOUBLE

// Copy data from gsl matrix to CUDA double array
bool coco::cuda_memcpy( double* gpu_target, gsl_matrix *M )
{
  size_t N = M->size1 * M->size2;
  CUDA_SAFE_CALL( cudaMemcpy( gpu_target, M->data, N*sizeof(double), cudaMemcpyHostToDevice ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}

// Copy data from gsl matrix to CUDA double array
bool coco::cuda_memcpy( double* gpu_target, double *d, size_t N )
{
  CUDA_SAFE_CALL( cudaMemcpy( gpu_target, d, N*sizeof(double), cudaMemcpyHostToDevice ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}



// Copy data from CUDA to gsl matrix
bool coco::cuda_memcpy( gsl_matrix *M, double* gpu_source )
{
  size_t N = M->size1 * M->size2;
  CUDA_SAFE_CALL( cudaMemcpy( M->data, gpu_source, N*sizeof(double), cudaMemcpyDeviceToHost ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}

// Copy data from gsl vector to CUDA double array
bool coco::cuda_memcpy( double* gpu_target, gsl_vector *v )
{
  size_t N = v->size;
  CUDA_SAFE_CALL( cudaMemcpy( gpu_target, v->data, N*sizeof(double), cudaMemcpyHostToDevice ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}

// Copy data from CUDA to gsl vector
bool coco::cuda_memcpy( gsl_vector *v, double* gpu_source )
{
  size_t N = v->size;
  CUDA_SAFE_CALL( cudaMemcpy( v->data, gpu_source, N*sizeof(double), cudaMemcpyDeviceToHost ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}



#endif
