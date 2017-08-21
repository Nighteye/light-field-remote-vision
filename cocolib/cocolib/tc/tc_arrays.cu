/* -*-c++-*- */
/** \file tc_arrays.cu
   Workspace handling and access code - array helper functions

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

#include <iostream>
#include "tc.h"
#include "tc.cuh"
#include "tc_arrays.cuh"

#include "../defs.h"
#include "../cuda/cuda_helper.h"


// Array handling functions
bool coco::cvl_alloc_array( tc_workspace* w, std::vector<stcflt*> &V )
{
  assert( V.size() == 0 );
  size_t N = w->_N;
  stcflt *vbase = NULL;
  CUDA_SAFE_CALL( cudaMalloc( &vbase, w->_Nf * N*N*N*N ));

  size_t i=0;
  for ( size_t y=0; y<N*N; y++ ) {
    for ( size_t z=0; z<N*N; z++ ) {

      stcflt *v = vbase + i * w->_W * w->_H;
      V.push_back( v );
      i++;

    }
  }

  return true;
}


bool coco::cvl_copy_array( tc_workspace* w, std::vector<stcflt*> &dest, std::vector<stcflt*> &src )
{
  size_t N = w->_N;
  assert( dest.size() == N*N*N*N );
  assert( src.size() == N*N*N*N );
  int N2 = w->_N2;
  for ( int y0=-N2; y0<=N2; y0++ ) {
    for ( int y1=-N2; y1<=N2; y1++ ) {

      for ( int z0=-N2; z0<=N2; z0++ ) {
	for ( int z1=-N2; z1<=N2; z1++ ) {
	  stcflt *d = cvl_get_3d_variable( w, dest, y0,y1, z0,z1 );
	  stcflt *s = cvl_get_3d_variable( w, src, y0,y1, z0,z1 );
	  CUDA_SAFE_CALL( cudaMemcpy( d, s, w->_Nf, cudaMemcpyDeviceToDevice ));
	}
      }
    }
  }

  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}

bool coco::cvl_copy_array_pointers( tc_workspace* w, std::vector<stcflt*> &dest, std::vector<stcflt*> &src )
{
  size_t N = w->_N;
  assert( dest.size() == 0 );
  assert( src.size() == N*N*N*N );
  dest.insert( dest.begin(), src.begin(), src.end() );
  return true;
}


bool coco::cvl_clear_array( tc_workspace* w, std::vector<stcflt*> &V )
{
  size_t N = w->_N;
  assert( V.size() == N*N*N*N );
  int N2 = w->_N2;
  for ( int y0=-N2; y0<=N2; y0++ ) {
    for ( int y1=-N2; y1<=N2; y1++ ) {

      for ( int z0=-N2; z0<=N2; z0++ ) {
	for ( int z1=-N2; z1<=N2; z1++ ) {

	  stcflt *v = cvl_get_3d_variable( w,V, y0,y1, z0,z1 );
	  CUDA_SAFE_CALL( cudaMemset( v, 0, w->_Nf ));

	}
      }
    }
  }

  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}



bool coco::cvl_free_array( tc_workspace* w, std::vector<stcflt*> &V )
{
  size_t N = w->_N;
  assert( V.size() == N*N*N*N );
  CUDA_SAFE_CALL( cudaFree( V[0] ));
  return true;
}


// y,z index the offset from the center ( -(N-1)/2 .... (N-1)/2 )
stcflt *coco::cvl_get_3d_variable( tc_workspace* w, std::vector<stcflt*> &V, int y0, int y1, int z0, int z1 )
{
  assert( y0>=-w->_N2 && y0<=w->_N2 );
  assert( z0>=-w->_N2 && z0<=w->_N2 );
  assert( y1>=-w->_N2 && y1<=w->_N2 );
  assert( z1>=-w->_N2 && z1<=w->_N2 );
  y0 += w->_N2;
  y1 += w->_N2;
  z0 += w->_N2;
  z1 += w->_N2;
  size_t N = w->_N;
  size_t index = (size_t)(y0 + N*( y1 + N*( z0 + N*z1 )));
  assert( index < V.size() );
  return V[index];
}




// Array handling functions
bool coco::cvl_cpu_alloc_array( tc_workspace* w, std::vector<stcflt*> &V )
{
  assert( V.size() == 0 );
  size_t N = w->_N;
  for ( size_t y=0; y<N*N; y++ ) {
    for ( size_t z=0; z<N*N; z++ ) {
      stcflt *v = new stcflt[ w->_W * w->_H ];
      if ( v == NULL ) {
	assert( false );
	return false;
      }
      V.push_back( v );
    }
  }
  return true;
}



bool coco::cvl_cpu_free_array( tc_workspace* w, std::vector<stcflt*> &V )
{
  size_t N = w->_N;
  assert( V.size() == N*N*N*N );
  int N2 = w->_N2;
  for ( int y0=-N2; y0<=N2; y0++ ) {
    for ( int y1=-N2; y1<=N2; y1++ ) {

      for ( int z0=-N2; z0<=N2; z0++ ) {
	for ( int z1=-N2; z1<=N2; z1++ ) {
	  stcflt *v = cvl_get_3d_variable( w, V, y0,y1, z0,z1 );
	  if ( v==NULL ) {
	    assert( false );
	  }
	  delete[] v;
	}
      }
    }
  }
  V.clear();
  return true;
}



bool coco::cvl_cpu_clear_array( tc_workspace* w, std::vector<stcflt*> &V )
{
  size_t N = w->_N;
  assert( V.size() == N*N*N*N );
  int N2 = w->_N2;
  for ( int y0=-N2; y0<=N2; y0++ ) {
    for ( int y1=-N2; y1<=N2; y1++ ) {

      for ( int z0=-N2; z0<=N2; z0++ ) {
	for ( int z1=-N2; z1<=N2; z1++ ) {

	  stcflt *v = cvl_get_3d_variable( w,V, y0,y1, z0,z1 );
	  memset( v, 0, w->_Nf );

	}
      }
    }
  }

  return true;
}




bool coco::cvl_copy_array_to_cpu( tc_workspace* w, std::vector<stcflt*> &cpu, std::vector<stcflt*> &gpu  )
{
  size_t N = w->_N;
  assert( cpu.size() == N*N*N*N );
  assert( gpu.size() == N*N*N*N );
  int N2 = w->_N2;
  for ( int y0=-N2; y0<=N2; y0++ ) {
    for ( int y1=-N2; y1<=N2; y1++ ) {

      for ( int z0=-N2; z0<=N2; z0++ ) {
	for ( int z1=-N2; z1<=N2; z1++ ) {

	  stcflt *v_cpu = cvl_get_3d_variable( w, cpu, y0,y1, z0,z1 );
	  stcflt *v_gpu = cvl_get_3d_variable( w, gpu, y0,y1, z0,z1 );
	  CUDA_SAFE_CALL( cudaMemcpy( v_cpu, v_gpu, w->_Nf, cudaMemcpyDeviceToHost ));

	}
      }
    }
  }

  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}


bool coco::cvl_copy_array_to_gpu( tc_workspace* w, std::vector<stcflt*> &gpu, std::vector<stcflt*> &cpu  )
{
  size_t N = w->_N;
  assert( cpu.size() == N*N*N*N );
  assert( gpu.size() == N*N*N*N );
  int N2 = w->_N2;
  for ( int y0=-N2; y0<=N2; y0++ ) {
    for ( int y1=-N2; y1<=N2; y1++ ) {

      for ( int z0=-N2; z0<=N2; z0++ ) {
	for ( int z1=-N2; z1<=N2; z1++ ) {

	  stcflt *v_cpu = cvl_get_3d_variable( w, cpu, y0,y1, z0,z1 );
	  stcflt *v_gpu = cvl_get_3d_variable( w, gpu, y0,y1, z0,z1 );
	  CUDA_SAFE_CALL( cudaMemcpy( v_gpu, v_cpu, w->_Nf, cudaMemcpyHostToDevice ));

	}
      }
    }
  }

  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  return true;
}




