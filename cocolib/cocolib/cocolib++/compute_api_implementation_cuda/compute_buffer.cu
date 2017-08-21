/* -*-c++-*- */
/** \file compute_buffer.cu

    Structure for memory buffer,
    i.e. as implemented by CUDA or OpenCL

    Copyright (C) 2014 Bastian Goldluecke.

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

#include "../../cuda/cuda_helper.h"
#include "../compute_api/compute_buffer.h"


using namespace coco;

// Construction and destruction
compute_buffer::compute_buffer( compute_engine* CE, size_t size )
{
  assert( size > 0 );
  _size = size;
  _parent = NULL;
  _engine = CE;
  CUDA_SAFE_CALL( cudaMalloc( &_data, size ));
  assert( _data != NULL );
}

compute_buffer::compute_buffer( compute_buffer* buffer, size_t start, size_t size )
{
  assert( start+size <= buffer->_size );
  assert( size > 0 );
  _size = size;
  _data = (float*)( ((char*)buffer->_data) + start );
  _parent = buffer;
  _engine = _parent->_engine;
}

compute_buffer::~compute_buffer()
{
  if ( _parent == NULL ) {
    CUDA_SAFE_CALL( cudaFree( _data ));
  }
}

// Low level memset
bool compute_buffer::memset( int v )
{
  CUDA_SAFE_CALL( cudaMemset( _data, v, _size ));
  return true;
}

// Low level memory transfer
bool compute_buffer::memcpy_from_cpu( const float *cpu_buffer)
{
  CUDA_SAFE_CALL( cudaMemcpy( _data, cpu_buffer, _size, cudaMemcpyHostToDevice ));
  return true;
}

bool compute_buffer::memcpy_to_cpu( float *cpu_buffer ) const
{
  CUDA_SAFE_CALL( cudaMemcpy( cpu_buffer, _data, _size, cudaMemcpyDeviceToHost ));
  return true;
}

bool compute_buffer::memcpy_from_engine( const compute_buffer *engine_buffer_src )
{
  assert( _size == engine_buffer_src->_size );
  CUDA_SAFE_CALL( cudaMemcpy( _data, engine_buffer_src->_data, _size, cudaMemcpyDeviceToDevice ));
  return true;
}


// Auto-cast to implementation-internal buffer object
compute_buffer::operator COMPUTE_API_BUFFER_TYPE()
{
  return (float*)_data;
}

compute_buffer::operator const COMPUTE_API_BUFFER_TYPE() const
{
  return (float*)_data;
}

