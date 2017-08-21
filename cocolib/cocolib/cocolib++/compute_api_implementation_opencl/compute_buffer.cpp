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

#include "../compute_api/compute_buffer.h"
#include "../compute_api/compute_engine.h"

#include "compute_api_implementation_opencl.h"


using namespace coco;

// Construction and destruction
compute_buffer::compute_buffer( compute_engine* CE, size_t size )
{
  assert( size > 0 );
  _size = size;
  _parent = NULL;
  _engine = CE;

  CL_CONTEXT( CE );
  _data = clCreateBuffer( context, CL_MEM_READ_WRITE, size, NULL, NULL );
  assert( _data != NULL );
}

compute_buffer::compute_buffer( compute_buffer* buffer, size_t start, size_t size )
{
  assert( start+size <= buffer->_size );
  assert( size > 0 );
  // cannot create sub-buffers within sub-buffers
  assert( buffer->_parent == NULL );
  _size = size;
  _cl_buffer_region region;
  region.origin = start;
  region.size = size;
  _data = clCreateSubBuffer( buffer->_data, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
			     &region, NULL );
  assert( _data != NULL );
  _parent = buffer;
  _engine = _parent->_engine;
}

compute_buffer::~compute_buffer()
{
  clReleaseMemObject( _data );
}

// Low level memset
const char *kernel_memset = "\n" \
  "__kernel void kernel_function(                                        \n" \
  "   const unsigned long size,                                          \n" \
  "   const char v,                                                      \n" \
  "   __global char *dst )                                               \n" \
  "{                                                                     \n" \
  "  int o = get_global_id(0);                                           \n" \
  "  if ( o>=size ) return;                                              \n" \
  "  dst[o] = v;                                                         \n" \
  "}";

bool compute_buffer::memset( int v )
{
  char pattern = v%255;
  CL_COMMAND_QUEUE( _engine );
  // OpenGL 1.2 -> not supported by nVidia?
  //  CL_SAFE_CALL( clEnqueueFillBuffer ( commands, _data, &pattern, 1, 0, _size, 0, NULL, NULL ));

  // slower
  static cl_kernel kernel = NULL;
  if ( kernel == NULL ) {
    kernel = kernel_compile( _engine,
			     "kernel_function",
			     ::kernel_memset );
    assert( kernel != NULL );
  }

  assert( sizeof( size_t ) == sizeof( unsigned long ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 0, sizeof(unsigned long), &_size ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 1, sizeof(char), &pattern ));
  CL_SAFE_CALL( clSetKernelArg(kernel, 2, sizeof(cl_mem), &_data ));

  size_t dimBlock = 256;
  CL_SAFE_CALL( clEnqueueNDRangeKernel(commands, kernel, 1, NULL,
				       &_size, &dimBlock, 0, NULL, NULL) );
  return true;
}

// Low level memory transfer
bool compute_buffer::memcpy_from_cpu( const float *cpu_buffer)
{
  CL_COMMAND_QUEUE( _engine );
  CL_SAFE_CALL( clEnqueueWriteBuffer( commands, _data, CL_TRUE, 0,
				      _size, cpu_buffer, 0, NULL, NULL) );
  return true;
}

bool compute_buffer::memcpy_to_cpu( float *cpu_buffer ) const
{
  CL_COMMAND_QUEUE( _engine );
  CL_SAFE_CALL( clFinish(commands) );
  CL_SAFE_CALL( clEnqueueReadBuffer( commands, _data, CL_TRUE, 0, _size, cpu_buffer, 0, NULL, NULL ));  
  return true;
}

bool compute_buffer::memcpy_from_engine( const compute_buffer *engine_buffer_src )
{
  CL_COMMAND_QUEUE( _engine );
  assert( _size == engine_buffer_src->_size );
  CL_SAFE_CALL( clEnqueueCopyBuffer( commands, engine_buffer_src->_data, _data, 0, 0, _size, 0, NULL, NULL ));
  return true;
}


// Auto-cast to implementation-internal buffer object
compute_buffer::operator COMPUTE_API_BUFFER_TYPE()
{
  return _data;
}

compute_buffer::operator const COMPUTE_API_BUFFER_TYPE() const
{
  return _data;
}

