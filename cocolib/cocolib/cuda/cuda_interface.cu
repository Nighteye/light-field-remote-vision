/* -*-c++-*- */
/** \file cuda_interface.cu

    Implements basic CUDA handling code and memory transfers.

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

#include "../defs.h"
#include "../common/debug.h"

#include "cuda_interface.h"
#include "cuda_helper.h"

#include <iostream>


// Get CUDA device count
size_t coco::cuda_device_count()
{
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  assert( deviceCount >= 0 );
  return deviceCount;
}

// Get CUDA device name
std::string coco::cuda_device_name( size_t device_id )
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);
  return deviceProp.name;
}


// Set CUDA device
void coco::cuda_set_device( size_t device_id )
{
  cudaSetDevice( device_id );
}



// Default block size on startup
namespace coco {
  static int __default_block_size_x = 16;
  static int __default_block_size_y = 16;
}

// Set global default block size
void coco::set_default_cuda_block_size( size_t dim_x, size_t dim_y )
{
  __default_block_size_x = dim_x;
  __default_block_size_y = dim_y;
  if ( (dim_x * dim_y % 32) != 0 ) {
    TRACE( "WARNING: default block size set to something which is not" << std::endl );
    TRACE( "         a multiple of warp size." << std::endl );
  }
}


// Returns current default block size
int coco::cuda_default_block_size_x()
{
  return __default_block_size_x;
}

int coco::cuda_default_block_size_y()
{
  return __default_block_size_y;
}
