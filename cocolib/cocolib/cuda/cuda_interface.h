/** \file cuda_interface.h

    Some helper functions for CUDA texture binding and memcopy operations.

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

#ifndef __CUDA_INTERFACE_H
#define __CUDA_INTERFACE_H

#include <string>
#include <iostream>
#include <assert.h>
#include <math.h>

// CUDA default floating point type
#ifdef CUDA_DOUBLE
typedef double cuflt;
#else
typedef float cuflt;
#endif

namespace coco {

  // Get CUDA device count
  size_t cuda_device_count();
  // Get CUDA device name
  std::string cuda_device_name( size_t device_id );
  // Set CUDA device
  void cuda_set_device( size_t device_id );


  /// Set global default block size
  /** All algorithm objects created afterwards use this block size.
      Current initial value is 16 x 16.
  */
  void set_default_cuda_block_size( size_t dim_x, size_t dim_y );
  
  // Returns current default block size
  int cuda_default_block_size_x();
  int cuda_default_block_size_y();
}


#endif
