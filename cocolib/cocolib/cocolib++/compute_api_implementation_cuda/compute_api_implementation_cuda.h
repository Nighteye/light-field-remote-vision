/* -*-c++-*- */
/** \file compute_api_implementation_cuda.h

    CUDA specific definitions for the compute API 

    Copyright (C) 2014 Bastian Goldluecke,
    
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

#ifndef __COMPUTE_API_IMPLEMENTAITON_CUDA_H
#define __COMPUTE_API_IMPLEMENTAITON_CUDA_H

#include "../compute_api/compute_array.h"

namespace coco {

  // Configure the layout for kernels for a compute array
  bool kernel_configure( const vector_valued_function_2D *U, dim3 &dimGrid, dim3 &dimBlock );

  // Configure the layout for kernels for a compute grid
  bool kernel_configure( const compute_grid *G, dim3 &dimGrid, dim3 &dimBlock );

};

#endif
