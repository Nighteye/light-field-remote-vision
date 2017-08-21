/* -*-c++-*- */
/** \file compute_api_implementation_opencl.h

    OpenCL specific definitions and includes for the compute API 

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

#ifndef __COMPUTE_API_IMPLEMENTAITON_OPENCL_H
#define __COMPUTE_API_IMPLEMENTAITON_OPENCL_H

#include <cl.h>

#include "../compute_api/compute_array.h"

#define CL_SAFE_CALL(call) if (CL_SUCCESS != call) {assert(false);}
#define CL_CONTEXT(CE) cl_context context = ((const coco::ce_data*)CE->internal_data())->_context
#define CL_COMMAND_QUEUE(CE) cl_command_queue commands = ((const coco::ce_data*)CE->internal_data())->_commands

typedef size_t dim3[3];
extern size_t __opencl_default_block_size_x;
extern size_t __opencl_default_block_size_y;

namespace coco {
  struct ce_data
  {
    cl_platform_id     _platform;
    cl_device_id       _device;
    cl_context         _context;
    cl_command_queue   _commands;
    std::vector<cl_program> _programs;
    std::vector<cl_kernel>  _kernels;
  };

  // Compile a kernel
  cl_kernel kernel_compile( const compute_engine *CE, const char *name, const char *source );

  // Configure the layout for kernels for a compute array
  bool kernel_configure( const vector_valued_function_2D *U, dim3 &dimGrid, dim3 &dimBlock );
  // Configure the layout for kernels for a compute grid
  bool kernel_configure( const compute_grid *G, dim3 &dimGrid, dim3 &dimBlock );

};

#endif
