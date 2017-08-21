/* -*-c++-*- */
/** \file compute_engine.h

    Structure for parallel grid computation engine,
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

#include "../compute_api/compute_engine.h"
#include "compute_api_implementation_opencl.h"
#include <string.h>

using namespace coco;
using namespace std;

// Construction and destruction
compute_engine::compute_engine( const map<string,string> *param_list )
{
  ce_data *ce = new ce_data;
  _implementation_data = (void*)ce;
  memset( ce, 0, sizeof( ce_data ));

  // Connect to the default compute device
  //
  cl_uint nps = 0;
  if ( clGetPlatformIDs( 1, &(ce->_platform), NULL ) != CL_SUCCESS ) {
    ERROR( "compute_engine(OpenCL): Failed to query platform ID." << endl );
    assert( false );
  }

  int err = clGetDeviceIDs(ce->_platform, CL_DEVICE_TYPE_ALL, 1, &ce->_device, NULL);
  if (err != CL_SUCCESS) {
    ERROR( "compute_engine(OpenCL): Failed to create a device." << endl );
    assert( false );
  }

  // Create a compute context 
  ce->_context = clCreateContext(0, 1, &ce->_device, NULL, NULL, NULL);
  if ( ce->_context == NULL ) {
    ERROR( "compute_engine(OpenCL): Failed to create a compute context!" << endl );
    assert( false );
  }

  // Create a command queue
  //
  ce->_commands = clCreateCommandQueue(ce->_context, ce->_device, 0, NULL );
  if (!ce->_commands) {
    ERROR( "compute_engine(OpenCL): failed to create a command commands!\n");
    assert( false );
  }
}

compute_engine::~compute_engine()
{
  ce_data *ce = (ce_data*)_implementation_data;

  // Relase all kernel objects
  for ( size_t i=0; i<ce->_kernels.size(); i++ ) {
    clReleaseKernel( ce->_kernels[i] );
  }

  // Release all program objects
  for ( size_t i=0; i<ce->_programs.size(); i++ ) {
    clReleaseProgram( ce->_programs[i] );
  }

  // Release rest
  clReleaseCommandQueue(ce->_commands);
  clReleaseContext(ce->_context);
  delete ce;
}


// Activation
// Necessary if multiple engines are in use by a program
bool compute_engine::set_active() const
{
  assert( false );
  return false;
}



// Necessary if multiple engines are in use by a program
const void *compute_engine::internal_data() const
{
  return _implementation_data;
}



// Compile a kernel
cl_kernel coco::kernel_compile( const compute_engine *CE, const char *name, const char *source )
{
  ce_data *ce = (ce_data*)CE->internal_data();
  CL_CONTEXT( CE );

  cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, NULL );
  if (!program) {
    ERROR( "kernel_compile(OpenCL): Failed to create compute program!\n");
    assert( false );
    return NULL;
  }
  ce->_programs.push_back( program );
  
  // Build the program executable
  //
  if ( CL_SUCCESS != clBuildProgram(program, 0, NULL, NULL, NULL, NULL)) {
    size_t len;
    char buffer[2048];
    
    ERROR( "OpenCL: failed to compile kernel program, error log follows." << endl << endl );
    clGetProgramBuildInfo(program, ce->_device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    printf("%s\n", buffer);
    assert( false );
    return NULL;
  }

  // Create the compute kernel in the program we wish to run
  //
  int err;
  cl_kernel kernel = clCreateKernel(program, name, &err);
  if (!kernel || err != CL_SUCCESS) {
    printf("Error: Failed to create compute kernel!\n");
    assert( false );
    return NULL;
  }
  ce->_kernels.push_back( kernel );

  return kernel;
}


