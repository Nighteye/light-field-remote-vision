/* -*-c++-*- */
/** \file compute_grid.cpp

    Grid data structure for a single computation grid.
    Primary parameter to be passed to any compute kernel.

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

#include "../compute_api/compute_grid.h"
#include "compute_api_implementation_opencl.h"

using namespace coco;


size_t __opencl_default_block_size_x = 16;
size_t __opencl_default_block_size_y = 16;

// Configure the grid for kernels for a compute array
bool coco::kernel_configure( const vector_valued_function_2D *U, dim3 &dimGrid, dim3 &dimBlock )
{
  return kernel_configure( U->grid(), dimGrid, dimBlock );
}


// Configure the grid for kernels for a compute array
bool coco::kernel_configure( const compute_grid *G, dim3 &dimGrid, dim3 &dimBlock )
{
  dimBlock[0] = __opencl_default_block_size_x;
  dimBlock[1] = __opencl_default_block_size_y;
  dimBlock[2] = 1;

  size_t blocks_w = G->W() / dimBlock[0];
  if ( G->W() % dimBlock[0] != 0 ) {
    blocks_w += 1;
  }
  size_t blocks_h = G->H() / dimBlock[1];
  if ( G->H() % dimBlock[1] != 0 ) {
    blocks_h += 1;
  }

  dimGrid[0] = blocks_w * dimBlock[0];
  dimGrid[1] = blocks_h * dimBlock[1];
  dimGrid[2] = dimBlock[2];

  if ( blocks_w==0 || blocks_h==0 ) {
    assert( false );
    return false;
  }

  return true;
}



// Construction and destruction
compute_grid::compute_grid( compute_engine *CE, int W, int H )
{
  assert( CE != NULL );
  _CE = CE;
  assert( W>0 );
  assert( H>0 );
  _W = W;
  _H = H;
  _workspace = NULL;
  _workspace_layers = 0;
  _workspace_current = 0;
  _internal = NULL;
};

compute_grid::~compute_grid()
{
  assert( _workspace == NULL );
  assert( _internal == NULL );
}


// Simple queries
int compute_grid::W() const
{
  return _W;
}

int compute_grid::H() const
{
  return _H;
}

int compute_grid::nbytes() const
{
  return _W*_H*sizeof(float);
}

compute_engine* compute_grid::engine() const
{
  return _CE;
}


// (Re-)allocate or free workspace buffer
// All suballocations must be released first
bool compute_grid::alloc_workspace( size_t nlayers )
{
  assert( false );
  return false;
}

bool compute_grid::free_workspace()
{
  assert( false );
  return false;
}

// Suballocate a number of layers
// Must be freed in the order they are reserved
compute_buffer *compute_grid::reserve_layers( size_t nlayers )
{
  assert( false );
  return NULL;
}

bool compute_grid::free_layers( compute_buffer * )
{
  assert( false );
  return false;
}

// Allocate layers independent of internal workspace
// Calls are passed to underlying compute engine and just for convenience
compute_buffer *compute_grid::alloc_layers( size_t nlayers ) const
{
  assert( nlayers > 0 );
  return new compute_buffer( _CE, nlayers * nbytes() );
}

