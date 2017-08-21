/** \file compute_grid.h

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

#ifndef __COCO_COMPUTE_GRID_H
#define __COCO_COMPUTE_GRID_H

#include "compute_engine.h"


namespace coco {

  /// Grid data structure for a single computation grid.
  /** This is the primary parameter to be passed to any compute kernel
      and contains all necessary information to launch it.

      Furthermore, the class offers a way to allocate a huge
      buffer of persistent workspace on the grid (organized in layers)
      for quick temporary suballocations.
  */
  struct compute_grid
  {
    // Construction and destruction
    compute_grid( compute_engine *CE, int W, int H );
    virtual ~compute_grid();

    // Simple queries
    int W() const;
    int H() const;
    int nbytes() const;
    compute_engine *engine() const;
    bool is_compatible( const compute_grid *G ) const;

    // (Re-)allocate or free workspace buffer
    // All suballocations must be released first
    bool alloc_workspace( size_t nlayers );
    bool free_workspace();

    // Suballocate a number of layers
    // Must be freed in the order they are reserved
    compute_buffer *reserve_layers( size_t nlayers );
    bool free_layers( compute_buffer* );

    // Allocate layers independent of internal workspace
    // Calls are passed to underlying compute engine, just for convenience
    compute_buffer *alloc_layers( size_t nlayers ) const;


  private:
    // Size
    int _W;
    int _H;

    // Underlying compute engine
    compute_engine* _CE;

    // Allocated workspace
    float *_workspace;
    // Allocated workspace layers
    int _workspace_layers;
    // First free layer
    int _workspace_current;

    // Internal implementation data
    void *_internal;
  };
};



#endif
