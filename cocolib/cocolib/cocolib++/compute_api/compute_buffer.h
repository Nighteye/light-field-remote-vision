/** \file compute_buffer.h

    Structure for memory buffer object
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

#ifndef __COCO_COMPUTE_BUFFER_H
#define __COCO_COMPUTE_BUFFER_H

#include "../../common/debug.h"


namespace coco {
  struct compute_engine;

  /// Grid compute buffer
  /** The idea is to put one layer of abstraction between anything
      related to CUDA and the computation algorithms. This way,
      a later reimplementation of the engine in e.g. OpenCL
      might become possible without too much effort.
  */
  struct compute_buffer
  {
    // Alloc primary buffer (static size)
    compute_buffer( compute_engine*, size_t size );
    // Alloc sub-buffer within a buffer (static size)
    compute_buffer( compute_buffer*, size_t offset, size_t size );
    // Release buffer
    ~compute_buffer();

    // Low level memset
    bool memset( int v );

    // Low level memory transfer
    bool memcpy_from_cpu( const float *cpu_buffer );
    bool memcpy_to_cpu( float *cpu_buffer ) const;
    bool memcpy_from_engine( const compute_buffer *src );

    // Auto-cast to implementation-internal buffer object
    operator COMPUTE_API_BUFFER_TYPE();
    operator const COMPUTE_API_BUFFER_TYPE() const;

  private:
    COMPUTE_API_BUFFER_TYPE _data;
    size_t _size;
    compute_engine *_engine;
    compute_buffer *_parent;
  };

};



#endif
