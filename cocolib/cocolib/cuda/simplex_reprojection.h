/** \file simplex_reprojection.h
    CUDA code to reproject a vector to the standard simplex

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

#ifndef __COCO_TV_SIMPLEX_REPROJECTION_H
#define __COCO_TV_SIMPLEX_REPROJECTION_H

#include <assert.h>



namespace coco {

  // Reprojection onto allowed subset: \sum u = 1, \sum v = 1.
  // W,H: Array size
  // G: Vector length
  bool simplex_reprojection( size_t W, size_t H, size_t G, float *u );

}

#endif
