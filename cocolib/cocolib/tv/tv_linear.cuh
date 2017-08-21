/* -*-c++-*- */
/** \file tv_linear.cuh
   Algorithms to solve the TV model with deblurring data term.

   Local CUDA workspace structure definition.

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

#include "../cuda/cuda_convolutions.h"

namespace coco {

  // Workspace structure, stores allocated GPU data
  struct tv_linear_workspace
  {
    // Linear factor
    float *_a;

    // CUDA block dimensions
    dim3 _dimBlock;
    dim3 _dimGrid;
  };


}
