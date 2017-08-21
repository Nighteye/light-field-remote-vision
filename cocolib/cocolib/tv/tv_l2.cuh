/* -*-c++-*- */
/** \file tv_l2.cuh
   Algorithms to solve the TV-L2 model.

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

namespace coco {

  // Manifold structure, stored on GPU
  struct tv_l2_workspace
  {
    // Primal variable
    float* _u;
    // Function to be denoised
    float* _f;
    // TV weight
    float* _g;
    // Dual variables XI
    float* _x1;
    float* _x2;
    // Dual variables XI (old state, for relaxation schemes)
    float* _x1e;
    float* _x2e;
    // Relaxation variables
    float* _y1;
    float* _y2;

    // CUDA block dimensions
    dim3 _dimBlock;
    dim3 _dimGrid;
  };

}
