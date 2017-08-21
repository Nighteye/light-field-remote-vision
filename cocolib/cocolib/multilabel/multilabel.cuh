/* -*-c++-*- */
/** \file multilabel.cuh

   CUDA-Only includes for multilabel solvers

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

  struct multilabel_workspace {

    // array for final integer solution (integer label number)
    int *_ur;
    // number of bytes in array
    int _urbytes;

    // Primal variable
    float *_u;
    // Primal variable lead
    float *_uq;
    // Precomputed data term
    float *_rho;

    // Spatial smoothness weight
    float *_g;
    // Lagrange multiplier for simplex constraint
    float *_sigma;

    // Memory alloc sizes
    size_t _nfbytes;
    size_t _nfbytes_sigma;

    // Primal/dual step sizes
    float _tau_u;
    float _sigma_p;
    float _sigma_s;

    // Block/grid sizes
    dim3 _dimBlock;
    dim3 _dimGrid;
  };

  // Init/free
  bool multilabel_workspace_init( multilabel_data *data, multilabel_workspace *w );
  bool multilabel_workspace_free( multilabel_data *data, multilabel_workspace *w );




  /*****************************************************************************
       Helper functions to fix memory layout
  *****************************************************************************/

  // Helper functions to compute indices
  // Memory layout for solution fields:
  // One (W*H) slice per label, first a stack of G1 slices, the a stack of G2 slices
  size_t multilabel_solution_index( multilabel_data* data,
				    size_t x, size_t y, size_t g );

  // Memory layout for data term:
  // One (W*H) slice per label, (G1*G2) stacks
  size_t multilabel_dataterm_index( multilabel_data* data,
				    size_t x, size_t y, size_t g );




}
