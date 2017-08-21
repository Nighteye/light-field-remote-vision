/* -*-c++-*- */
/** \file potts_multilabel_simplex.cuh

   CUDA-Only includes for potts_multilabel solvers
   Experimental code for 2D label space

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

#include "multilabel.cuh"


namespace coco {

  struct potts_multilabel_simplex_workspace : public multilabel_workspace {

    // Lipschitz constant for data term
    float _L;

    // Overrelaxation constant
    float _theta;
    // Current iteration
    size_t _simplex_iter;

    // Dual variables
    float *_x1;
    float *_x2;
  };

  // Init multilabel problem workspace for Potts model
  bool potts_multilabel_workspace_init( potts_multilabel_simplex_data*, potts_multilabel_simplex_workspace * );


  /*****************************************************************************
       TV-Multilabel 2D algorithm: Chambolle/Pock '10
       special case suitable only for Potts model
  *****************************************************************************/

  // Compute primal prox operator
  bool potts_multilabel_simplex_primal_prox( potts_multilabel_simplex_data *data );
  // Compute dual prox operator
  bool potts_multilabel_simplex_dual_prox( potts_multilabel_simplex_data *data );
  // Update overrelaxation
  bool potts_multilabel_simplex_update_overrelaxation( potts_multilabel_simplex_data *data );


}
