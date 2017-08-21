/** \file potts_multilabel_simplex.h
   Experimental algorithms to solve a multilabel model
   with 2D-label space.

   argmin_u J(u) + \int_\Omega \rho( u(x), x ) \dx

   where u:\Omega\to\Gamma\subset\R{2}, and
   Gamma is a rectangle.

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

#ifndef __COCO_POTTS_MULTILABEL_SIMPLEX_H
#define __COCO_POTTS_MULTILABEL_SIMPLEX_H

#include <map>
#include <vector>
#include <assert.h>

#include "multilabel.h"



namespace coco {

  // Data structure for problem processing
  struct potts_multilabel_simplex_data : public multilabel_data {
    // Workspace
    struct potts_multilabel_simplex_workspace *_potts_w;
  };



  /*****************************************************************************
       Workspace creation / access
  *****************************************************************************/

  // Alloc multilabel problem structure for Potts model
  potts_multilabel_simplex_data* potts_multilabel_simplex_data_alloc( size_t W, size_t H, size_t G ); 

  // Free multilabel problem structure
  bool potts_multilabel_simplex_data_free( potts_multilabel_simplex_data* data );

  // Return mem requirements
  size_t potts_multilabel_simplex_workspace_size( potts_multilabel_simplex_data *data );

  // Init algorithm with zero solution
  bool potts_multilabel_simplex_init( potts_multilabel_simplex_data* data );

  // Set current solution data
  bool potts_multilabel_simplex_set_solution( potts_multilabel_simplex_data* data,
					      const gsl_matrix *u );

  // Get current solution (given continuous range)
  bool potts_multilabel_simplex_get_solution( potts_multilabel_simplex_data *data,
					      gsl_matrix *u );

  // Compute current energy
  double potts_multilabel_simplex_energy( potts_multilabel_simplex_data *data );




  /*****************************************************************************
       TV-Multilabel 2D algorithm: Chambolle/Pock '10
       special case suitable only for Potts model
  *****************************************************************************/

  // Perform one multilabel FISTA iteration (outer loop)
  bool potts_multilabel_simplex_iteration( potts_multilabel_simplex_data *data );

}


#endif
