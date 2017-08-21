/** \file tv_linear.h
   Specialization of FISTA-FGP to solve the TV model with linear
   data term.

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

#ifndef __COCO_TV_LINEAR_H
#define __COCO_TV_LINEAR_H

#include "tv_convex.h"

namespace coco {

  // Workspace structure
  struct tv_linear_workspace;

  // Helper structure to set all parameters
  struct tv_linear_data
  {
    // Local copy of linear factor
    gsl_matrix *_a;
    // Workspace data
    tv_linear_workspace* _workspace;
    // Workspace for tv_convex functional
    tv_convex_data *_tv_convex;
  };


  /*****************************************************************************
       Workspace creation / access
  *****************************************************************************/

  // Alloc PDE data with sensible defaults
  tv_linear_data* tv_linear_data_alloc( gsl_matrix *a, gsl_matrix *g=NULL );
  // Free up PDE data
  bool tv_linear_data_free( tv_linear_data *data );

  // Initialize workspace with current solution
  bool tv_linear_initialize( tv_linear_data *data,
			     gsl_matrix* u );

  // Get current solution
  bool tv_linear_get_solution( tv_linear_data *data,
			       gsl_matrix* u );

  // Compute current energy (slow)
  double tv_linear_energy( tv_linear_data *data );


  /*****************************************************************************
       TV-Linear algorithm I: Specialized FISTA (Beck/Teboulle 2008)
  *****************************************************************************/

  // Perform one full iteration
  bool tv_linear_iteration_fista( tv_linear_data *data );

}


#endif
