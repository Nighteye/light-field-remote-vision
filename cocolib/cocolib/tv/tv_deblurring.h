/** \file tv_deblurring.h
   Specialization of FISTA-FGP to solve the TV model with deblurring
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

#ifndef __COCO_TV_DEBLURRING_H
#define __COCO_TV_DEBLURRING_H

#include "tv_convex.h"

namespace coco {

  // Workspace structure
  struct tv_deblurring_workspace;

  // Helper structure to set all parameters
  struct tv_deblurring_data
  {
    // Smoothness parameter
    float _lambda;

    // Workspace data
    tv_deblurring_workspace* _workspace;
    // Workspace for tv_l2 functional
    tv_convex_data *_tv_convex;
  };


  /*****************************************************************************
       Workspace creation / access
  *****************************************************************************/

  // Alloc PDE data with sensible defaults
  tv_deblurring_data* tv_deblurring_data_alloc( gsl_vector *filter, gsl_matrix *f );
  // Free up PDE data
  bool tv_deblurring_data_free( tv_deblurring_data *data );

  // Initialize workspace with current solution
  bool tv_deblurring_initialize( tv_deblurring_data *data,
				 gsl_matrix* u );

  // Get current solution
  bool tv_deblurring_get_solution( tv_deblurring_data *data,
				   gsl_matrix* u );

  // Compute current energy (slow)
  double tv_deblurring_energy( tv_deblurring_data *data );


  /*****************************************************************************
       TV-Deblurring algorithm I: Specialized FISTA (Beck/Teboulle 2008)
  *****************************************************************************/

  // Perform one full iteration
  bool tv_deblurring_iteration_fista( tv_deblurring_data *data );

}


#endif
