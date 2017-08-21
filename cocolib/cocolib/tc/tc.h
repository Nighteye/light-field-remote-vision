/** \file tc.h
    Solver for Total Curvature

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

#ifndef __COCO_TC_H
#define __COCO_TC_H

#include "../defs.h"
#include "../cuda/cuda_interface.h"
#include "../common/gsl_matrix_helper.h"

#define stcflt cuflt


namespace coco {

  // Workspace structure
  struct tc_workspace;

  // Helper structure to set all parameters
  struct tc_data
  {
    // Problem dimension
    size_t _W;
    size_t _H;
    // Curvature window
    size_t _N;

    // Current iteration
    size_t _iteration;

    // Curvature weight
    stcflt _alpha;
    // Curvature exponent
    stcflt _p;

    // Data term weight
    stcflt _lambda;
    // TV weight
    stcflt _tv_lambda;

    // Number of inner iterations
    size_t _inner_iterations;

    // Local workspace data
    tc_workspace* _workspace;
  };


  /*****************************************************************************
       Workspace creation / access
  *****************************************************************************/

  // Alloc PDE data with sensible defaults
  tc_data* tc_data_alloc( size_t N, gsl_matrix *a );
  // Free up PDE data
  bool tc_data_free( tc_data *data );

  // Initialize workspace with current solution
  bool tc_initialize( tc_data *data,
		      gsl_matrix* u );
  
  // Set mask for inpainting model
  bool tc_set_inpainting_mask( tc_data *data, gsl_matrix *mask );

  // Set kernels for deblurring
  bool tc_set_kernel( tc_data *data, gsl_matrix *kernel );
  bool tc_set_separable_kernel( tc_data *data, gsl_vector *kernel );

  // Get current solution
  bool tc_get_solution( tc_data *data,
			gsl_matrix* u );

  // Init product field from single layer
  bool tc_init_product_field( tc_data *data,
			      std::vector<stcflt*> &v, stcflt *u );

}


#endif
