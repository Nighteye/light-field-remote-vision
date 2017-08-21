/** \file tv_convex.h
   Algorithms to solve the TV model with convex data term.

   1. FISTA (requires convex and differentiable data term)
   Can use any iterative ROF solver for inner iterations, preset (and recommended) is FGP
   Implements Beck/Teboulle 2008,
   "A fast iterative shrinkage thresholding algorithm for linear inverse problems".

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

#ifndef __COCO_TV_CONVEX_H
#define __COCO_TV_CONVEX_H

#include "../tv/tv_l2.h"

namespace coco {

  // Workspace structure with CUDA implementation details
  struct tv_convex_workspace;

  // Function to compute a CUDA matrix from another one
  typedef bool (*fn_compute_matrix_callback)( void* context,
					      size_t W, size_t H,
					      float *in, float *out );

  // Helper structure to set all parameters
  struct tv_convex_data
  {
    // Field size
    size_t _W;
    size_t _H;
    size_t _N;

    // Lipschitz constant for derivative of data term
    float _L;
    // Function to compute the derivative of the data term
    fn_compute_matrix_callback _fn_grad_data_term;
    // Function to compute the data term (optional)
    fn_compute_matrix_callback _fn_data_term;
    // Callback context
    void *_callback_context;

    // Step size, inner iterations
    float _tau;
    // Number of inner iterations
    size_t _gp_iter;

    // Current FISTA relaxation factor
    float _alpha;
    // Number of bytes per image float layer
    size_t _nfbytes;
    // Workspace data
    tv_convex_workspace* _workspace;
    // Workspace for tv_l2 functional
    tv_l2_data *_rof;
  };


  /*****************************************************************************
       Workspace creation / access
  *****************************************************************************/

  // Alloc PDE data with sensible defaults
  tv_convex_data* tv_convex_data_alloc( size_t W, size_t H,
					fn_compute_matrix_callback fn_grad_f,
					void *callback_context,
					gsl_matrix *g = NULL );
  // Free up PDE data
  bool tv_convex_data_free( tv_convex_data *data );

  // Initialize workspace with current solution
  bool tv_convex_initialize( tv_convex_data *data,
			     gsl_matrix* u );

  // Get current solution
  bool tv_convex_get_solution( tv_convex_data *data,
			       gsl_matrix* u );

  // Compute current energy (slow)
  double tv_convex_energy( tv_convex_data *data );


  /*****************************************************************************
       TV-Convex algorithm I: FISTA (Beck/Teboulle 2008)
  *****************************************************************************/

  // Perform one full iteration
  bool tv_convex_iteration_fista( tv_convex_data *data );

}


#endif
