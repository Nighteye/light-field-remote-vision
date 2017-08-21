/** \file tv_l2.h
   Algorithms to solve the TV-L2 model:

   argmin_u J(u) + \frac{1}{2\lambda}\norm{u-f}_2^2

   Weighted TV with J(u) = \int g\enorm{\grad u} \dx is supported.

   1. Primal-dual semi-implicit descent,
   Implements Chambolle 2004,
   "An algorithm for total variation minimization and applications".

   2. Primal-dual projection
   Implements Chambolle 2005,
   "Total variation minimization and a class of binary MRF models".

   3. Fast gradient projection
   Implements FGP from Beck/Teboulle 2009,
   "Fast Gradient-based algorithms for constrained total variation
   image denoising and deblurring problems."

   For most problems, algorithm (3) should work fastest.

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

#ifndef __COCO_TV_L2_H
#define __COCO_TV_L2_H

#include <vector>
#include <assert.h>

#include "../common/gsl_matrix_helper.h"
#include "../cuda/cuda_interface.h"



namespace coco {

  // Workspace structure with CUDA-specific definitions
  struct tv_l2_workspace;

  // Helper structure to set all parameters
  struct tv_l2_data
  {
    // Field size
    size_t _W;
    size_t _H;
    size_t _N; // W*H

    // Smoothness parameter
    float _lambda;

    // Step size for dual gradient descent
    float _tau;
    // Current FGP relaxation factor
    // Initialized with 1.0
    float _alpha;

    // Number of bytes per image float layer
    // usually W*H*sizeof(float)
    size_t _nfbytes;

    // Local CPU copy of the approximated function f
    gsl_matrix *_f;
    // Local CPU copy of the TV weight g
    gsl_matrix *_g;

    // Workspace data
    tv_l2_workspace* _workspace;
  };


  /*****************************************************************************
       Workspace creation / access
  *****************************************************************************/

  // Alloc PDE data with sensible defaults
  tv_l2_data* tv_l2_data_alloc( gsl_matrix* f, gsl_matrix *g=NULL );
  // Free up PDE data
  bool tv_l2_data_free( tv_l2_data *data );

  // Initialize workspace with current solution
  bool tv_l2_initialize( tv_l2_data *data,
			    gsl_matrix* u );

  // Get current solution
  bool tv_l2_get_solution( tv_l2_data *data,
			      gsl_matrix* u );

  // Get dual variable XI (vector of dimension 2)
  bool tv_l2_get_dual_xi( tv_l2_data *data,
			     std::vector<gsl_matrix*> &XI );

  // Compute current energy (slow)
  double tv_l2_energy( tv_l2_data *data );




  /*****************************************************************************
       ROF algorithm I: Primal-dual with semi-implicit descent (Chambolle 2004)
  *****************************************************************************/

  // Perform one full iteration
  bool tv_l2_iteration_pd_semi_implicit( tv_l2_data *data );

  // Perform one primal step
  bool tv_l2_primal_step_pd_semi_implicit( tv_l2_data *data );
  // Perform one dual step
  bool tv_l2_dual_step_pd_semi_implicit( tv_l2_data *data );



  /*****************************************************************************
       ROF algorithm II: Primal-dual with simple projection (Chambolle 2005)
  *****************************************************************************/

  // Perform one full iteration
  bool tv_l2_iteration_pd_project( tv_l2_data *data );

  // Perform one primal step
  bool tv_l2_primal_step_pd_project( tv_l2_data *data );
  // Perform one dual step
  bool tv_l2_dual_step_pd_project( tv_l2_data *data );



  /*****************************************************************************
       ROF algorithm III: Fast gradient projection (Beck/Teboulle 2009)
  *****************************************************************************/

  // Perform one full iteration
  bool tv_l2_iteration_fgp( tv_l2_data *data );

  // Perform one primal step
  bool tv_l2_primal_step_fgp( tv_l2_data *data );
  // Perform one dual step
  bool tv_l2_dual_step_fgp( tv_l2_data *data );
  // Perform update of the fgp relaxation variables
  bool update_fgp_relaxation( tv_l2_data *data, float r ); 

}


#endif
