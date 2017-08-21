/** \file tv_multilabel.h
   Algorithms to solve the TV-Multilabel model:

   argmin_u J(u) + \int_\Omega \rho( u(x), x ) \dx

   Weighted TV with J(u) = \int g\enorm{\grad u} \dx is supported.
   Experimental anisotropic TV on each layer is supported

   1. Primal-dual projection algorithm
   Implements Pock et al. 2008

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

#ifndef __COCO_TV_MULTILABEL_H
#define __COCO_TV_MULTILABEL_H

#include "multilabel.h"



namespace coco {

  // Data structure for problem processing
  struct tv_multilabel_data : public multilabel_data
  {
    // Workspace
    struct tv_multilabel_workspace *_tv_w;
  };



  /*****************************************************************************
       Workspace creation / access
  *****************************************************************************/

  // Alloc multilabel problem structure
  tv_multilabel_data* tv_multilabel_data_alloc( size_t W, size_t H, size_t L ); 
  // Free multilabel problem structure
  bool tv_multilabel_data_free( tv_multilabel_data* data );

  // Set current solution data
  bool tv_multilabel_set_solution( tv_multilabel_data* data,
				   const gsl_matrix *u );

  // Get current solution
  bool tv_multilabel_get_solution( tv_multilabel_data *data,
				   gsl_matrix *u,
				   const float threshold = 0.5f );


  // Set current solution data
  bool tv_multilabel_set_solution( tv_multilabel_data* data,
				   const float *u );

  // Get current solution
  bool tv_multilabel_get_solution( tv_multilabel_data *data,
				   float *u,
				   const float threshold = 0.5f );

  // Compute current energy
  double tv_multilabel_energy( tv_multilabel_data *data );



  /*****************************************************************************
       Compute different predefined data terms for common problems
  *****************************************************************************/

  // Compute data term for multilabel segmentation (equidistant)
  bool tv_multilabel_set_dataterm_segmentation( tv_multilabel_data* data,
						const gsl_matrix *image,
						const float lambda );

  // Compute data term for multilabel segmentation (equidistant)
  bool tv_multilabel_set_dataterm_disparity( tv_multilabel_data* data,
					     gsl_image *image_left,
					     gsl_image *image_right,
					     float lambda,
					     size_t disp_min,
					     matching_score m );



  /*****************************************************************************
       TV-Multilabel algorithm I: Pock et al. 2008
  *****************************************************************************/

  // Init algorithm (step sizes)
  bool tv_multilabel_init( tv_multilabel_data* data );
  // Init dual variables (corresponds to a dual step with size 1)
  bool tv_multilabel_dual_init( tv_multilabel_data *data );

  // Perform one multilabel iteration
  bool tv_multilabel_iteration( tv_multilabel_data *data );

  // Perform one primal step
  bool tv_multilabel_primal_step( tv_multilabel_data *data );
  // Compute leading solution (overrelaxation)
  bool tv_multilabel_compute_lead( tv_multilabel_data *data );
  // Perform one dual step
  bool tv_multilabel_dual_step( tv_multilabel_data *data );

}


#endif
