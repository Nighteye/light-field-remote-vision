/** \file multilabel.h
   Base structures for multilabel problems of the form

   argmin_u J(u) + \sum_i \int_\Omega c_i u_i \dx,

   where each label is represented by an indicator function.
   Derived structures use functionality in this header.
   
   Copyright (C) 2012 Bastian Goldluecke,
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

#ifndef __COCO_MULTILABEL_H
#define __COCO_MULTILABEL_H

#include <map>
#include <vector>
#include <assert.h>

#include "../common/gsl_image.h"



namespace coco {

  // Data structure for problem processing
  struct multilabel_data {
    // Image size
    size_t _W;
    size_t _H;
    size_t _N;

    // Label space size and values
    size_t _G;
    std::vector<float> _labels;

    // Regularization directions (for anisotropic TV)
    std::vector<float> _nx;
    std::vector<float> _ny;

    // Regularizer weight
    float _lambda;

    // Workspace
    struct multilabel_workspace *_w;
  };



  /*****************************************************************************
       Workspace creation / access
  *****************************************************************************/

  // Init structure
  bool multilabel_data_init( multilabel_data *data, size_t W, size_t H, size_t L ); 
  // Free multilabel problem structure
  bool multilabel_data_free( multilabel_data* data );


  // Set equidistant labels
  bool multilabel_set_label_range( multilabel_data *data,
				   float gmin, float gmax );
  
  // Set local spatial smoothness weight
  bool multilabel_set_spatial_smoothness_weight( multilabel_data* data,
						 const gsl_matrix *g );

  // Set current solution data to point-wise optimum of data term (lifting method)
  bool multilabel_set_solution_pointwise_optimum_lifting( multilabel_data* data );
  // Set current solution data to point-wise optimum of data term (indicator functions)
  bool multilabel_set_solution_pointwise_optimum_indicator( multilabel_data* data );

  // Set current solution data (lifting method)
  bool multilabel_set_solution_lifting( multilabel_data* data,
					const gsl_matrix *u );
  
  // Get current solution (lifting method)
  bool multilabel_get_solution_lifting( multilabel_data *data,
					gsl_matrix *u,
					const float threshold = 0.5f );

  // Set current solution data (lifting method)
  bool multilabel_set_solution_lifting( multilabel_data* data,
					const float *u );
  
  // Get current solution (lifting method)
  bool multilabel_get_solution_lifting( multilabel_data *data,
					float *u,
					const float threshold = 0.5f );

  // Set current solution data (indicator functions)
  bool multilabel_set_solution_indicator( multilabel_data* data,
					  const gsl_matrix *u );
  
  // Project current relaxed indicator function solution onto integer values
  bool multilabel_project_solution_indicator( multilabel_data *data );

  // Get current solution (indicator functions)
  bool multilabel_get_solution_indicator( multilabel_data *data,
					  gsl_matrix *u );

  // Get current solution (integer labeling after projection)
  bool multilabel_get_solution( multilabel_data *data,
				int *ur );

  // Set current solution (integer labeling after projection)
  bool multilabel_set_solution( multilabel_data *data,
				int *ur );

  // Set precomputed data term as W x H x L 3D array
  bool multilabel_set_data_term( multilabel_data* data,
				 const float *rho );

  // Return memory requirements for current workspace
  size_t multilabel_workspace_size( multilabel_data *data );

  // Compute current energy (CPU, very slow)
  double multilabel_energy( multilabel_data *data );

}


#endif
