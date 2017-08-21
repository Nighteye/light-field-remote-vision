/** \file anisotropic_diffusion.h
    Perona-Malik isotropic and Weickert's coherence-enhancing diffusion,
    different discretizations,
    inpainting models.

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

#ifndef __COCO_DIFFUSION_H
#define __COCO_DIFFUSION_H

#include <map>
#include <vector>
#include <assert.h>

#include "../cuda/cuda_interface.h"
#include "../defs.h"
#include "../modules.h"
#include "../common/gsl_image.h"



namespace coco {

  /*** CORE PDE METHODS ***/
  struct coco_diffusion_workspace;

  // Anisotropic diffusion workspace
  struct coco_diffusion_data
  {
    // Field size
    size_t _W;
    size_t _H;
    size_t _N;
    // Number of channels
    size_t _nchannels;

    // Step size
    double _tau;

    // Value range
    double _rmin;
    double _rmax;

    // Discretization
    enum diffusion_discretization {
      DISCRETIZATION_STANDARD,
      DISCRETIZATION_NON_NEGATIVITY,
      DISCRETIZATION_ROTATION_INVARIANT,
    };
    diffusion_discretization _discretization;

    // Rotation-invariant discretization: 
    // requires extra filter to get rid of checkerboard artifacts
    // sensible weight is around 0.02
    double _roi_filter_weight;

    // Data term model
    enum diffusion_type {
      DIFFUSION_PLAIN,
      DIFFUSION_L2,
      DIFFUSION_INPAINTING,
    };
    diffusion_type _type;

    // Diffusion tensor model
    enum diffusion_tensor {
      TENSOR_IDENTITY,
      TENSOR_TV,
      TENSOR_PERONA_MALIK,
      TENSOR_COHERENCE_ENHANCING,
    };
    diffusion_tensor _tensor;

    // Diffusion tensor parameters
    // Global weight
    float _lambda;
    // For Perona-Malik and coherence-enhancing: edge strength constant
    float _K;
    // For coherence-enhancing: strength of across-edge diffusion
    float _c1;
    // Outer scale for structure tensor
    float _outer_scale;
    // Inner scale for structure tensor
    float _inner_scale;

    // Workspace data
    coco_diffusion_workspace* _workspace;
  };




  // Alloc PDE data with sensible defaults
  coco_diffusion_data* coco_diffusion_data_alloc( std::vector<gsl_matrix*> F );
  // Free up PDE data
  bool coco_diffusion_data_free( coco_diffusion_data *data );

  // Initialize workspace with current solution
  bool coco_diffusion_set_solution( coco_diffusion_data *data,
      std::vector<gsl_matrix*> &U );

  // Get current solution
  bool coco_diffusion_get_solution( coco_diffusion_data *data,
      std::vector<gsl_matrix*> &U );

  // Init inpainting stencil
  bool coco_diffusion_set_stencil( coco_diffusion_data *data, gsl_matrix *stencil );



  /*****************************************************************************
       ANISOTROPIC DIFFUSION ITERATIONS
  *****************************************************************************/

  // Compute structure tensor for all channels
  bool coco_diffusion_compute_structure_tensor( coco_diffusion_data *data );

  // Set fixed diffusion tensor (very inefficient during iterations)
  bool coco_diffusion_set_diffusion_tensor( coco_diffusion_data *data,
					    gsl_matrix *a, gsl_matrix *b, gsl_matrix *c );

  // Compute diffusion tensor according to pre-defined parameters
  bool coco_diffusion_compute_diffusion_tensor( coco_diffusion_data *data );

  // Perform one diffusion iteration for currently activated scheme
  bool coco_diffusion_iteration( coco_diffusion_data *data );

}


#endif
