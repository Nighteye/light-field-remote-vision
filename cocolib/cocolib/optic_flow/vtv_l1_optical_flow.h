/* -*-c++-*- */
/** \of_tv_l1.h

	optical flow detection
	Implemented from Zach/Pock,Bischof 2007,
	"A Duality Based Approach for Realtime TV-L1 Optical Flow"
	combined with vectorial total variation

    Copyright (C) 2012 Ole Johannsen,
    <first name>.<last name>ATberlin.de

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
#ifndef __COCO_OF_TV_L1_H
#define __COCO_OF_TV_L1_H

#include <vector>
#include <assert.h>

#include "../common/gsl_image.h"
#include "../cuda/cuda_interface.h"


namespace coco {

  // Workspace structure with CUDA-specific definitions
  struct of_tv_l1_workspace;

  // Helper structure to set all parameters
  struct of_tv_l1_data
  {
    // Field size
    size_t _W;
    size_t _H;
    size_t _N; // W*H
    size_t _nfbytes;

    // data attachment weight
    float _lambda;

    // time step
    float _tau;

    // tightness
    float _theta;

    // stopping threshold
    float _stopping_threshold;

    // number of warps
    int _warps;

    // number of scales
    int _scales;

    // scale factor
    float _factor;

    int _iterations;

    std::string _output_dir;

    // Regularizer
    // 0: TV
    // 1: VTV_J
    size_t _regularizer;

    // Number of bytes per image float layer
    // usually W*H*sizeof(float)

    // Local CPU copy of the approximated function im0
    gsl_matrix *_im0;
    // Local CPU copy of the approximated function im1
    gsl_matrix *_im1;
    // Local CPU copy of the optical flow u
    gsl_matrix *_u;
    // Loca CPU copy of ground Truth
    gsl_matrix *_gt;

    // Workspace data
    of_tv_l1_workspace* _workspace;
  };


  /*****************************************************************************
       Workspace creation / access
  *****************************************************************************/

  // Alloc PDE data with sensible defaults
  of_tv_l1_data* of_tv_l1_data_alloc( gsl_matrix* im0, gsl_matrix* im1, size_t scales,double zfactor);

  // Free up PDE data
  bool of_tv_l1_data_free( of_tv_l1_data *data );


  /*****************************************************************************
       Optic flow algorithm
  *****************************************************************************/

  // Initialize workspace with pyramids
  bool of_tv_l1_initialize( of_tv_l1_data *data);

  // Calculate pyramid solutions
  bool of_tv_l1_calculate( of_tv_l1_data *data);

  // Get current solution (as raw flow field)
  bool of_tv_l1_get_solution( of_tv_l1_data *data,
  			      gsl_matrix *fx, gsl_matrix *fy );

  // Get current solution (as color-coded image)
  bool of_tv_l1_get_solution( of_tv_l1_data *data,
  			      gsl_image* u);



  /*****************************************************************************
       Auxiliary functions
  *****************************************************************************/

  // Compare flow field to ground truth
  bool optic_flow_compare_gt( gsl_matrix *fx, gsl_matrix *fy,
			      gsl_matrix *gtx, gsl_matrix *gty,
			      double &average_endpoint_error,
			      double &average_angular_error );

  // Helper function to convert image to grayscale
  void image_to_grayscale( size_t W, size_t H, gsl_image *im, gsl_matrix *mim );

  // Convert flow field to color image using Middlebury color wheel
  gsl_image *flow_field_to_image( gsl_matrix *u1, gsl_matrix *u2 );

  // Convert flow field to color image using Middlebury color wheel
  gsl_image *flow_field_to_image( gsl_matrix *u1, gsl_matrix *u2 );
}
#endif
