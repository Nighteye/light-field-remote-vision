/** \file multilabel_dataterms.h
   Helper functions: compute various dataterms for multilabel problems
   
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

#ifndef __COCO_MULTILABEL_DATATERMS_H
#define __COCO_MULTILABEL_DATATERMS_H

#include <map>
#include <vector>
#include <assert.h>

#include "multilabel.h"
#include "vectorial_multilabel.h"

#include "../common/color_spaces.h"
#include "../common/linalg3d.h"



namespace coco {

  /*****************************************************************************
       Compute different predefined data terms for common problems
  *****************************************************************************/

  // Compute data term for multilabel grayscale segmentation (equidistant)
  float* multilabel_dataterm_segmentation( multilabel_data* data,
					   const gsl_matrix *image );

  // Disparity (stereo) dataterm. Labels equal disparity values used for matching.
  float *multilabel_dataterm_stereo( multilabel_data* tvm,
				     float lambda,
				     gsl_image *im0,
				     gsl_image *im1 );

  // Compute data term for multilabel stereo
  // Labels equal disparity values used for matching
  float *multilabel_dataterm_stereo( size_t W, size_t H,
				     const std::vector<float> &labels,
				     float lambda,
				     const gsl_image *im0,
				     const gsl_image *im1 );

  // VECTORIAL MULTILABEL SEGMENTATION

  // Vectorial segmentation dataterm. Three dimensions (RGB components).
  // Equidistant labels, absolute difference distance.
  float *vml_dataterm_segmentation( vml_data *data,
				    gsl_image *I,
				    int *rlabels,
				    int *glabels,
				    int *blabels );

  // Compute RGB color value list, one number per label
  // Corresponds to exact dataterm numbering
  bool vml_dataterm_segmentation_labels( vml_data *data, std::vector<color> &labels );

  // Convert vectorial labeling result to image
  gsl_image *vml_segmentation_result( vml_data *data, int *rlabel, int *glabel, int *blabel );

  // Convert linear labeling result to RGB image, given label color list
  gsl_image *vml_segmentation_result( size_t W, size_t H, const std::vector<color> &colors, int *labels );


  // VECTORIAL MULTILABEL OPTIC FLOW

  // Vectorial segmentation dataterm. Three dimensions (RGB components).
  // Equidistant labels, absolute difference distance.
  float *vml_dataterm_opticflow( vml_data *data,
				 gsl_image *I0,
				 gsl_image *I1,
				 double rho,
				 double fx_min, double fx_max,
				 double fy_min, double fy_max,
				 int *ulabels, int *vlabels );

  // Compute RGB color value list, one number per label
  // Corresponds to exact dataterm numbering
  bool vml_dataterm_opticflow_labels( vml_data *data, std::vector<Vec2f> &labels );

  // Convert vectorial labeling result to image
  gsl_image *vml_opticflow_result( vml_data *data, int *ulabel, int *vlabel );

  // Convert linear labeling result to RGB image, given label color list
  gsl_image *vml_opticflow_result( size_t W, size_t H, const std::vector<Vec2f> &vecs, int *labels );



/*****************************************************************************

    Nieuwenhuis and Cremers TPAMI 2012:
    Spatially varying color distributions

    Input:   I      - image on which the scribbles are defined
             masks  - vector of binary masks (one for each label)
             S      - image to be segmented

             sigma - standard deviation of color difference kernel
             alpha - standard deviation scale of distance difference kernel

    Output:  GPU array rho of label costs (layered by label index)
             will be filled (must be allocated on call)

*******************************************************************************/
  bool vml_dataterm_segmentation_nieuwenhuis( gsl_image *I,
					      std::vector<gsl_matrix*> &masks,
					      gsl_image *S,
					      double sigma,
					      double alpha,
					      float *rho );

}


#endif
