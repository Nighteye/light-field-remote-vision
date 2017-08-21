/** \file tv.h

    Spectral TV decomposition
    after Gilboa's Techreport 2013

    Copyright (C) 2013 Bastian Goldluecke,
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

#ifndef __COCO_SPECTRAL_VTV_H
#define __COCO_SPECTRAL_VTV_H

#include <map>
#include <vector>
#include <assert.h>

#include <gsl/gsl_matrix.h>

#include "../cuda/cuda_interface.h"
#include "../defs.h"
#include "../modules.h"
#include "../common/gsl_image.h"



namespace coco {
  struct coco_vtv_spectrum_workspace;

  // Helper structure to set all parameters
  struct coco_vtv_spectrum_data
  {
    // Image size and channels
    size_t _W;
    size_t _H;
    size_t _N;

    // Time step
    double _dt;
    // Number of iterations (time steps)
    // If zero, proceeds until residual is zero
    int _iterations;

    // Number of inner iterations (fista)
    size_t _inner_iterations;

    // Regularizer
    // 0: TV_S
    // 1: TV_F
    // 2: TV_J
    size_t _regularizer;

    // Workspace data
    coco_vtv_spectrum_workspace* _workspace;
  };




  // Alloc VTV spectral decomposition for an image
  coco_vtv_spectrum_data* coco_vtv_spectrum_alloc( vector<gsl_matrix*> F );
  // Free up VTV spectral data
  bool coco_vtv_spectrum_free( coco_vtv_spectrum_data *data );



  /*****************************************************************************
       Spectral decomposition
  *****************************************************************************/

  // Perform full decomposition within given range
  bool coco_vtv_spectrum_decomposition( coco_vtv_spectrum_data *data );

  // Perform single decomposition iteration
  bool coco_vtv_spectrum_decomposition_iteration( coco_vtv_spectrum_data *data );

  // Return current spectrum
  bool coco_vtv_spectrum_get_spectrum( coco_vtv_spectrum_data *data, vector<float> &spectrum );

  // Return single mode
  bool coco_vtv_spectrum_get_mode( coco_vtv_spectrum_data *data, int index, vector<gsl_matrix*> &mode );

  // Return reconstruction with given coefficients
  bool coco_vtv_spectrum_reconstruction( coco_vtv_spectrum_data *data,
					 vector<float> &coefficients,
					 bool add_residual,
					 vector<gsl_matrix*> &result );


}


#endif
