/** \file data_term_deconvolution.h

    Base data structure for the data term of an L^p deconvolution problem.

    Copyright (C) 2014 Bastian Goldluecke.

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

#ifndef __CUDA_DATA_TERM_DECONVOLUTION_H
#define __CUDA_DATA_TERM_DECONVOLUTION_H

#include "denoising.h"
#include "../compute_api/convolutions.h"

namespace coco {

  /// Data term of an L^p denoising model
  /***************************************************************
  Allocates data term data structures which are not
  algorithm specific.

  Implements basic building blocks for algorithms.

  In contrast to ROF data term, implements an L^p norm,
  where p=1,2. Thus, is more general, but also less efficient,
  as it requires an additional dual variable. Basis for the
  general linear inverse problem.
  ****************************************************************/
  struct data_term_deconvolution : public data_term_denoising
  {
    // Construction and destruction
    data_term_deconvolution();
    virtual ~data_term_deconvolution();

    // Init deconvolution kernel
    bool set_kernel( const gsl_matrix *kernel );
    bool set_separable_kernel( const gsl_vector *kernel_x, const gsl_vector* kernel_y );

    // Resize problem
    virtual bool resize( compute_grid *G, int N );

    // Perform primal update, i.e. gradient step + reprojection
    virtual bool primal_update( vector_valued_function_2D *U,
				float *stepU,
				vector_valued_function_2D *Q=NULL );
    // Perform dual update, i.e. gradient step + reprojection
    virtual bool dual_update( const vector_valued_function_2D *U,
			      vector_valued_function_2D *Q,
			      float *step_Q );

  protected:
    // Kernel and transpose
    convolution_kernel *_b, *_bq;
    // Auxiliary space
    vector_valued_function_2D *_tmp;
  };

};



#endif
