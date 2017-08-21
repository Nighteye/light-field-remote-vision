/** \file data_term_denoising.h

    Base data structure for the data term of an L^p denoising problem.

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

#ifndef __CUDA_DATA_TERM_DENOISING_H
#define __CUDA_DATA_TERM_DENOISING_H

#include "rof.h"

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
  struct data_term_denoising : public data_term_rof
  {
    // Construction and destruction
    data_term_denoising();
    virtual ~data_term_denoising();

    // Resize problem
    virtual bool resize( compute_grid *G, int N );
    
    // Algorithm implementation
    // Operator row sum norm of B for primal variables
    virtual bool accumulate_operator_norm_U( float *norm_U ) const;
    // Operator column sum norm of B for dual variables
    virtual bool accumulate_operator_norm_Q( float *norm_Q ) const;
    
    // Perform primal update, i.e. gradient step + reprojection
    virtual bool primal_update( vector_valued_function_2D *U,
				float *stepU,
				vector_valued_function_2D *Q=NULL );
    // Perform dual update, i.e. gradient step + reprojection
    virtual bool dual_update( const vector_valued_function_2D *U,
			      vector_valued_function_2D *Q,
			      float *step_Q );

    // Perform dual reprojection (called by dual update)
    virtual bool dual_reprojection( vector_valued_function_2D *Q,
				    float *step_Q );
  };

};



#endif
