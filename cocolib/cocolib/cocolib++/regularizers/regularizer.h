/** \file regularizer.h

    Base data structure for the regularizer of an inverse problem.

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

#ifndef __CUDA_REGULARIZER_H
#define __CUDA_REGULARIZER_H

#include <map>
#include "../compute_api/compute_array.h"

namespace coco {

  /// Base class for regularizer
  /***************************************************************
  Allocates regularizer data structures which are not
  algorithm specific.

  Implements basic building blocks for algorithms.

  The type of regularizer for a function u considered is

  max_{p,q} min_v { \alpha (Ku,p) + \beta (v,p) + \gamma (Lv,q) }

  in the most general form, thus it includes for example TGV_2.
  ****************************************************************/
  struct regularizer
  {
    // Construction and destruction
    regularizer();
    virtual ~regularizer();

    // Resize problem
    virtual bool resize( compute_grid *G, int N );

    // Get problem dimension
    int N() const;
    // Get problem grid
    compute_grid* grid() const;

    // Queries for algorithm implementation

    // Total dimension of dual variables
    int dual_dimension() const;
    // Total dimension of additional primal variables w (excludes solution)
    int extra_primal_dimension() const;

    // Check for valid dimensions of paramter arrays
    bool assert_valid_dims( const vector_valued_function_2D *U,
			    const vector_valued_function_2D *P,
			    const vector_valued_function_2D *V ) const;

    // Operator row sum norm of K for primal variables
    virtual bool accumulate_operator_norm_U( float *norm_U ) const;
    // Operator column sum norms of K and L for dual variables
    virtual bool accumulate_operator_norm_P( float *norm_P ) const;
    // Operator row sum norm of L for extra primal variables
    virtual bool accumulate_operator_norm_V( float *norm_V ) const;

    // Regularizer parameter set
    bool set_parameter( const  std::string &param, float value );
    // Regularizer parameter get
    float get_parameter( const std::string &param ) const;

    // Algorithm implementation (all functions inplace)
    // Gives sufficient functionality for a generic implementation
    // of Chambolle/Pock 2010
    
    // Perform primal step
    virtual bool primal_step( vector_valued_function_2D *U,
			      float *step_U,
			      const vector_valued_function_2D *P,
			      vector_valued_function_2D *V = NULL,
			      float *step_V = NULL );

    // Perform primal reprojection
    // (solution range only affected by data term, so data term performs projection for U)
    virtual bool primal_prox( vector_valued_function_2D *V = NULL );
    
    // Perform dual step
    virtual bool dual_step( const vector_valued_function_2D *U,
			    vector_valued_function_2D *P,
			    float *step_P,
			    const vector_valued_function_2D *V = NULL );

    // Perform dual reprojection
    virtual bool dual_prox( vector_valued_function_2D* P );


  protected:
    // params
    std::map<std::string,float> _params;

    // dual dimension
    int _K;
    // extra primal dimension
    int _M;
    // problem dimension
    int _N;
    // grid
    compute_grid *_G;
  };

};



#endif
