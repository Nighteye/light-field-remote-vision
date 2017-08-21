/** \file data_term.h

    Base data structure for the data term of an inverse problem.

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

#ifndef __CUDA_DATA_TERM_H
#define __CUDA_DATA_TERM_H

#include <map>
#include "../compute_api/compute_array.h"


namespace coco {

  /// Base class for data term
  /***************************************************************
  Allocates data term data structures which are not
  algorithm specific.

  Implements basic building blocks for algorithms.

  The type of data term for a function u considered is
  any convex lsc functional F.

  Supports dual form,
    e.g. F(u) = max_q{ \alpha (Bu,q) + \beta (f,q)}
    for L^1 data terms.
  ****************************************************************/
  struct data_term
  {
    // Construction and destruction
    data_term();
    virtual ~data_term();

    // Resize problem
    virtual bool resize( compute_grid *G, int N );
    
    // Get problem dimension
    int N() const;
    // Get problem grid
    compute_grid* grid() const;

    // Dataterm parameter set
    bool set_parameter( const std::string &param, float value );
    // Dataterm parameter get
    float get_parameter( const std::string &param ) const;

    // Queries for algorithm implementation

    // Check for valid dimensions of paramter arrays
    bool assert_valid_dims( const vector_valued_function_2D *U,
			    const vector_valued_function_2D *Q ) const;

    // Total dimension of additional dual variables
    int dual_dimension() const;

    // Operator row sum norm of B for primal variables
    virtual bool accumulate_operator_norm_U( float *norm_U ) const;
    // Operator column sum norm of B for dual variables
    virtual bool accumulate_operator_norm_Q( float *norm_Q ) const;

    // Algorithm implementation
    
    // Perform primal update, i.e. gradient step + reprojection
    virtual bool primal_update( vector_valued_function_2D *U,
				float *stepU,
				vector_valued_function_2D *Q=NULL );
    
    // Perform dual update, i.e. gradient step + reprojection
    virtual bool dual_update( const vector_valued_function_2D *U,
			      vector_valued_function_2D *Q,
			      float *step_Q );

  protected:
    std::map<std::string,float> _params;
    compute_grid *_G;
    int _N;
    int _K;
  };

};



#endif
