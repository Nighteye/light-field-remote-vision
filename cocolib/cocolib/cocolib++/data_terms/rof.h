/** \file data_term_rof.h

    Base data structure for the data term of the ROF problem.

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

#ifndef __CUDA_DATA_TERM_ROF_H
#define __CUDA_DATA_TERM_ROF_H

#include "data_term.h"

namespace coco {

  /// Data term of the ROF model
  /***************************************************************
  Allocates data term data structures which are not
  algorithm specific.

  Implements basic building blocks for algorithms.
  ****************************************************************/
  struct data_term_rof : public data_term
  {
    // Construction and destruction
    data_term_rof();
    virtual ~data_term_rof();

    // Resize problem
    virtual bool resize( compute_grid *G, int N );
    
    // Return prox function F
    const vector_valued_function_2D *F() const;
    vector_valued_function_2D *F();

    // Algorithm implementation
    
    // Perform primal update, i.e. gradient step + reprojection
    virtual bool primal_update( vector_valued_function_2D *U,
				float *stepU,
				vector_valued_function_2D *Q=NULL );

  protected:
    // Prox function
    vector_valued_function_2D *_F;
  };

};



#endif
