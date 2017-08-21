/** \file regularizer_vtv_j.h

    Base data structure for vectorial total variation regularizer
    Operator norm (generalized Jacobian)

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

#ifndef __CUDA_REGULARIZER_VTV_J_H
#define __CUDA_REGULARIZER_VTV_J_H

#include "regularizer.h"

namespace coco {

  /// Base class for regularizer
  /***************************************************************
  Allocates VTV regularizer data structures which are not
  algorithm specific.

  Implements basic building blocks for algorithms.
  ****************************************************************/
  struct regularizer_vtv_j : public regularizer
  {
    // Construction and destruction
    regularizer_vtv_j();
    virtual ~regularizer_vtv_j();

    // Algorithm implementation
    
    // Perform dual reprojection
    virtual bool dual_prox( vector_valued_function_2D* P );
  };

};



#endif
