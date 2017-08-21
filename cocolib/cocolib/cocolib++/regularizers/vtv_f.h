/** \file regularizer_tv.h

    Base data structure for the total variation regularizer

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

#ifndef __CUDA_REGULARIZER_VTV_F_H
#define __CUDA_REGULARIZER_VTV_F_H

#include "regularizer.h"

namespace coco {

  /// Base class for regularizer
  /***************************************************************
  Allocates VTV regularizer data structures which are not
  algorithm specific.

  Implements basic building blocks for algorithms.
  ****************************************************************/
  struct regularizer_vtv_f : public regularizer
  {
    // Construction and destruction
    regularizer_vtv_f();
    virtual ~regularizer_vtv_f();

    // Algorithm implementation
    
    // Perform dual reprojection
    virtual bool dual_prox( vector_valued_function_2D *P );
  };

};



#endif
