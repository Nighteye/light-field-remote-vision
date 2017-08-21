/** \file inverse_problem.h

    Base data structure for inverse problem solvers,
    consisting of regularizer and data term.

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

#ifndef __CUDA_INVERSE_PROBLEM_H
#define __CUDA_INVERSE_PROBLEM_H

#include "variational_model.h"

namespace coco {

  /// Base class for inverse problem definition
  /***************************************************************
  Allocates common problem data structures, which are not
  algorithm-specific.

  The type of problem considered is min_u{ J(u) + F(u) },
  where J is a (convex) regularizer and F a (convex)
  data term.
  ****************************************************************/
  struct inverse_problem : public variational_model
  {
    // Construction and destruction
    inverse_problem( regularizer *J, data_term *F );
    virtual ~inverse_problem();
  };

};



#endif
