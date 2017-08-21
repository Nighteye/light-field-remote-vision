/** \file multilabel_problem.h

    Base data structure for a multilabel problem.
      - Each component is interpreted as indicator function
      - Data term is usually linear (assignment costs)

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

#ifndef __COCO_MULTILABEL_PROBLEM_H
#define __COCO_MULTILABEL_PROBLEM_H

#include "variational_model.h"

namespace coco {

  /// Base class for multilabel problem definition
  /***************************************************************
  Allocates common problem data structures, which are not
  algorithm-specific.

  The type of problem considered is min_u{ J(u) + F(u) },
  where J is a (convex) regularizer and F a (convex)
  data term.
  ****************************************************************/
  struct multilabel_problem : public variational_model
  {
    // Construction and destruction
    multilabel_problem( regularizer *J, data_term *F );
    virtual ~multilabel_problem();

    // Set/access solution vectors
    bool set_pointwise_optimal_labeling( const vector_valued_function_2D *data_term,
					 vector_valued_function_2D *U );
    /*
    bool project_labeling( const vector_valued_function_2D *U,
			   const std::vector<float> &labels,
			   compute_buffer *solution );
    */
    bool project_labeling_to_cpu( const vector_valued_function_2D *U,
				  const std::vector<float> &labels,
				  float *solution );
  };

};



#endif
