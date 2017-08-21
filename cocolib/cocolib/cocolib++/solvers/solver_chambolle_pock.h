/** \file solver_chambolle_pock.cu

    Data structure for inverse problem solver based on
    Chambolle/Pock SIIMS 2010.

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

#ifndef __CUDA_SOLVER_CP_H
#define __CUDA_SOLVER_CP_H

#include "solver.h"

namespace coco {

  /// Base class for inverse problem solver
  /***************************************************************
  Allocates solver data structures for a given problem.

  The type of problem considered is min_u{ J(u) + F(u) },
  where J is a (convex) regularizer and F a (convex)
  data term.
  ****************************************************************/
  struct solver_chambolle_pock : public solver
  {
    // Construction and destruction
  public:
    solver_chambolle_pock( variational_model * );
    virtual ~solver_chambolle_pock();

    // Access
  public:

    // Implementation
  public:

    // Helper functions
  public:
    // The following helper functions are called by "solve"

    // Re-initialize solver using current solution as starting point
    virtual bool initialize();
    // Perform one iteration
    virtual bool iterate();


  protected:
    // Extragradient variables
    vector_valued_function_2D *_Uq;
    vector_valued_function_2D *_Vq;
  };
};



#endif
