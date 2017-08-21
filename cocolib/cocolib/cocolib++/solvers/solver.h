/* -*-c++-*- */
/** \file solver.h

    Base data structure for inverse problem solvers.

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

#ifndef __CUDA_SOLVER_H
#define __CUDA_SOLVER_H

#include <map>

#include "../compute_api/compute_array.h"
#include "stopping_criterion.h"
#include "../models/variational_model.h"


namespace coco {

  /// Base class for inverse problem solver
  /***************************************************************
  Allocates solver data structures for a given problem.

  The type of problem considered is min_u{ J(u) + F(u) },
  where J is a (convex) regularizer and F a (convex)
  data term.
  ****************************************************************/
  struct solver
  {
    // Construction and destruction
  public:
    solver( variational_model * );
    virtual ~solver();


    // Access
  public:

    // Get problem
    variational_model *problem();

    // Get solution state
    vector_valued_function_2D *U();
    const vector_valued_function_2D *U() const;

    // Get aux variable states
    // TODO: Think about which ones we want to provide (if any)



    // Parameters
  public:
    
    // Set value of a parameter
    bool set_parameter( const std::string &name, double value );
    // Get value of a parameter (defaults to 0.0 if not set explicitly)
    double get_parameter( const std::string &name ) const;


    // Implementation
  public:

    // Solve the problem according to the given stopping criterion
    virtual bool solve( stopping_criterion *sc );



    // Helper functions
  public:
    // The following helper functions are called by "solve"

    // Re-initialize solver using current solution as starting point
    virtual bool initialize();
    // Perform one iteration
    virtual bool iterate();

    // The following helper functions can be used in derived algorithms

    // Extragradient step.
    //   In:  old value in V, updated value in Vq
    //   Out: updated value in V, extragradient value in Vq
    virtual bool extragradient_step( float theta, vector_valued_function_2D *V, vector_valued_function_2D *Vq );



    // Debugging
  public:
    // Full trace of a single pixel (dumps all data to console)
    bool trace_pixel( int x, int y ) const;


  protected:
    // Problem size (copy for convenience)
    int _N; // primal dim
    int _K; // dual regularizer dim
    int _M; // extra primal regularizer dim
    int _L; // dual dataterm dim
    compute_grid *_G; // Computation grid
    
    // Step sizes stored for each variable dimension
    float *_step_U;
    float *_step_P;
    float *_step_V;
    float *_step_Q;

    // Primal and dual variables for the problem
    // Solution
    vector_valued_function_2D *_U;

    // Regularizer duals
    vector_valued_function_2D* _P;
    // Regularizer extra primals
    vector_valued_function_2D* _V;

    // Data term duals
    vector_valued_function_2D* _Q;



  private:
    // Target inverse problem
    variational_model *_problem;

    // Algorithm parameters
    std::map<std::string,double> _params;


    // Copying and assigning forbidden.
    // These are inherently inefficient, or, if implemented efficiently via pointer copies,
    // can easily lead to unintended side effects. Copy operations must always be initiated
    // explicitly.
    virtual solver &operator= ( const solver & );
    solver( const solver & );
  };
};



#endif
