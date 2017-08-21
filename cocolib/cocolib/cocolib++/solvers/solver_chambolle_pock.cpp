/* -*-c++-*- */
/** \file solver_chambolle_pock.cpp

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

#include "solver_chambolle_pock.h"

using namespace coco;

solver_chambolle_pock::solver_chambolle_pock( variational_model *problem )
  : solver( problem )
{
  // Alloc extragradient variables
  _Uq = new vector_valued_function_2D;
  _Uq->alloc( _G,_N );
  TRACE5( "  alloc U_q " << _N << " layers." << std::endl );
  if ( _M > 0 ) {
    _Vq = new vector_valued_function_2D;
    _Vq->alloc( _G,_M );
    TRACE5( "  alloc V_q " << _M << " layers." << std::endl );
  }
  else {
    _Vq = NULL;
  }
}


solver_chambolle_pock::~solver_chambolle_pock()
{
  delete _Uq;
  delete _Vq;
}



// Re-initialize solver using current solution as starting point
bool solver_chambolle_pock::initialize()
{
  _Uq->dump();
  solver::initialize();

  // Set all dual and aux variables to zero
  if ( _P != NULL ) {
    _P->set_zero();
  }
  if ( _Q != NULL ) {
    _Q->set_zero();
  }
  if ( _V != NULL ) {
    _V->set_zero();
    _Vq->set_zero();
  }

  // Set Uq to starting value of U
  _Uq->dump();
  _U->dump();
  _Uq->copy_from_gpu( _U );
  return true;
}

// Perform one iteration
bool solver_chambolle_pock::iterate()
{
  variational_model *IP = problem();
  assert( IP != NULL );
  regularizer *J = IP->J();
  assert( J != NULL );
  data_term *F = IP->F();
  assert( F != NULL );

  // Dual prox operator
  J->dual_step( _Uq, _P, _step_P, _Vq );
  J->dual_prox( _P );
  F->dual_update( _Uq, _Q, _step_Q );
  // Primal prox operator
  _Uq->copy_from_gpu( U() );
  if ( _M>0 ) {
    _Vq->copy_from_gpu( _V );
  }
  J->primal_step( _Uq, _step_U, _P, _Vq, _step_V );
  J->primal_prox( _V );
  F->primal_update( _Uq, _step_U, _Q );
  // Extragradient step in U,V
  extragradient_step( 1.0f, U(), _Uq );
  if ( _M>0 ) {
    extragradient_step( 1.0f, _V, _Vq );
  }

  return true;
}
