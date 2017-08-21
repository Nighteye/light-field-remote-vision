/* -*-c++-*- */
/** \file solver.cu

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

#include <string.h>
#include "solver.h"
#include "../compute_api/kernels_vtv.h"


using namespace coco;
using namespace std;


solver::solver( variational_model *problem )
{
  assert( problem != NULL );
  regularizer *J = problem->J();
  assert( J != NULL );
  data_term *F = problem->F();
  assert( F != NULL );
  _problem = problem;

  // Test compability of regularizer and data term
  // Validate problem size
  int N = J->N();
  assert( N>0 );
  assert( N == F->N() );
  _G = problem->grid();
  _N = N;

  // Alloc solution
  _U = new vector_valued_function_2D;
  _U->alloc( _G,N );
  _step_U = new float[N];
  memset( _step_U, 0.0, N * sizeof(float) );
  TRACE5( "  alloc U " << N << " layers." << endl );

  // Alloc dual variables
  _step_P = NULL;
  _P = NULL;
  _K = J->dual_dimension();
  if ( _K>0 ) {
    _P = new vector_valued_function_2D;
    _P->alloc( _G,_K );
    _step_P = new float[_K];
    TRACE5( "  alloc P " << _K << " layers." << endl );
  }

  // Alloc extra regularizer primals
  _step_V = NULL;
  _V = NULL;
  _M = J->extra_primal_dimension();
  if ( _M>0 ) {
    _V = new vector_valued_function_2D;
    _V->alloc( _G,_M );
    _step_V = new float[_M];
    TRACE5( "  alloc V " << _M << " layers." << endl );
  }

  // Alloc extra data term duals
  _step_Q = NULL;
  _Q = NULL;
  _L = F->dual_dimension();
  if ( _L>0 ) {
    _Q = new vector_valued_function_2D;
    _Q->alloc( _G,_L );
    _step_Q = new float[_L];
    TRACE5( "  alloc Q " << _L << " layers." << endl );
  }
}

solver::~solver()
{
  delete _U;
  delete _V;
  delete _P;
  delete _Q;
  delete[] _step_U;
  delete[] _step_V;
  delete[] _step_P;
  delete[] _step_Q;
}


// Get problem
variational_model *solver::problem()
{
  return _problem;
}


// Get solution state
vector_valued_function_2D *solver::U()
{
  return _U;
}

const vector_valued_function_2D *solver::U() const
{
  return _U;
}

    
// Set value of a parameter
bool solver::set_parameter( const string &name, double value )
{
  _params[name] = value;
  return true;
}

// Get value of a parameter (defaults to 0.0 if not set explicitly)
double solver::get_parameter( const string &name ) const
{
  map<string,double>::const_iterator it = _params.find( name );
  if ( it == _params.end() ) {
    TRACE5( "solver :: queried undefined parameter " << name << endl );
    return 0.0;
  }
  return (*it).second;
}


// Solve the problem according to the given stopping criterion
bool solver::solve( stopping_criterion *sc )
{
  initialize();
  int iteration = 0;
  while ( !sc->stop( iteration, U() )) {
    iterate();
    iteration ++;
  }
  return true;
}

// Re-initialize solver using current solution as starting point
bool solver::initialize()
{
  variational_model *IP = problem();
  assert( IP != NULL );
  regularizer *J = IP->J();
  assert( J != NULL );
  data_term *F = IP->F();
  assert( F != NULL );


  // init preconditioned step sizes
  memset( _step_U, 0, _U->N() * sizeof(float) );
  J->accumulate_operator_norm_U( _step_U );
  F->accumulate_operator_norm_U( _step_U );
  for ( int i=0; i<_U->N(); i++ ) {
    assert( _step_U[i] > 0.0f );
    _step_U[i] = 1.0f / _step_U[i];
    TRACE6( "  stepsize U " << i << "  = " << _step_U[i] << endl );
  }

  if ( _step_P != NULL ) {
    memset( _step_P, 0, _P->N() * sizeof(float) );
    J->accumulate_operator_norm_P( _step_P );
    for ( int i=0; i<_P->N(); i++ ) {
      assert( _step_P[i] > 0.0f );
      _step_P[i] = 1.0f / _step_P[i];
      TRACE6( "  stepsize P " << i << "  = " << _step_P[i] << endl );
    }
  }
  if ( _step_V != NULL ) {
    memset( _step_V, 0, _V->N() * sizeof(float) );
    J->accumulate_operator_norm_V( _step_V );
    for ( int i=0; i<_V->N(); i++ ) {
      assert( _step_V[i] > 0.0f );
      _step_V[i] = 1.0f / _step_V[i];
      TRACE6( "  stepsize V " << i << "  = " << _step_V[i] << endl );
    }
  }
  if ( _step_Q != NULL ) {
    memset( _step_Q, 0, _Q->N() * sizeof(float) );
    F->accumulate_operator_norm_Q( _step_Q );
    for ( int i=0; i<_Q->N(); i++ ) {
      assert( _step_Q[i] > 0.0f );
      _step_Q[i] = 1.0f / _step_Q[i];
      TRACE6( "  stepsize Q " << i << "  = " << _step_Q[i] << endl );
    }
  }

  // nothing else should be reasonably done by default
  return true;
}

// Perform one iteration
bool solver::iterate()
{
  // no sensible default implementation
  assert( false );
  return false;
}



// Extragradient step.
//   In:  old value in V, updated value in Vq
//   Out: updated value in V, extragradient value in Vq
bool solver::extragradient_step( float theta, vector_valued_function_2D *V, vector_valued_function_2D *Vq )
{
  // Default is one gradient operator step
  assert( V != NULL );
  assert( V->equal_dim( Vq ));

  // Assumed memory layout is consecutive storage of dual variables
  // for each dimension, starting at index 0

  // Kernel call for each channel
  for ( int i=0; i<V->N(); i++ ) {
    kernel_extragradient_step
      ( _G,
	theta,
	V->channel(i),
	Vq->channel(i) );
  }

  return true;
}



bool solver::trace_pixel( int x, int y ) const
{
  // trace for all functions
  TRACE( "Pixel trace (" << x << " " << y << ") " << endl );
  TRACE( "  Primal Variables" << endl );
  _U->trace_pixel( x,y );
  _V->trace_pixel( x,y );
  TRACE( "  Dual Variables" << endl );
  _P->trace_pixel( x,y );
  _Q->trace_pixel( x,y );
  return true;
}



// Copying and assigning forbidden.
// These are inherently inefficient, or, if implemented efficiently via pointer copies,
// can easily lead to unintended side effects. Copy operations must always be initiated
// explicitly.
solver &solver::operator= ( const solver & )
{
  assert(false);
  return *this;
}

solver::solver( const solver & )
{
  assert(false);
}
