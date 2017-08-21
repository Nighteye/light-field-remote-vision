#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <iostream>
#include <sys/stat.h>

#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_linalg.h>

#include "../common/gsl_image.h"
#include "../common/gsl_matrix_helper.h"
#include "../common/gsl_matrix_derivatives.h"

#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>

//#include "multidim_tv.h"

//#include "config.h"
#include "vtv_rof_solvers.h"

using namespace gov;
using namespace std;



// Map solver to images
multidim_tv_rof_solver_data* multidim_tv_rof_solver_create( gsl_image *U, gsl_image *F )
{
  multidim_tv_rof_solver_data *data = new multidim_tv_rof_solver_data;
  data->_U.push_back( U->_r );
  data->_U.push_back( U->_g );
  data->_U.push_back( U->_b );
  data->_F.push_back( F->_r );
  data->_F.push_back( F->_g );
  data->_F.push_back( F->_b );
  data->_energy = 0.0;
  data->_energy_smooth = 0.0;
  data->_energy_data = 0.0;
  data->_lambda = 1.0;
  data->_maxiter = 100;
  return data;
}

// Destroy solver
bool multidim_tv_rof_solver_free( multidim_tv_rof_solver_data *data )
{
  if ( data == NULL ) {
    assert( false );
    return false;
  }
  delete data;
  return true;
}



#ifdef __UNDEFINED

/****************************************************************************************
CPU SOLVER FOR MTV STEEPEST DESCENT
*****************************************************************************************/

/*
struct sv_gradient_workspace
{
  gsl_matrix *_U;
  gsl_matrix *_V;
  gsl_vector *_work;
  gsl_vector *_s;
  size_t _N;
  size_t _M;
};

sv_gradient_workspace *sv_gradient_workspace_alloc( size_t N, size_t M )
{
  sv_gradient_workspace *w = new sv_gradient_workspace;
  w->_U = gsl_matrix_alloc( N, M );
  w->_V = gsl_matrix_alloc( M, M );
  w->_work = gsl_vector_alloc( M );
  w->_s = gsl_vector_alloc( M );
  w->_N = N;
  w->_M = M;
  return w;
}

bool sv_gradient_workspace_free( sv_gradient_workspace *w )
{
  if ( w==NULL ) {
    assert( false );
    return false;
  }
  gsl_matrix_free( w->_U );
  gsl_matrix_free( w->_V );
  gsl_vector_free( w->_s );
  gsl_vector_free( w->_work );
  delete w;
  return true;
}

double compute_largest_sv( sv_gradient_workspace* w, const gsl_matrix *A, size_t i, size_t j, double a_ij )
{
  gsl_matrix_copy( A, w->_U );
  if ( i<w->_N && j<w->_M ) {
    // Adjust for derivative
    gsl_matrix_set( w->_U, i,j, a_ij );
  }
  gsl_linalg_SV_decomp( w->_U,w->_V, w->_s, w->_work );
  return w->_s->data[0];
}

double compute_sv_gradient( sv_gradient_workspace *w, const gsl_matrix *A, gsl_matrix *D )
{
  // First perform a singular value decomposition on A
  // Currently only rectangular A supported.
  assert( w != NULL );
  assert( A != NULL );
  assert( D != NULL );
  assert( A->size1 == w->_N );
  assert( A->size2 == w->_M );
  double s_A = compute_largest_sv( w, A, w->_N, w->_M, 0.0 );

  // Compute derivative for largest SV
  // Selector is the "0" below.
  for ( size_t i=0; i<w->_N; i++ ) {
    for ( size_t j=0; j<w->_M; j++ ) {
      // Add a small variation to A at ij, compute new max SV
      // Compute difference quotient as first-order approximation
      double a_ij = gsl_matrix_get( A, i,j );
      double eps = max( 0.01, a_ij / 10.0 );
      double s_ij = compute_largest_sv( w, A, i,j, a_ij + eps );
      gsl_matrix_set( D, i,j, ( s_ij - s_A ) / eps );
    }
  }

  return s_A;
}

static sv_gradient_workspace *__sv_workspace = NULL;
double sv_gradient_flux( const gsl_matrix *A, gsl_matrix *D )
{
  assert( __sv_workspace != NULL );
  return compute_sv_gradient( __sv_workspace, A,D );
}


void test_largest_sv_gradient()
{
  // Given a matrix, find the descent direction such
  // that the largest singular value becomes smaller.

  // Create random NxM test matrix A
  size_t M = 2;
  size_t N = 3;
  sv_gradient_workspace *w = sv_gradient_workspace_alloc( 3,2 );
  gsl_matrix *A = gsl_matrix_alloc( N,N );
  gsl_matrix_set_all( A, 0.0 );
  for ( size_t i=0; i<N; i++ ) {
    for ( size_t j=0; j<M; j++ ) {
      gsl_matrix_set( A, i,j, get_clamped_normal_distribution( 1.0, 2.0 ));
    }
  }

  // Test gradient descent
  gsl_matrix *D = gsl_matrix_alloc( N,N );
  gsl_matrix *Dnew = gsl_matrix_alloc( N,N );
  gsl_matrix *Anew = gsl_matrix_alloc( N,N );
  double max_s = compute_sv_gradient( w, A,D );
  cout << "initial sv " << max_s << endl;
  double t = 10.0;
  size_t iter = 0;
  do {
  __repeat_test:
    if ( t < 1e-5 ) break;
    gsl_matrix_add_scaled( 1.0, A, -t, D, Anew );
    {
      cout << "  testing step size " << t << " current max " << max_s << endl;
      double new_s = compute_sv_gradient( w, Anew, Dnew );
      if ( new_s < max_s ) {
	// Accept step
	gsl_matrix_copy( Anew, A );
	gsl_matrix_copy( Dnew, D );
	max_s = new_s;
	cout << "success, new max " << new_s << endl;
      }
      else {
	// Half ts, repeat
	t = t/2.0;
	cout << "    failed with ns = " << new_s << endl;
	goto __repeat_test;
      }
    }
  } while (iter++<20);

  sv_gradient_workspace_free( w );
  gsl_matrix_free( A );
  gsl_matrix_free( D );
  gsl_matrix_free( Anew );
  gsl_matrix_free( Dnew );
}


typedef double (*flux_function)( const gsl_matrix *A, gsl_matrix *D );

double gsl_matrix_compute_flux( const vector<gsl_matrix*> &HX_DX,
				const vector<gsl_matrix*> &HX_DY,
				const vector<gsl_matrix*> &HY_DX,
				const vector<gsl_matrix*> &HY_DY,
				const vector<gsl_matrix*> &JX,
				const vector<gsl_matrix*> &JY,
				flux_function flux_from_jacobian )
{
  size_t nc = HX_DX.size();
  assert( HX_DY.size() == nc );
  assert( HY_DX.size() == nc );
  assert( HY_DY.size() == nc );
  assert( JX.size() == nc );
  assert( JY.size() == nc );
  // Hardcoded loops atm.
  assert( nc == 3 );
  size_t W = HX_DX[0]->size2-1;
  size_t H = HX_DX[0]->size1;
  for ( size_t i=0; i<nc; i++ ) {
    assert( HX_DX[i]->size2 == W+1 );
    assert( HX_DX[i]->size1 == H );
    assert( HX_DY[i]->size2 == W+1 );
    assert( HX_DY[i]->size1 == H );

    assert( HY_DX[i]->size2 == W );
    assert( HY_DX[i]->size1 == H+1 );
    assert( HY_DY[i]->size2 == W );
    assert( HY_DY[i]->size1 == H+1 );

    assert( JX[i]->size2 == W+1 );
    assert( JX[i]->size1 == H );
    assert( JY[i]->size2 == W );
    assert( JY[i]->size1 == H+1 );
  }

  // Temp matrices
  double energy = 0.0;
  gsl_matrix *A = gsl_matrix_alloc( 3,2 );
  gsl_matrix *D = gsl_matrix_alloc( 3,2 );
  gsl_matrix_set_all( A,0.0 );

  // 1. Halfway flux X-direction
  size_t index = 0;
  for ( size_t y=0; y<H; y++ ) {
    for ( size_t x=0; x<=W; x++ ) {
      // Construct DU
      gsl_matrix_set( A, 0,0, HX_DX[0]->data[index] );
      gsl_matrix_set( A, 0,1, HX_DY[0]->data[index] );
      gsl_matrix_set( A, 1,0, HX_DX[1]->data[index] );
      gsl_matrix_set( A, 1,1, HX_DY[1]->data[index] );
      gsl_matrix_set( A, 2,0, HX_DX[2]->data[index] );
      gsl_matrix_set( A, 2,1, HX_DY[2]->data[index] );
      // Compute SVD of DU
      // Compute derivative matrix of largest SV at DU
      energy += flux_from_jacobian( A,D );

      // X-direction used only.
      JX[0]->data[index] = gsl_matrix_get( D, 0,0 );
      JX[1]->data[index] = gsl_matrix_get( D, 1,0 );
      JX[2]->data[index] = gsl_matrix_get( D, 2,0 );

      // Done
      index++;
    }
  }

  // 1. Halfway flux Y-direction
  index = 0;
  for ( size_t y=0; y<=H; y++ ) {
    for ( size_t x=0; x<W; x++ ) {
      // Construct DU
      gsl_matrix_set( A, 0,0, HY_DX[0]->data[index] );
      gsl_matrix_set( A, 0,1, HY_DY[0]->data[index] );
      gsl_matrix_set( A, 1,0, HY_DX[1]->data[index] );
      gsl_matrix_set( A, 1,1, HY_DY[1]->data[index] );
      gsl_matrix_set( A, 2,0, HY_DX[2]->data[index] );
      gsl_matrix_set( A, 2,1, HY_DY[2]->data[index] );
      // Compute SVD of DU
      // Compute derivative matrix of largest SV at DU
      energy += flux_from_jacobian( A,D );

      // Y-direction used only.
      JY[0]->data[index] = gsl_matrix_get( D, 0,1 );
      JY[1]->data[index] = gsl_matrix_get( D, 1,1 );
      JY[2]->data[index] = gsl_matrix_get( D, 2,1 );

      // Done
      index++;
    }
  }

  gsl_matrix_free( D );
  gsl_matrix_free( A );
  return energy / 2.0;
}
*/


// Perform steepest descent for TV-ROF model with spectral norm
// Nonnegativity discretization
/*
bool cpu_multidim_tv_rof_spectral_descent( multidim_tv_rof_solver_data &data )
{
  //  test_largest_sv_gradient();
  //assert( false );
  double dt = 0.005;

  // Input validation
  size_t nc = data._U.size();
  assert( nc > 0 );
  assert( data._F.size() == nc );
  size_t W = data._U[0]->size2;
  size_t H = data._U[0]->size1;
  for ( size_t i=0; i<nc; i++ ) {
    if ( data._U[i]->size2 != W || data._U[i]->size1 != H ) {
      assert( false );
    }
    if ( data._F[i]->size2 != W || data._F[i]->size1 != H ) {
      assert( false );
    }
  }

  assert( data._F.size() == nc );

  // Compute gradients of U at halfway points
  vector<gsl_matrix*> HX_UX;
  vector<gsl_matrix*> HX_UY;
  vector<gsl_matrix*> HY_UX;
  vector<gsl_matrix*> HY_UY;
  for ( size_t i=0; i<nc; i++ ) {
    gsl_matrix *hx_ux = gsl_matrix_alloc( H,W+1 );
    gsl_matrix *hx_uy = gsl_matrix_alloc( H,W+1 );
    gsl_matrix *hy_ux = gsl_matrix_alloc( H+1,W );
    gsl_matrix *hy_uy = gsl_matrix_alloc( H+1,W );

    gsl_matrix_gradient_halfway_x_dx_neumann( 1.0, data._U[i], hx_ux );
    gsl_matrix_gradient_halfway_x_dy_neumann( 1.0, data._U[i], hx_uy );
    gsl_matrix_gradient_halfway_y_dx_neumann( 1.0, data._U[i], hy_ux );
    gsl_matrix_gradient_halfway_y_dy_neumann( 1.0, data._U[i], hy_uy );

    HX_UX.push_back( hx_ux );
    HX_UY.push_back( hx_uy );
    HY_UX.push_back( hy_ux );
    HY_UY.push_back( hy_uy );
  }

  // Compute Euler-Lagrange equations:
  // In each point, derivative matrix of f=largest singular value of A at DU times DU
  // Alloc temp matrices for SVD
  sv_gradient_workspace *w = sv_gradient_workspace_alloc( 3,2 );
  __sv_workspace = w;
  // Alloc temp matrices for flux components
  vector<gsl_matrix*> JX;
  vector<gsl_matrix*> JY;
  for ( size_t i=0; i<nc; i++ ) {
    gsl_matrix *jx = gsl_matrix_alloc( H,W+1 );
    gsl_matrix *jy = gsl_matrix_alloc( H+1,W );
    JX.push_back( jx );
    JY.push_back( jy );
  }

  // Compute flux components
  data._energy_smooth = gsl_matrix_compute_flux( HX_UX, HX_UY, HY_UX, HY_UY, JX, JY,
						 &sv_gradient_flux );

  // Finally, compute divergence to get Frechet derivative
  vector<gsl_matrix*> D;
  for ( size_t i=0; i<nc; i++ ) {
    D.push_back( gsl_matrix_alloc( H,W ) );
  }

  for ( size_t i=0; i<nc; i++ ) {
    gsl_matrix *jx = JX[i];
    gsl_matrix *jy = JY[i];
    gsl_matrix *d = D[i];
    for ( size_t y=0; y<H; y++ ) {
      for ( size_t x=0; x<W; x++ ) {
	double div_x = gsl_matrix_get( jx, y,x+1 ) - gsl_matrix_get( jx, y,x );
	double div_y = gsl_matrix_get( jy, y+1,x ) - gsl_matrix_get( jy, y,x );
	gsl_matrix_set( d, y,x, div_x + div_y );
      }
    }
    gsl_matrix_stats stats = gsl_matrix_get_stats( D[i] );
    //    gsl_matrix_get_stats( UX[i], stats );
    TRACE6( "stats derivative " << i << ":" << endl );
    TRACE6( stats << endl );
  }

  // UX[i] now holds Frechet derivative for each channel.
  TRACE7( "Current MTV: " << data._energy_smooth << endl );

  // Perform gradient descent
  data._energy = data._energy_smooth;
  for ( size_t i=0; i<nc; i++ ) {
    gsl_matrix *mtv_el = D[i];
    gsl_matrix *u = data._U[i];
    gsl_matrix *f = data._F[i];
    size_t index = 0;
    for ( size_t y=0; y<H; y++ ) {
      for ( size_t x=0; x<W; x++ ) {
	double step = mtv_el->data[index];

	double diff = u->data[index] - f->data[index];
	data._energy += 2.0 / data._lambda * diff*diff;
	step -= diff / data._lambda;

	double unew = u->data[index] + dt * step;
	unew = clamp( unew, 0.0, 1.0 );
	u->data[index] = unew;
	index++;
      }
    }
  }

  // Cleanup
  sv_gradient_workspace_free( w );
  __sv_workspace = NULL;
  for ( size_t i=0; i<nc; i++ ) {
    gsl_matrix_free( HX_UX[i] );
    gsl_matrix_free( HX_UY[i] );
    gsl_matrix_free( HY_UX[i] );
    gsl_matrix_free( HY_UY[i] );
    gsl_matrix_free( JX[i] );
    gsl_matrix_free( JY[i] );
    gsl_matrix_free( D[i] );
  }
  HX_UX.clear();
  HX_UY.clear();
  HY_UX.clear();
  HY_UY.clear();
  JX.clear();
  JY.clear();
  D.clear();


  // Normalize energies and return
  data._energy_smooth /= double(W*H);
  data._energy /= double(W*H);
  data._energy_data = data._energy - data._energy_smooth;
  return true;
}
*/
  




/****************************************************************************************
GPU SOLVER FOR MTV PRIMAL DUAL
*****************************************************************************************/


double projection_error( gsl_vector *v, gsl_vector *xi, gsl_vector *eta )
{
  size_t M = xi->size;
  size_t N = eta->size;
  assert( v->size == N*M );
  gsl_vector *tmp = gsl_vector_alloc( M*N );
  for ( size_t n=0; n<N; n++ ) {
    for ( size_t m=0; m<M; m++ ) {
      double z = gsl_vector_get( xi, m ) * gsl_vector_get( eta, n );
      z -= gsl_vector_get( v, n*M + m );
      gsl_vector_set( tmp, n*M + m, z );
    }
  }
  double err = gsl_vector_norm( tmp );
  gsl_vector_free( tmp );
  return err;
}


bool tensor_projection( gsl_vector *v, gsl_matrix *V, gsl_vector *xi_out, gsl_vector *eta_out )
{
  bool ok = false;
  size_t M = xi_out->size;
  size_t N = eta_out->size;
  gsl_vector *xi = gsl_vector_alloc( xi_out->size );
  gsl_vector *eta = gsl_vector_alloc( eta_out->size );

  // 3. New projection: Eigenvector directions of VV^T and V^T V
  // Transpose and multiply V
  gsl_matrix *Vt = gsl_matrix_alloc( M,N );
  gsl_matrix_transpose_memcpy( Vt, V );
  gsl_matrix *VtV = gsl_matrix_alloc( M,M );
  gsl_matrix_product( VtV, Vt, V );
  gsl_matrix *VVt = gsl_matrix_alloc( N,N );
  gsl_matrix_product( VVt, V, Vt );
  // Eigenvalue decomposition for xi
  gsl_matrix *evecM = gsl_matrix_alloc( M,M );
  gsl_vector *evalM = gsl_vector_alloc( M );
  gsl_eigen_symmv_workspace *wM = gsl_eigen_symmv_alloc( M );
  gsl_eigen_symmv( VtV, evalM, evecM, wM );
  gsl_eigen_symmv_free( wM );
  // Test: Compare to result of simple lib
  {
    Mat22d M;
    M._11 = gsl_matrix_get( VtV, 0,0 );
    M._12 = gsl_matrix_get( VtV, 0,1 );
    M._21 = gsl_matrix_get( VtV, 1,0 );
    M._22 = gsl_matrix_get( VtV, 1,1 );
    Mat22d E;
    gsl_matrix_eigensystem( M,E );
    cout << "test eigensystem: " << M << endl;
  }

  // Eigenvalue decomposition for eta
  gsl_matrix *evecN = gsl_matrix_alloc( N,N );
  gsl_vector *evalN = gsl_vector_alloc( N );
  gsl_eigen_symmv_workspace *wN = gsl_eigen_symmv_alloc( N );
  gsl_eigen_symmv( VVt, evalN, evecN, wN );
  gsl_eigen_symmv_free( wN );
  // Check distances of all decompositions
  double err_min = DOUBLE_MAX;
  for ( size_t m=0; m<M; m++ ) {
    for ( size_t i=0; i<M; i++ ) {
      gsl_vector_set( xi, i, gsl_matrix_get( evecM, i,m ));
    }

    // Variant 0: Standard optimization for eta
    double L = max( 0.01, gsl_vector_norm( xi ));
    gsl_vector_set_all( eta, 0.0 );
    for ( size_t n=0; n<N; n++ ) {
      double sp = 0.0;
      for ( size_t m=0; m<M; m++ ) {
	sp += gsl_vector_get( xi, m ) * gsl_matrix_get( V, n,m );
      }
      sp /= L;
      gsl_vector_set( eta, n, sp );
    }
    gsl_vector_reproject( eta );

    double err = projection_error( v, xi, eta );
    if ( err < err_min ) {
      err_min = err;
      ok = true;
      memcpy( xi_out->data, xi->data, M*sizeof(double) );
      memcpy( eta_out->data, eta->data, N*sizeof(double) );
    }
  }
  gsl_vector_free( evalN );
  gsl_vector_free( evalM );
  gsl_matrix_free( evecM );
  gsl_matrix_free( evecN );

  // Free up stuff
  gsl_matrix_free( VtV );
  gsl_matrix_free( VVt );
  gsl_matrix_free( Vt );
  gsl_vector_free( xi );
  gsl_vector_free( eta );
  return ok;
}


void test_tensor_projection()
{
  srand( time(NULL) );

  // Channels/Space
  size_t N = 3;
  size_t M = 2;
  // Create random space direction per channel, stack into vector and matrix
  gsl_vector *v = gsl_vector_alloc( N*M );
  gsl_matrix *V = gsl_matrix_alloc( N,M );
  for ( size_t n=0; n<N; n++ ) {
    for ( size_t m=0; m<M; m++ ) {
      double z = ( double(rand() % 2000) - 1000.0 ) / 1000.0;
      gsl_vector_set( v, n*M + m, z );
      gsl_matrix_set( V, n,m, z );
    }
  }
  cout << "v  : ";
  gsl_vector_out( v );
  cout << endl;

  // 1. Naive projection test: average of directions, project onto average
  gsl_vector *xi = gsl_vector_alloc( M );
  gsl_vector_set_all( xi, 0.0 );
  for ( size_t m=0; m<M; m++ ) {
    for ( size_t n=0; n<N; n++ ) {
      double v = gsl_vector_get( xi, m ) + gsl_matrix_get( V, n,m );
      gsl_vector_set( xi, m, v );
    }
  }
  gsl_vector_reproject( xi );
  double L = max( 0.01, gsl_vector_norm( xi ));
  gsl_vector *eta = gsl_vector_alloc( N );
  gsl_vector_set_all( eta, 0.0 );
  for ( size_t n=0; n<N; n++ ) {
    double sp = 0.0;
    for ( size_t m=0; m<M; m++ ) {
      sp += gsl_vector_get( xi, m ) * gsl_matrix_get( V, n,m ) / double(N);
    }
    sp /= L;
    gsl_vector_set( eta, n, sp );
  }
  gsl_vector_reproject( eta );
  double err = projection_error( v, xi, eta );

  cout << "eta: ";
  gsl_vector_out( eta );
  cout << endl;
  cout << "xi : ";
  gsl_vector_out( xi );
  cout << endl;
  cout << "Projection error (t0) = " << err << endl;
  cout << endl;

  // 2. Naive projection test: average of normalized directions, project onto average
  gsl_vector_set_all( xi, 0.0 );
  for ( size_t n=0; n<N; n++ ) {
    double vl = 0.0;
    for ( size_t m=0; m<M; m++ ) {
      vl += square( gsl_matrix_get( V, n,m ) );
    }
    vl = sqrt( vl );
    for ( size_t m=0; m<M; m++ ) {
      double v = gsl_vector_get( xi, m ) + gsl_matrix_get( V, n,m ) / ( vl * N );
      gsl_vector_set( xi, m, v );
    }
  }
  gsl_vector_reproject( xi );
  L = max( 0.01, gsl_vector_norm( xi ));
  gsl_vector_set_all( eta, 0.0 );
  for ( size_t n=0; n<N; n++ ) {
    double sp = 0.0;
    for ( size_t m=0; m<M; m++ ) {
      sp += gsl_vector_get( xi, m ) * gsl_matrix_get( V, n,m );
    }
    sp /= L;
    gsl_vector_set( eta, n, sp );
  }
  gsl_vector_reproject( eta );
  err = projection_error( v, xi, eta );

  cout << "eta: ";
  gsl_vector_out( eta );
  cout << endl;
  cout << "xi : ";
  gsl_vector_out( xi );
  cout << endl;
  cout << "Projection error (t1) = " << err << endl;


  // (Hopefully) correct projection
  tensor_projection( v, V, xi, eta );
  err = projection_error( v, xi, eta );
  cout << endl;
  cout << " EVP test" << endl;
  cout << "  eta: ";
  gsl_vector_out( eta );
  cout << endl;
  cout << "  xi : ";
  gsl_vector_out( xi );
  cout << endl;
  cout << "  eps: " << err;

  gsl_vector_free( xi );
  gsl_vector_free( eta );
  gsl_vector_free( v );
  gsl_matrix_free( V );
}


/****************************** COMBINED, DUAL ASCEND ************************************/
// Perform primal-dual optimization for TV-ROF model with spectral norm solver
bool cpu_multidim_tv_rof_spectral_pd( multidim_tv_rof_solver_data &input )
{
  cuda_multidim_tv_data* data = cuda_multidim_tv_alloc( 3, input._F );
  cuda_multidim_tv_initialize( data, input._U );
  //cuda_multidim_tv_set_dual_eta( data, input._O );
  data->_lambda = input._lambda;

  // Iterate
  TRACE1( "Multidim TV (GPU, combined channels) ..." << endl );
  TRACE1( "  [" );
  for ( size_t iter=0; iter<input._maxiter; iter++ ) {
    if ( (iter% (input._maxiter/10) )==0 ) {
      TRACE1( "." );
    }

    /*
    cuda_multidim_tv_rof_combined_dual_step( data );
    cuda_multidim_tv_rof_combined_primal_step( data );
    */

    cuda_multidim_tv_rof_pg_tensor_dual_step( data );
    cuda_multidim_tv_rof_pg_tensor_reproject( data );
    cuda_multidim_tv_rof_pg_tensor_primal_step( data );
  }
  TRACE1( "] done." << endl );

  // Write result
  cuda_multidim_tv_get_solution( data, input._U );
  cuda_multidim_tv_free( data );
  return true;
}






/****************************** CHANNEL-BY-CHANNEL SOLVER ************************************/

// Perform primal-dual optimization for TV-ROF model with channel-by-channel norm
bool cpu_multidim_tv_rof_cbc_pd( multidim_tv_rof_solver_data &input )
{
  cuda_multidim_tv_data* data = cuda_multidim_tv_alloc( 3, input._F );
  cuda_multidim_tv_initialize( data, input._U );
  data->_lambda = input._lambda;

  // Iterate separate channels
  TRACE1( "Multidim TV (GPU, separate channels) ..." << endl );
  TRACE1( "[" );
  for ( size_t iter=0; iter<input._maxiter; iter++ ) {
    if ( (iter% (input._maxiter/10) )==0 ) {
      TRACE1( "." );
    }
    cuda_multidim_tv_rof_separated_dual_step( data );
    cuda_multidim_tv_rof_separated_primal_step( data );
  }
  TRACE1( "] done." << endl );

  // Write result
  cuda_multidim_tv_get_solution( data, input._U );
  cuda_multidim_tv_free( data );
  return true;
}




/****************************** SINGLE DIM, OPERATE ON NORM ************************************/
    
// Perform primal-dual optimization for TV-ROF model on black/white image
bool cpu_multidim_tv_rof_bw_pd( multidim_tv_rof_solver_data &input )
{
  size_t W = input._F[0]->size2;
  size_t H = input._F[0]->size1;
  // TODO: Input validation.
  gsl_matrix *Mu = gsl_matrix_alloc( H,W );
  gsl_matrix *Mf = gsl_matrix_alloc( H,W );
  vector<gsl_matrix*> Su;
  Su.push_back( Mu );
  vector<gsl_matrix*> Sf;
  Sf.push_back( Mf );
  size_t index = 0;
  for ( size_t y=0; y<H; y++ ) {
    for ( size_t x=0; x<W; x++ ) {
      double r = input._F[0]->data[index];
      double g = input._F[1]->data[index];
      double b = input._F[2]->data[index];
      double v = sqrt( r*r + g*g + b*b );
      Mu->data[index] = v;
      Mf->data[index] = v;
      index++;
    }
  }
    
  // Iterate separate channels
  cuda_multidim_tv_data* sdata = cuda_multidim_tv_alloc( 1, Sf );
  cuda_multidim_tv_initialize( sdata, Su );
  sdata->_lambda = input._lambda;
  
  TRACE1( "Singledim TV (GPU) ..." << endl );
  TRACE1( "[" );
  for ( size_t iter=0; iter<input._maxiter; iter++ ) {
    if ( (iter% (input._maxiter/10) )==0 ) {
      TRACE1( "." );
    }
    cuda_multidim_tv_rof_separated_dual_step( sdata );
    cuda_multidim_tv_rof_separated_primal_step( sdata );
  }
  TRACE1( "] done." << endl );
  
  // Write result
  cuda_multidim_tv_get_solution( sdata, Su );
  // Copy to channels of input
  gsl_matrix_copy( Mu, input._U[0] );
  gsl_matrix_copy( Mu, input._U[1] );
  gsl_matrix_copy( Mu, input._U[2] );
  gsl_matrix_free( Mu );
  gsl_matrix_free( Mf );
  cuda_multidim_tv_free( sdata );
  return true;
}




/****************************** COMBINED, DAV ************************************/

// Perform primal-dual optimization for TV-ROF model with DAV solver
bool cpu_multidim_tv_rof_dav_pd( multidim_tv_rof_solver_data &input )
{
  cuda_multidim_tv_data* data = cuda_multidim_tv_alloc( 3, input._F );
  cuda_multidim_tv_initialize( data, input._U );
  data->_lambda = input._lambda;

  TRACE1( "Multidim TV (GPU, combined channels, DAV method) ..." << endl );
  TRACE1( "[" );
  for ( size_t iter=0; iter<input._maxiter; iter++ ) {
    if ( (iter% (input._maxiter/10) )==0 ) {
      cout << "."; cout.flush();
    }
    cuda_multidim_tv_rof_dav_dual_step( data );
    cuda_multidim_tv_rof_dav_primal_step( data );
  }
  TRACE1( "] done." << endl );

  // Write result
  cuda_multidim_tv_get_solution( data, input._U );
  cuda_multidim_tv_free( data );
  return true;
}



/****************************** COMBINED, CVH ************************************/

// Perform primal-dual optimization for TV-ROF model with DAV solver
bool cpu_multidim_tv_rof_cvh( multidim_tv_rof_solver_data &input )
{
  cuda_multidim_tv_data* data = cuda_multidim_tv_alloc( 3, input._F );
  data->_lambda = input._lambda;
  cuda_multidim_tv_initialize( data, input._U );

  TRACE1( "Multidim TV (GPU, combined channels, CVH method) ..." << endl );
  TRACE1( "[" );
  for ( size_t iter=0; iter<input._maxiter; iter++ ) {
    if ( (iter% (input._maxiter/10) )==0 ) {
      cout << "."; cout.flush();
    }
    //    cuda_multidim_tv_rof_cvh_iteration_bermudez_morena( data );
    //cuda_multidim_tv_rof_cvh_iteration_chambolle_pock_1( data );
    cuda_multidim_tv_rof_cvh_iteration_chambolle_pock_2( data );
    //cuda_multidim_tv_rof_cvh_dual_step( data );
    //cuda_multidim_tv_rof_cvh_primal_step( data );
  }
  TRACE1( "] done." << endl );

  // Write result
  cuda_multidim_tv_get_solution( data, input._U );
  cuda_multidim_tv_free( data );
  return true;
}

#endif
