/** \file gsl_matrix_derivatives.h

    File imported from "common" lib, use if this library is not available.
    Uses "gov" namespace (Graphics-optics-vision, MPI).

    Compute numerical derivatices of matrices using finite differences.
    For all functions, h is the grid cell size.
    
    Copyright (C) 2008 Bastian Goldluecke,
    <first name>AT<last name>.net

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

#include <assert.h>
#include <math.h>

#include "gsl_matrix_derivatives.h"
#include "gsl_matrix_convolutions.h"
#include "gsl_matrix_helper.h"
#include "defs.h"

#include "linalg3d.h"


using namespace std;
using namespace coco;

// Compute numerical derivatices of matrices using finite differences
// For all functions, h is the grid scaling factor

// Derivative in x-direction, central, Neumann boundary conditions
bool coco::gsl_matrix_dx( double hx, const gsl_matrix *in, gsl_matrix *out )
{
  size_t h = in->size1;
  size_t w = in->size2;
  if ( h != out->size1 || w != out->size2 ) {
    assert( false );
    return false;
  }

  double h2 = 2.0*hx;
  double *s = in->data;
  double *d = out->data;
  for ( size_t y=0; y<h; y++ ) {
    // First entry: One-sided derivative
    *(d++) = (*(s+1) - *s) / h2;
    s++;

    // Next entries until last: Central difference
    for ( size_t x=1; x<w-1; x++ ) {
      *(d++) = (*(s+1) - *(s-1)) / h2;
      s++;
    }

    // Last entry: One-sided derivative
    *(d++) = (*s - *(s-1)) / h2;
    s++;
  }
  
  return true;
}

// Derivative in y-direction
bool coco::gsl_matrix_dy( double hy, const gsl_matrix *in, gsl_matrix *out )
{
  size_t h = in->size1;
  size_t w = in->size2;
  if ( h != out->size1 || w != in->size2 ) {
    assert( false );
    return false;
  }

  double h2 = 2.0*hy;
  double *s = in->data;
  double *d = out->data;

  // First row: One-sided derivative
  for ( size_t x=0; x<w; x++ ) {
    *(d++) = (*(s+w) - *s) / h2;
    s++;
  }

  // Next rows until last: Central difference
  for ( size_t y=1; y<h-1; y++ ) {
    for ( size_t x=0; x<w; x++ ) {
      *(d++) = (*(s+w) - *(s-w)) / h2;
      s++;
    }
  }
 
  // Last row: One-sided derivative
  for ( size_t x=0; x<w; x++ ) {
    *(d++) = (*s - *(s-w)) / h2;
    s++;
  }
 
  return true;
}


// Normalize gradient
bool coco::gsl_matrix_norm_grad( gsl_matrix *dx, gsl_matrix *dy, gsl_matrix* norm=NULL )
{
  size_t h = dy->size1;
  size_t w = dy->size2;
  if ( h != dx->size1 || w != dx->size2 ) {
    assert( false );
    return false;
  }

  const double eps = 0.001;
  double *x = dx->data;
  double *y = dy->data;
  double *m = NULL;
  if ( norm != NULL ) {
    m = norm->data;
  }
  size_t N = w*h;
  for ( size_t i=0; i<N; i++ ) {
    double l = hypot( *x, *y );
    double n = max( eps, l );
    *x /= n;
    *y /= n;

    x++;
    y++;
    if ( m != NULL ) {
      *(m++) = l;
    }
  }

  return true;
}




// Laplacian directly
bool coco::gsl_matrix_laplacian( double hx, double hy, const gsl_matrix *in, gsl_matrix *out )
{
  assert( hx==1.0 );
  assert( hy==1.0 );
  assert( in != NULL );
  assert( out != NULL );

  // Not yet implemented, compute via divergence.
  return false;
}

// Divergence of a vector field
bool coco::gsl_matrix_divergence( double hx, double hy, 
			   const gsl_matrix *dx,
			   const gsl_matrix *dy,
			   gsl_matrix *out )
{
  size_t h = out->size1;
  size_t w = out->size2;
  if ( h != dx->size1 || w != dx->size2 ) {
    assert( false );
    return false;
  }
  if ( h != dy->size1 || w != dy->size2 ) {
    assert( false );
    return false;
  }

  // Second derivatives
  gsl_matrix *ddx = gsl_matrix_alloc( h,w );
  if ( ddx==NULL ) {
    return false;
  }
  if ( !gsl_matrix_dx( hx, dx,ddx )) {
    gsl_matrix_free( ddx );
    return false;
  }

  gsl_matrix *ddy = gsl_matrix_alloc( h,w );
  if ( ddy==NULL ) {
    gsl_matrix_free( ddx );
    return false;
  }
  if ( !gsl_matrix_dy( hy, dy,ddy )) {
    gsl_matrix_free( ddx );
    gsl_matrix_free( ddy );
    return false;
  }

  // Sum up to compute Laplacian
  double *sx = ddx->data;
  double *sy = ddy->data;
  double *d = out->data;
  size_t N = w*h;
  for ( size_t i=0; i<N; i++ ) {
    *(d++) = *(sx++) + *(sy++);
  }

  gsl_matrix_free( ddx );
  gsl_matrix_free( ddy );
  return true;
}




// Derivative in x-direction, forward differences
bool coco::gsl_matrix_dx_forward( double hx, const gsl_matrix *in, gsl_matrix *out )
{
  size_t h = in->size1;
  size_t w = in->size2;
  if ( h != out->size1 || w != out->size2 ) {
    assert( false );
    return false;
  }

  double *s = in->data;
  double *d = out->data;
  for ( size_t y=0; y<h; y++ ) {
    // Entries until last: Forward difference
    for ( size_t x=0; x<w-1; x++ ) {
      *(d++) = (*(s+1) - *s) / hx;
      s++;
    }

    // Last entry: zero (Neumann conditions)
    *(d++) = 0.0;
    s++;
  }
  
  return true;
}

// Derivative in y-direction
bool coco::gsl_matrix_dy_forward( double hy, const gsl_matrix *in, gsl_matrix *out )
{
  size_t h = in->size1;
  size_t w = in->size2;
  if ( h != out->size1 || w != in->size2 ) {
    assert( false );
    return false;
  }

  double *s = in->data;
  double *d = out->data;

  // Until last row: Forward difference
  for ( size_t y=0; y<h-1; y++ ) {
    for ( size_t x=0; x<w; x++ ) {
      *(d++) = (*(s+w) - *s) / hy;
      s++;
    }
  }
 
  // Last row: Zero ( Neumann condition )
  for ( size_t x=0; x<w; x++ ) {
    *(d++) = 0.0;
    s++;
  }
 
  return true;
}



// Derivative in x-direction, backward differences (Dirichlet)
bool coco::gsl_matrix_dx_backward( double hx, const gsl_matrix *in, gsl_matrix *out )
{
  size_t h = in->size1;
  size_t w = in->size2;
  if ( h != out->size1 || w != out->size2 ) {
    assert( false );
    return false;
  }

  double *s = in->data;
  double *d = out->data;
  for ( size_t y=0; y<h; y++ ) {
    // First entry: Copy
    *(d++) = *(s++) / hx;

    // Entries until last: Backward difference
    for ( size_t x=0; x<w-1; x++ ) {
      *(d++) = (*s - *(s-1)) / hx;
      s++;
    }
  }
  
  return true;
}

// Derivative in y-direction, backward differencing (Dirichlet)
bool coco::gsl_matrix_dy_backward( double hy, const gsl_matrix *in, gsl_matrix *out )
{
  size_t h = in->size1;
  size_t w = in->size2;
  if ( h != out->size1 || w != in->size2 ) {
    assert( false );
    return false;
  }

  double *s = in->data;
  double *d = out->data;

  // First row: Copy
  for ( size_t x=0; x<w; x++ ) {
    *(d++) = *(s++) / hy;
  }

  // Until last row: Backward difference
  for ( size_t y=0; y<h-1; y++ ) {
    for ( size_t x=0; x<w; x++ ) {
      *(d++) = (*s - *(s-w)) / hy;
      s++;
    }
  }
 
  return true;
}



// Computes Euler-Lagrange equation for total variation term
// Neumann boundary conditions, see
//
// A ROBUST ALGORITHM FOR TOTAL VARIATION DEBLURRING AND DENOISING
// Chan,Chen,Wang,Xu
//
double coco::gsl_matrix_tv_derivative( double hx, double hy, const gsl_matrix *u, gsl_matrix *td )
{
  size_t W = u->size2;
  size_t H = u->size1;
  if ( td->size2 != W || td->size1 != H ) {
    assert( false );
    return false;
  }
  const double eps = 0.001;
  double tv = 0.0;

  double *tdd = td->data;
  const double *ud = u->data;
  for ( size_t y=0; y<H; y++ ) {
    for ( size_t x=0; x<W; x++ ) {

      // Get values in neighbourhood considering boundary conditions
      double uxcyc = *ud;
      double uxpym = uxcyc;
      double uxpyc = uxcyc;
      double uxpyp = uxcyc;
      double uxcym = uxcyc;
      double uxcyp = uxcyc;
      double uxmym = uxcyc;
      double uxmyc = uxcyc;
      double uxmyp = uxcyc;

      // Center column
      if ( y>0 ) {
	uxcym = *(ud-W);
      }
      else {
	uxcym = *ud;
      }
      if ( y<H-1 ) {
	uxcyp = *(ud+W);
      }
      else {
	uxcyp = *ud;
      }

      // Left column
      if ( x>0 ) {
	uxmyc = *(ud-1);
	if ( y>0 ) {
	  uxmym = *(ud-1-W);
	}
	else {
	  uxmym = uxmyc;
	}
	if ( y<H-1 ) {
	  uxmyp = *(ud-1+W);
	}
	else {
	  uxmyp = uxmyc;
	}
      }
      else {
	// At left border, replicate center column
	uxmym = uxcym;
	uxmyc = uxcyc;
	uxmyp = uxcyp;
      }

      // Right column
      if ( x<W-1 ) {
	uxpyc = *(ud+1);
	if ( y>0 ) {
	  uxpym = *(ud+1-W);
	}
	else {
	  uxpym = uxpyc;
	}
	if ( y<H-1 ) {
	  uxpyp = *(ud+1+W);
	}
	else {
	  uxpyp = uxpyc;
	}
      }
      else {
	// At right border, replicate center column
	uxpym = uxcym;
	uxpyc = uxcyc;
	uxpyp = uxcyp;
      }


      // Neighbourhood initialized.
      // now compute difference scheme
      double dx_cp = (0.25/hx) * ( uxpyp - uxmyp + uxpyc - uxmyc );
      double dy_cp = (uxcyp - uxcyc) / hy;

      double dx_cm = (0.25/hx) * ( uxpym - uxmym + uxpyc - uxpyc );
      double dy_cm = (uxcyc - uxcym) / hy;

      double dx_pc = (uxpyc - uxcyc) / hx;
      double dy_pc = (0.25/hy) * ( uxpyp - uxpym + uxcyp - uxcym );

      double dx_mc = (uxcyc - uxmyc) / hx;
      double dy_mc = (0.25/hy) * ( uxmyp - uxmym + uxcyp - uxcym );

      double c_cp = 1.0 / sqrt( square( dx_cp ) + square( dy_cp ) + eps );
      double c_cm = 1.0 / sqrt( square( dx_cm ) + square( dy_cm ) + eps );
      double c_pc = 1.0 / sqrt( square( dx_pc ) + square( dy_pc ) + eps );
      double c_mc = 1.0 / sqrt( square( dx_mc ) + square( dy_mc ) + eps );

      *tdd = -c_pc*dx_pc + c_mc*dx_mc - c_cp*dy_cp + c_cm*dy_cm;
      tv += hypot( dx_pc, dy_cp );

      tdd++;
      ud++;
    }
  }


  return tv;
}


typedef Vec2d( *diffusion_tensor )( double dx, double dy, const std::vector<double> &params );



// Compute Eigenvectors / values of a 2x2 matrix
int coco::gsl_matrix_eigensystem( Mat22d &M, Mat22d &E )
{
  // Compute larger Eigenvalue (= square of largest singular value)
  double d11 = M._11;
  double d12 = M._21;
  double d21 = M._12;
  if ( d21 != d12 ) {
    cout << "Eigensystem computation for non-symmetric matrix " << d12 << " " << d21 << endl;
  }
  double d22 = M._22;
  double trace = d11 + d22;
  double det = d11*d22 - d12*d12;
  double d = sqrt( 0.25*trace*trace - det );
  double lmax = max( 0.0, 0.5 * trace + d );
  double lmin = max( 0.0, 0.5 * trace - d );
  double smax = sqrt( lmax );
  double smin = sqrt( lmin );
  
  // Compute Eigensystem which fits the Eigenvectors
  double v11, v12, v21, v22;
  if ( d12 == 0.0 ) {
    if ( d11 >= d22 ) {
      v11 = 1.0; v21 = 0.0; v12 = 0.0; v22 = 1.0;
    }
    else {
      v11 = 0.0; v21 = 1.0; v12 = 1.0; v22 = 0.0;
    }
  }
  else {
    v11 = lmax - d22; v21 = d12;
    double l1 = hypot( v11, v21 );
    v11 /= l1; v21 /= l1;
    v12 = lmin - d22; v22 = d12;
    double l2 = hypot( v12, v22 );
    v12 /= l2; v22 /= l2;
  }

  // Write back result
  E._11 = smax;
  E._22 = smin;
  E._12 = 0.0;
  E._21 = 0.0;
  M._11 = v11;
  M._12 = v21;
  M._21 = v12;
  M._22 = v22;
  return (smax != 0.0) + (smin != 0.0);
}




// Diffusion tensor which applies PM-Diffusivity to
// Eigenvalues of structure tensor
Vec2d coco::diffusion_perona_malik_anisotropic( double dx, double dy, const vector<double> &params )
{
  assert( params.size() > 0 );
  // Compute Eigenvalues and Eigenvectors of structure tensor
  Mat22d M;
  M._11 = dx*dx; M._12 = dx*dy;
  M._21 = M._12; M._22 = dy*dy;
  Mat22d E;
  int nv = gsl_matrix_eigensystem( M,E );
  if ( nv != 2 ) {
    //    ERROR( "Eigenvalue decomposition failed." << endl );
    return diffusion_perona_malik_isotropic( dx, dy, params );
  }

  // Perona-Malik type reduction of first Eigenvalue
  double g = 1.0 / sqrt( 1.0 + (square(dx) + square(dy)) / square(params[0]) );
  E._11 = g * E._11;

  // Construct diffusion tensor from new Eigenvalues
  Mat22d D = E * M;
  M.transpose();
  D.preMultiply( M );
  return D * Vec2d(dx,dy);
}


// Diffusion tensor which applies PM-Diffusivity to gradient squared
Vec2d coco::diffusion_perona_malik_isotropic( double dx, double dy, const vector<double> &params )
{
  assert( params.size() > 0 );
  double g = 1.0 / sqrt( 1.0 + (square(dx) + square(dy)) / square(params[0]) );
  return Vec2d( dx*g, dy*g );
}



//
// Computes anisotropic diffusion flow field
// Neumann boundary conditions
//
// See
// PDE-Based Deconvolution with Forward-Backward Diffusivities and Diffusion Tensors
// Brox/Weickert et al
//
double coco::gsl_matrix_anisotropic_diffusion_flow( double hx, double hy,
					      const gsl_matrix *u,
					      diffusion_tensor D,
					      const vector<double> &params,
					      gsl_matrix *flow )
{
  size_t W = u->size2;
  size_t H = u->size1;
  if ( flow->size2 != W || flow->size1 != H ) {
    assert( false );
    return false;
  }
  double tv = 0.0;

  double *flow_d = flow->data;
  const double *ud = u->data;
  for ( size_t y=0; y<H; y++ ) {
    for ( size_t x=0; x<W; x++ ) {

      // Get values in neighbourhood considering boundary conditions
      double uxcyc = *ud;
      double uxpym = uxcyc;
      double uxpyc = uxcyc;
      double uxpyp = uxcyc;
      double uxcym = uxcyc;
      double uxcyp = uxcyc;
      double uxmym = uxcyc;
      double uxmyc = uxcyc;
      double uxmyp = uxcyc;

      // Center column
      if ( y>0 ) {
	uxcym = *(ud-W);
      }
      else {
	uxcym = *ud;
      }
      if ( y<H-1 ) {
	uxcyp = *(ud+W);
      }
      else {
	uxcyp = *ud;
      }

      // Left column
      if ( x>0 ) {
	uxmyc = *(ud-1);
	if ( y>0 ) {
	  uxmym = *(ud-1-W);
	}
	else {
	  uxmym = uxmyc;
	}
	if ( y<H-1 ) {
	  uxmyp = *(ud-1+W);
	}
	else {
	  uxmyp = uxmyc;
	}
      }
      else {
	// At left border, replicate center column
	uxmym = uxcym;
	uxmyc = uxcyc;
	uxmyp = uxcyp;
      }

      // Right column
      if ( x<W-1 ) {
	uxpyc = *(ud+1);
	if ( y>0 ) {
	  uxpym = *(ud+1-W);
	}
	else {
	  uxpym = uxpyc;
	}
	if ( y<H-1 ) {
	  uxpyp = *(ud+1+W);
	}
	else {
	  uxpyp = uxpyc;
	}
      }
      else {
	// At right border, replicate center column
	uxpym = uxcym;
	uxpyc = uxcyc;
	uxpyp = uxcyp;
      }


      // Neighbourhood initialized.
      // now compute difference scheme
      double dx_cp = (0.25/hx) * ( uxpyp - uxmyp + uxpyc - uxmyc );
      double dy_cp = (uxcyp - uxcyc) / hy;

      double dx_cm = (0.25/hx) * ( uxpym - uxmym + uxpyc - uxpyc );
      double dy_cm = (uxcyc - uxcym) / hy;

      double dx_pc = (uxpyc - uxcyc) / hx;
      double dy_pc = (0.25/hy) * ( uxpyp - uxpym + uxcyp - uxcym );

      double dx_mc = (uxcyc - uxmyc) / hx;
      double dy_mc = (0.25/hy) * ( uxmyp - uxmym + uxcyp - uxcym );

      const Vec2d &c_cp = D( dx_cp, dy_cp, params );
      const Vec2d &c_cm = D( dx_cm, dy_cm, params );
      const Vec2d &c_pc = D( dx_pc, dy_pc, params );
      const Vec2d &c_mc = D( dx_mc, dy_mc, params );

      // Compute divergence
      *flow_d = (-c_pc.x + c_mc.x)/hx + (-c_cp.y + c_cm.y) / hy;

      tv += hypot( dx_pc, dy_cp );

      flow_d++;
      ud++;
    }
  }


  return tv;
}

// Filter is
//   1   [-3  0  3]
//  --   [-10 0 10]
//  32   [-3  0  3]
bool coco::gsl_matrix_dx_roi( const gsl_matrix *u, gsl_matrix *ux )
{
  size_t h = u->size1;
  size_t w = u->size2;
  if ( h != ux->size1 || w != ux->size2 ) {
    assert( false );
    return false;
  }
  gsl_matrix *kernel = gsl_matrix_alloc( 3,3 );
  gsl_matrix_set( kernel, 0,0, -3.0 / 32.0 );
  gsl_matrix_set( kernel, 0,1, 0.0 );
  gsl_matrix_set( kernel, 0,2,  3.0 / 32.0 );

  gsl_matrix_set( kernel, 1,0, -10.0 / 32.0 );
  gsl_matrix_set( kernel, 1,1, 0.0 );
  gsl_matrix_set( kernel, 1,2, 10.0 / 32.0 );

  gsl_matrix_set( kernel, 2,0, -3.0 / 32.0 );
  gsl_matrix_set( kernel, 2,1, 0.0 );
  gsl_matrix_set( kernel, 2,2, 3.0 / 32.0 );

  // Perform convolution
  gsl_matrix_convolution_3x3( u, kernel, ux );
  gsl_matrix_free( kernel );
  return true;
}


// Filter is
//   1   [ 3  10  3]
//  --   [ 0   0  0]
//  32   [-3 -10 -3]
bool coco::gsl_matrix_dy_roi( const gsl_matrix *u, gsl_matrix *uy )
{
  size_t h = u->size1;
  size_t w = u->size2;
  if ( h != uy->size1 || w != uy->size2 ) {
    assert( false );
    return false;
  }

  // Build kernel
  gsl_matrix *kernel = gsl_matrix_alloc( 3,3 );
  gsl_matrix_set( kernel, 0,0, 3.0 / 32.0 );
  gsl_matrix_set( kernel, 0,1, 10.0 / 32.0 );
  gsl_matrix_set( kernel, 0,2, 3.0 / 32.0 );

  gsl_matrix_set( kernel, 1,0, 0.0 );
  gsl_matrix_set( kernel, 1,1, 0.0 );
  gsl_matrix_set( kernel, 1,2, 0.0 );

  gsl_matrix_set( kernel, 2,0, -3.0 / 32.0 );
  gsl_matrix_set( kernel, 2,1, -10.0 / 32.0 );
  gsl_matrix_set( kernel, 2,2, -3.0 / 32.0 );

  // Perform convolution
  gsl_matrix_convolution_3x3( u, kernel, uy );
  gsl_matrix_free( kernel );
  return true;
}


static bool gsl_matrix_diffusion_tensor_roi( const gsl_matrix *dx, const gsl_matrix *dy,
					     gsl_matrix *a, gsl_matrix *b, gsl_matrix *c )
{
  size_t h = dx->size1;
  size_t w = dx->size2;
  if ( h != dy->size1 || w != dy->size2 ) {
    assert( false );
    return false;
  }
  if ( h != a->size1 || w != a->size2 ) {
    assert( false );
    return false;
  }
  if ( h != b->size1 || w != b->size2 ) {
    assert( false );
    return false;
  }
  if ( h != c->size1 || w != c->size2 ) {
    assert( false );
    return false;
  }

  const double eps = 0.001;
  size_t index = 0;
  for ( size_t y=0; y<h; y++ ) {
    for ( size_t x=0; x<w; x++ ) {
      double dxv = dx->data[index];
      double dyv = dy->data[index];
      double n = hypot( dxv, dyv );

      // Normal TV flow
      double av = 1.0 / max( n, eps );
      double bv = 0.0;
      double cv = av;
    
      if ( n != 0.0 ) {
	// Compute Eigenvalues and Eigenvectors of structure tensor
	Mat22d M;
	M._11 = dxv*dxv; M._12 = dxv*dyv;
	M._21 = M._12; M._22 = dyv*dyv;
	Mat22d E;
	int nv = coco::gsl_matrix_eigensystem( M,E );
	if ( nv == 2 ) {
	  // Perona-Malik type reduction of first Eigenvalue
	  /*
	    const double K = 1.0;
	    //double g = 1.0 - exp( -2.33666 / square( n*n / (K*K) ));
	    double g = 1.0 / sqrt( 1.0 + square(n) / square(K) );
	    E._11 = g/n;
	    E._22 = 1.0/n;
	  */

	  // Weickert, coherence-enhancing
	  const double C = 2.0;
	  const double delta = E._11 - E._22;
	  double f = 0.0;
	  if ( delta > 0.0 ) {
	    f = exp( -C / square(delta) );
	  }
	  TRACE( "f: " << f << " from delta: " << delta << endl );

	  const double alpha = 0.2;
	  E._11 = alpha / n;
	  E._22 = (alpha + (1.0 - alpha) * f) / n;

	  // Construct diffusion tensor from new Eigenvalues
	  Mat22d D = E * M;
	  M.transpose();
	  D.preMultiply( M );
	  /*
	    TRACE( "Testing structure tensor decomposition: " << endl );
	    TRACE( "  a: " << D._11 << " = " << dxv*dxv << endl );
	    TRACE( "  b: " << D._12 << " = " << dxv*dyv << endl );
	    TRACE( "  c: " << D._22 << " = " << dyv*dyv << endl );
	  */
	  /*
	  Mat22d M;
	  M._11 = dxv / n; M._12 = dyv / n;
	  M._21 = -M._12; M._22 = M._11;
	  
	  // Perona-Malik type reduction of first Eigenvalue
	  Mat22d E;
	  E._11 = g * E._11;
	  E._22 = 1.0;
    
	  // Construct diffusion tensor from new Eigenvalues
	  Mat22d D = E * M;
	  M.transpose();
	  D.preMultiply( M );
	  */
	  av = D._11;
	  bv = D._12;
	  cv = D._22;
	  TRACE9( "Final diff tensor " << av << " " << bv << " " << cv << "   norm: " << n << endl );
	  TRACE9( "  Applied value: " << av * dxv + bv * dyv << " " << bv*dxv + cv*dyv << endl );
	}
	else {
	  // Think about something.
	}
      }

      a->data[index] = av;
      b->data[index] = bv;
      c->data[index] = cv;
      index++;
    }
  }

  return true;
}


bool coco::gsl_matrix_structure_tensor( const gsl_matrix *dx, const gsl_matrix *dy,
				       gsl_matrix *a, gsl_matrix *b, gsl_matrix *c )
{
  size_t h = dx->size1;
  size_t w = dx->size2;
  if ( h != dy->size1 || w != dy->size2 ) {
    assert( false );
    return false;
  }
  if ( h != a->size1 || w != a->size2 ) {
    assert( false );
    return false;
  }
  if ( h != b->size1 || w != b->size2 ) {
    assert( false );
    return false;
  }
  if ( h != c->size1 || w != c->size2 ) {
    assert( false );
    return false;
  }

  size_t index = 0;
  for ( size_t y=0; y<h; y++ ) {
    for ( size_t x=0; x<w; x++ ) {
      double dxv = dx->data[index];
      double dyv = dy->data[index];
      a->data[index] = dxv * dxv;
      b->data[index] = dxv * dyv;
      c->data[index] = dyv * dyv;
      index++;
    }
  }

  return true;
}



static void eigenvalue_reduction( double &e1, double &e2, double n, const vector<double> &params )
{
  switch( int(params[0]) ) {
  case 1: // Perona-Malik
    {
      const double K = params[1];
      double g = 1.0 - exp( -params[2] / square( n*n / (K*K) ));
      //double g = 1.0 / sqrt( 1.0 + square(n) / square(K) );
      e1 = g/n;
      e2 = 1.0/n;
    }
    break;

  case 2: // Weickert, coherence-enhancing
    {
      const double C = params[1];
      const double delta = e1-e2;
      double f = 0.0;
      if ( delta > 0.0 ) {
	f = exp( -square(C/delta) );
      }
      
      const double alpha = params[2];
      e1 = alpha / n;
      e2 = (alpha + (1.0 - alpha) * f) / n;
    }
    break;

  default:
    {
      ERROR( "Unknown regularizer " << params[0] << endl );
      assert( false );
    }
  }
}


bool coco::gsl_matrix_diffusion_tensor_perona_malik( gsl_matrix *a, gsl_matrix *b, gsl_matrix *c,
						    const vector<double> &params,
						    gsl_matrix *edgedir_x, gsl_matrix *edgedir_y,
						    gsl_matrix *ev_max, gsl_matrix *ev_min )
{
  size_t h = a->size1;
  size_t w = b->size2;
  if ( h != b->size1 || w != b->size2 ) {
    assert( false );
    return false;
  }
  if ( h != c->size1 || w != c->size2 ) {
    assert( false );
    return false;
  }

  const double eps = 0.001;
  size_t index = 0;
  for ( size_t y=0; y<h; y++ ) {
    for ( size_t x=0; x<w; x++ ) {
      // Compute Eigenvalues and Eigenvectors of structure tensor
      Mat22d M;
      M._11 = a->data[index]; M._12 = b->data[index];
      M._21 = M._12; M._22 = c->data[index];
      double n = sqrt(M._11 + M._22);
      Mat22d E;
      double av = 1.0 / max( n, eps );
      double bv = 0.0;
      double cv = av;
      
      if ( params[0] != 0.0 ) {
	TRACE9( "Computing Eigensystem for " << M << endl );
	int nv = gsl_matrix_eigensystem( M,E );
	if ( edgedir_x != NULL && edgedir_y != NULL ) {
	  edgedir_x->data[index] = M._21;
	  edgedir_y->data[index] = M._22;
	}
	if ( ev_max != NULL && ev_min != NULL ) {
	  ev_max->data[index] = E._11;
	  ev_min->data[index] = E._22;
	}

	if ( nv == 2 ) {
	  // Perona-Malik type reduction of first Eigenvalue
	  eigenvalue_reduction( E._11, E._22, n, params );
	  
	  // Construct diffusion tensor from new Eigenvalues
	  Mat22d D = E * M;
	  M.transpose();
	  D.preMultiply( M );
	  av = D._11;
	  bv = D._12;
	  cv = D._22;
	  TRACE9( "PMR " << av << " " << bv << " " << cv << endl );
	}
	else {
	  // Think about something (different from "no flow")
	  //TRACE( "Eigensystem dimension " << nv << endl );
	}
      }

      a->data[index] = av;
      b->data[index] = bv;
      c->data[index] = cv;
      index++;
    }
  }

  return true;
}




// Anisotropic diffusion, improved rotation invariance
double coco::gsl_matrix_anisotropic_diffusion_flow_roi( double hx, double hy,
						       const gsl_matrix *u,
						       gsl_matrix *flow )
{
  // Not implemented for different cell widths
  assert( hx==1.0 );
  assert( hy==1.0 );

  size_t W = u->size2;
  size_t H = u->size1;
  if ( flow->size2 != W || flow->size1 != H ) {
    assert( false );
    return false;
  }

  // Calculate structure tensor components using optimized derivative filter
  gsl_matrix *a = gsl_matrix_alloc( H,W );
  gsl_matrix *b = gsl_matrix_alloc( H,W );
  gsl_matrix *c = gsl_matrix_alloc( H,W );
  gsl_matrix *ux = gsl_matrix_alloc( H,W );
  gsl_matrix_dx_roi( u,ux );
  gsl_matrix *uy = gsl_matrix_alloc( H,W );
  gsl_matrix_dy_roi( u,uy );

  // Compute diffusion tensor as a function of the structure tensor
  //  vector<double> params;
  gsl_matrix_diffusion_tensor_roi( ux,uy, a,b,c ); //, ev_transformation_perona_malik, params );

  // Compute flux components j1 = a dx(u) + b dy(u), j2 = b dx(u) + c dy(u)
  gsl_matrix *j1 = gsl_matrix_alloc( H,W );
  gsl_matrix *j2 = gsl_matrix_alloc( H,W );
  size_t index = 0;
  double tv = 0.0;
  for ( size_t y=0; y<H; y++ ) {
    for ( size_t x=0; x<W; x++ ) {
      j1->data[index] = a->data[index] * ux->data[index] + b->data[index] * uy->data[index];
      j2->data[index] = b->data[index] * ux->data[index] + c->data[index] * uy->data[index];
      tv += hypot( ux->data[index], uy->data[index] );
      index++;
    }
  }

  // Compute divergence dx(j1) + dy(j2)
  gsl_matrix_dx_roi( j1,ux );
  gsl_matrix_dy_roi( j2,flow );
  gsl_matrix_add( flow, ux );
  gsl_matrix_scale( flow, -1.0 );

  // Cleanup 
  gsl_matrix_free( ux );
  gsl_matrix_free( uy );
  gsl_matrix_free( j1 );
  gsl_matrix_free( j2 );
  gsl_matrix_free( a );
  gsl_matrix_free( b );
  gsl_matrix_free( c );
  return tv;
}




bool coco::gsl_matrix_multichannel_structure_tensor( const vector<gsl_matrix*> &U,
						    float tau,
						    gsl_matrix *edgedir_x, gsl_matrix *edgedir_y,
						    gsl_matrix *ev_max, gsl_matrix *ev_min )
{
  assert( U.size() > 0 );
  size_t H = U[0]->size1;
  size_t W = U[0]->size2;
  gsl_matrix *a = gsl_matrix_alloc( H,W );
  gsl_matrix *b = gsl_matrix_alloc( H,W );
  gsl_matrix *c = gsl_matrix_alloc( H,W );
  gsl_matrix *A = gsl_matrix_alloc( H,W );
  gsl_matrix *B = gsl_matrix_alloc( H,W );
  gsl_matrix *C = gsl_matrix_alloc( H,W );
  memset( A->data, 0, W*H*sizeof(double) );
  memset( B->data, 0, W*H*sizeof(double) );
  memset( C->data, 0, W*H*sizeof(double) );

  // Accumulate structure tensors for each component
  gsl_matrix *ux = gsl_matrix_alloc( H,W );
  gsl_matrix *uy = gsl_matrix_alloc( H,W );
  for ( size_t i=0; i<U.size(); i++ ) {
    gsl_matrix_dx_roi( U[i],ux );
    gsl_matrix_dy_roi( U[i],uy );

    gsl_matrix_structure_tensor( ux,uy, a,b,c );

    gsl_matrix_gauss_filter( a, a, tau );
    gsl_matrix_gauss_filter( b, b, tau );
    gsl_matrix_gauss_filter( c, c, tau );

    gsl_matrix_add( A,a );
    gsl_matrix_add( B,b );
    gsl_matrix_add( C,c );
  }

  gsl_matrix_free( a );
  gsl_matrix_free( b );
  gsl_matrix_free( c );
  gsl_matrix_free( ux );
  gsl_matrix_free( uy );

  // Compute Eigenvalues and Eigenvectors of structure tensor
  size_t index = 0;
  for ( size_t y=0; y<H; y++ ) {
    for ( size_t x=0; x<W; x++ ) {

      Mat22d M;
      M._11 = A->data[index];
      M._12 = B->data[index];
      M._21 = M._12;
      M._22 = C->data[index];
      Mat22d E;
      TRACE9( "Computing Eigensystem for " << M << endl );
      
      gsl_matrix_eigensystem( M,E );
      if ( edgedir_x != NULL ) {
	edgedir_x->data[index] = M._21;
      }
      if ( edgedir_y != NULL ) {
	edgedir_y->data[index] = M._22;
      }
      if ( ev_max != NULL ) {
	ev_max->data[index] = E._11;
      }
      if ( ev_min != NULL ) {
	ev_min->data[index] = E._22;
      }

      index++;
    }
  }

  gsl_matrix_free( A );
  gsl_matrix_free( B );
  gsl_matrix_free( C );
  return true;
}



// Anisotropic diffusion, improved rotation invariance
double coco::gsl_matrix_anisotropic_diffusion_flow_roi_multichannel( double hx, double hy,
								    const vector<gsl_matrix*> &U,
								    vector<gsl_matrix*> &F,
								    const vector<double> &params,
								    gsl_matrix *lambda,
								    gsl_matrix *edgedir_x,
								    gsl_matrix *edgedir_y,
								    gsl_matrix *ev_max,
								    gsl_matrix *ev_min )
{
  // Sizes are assumed fixed here.
  assert( hx == 1.0 );
  assert( hy == 1.0 );

  size_t nc = U.size();
  // Input validation
  assert( nc > 0 );
  assert( F.size() == nc );
  size_t W = U[0]->size2;
  size_t H = U[0]->size1;
  for ( size_t i=0; i<nc; i++ ) {
    if ( U[i]->size2 != W || U[i]->size1 != H ) {
      assert( false );
    }
    if ( F[i]->size2 != W || F[i]->size1 != H ) {
      assert( false );
    }
  }

  // Calculate structure tensor components using optimized derivative filter
  gsl_matrix *a = gsl_matrix_alloc( H,W );
  gsl_matrix *b = gsl_matrix_alloc( H,W );
  gsl_matrix *c = gsl_matrix_alloc( H,W );
  gsl_matrix *A = gsl_matrix_alloc( H,W );
  gsl_matrix *B = gsl_matrix_alloc( H,W );
  gsl_matrix *C = gsl_matrix_alloc( H,W );
  memset( A->data, 0, sizeof(double)*W*H );
  memset( B->data, 0, sizeof(double)*W*H );
  memset( C->data, 0, sizeof(double)*W*H );
  vector<gsl_matrix*> UX;
  vector<gsl_matrix*> UY;
  for ( size_t i=0; i<nc; i++ ) {
    gsl_matrix *ux = gsl_matrix_alloc( H,W );
    gsl_matrix *uy = gsl_matrix_alloc( H,W );
    gsl_matrix_dx_roi( U[i],ux );
    gsl_matrix_dy_roi( U[i],uy );
    UX.push_back( ux );
    UY.push_back( uy );
    // Accumulate structure tensors in each component
    gsl_matrix_structure_tensor( ux,uy, a,b,c );
    gsl_matrix_add( A,a );
    gsl_matrix_add( B,b );
    gsl_matrix_add( C,c );
  }

  gsl_matrix_diffusion_tensor_perona_malik( A,B,C, params, edgedir_x, edgedir_y, ev_max, ev_min );
  if ( lambda != NULL ) {
    gsl_matrix_mul_with( A,lambda );
    gsl_matrix_mul_with( B,lambda );
    gsl_matrix_mul_with( C,lambda );
    //TRACE( "Diffusion tensor scaling, min " << gsl_matrix_min( lambda )
    //   << "  max " << gsl_matrix_max( lambda ) << endl );  
  }


  // Use common averaged diffusion tensor to compute divergence
  double tv_total = 0.0;
  gsl_matrix *j1 = gsl_matrix_alloc( H,W );
  gsl_matrix *j2 = gsl_matrix_alloc( H,W );
  for ( size_t i=0; i<nc; i++ ) {
    // Compute flux components j1 = a dx(u) + b dy(u), j2 = b dx(u) + c dy(u)
    gsl_matrix *ux = UX[i];
    gsl_matrix *uy = UY[i];
    size_t index = 0;
    double tv = 0.0;
    for ( size_t y=0; y<H; y++ ) {
      for ( size_t x=0; x<W; x++ ) {
	j1->data[index] = A->data[index] * ux->data[index] + B->data[index] * uy->data[index];
	j2->data[index] = B->data[index] * ux->data[index] + C->data[index] * uy->data[index];
	tv += hypot( ux->data[index], uy->data[index] );
	index++;
      }
    }
    TRACE8( "  TV channel #" << i << " : " << tv << endl );
    tv_total += tv;

    // Compute divergence dx(j1) + dy(j2)
    gsl_matrix_dx_roi( j1,ux );
    gsl_matrix_dy_roi( j2,F[i] );
    gsl_matrix_add( F[i], ux );
    gsl_matrix_scale( F[i], -1.0 );
    //TRACE8( "    flow min " << gsl_matrix_min(F[i]) << "   max " << gsl_matrix_max( F[i] ) << endl );
  }
  gsl_matrix_free( j1 );
  gsl_matrix_free( j2 );
  TRACE8( "TV total: " << tv_total << endl );

  // Cleanup 
  for ( size_t i=0; i<nc; i++ ) {
    gsl_matrix_free( UX[i] );
    gsl_matrix_free( UY[i] );
  }
  gsl_matrix_free( a );
  gsl_matrix_free( b );
  gsl_matrix_free( c );
  gsl_matrix_free( A );
  gsl_matrix_free( B );
  gsl_matrix_free( C );
  return tv_total;
}



// Compute gradient at cell boundaries for non-negativity scheme
// Dirichlet boundary conditions.
void coco::gsl_matrix_gradient_halfway_x_dx( double hx, const gsl_matrix *u, gsl_matrix *dx_hx )
{
  assert( u != NULL );
  assert( dx_hx != NULL );
  size_t W = u->size2;
  size_t H = u->size1;
  assert( dx_hx->size2 == W+1 );
  assert( dx_hx->size1 == H );

  // First and last column
  for ( size_t y=0; y<H; y++ ) {
    double v0 = gsl_matrix_get( u, y,0 ) / hx;
    gsl_matrix_set( dx_hx, y,0, v0 );

    double vW = -gsl_matrix_get( u, y,W-1 ) / hx;
    gsl_matrix_set( dx_hx, y,W, vW );
  }

  // Middle columns
  for ( size_t x=1; x<W; x++ ) {
    for ( size_t y=0; y<H; y++ ) {
      double v = (gsl_matrix_get( u, y,x ) - gsl_matrix_get( u, y,x-1 )) / hx;
      gsl_matrix_set( dx_hx, y,x, v );
    }
  }
}


// Compute gradient at cell boundaries for non-negativity scheme
void coco::gsl_matrix_gradient_halfway_x_dy( double hy, const gsl_matrix *u, gsl_matrix *dx_hy )
{
  assert( u != NULL );
  assert( dx_hy != NULL );
  size_t W = u->size2;
  size_t H = u->size1;
  assert( dx_hy->size2 == W+1 );
  assert( dx_hy->size1 == H );
  double hy4 = hy*4.0;

  // Corners
  gsl_matrix_set( dx_hy, 0,0, gsl_matrix_get( u, 0,0 ) / hy4 );
  gsl_matrix_set( dx_hy, 0,W, -gsl_matrix_get( u, 0,W-1 ) / hy4 );
  gsl_matrix_set( dx_hy, H-1,0, gsl_matrix_get( u, H-1,0 ) / hy4 );
  gsl_matrix_set( dx_hy, H-1,W, -gsl_matrix_get( u, H-1,W-1 ) / hy4 );
  // First and last column
  for ( size_t y=1; y<H-1; y++ ) {
    double v0p = gsl_matrix_get( u, y+1,0 );
    double v0m = gsl_matrix_get( u, y-1,0 );
    gsl_matrix_set( dx_hy, y,0, (v0p - v0m) / hy4 );

    double vWp = gsl_matrix_get( u, y+1,W-1 );
    double vWm = gsl_matrix_get( u, y-1,W-1 );
    gsl_matrix_set( dx_hy, y,W, (vWp - vWm) / hy4 );
  }

  // Middle columns
  // Top and bottom row
  for ( size_t x=1; x<W; x++ ) {
    double v0 = (gsl_matrix_get( u, 0,x-1 ) + gsl_matrix_get( u, 0,x )) / hy4;
    gsl_matrix_set( dx_hy, 0,x, v0 );
    double vH = - (gsl_matrix_get( u, H-1,x-1 ) + gsl_matrix_get( u, H-1,x )) / hy4;
    gsl_matrix_set( dx_hy, H-1,x, vH );
  }

  // Center rows
  for ( size_t x=1; x<W; x++ ) {
    for ( size_t y=1; y<H-1; y++ ) {
      double vp = (gsl_matrix_get( u, y+1,x ) + gsl_matrix_get( u, y+1,x-1 ));
      double vm = (gsl_matrix_get( u, y-1,x ) + gsl_matrix_get( u, y-1,x-1 ));
      gsl_matrix_set( dx_hy, y,x, (vp-vm) / hy4 );
    }
  }
}

// Compute gradient at cell boundaries for non-negativity scheme
void coco::gsl_matrix_gradient_halfway_y_dx( double hx, const gsl_matrix *u, gsl_matrix *dy_hx )
{
  assert( u != NULL );
  assert( dy_hx != NULL );
  size_t W = u->size2;
  size_t H = u->size1;
  assert( dy_hx->size2 == W );
  assert( dy_hx->size1 == H+1 );
  double hx4 = hx*4.0;

  // Corners
  gsl_matrix_set( dy_hx, 0,0, gsl_matrix_get( u, 0,0 ) / hx4 );
  gsl_matrix_set( dy_hx, 0,W-1, -gsl_matrix_get( u, 0,W-1 ) / hx4 );
  gsl_matrix_set( dy_hx, H,0, gsl_matrix_get( u, H-1,0 ) / hx4 );
  gsl_matrix_set( dy_hx, H,W-1, -gsl_matrix_get( u, H-1,W-1 ) / hx4 );
  // First and last row
  for ( size_t x=1; x<W-1; x++ ) {
    double v0p = gsl_matrix_get( u, 0,x+1 );
    double v0m = gsl_matrix_get( u, 0,x-1 );
    gsl_matrix_set( dy_hx, 0,x, (v0p - v0m) / hx4 );

    double vWp = -gsl_matrix_get( u, H-1,x+1 );
    double vWm = -gsl_matrix_get( u, H-1,x-1 );
    gsl_matrix_set( dy_hx, H,x, (vWp - vWm) / hx4 );
  }

  // Middle columns
  // Top and bottom row
  for ( size_t y=1; y<H; y++ ) {
    double v0 = (gsl_matrix_get( u, y-1,0 ) + gsl_matrix_get( u, y,0 )) / hx4;
    gsl_matrix_set( dy_hx, y,0, v0 );
    double vH = - (gsl_matrix_get( u, y-1,W-1 ) + gsl_matrix_get( u, y,W-1 )) / hx4;
    gsl_matrix_set( dy_hx, y,W-1, vH );
  }

  // Center rows
  for ( size_t y=1; y<H; y++ ) {
    for ( size_t x=1; x<W-1; x++ ) {
      double vp = (gsl_matrix_get( u, y,x+1 ) + gsl_matrix_get( u, y-1,x+1 ));
      double vm = (gsl_matrix_get( u, y,x-1 ) + gsl_matrix_get( u, y-1,x-1 ));
      gsl_matrix_set( dy_hx, y,x, (vp-vm) / hx4 );
    }
  }
}

// Compute gradient at cell boundaries for non-negativity scheme
void coco::gsl_matrix_gradient_halfway_y_dy( double hy, const gsl_matrix *u, gsl_matrix *dy_hy )
{
  assert( u != NULL );
  assert( dy_hy != NULL );
  size_t W = u->size2;
  size_t H = u->size1;
  assert( dy_hy->size2 == W );
  assert( dy_hy->size1 == H+1 );

  // First and last row
  for ( size_t x=0; x<W; x++ ) {
    double v0 = gsl_matrix_get( u, 0,x ) / hy;
    gsl_matrix_set( dy_hy, 0,x, v0 );

    double vH = -gsl_matrix_get( u, H-1,x ) / hy;
    gsl_matrix_set( dy_hy, H,x, vH );
  }

  // Middle columns
  for ( size_t x=0; x<W; x++ ) {
    for ( size_t y=1; y<H; y++ ) {
      double v = (gsl_matrix_get( u, y,x ) - gsl_matrix_get( u, y-1,x )) / hy;
      gsl_matrix_set( dy_hy, y,x, v );
    }
  }
}





/*** Same with Neumann ***/

// Compute gradient at cell boundaries for non-negativity scheme
void coco::gsl_matrix_gradient_halfway_x_dx_neumann( double hx, const gsl_matrix *u, gsl_matrix *dx_hx )
{
  assert( u != NULL );
  assert( dx_hx != NULL );
  size_t W = u->size2;
  size_t H = u->size1;
  assert( dx_hx->size2 == W+1 );
  assert( dx_hx->size1 == H );

  // First and last column
  for ( size_t y=0; y<H; y++ ) {
    gsl_matrix_set( dx_hx, y,0, 0.0 );
    gsl_matrix_set( dx_hx, y,W, 0.0 );
  }

  // Middle columns
  for ( size_t x=1; x<W; x++ ) {
    for ( size_t y=0; y<H; y++ ) {
      double v = (gsl_matrix_get( u, y,x ) - gsl_matrix_get( u, y,x-1 )) / hx;
      gsl_matrix_set( dx_hx, y,x, v );
    }
  }
}


// Compute gradient at cell boundaries for non-negativity scheme
void coco::gsl_matrix_gradient_halfway_x_dy_neumann( double hy, const gsl_matrix *u, gsl_matrix *dx_hy )
{
  assert( u != NULL );
  assert( dx_hy != NULL );
  size_t W = u->size2;
  size_t H = u->size1;
  assert( dx_hy->size2 == W+1 );
  assert( dx_hy->size1 == H );
  double hy4 = hy*4.0;

  // First and last column
  for ( size_t y=0; y<H; y++ ) {
    gsl_matrix_set( dx_hy, y,0, 0.0 );
    gsl_matrix_set( dx_hy, y,W, 0.0 );
  }

  // Middle columns
  // Top and bottom row
  for ( size_t x=1; x<W; x++ ) {
    double v0p = (gsl_matrix_get( u, 1,x-1 ) + gsl_matrix_get( u, 1,x ));
    double v0m = (gsl_matrix_get( u, 0,x-1 ) + gsl_matrix_get( u, 0,x ));
    gsl_matrix_set( dx_hy, 0,x, (v0p-v0m) / hy4 );
    double vHp = (gsl_matrix_get( u, H-1,x-1 ) + gsl_matrix_get( u, H-1,x ));
    double vHm = (gsl_matrix_get( u, H-2,x-1 ) + gsl_matrix_get( u, H-2,x ));
    gsl_matrix_set( dx_hy, H-1,x, (vHp-vHm) / hy4 );
  }

  // Center rows
  for ( size_t x=1; x<W; x++ ) {
    for ( size_t y=1; y<H-1; y++ ) {
      double vp = (gsl_matrix_get( u, y+1,x ) + gsl_matrix_get( u, y+1,x-1 ));
      double vm = (gsl_matrix_get( u, y-1,x ) + gsl_matrix_get( u, y-1,x-1 ));
      gsl_matrix_set( dx_hy, y,x, (vp-vm) / hy4 );
    }
  }
}

// Compute gradient at cell boundaries for non-negativity scheme
void coco::gsl_matrix_gradient_halfway_y_dx_neumann( double hx, const gsl_matrix *u, gsl_matrix *dy_hx )
{
  assert( u != NULL );
  assert( dy_hx != NULL );
  size_t W = u->size2;
  size_t H = u->size1;
  assert( dy_hx->size2 == W );
  assert( dy_hx->size1 == H+1 );
  double hx4 = hx*4.0;

  // Top and bottom row
  for ( size_t x=0; x<W; x++ ) {
    gsl_matrix_set( dy_hx, 0,x, 0.0 );
    gsl_matrix_set( dy_hx, H,x, 0.0 );
  }

  // Middle columns
  // First and last column
  for ( size_t y=1; y<H; y++ ) {
    double v0p = (gsl_matrix_get( u, y-1,1 ) + gsl_matrix_get( u, y,1 ));
    double v0m = (gsl_matrix_get( u, y-1,0 ) + gsl_matrix_get( u, y,0 ));
    gsl_matrix_set( dy_hx, y,0, (v0p-v0m) / hx4 );
    double vHp = (gsl_matrix_get( u, y-1,W-1 ) + gsl_matrix_get( u, y,W-1 ));
    double vHm = (gsl_matrix_get( u, y-1,W-2 ) + gsl_matrix_get( u, y,W-2 ));
    gsl_matrix_set( dy_hx, y,W-1, (vHp-vHm) / hx4 );
  }

  // Center rows
  for ( size_t y=1; y<H; y++ ) {
    for ( size_t x=1; x<W-1; x++ ) {
      double vp = (gsl_matrix_get( u, y,x+1 ) + gsl_matrix_get( u, y-1,x+1 ));
      double vm = (gsl_matrix_get( u, y,x-1 ) + gsl_matrix_get( u, y-1,x-1 ));
      gsl_matrix_set( dy_hx, y,x, (vp-vm) / hx4 );
    }
  }
}

// Compute gradient at cell boundaries for non-negativity scheme
void coco::gsl_matrix_gradient_halfway_y_dy_neumann( double hy, const gsl_matrix *u, gsl_matrix *dy_hy )
{
  assert( u != NULL );
  assert( dy_hy != NULL );
  size_t W = u->size2;
  size_t H = u->size1;
  assert( dy_hy->size2 == W );
  assert( dy_hy->size1 == H+1 );

  // First and last row
  for ( size_t x=0; x<W; x++ ) {
    gsl_matrix_set( dy_hy, 0,x, 0.0 );
    gsl_matrix_set( dy_hy, H,x, 0.0 );
  }

  // Middle columns
  for ( size_t x=0; x<W; x++ ) {
    for ( size_t y=1; y<H; y++ ) {
      double v = (gsl_matrix_get( u, y,x ) - gsl_matrix_get( u, y-1,x )) / hy;
      gsl_matrix_set( dy_hy, y,x, v );
    }
  }
}



// Compute square root of a (symmetric, positive semi-definite) matrix
bool coco::gsl_matrix_sqrt( const Mat22d &M, Mat22d &R )
{
  Mat22d tmp( M );
  Mat22d E;
  int nv = gsl_matrix_eigensystem( tmp,E );
  if ( nv != 2 ) {
    // Decomposition failed for some reason.
    R = M;
    return false;
  }
  if ( E._11<0.0f || E._22<0.0f ) {
    R = M;
    return false;
  }

  // Square roots of Eigenvalues
  E._11 = sqrt( E._11 );
  E._22 = sqrt( E._22 );

  // Construct root tensor from new Eigenvalues
  R = E * tmp;
  tmp.transpose();
  R.preMultiply( tmp );
  return true;
}
