/** \file menger_curvature.cpp

    Computes Menger curvature radius/weight etc.
    
    Copyright (C) 2010 Bastian Goldluecke,
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

#include <complex>

#include "menger_curvature.h"
#include "gsl_matrix_derivatives.h"

using namespace std;

// Compute Cauchy kernel 1/z
complex<double> coco::cauchy_curvature_weight( double z_re, double z_im )
{
  if ( z_re==0.0 && z_im==0.0 ) {
    return 0.0;
  }
  complex<double> z( z_re, z_im );
  return 1.0 / z;
}


// Compute Cauchy kernel 1/z
double coco::stc_curvature_weight( double y0, double y1 )
{
  double R = hypot( y0, y1 );
  if ( R==0.0 ) {
    // TODO: clamp to something sensible
    return 0.0;
  }
  return 1.0 / R;
}


// See http://mathworld.wolfram.com/Circle.html
// for the circle through three points
// here special case (x_1,y_1) = 0
double coco::menger_curvature_weight( double x2, double y2, double x3, double y3 )
{
  double a = fabs( x2*y3 - x3*y2 );
  if ( a < 0.000001 ) {
    TRACE9( "  collinear " << x2 << " " << y2 << " " << x3 << " " << y3 << endl );
    TRACE9( "    a=" << a << endl );
    // Collinear
    return 0.0;
  }

  double r2 = pow( x2, 2.0 ) + pow( y2, 2.0 );
  double r3 = pow( x3, 2.0 ) + pow( y3, 2.0 );

  double d = r2*x3 - x2*r3;
  double e = r2*y3 - y2*r3;

  double r = hypot( d, e ) / (2.0 * a);
  TRACE9( "    non-collinear" );
  TRACE9( "      a=" << a << endl );
  TRACE9( "      d=" << d << endl );
  TRACE9( "      e=" << e << endl );
  TRACE9( "      r=" << r << endl );

  if ( r<0.01 ) {
    TRACE( "Warning: circle very small (" << r << ")" << endl );
  }

  return 1.0 / r;
}

double coco::local_menger_curvature( gsl_matrix *g, int x, int y, double p, int radius )
{
  int W = (int)g->size2;
  int H = (int)g->size1;
  double w1 = gsl_matrix_get( g, y,x );
  if ( w1==0.0 ) {
    return 0.0;
  }

  // Outer loop for second point
  double mc = 0.0;
  for ( int i2=-radius; i2<=radius; i2++ ) {
    for ( int j2=-radius; j2<=radius; j2++ ) {
      int x2 = x+i2;
      int y2 = y+j2;
      if ( x2<0 || x2>= W || y2<0 || y2>=H ) {
	continue;
      }
      double w2 = gsl_matrix_get( g, y2,x2 );
      if ( w2 == 0.0 ) {
	continue;
      }

      // Inner loop for third point
      for ( int i3=-radius; i3<=radius; i3++ ) {
	for ( int j3=-radius; j3<=radius; j3++ ) {
	  int x3 = x+i3;
	  int y3 = y+j3;
	  if ( x3<0 || x3>= W || y3<0 || y3>=H ) {
	    continue;
	  }
	  double w3 = gsl_matrix_get( g, y3,x3 );
	  if ( w3 == 0.0 ) {
	    continue;
	  }

	  // Compute curvature weight (homogenous coords)
	  //double cv = menger_curvature_weight( i2/double(W), j2/double(H), i3/double(W), j3/double(H) );
	  double cv = menger_curvature_weight( i2, j2, i3, j3 );

	  // Add to total
	  TRACE8( "nonzero weight at " << i2 << " " << j2 << " : " << i3 << " " << j3 << endl );
	  TRACE8( "  menger radius " << cv << endl );
	  mc += pow( cv, p ) * fabs( w1 * w2 * w3 );
	}
      }
      // End inner loop

    }
  }

  return mc;
}


double coco::total_menger_curvature( gsl_matrix *M, double p, int radius, vector<double> *values )
{
  size_t W = M->size2;
  size_t H = M->size1;
  gsl_matrix *dxp = gsl_matrix_alloc( H,W );
  gsl_matrix *dyp = gsl_matrix_alloc( H,W );
  gsl_matrix *G = gsl_matrix_alloc( H,W );
  gsl_matrix_dx_forward( 1.0, M, dxp );
  gsl_matrix_dy_forward( 1.0, M, dyp );
  gsl_matrix_norm_grad( dxp, dyp, G );

  double tc = 0.0;
  for ( size_t y=0; y<H; y++ ) {
    if ( (y%10)==0 ) TRACE( "." );
    for ( size_t x=0; x<W; x++ ) {
      double v = local_menger_curvature( G, x,y, p, radius );
      if ( values != NULL ) {
	values->push_back( v );
      }
      tc += v;
    }
  }

  gsl_matrix_free( G );
  gsl_matrix_free( dxp );
  gsl_matrix_free( dyp );

  return tc / double( W*H );
}


double coco::local_menger_curvature_relaxation( double alpha, gsl_matrix *g, int x, int y, double p, int radius )
{
  int W = (int)g->size2;
  int H = (int)g->size1;
  double w1 = gsl_matrix_get( g, y,x );

  // Outer loop for second point
  double mc = 0.0;
  for ( int i2=-radius; i2<=radius; i2++ ) {
    for ( int j2=-radius; j2<=radius; j2++ ) {
      int x2 = x+i2;
      int y2 = y+j2;
      double w2 = 0.0;
      if ( x2>=0 && x2<W && y2>=0 && y2<H ) {
	w2 = gsl_matrix_get( g, y2,x2 );
      }

      // Inner loop for third point
      for ( int i3=-radius; i3<=radius; i3++ ) {
	for ( int j3=-radius; j3<=radius; j3++ ) {
	  int x3 = x+i3;
	  int y3 = y+j3;
	  double w3 = 0.0;
	  if ( x3>=0 && x3<W && y3>=0 && y3<H ) {
	    w3 = gsl_matrix_get( g, y3,x3 );
	  }

	  // Compute curvature weight (homogenous coords)
	  //double cv = menger_curvature_weight( i2/double(W), j2/double(H), i3/double(W), j3/double(H) );
	  double cv = menger_curvature_weight( i2, j2, i3, j3 );
	  // Add to total
	  mc += pow( cv, p ) * max( 0.0, w1 + w2 + w3 - alpha );
	}
      }
      // End inner loop

    }
  }

  return mc;
}


double coco::total_menger_curvature_relaxation( double alpha, gsl_matrix *M,
						double p, int radius,
						vector<double> *values )
{
  size_t W = M->size2;
  size_t H = M->size1;
  gsl_matrix *dxp = gsl_matrix_alloc( H,W );
  gsl_matrix *dyp = gsl_matrix_alloc( H,W );
  gsl_matrix *G = gsl_matrix_alloc( H,W );
  gsl_matrix_dx_forward( 1.0, M, dxp );
  gsl_matrix_dy_forward( 1.0, M, dyp );
  gsl_matrix_norm_grad( dxp, dyp, G );

  double tc = 0.0;
  for ( size_t y=0; y<H; y++ ) {
    if ( (y%10)==0 ) TRACE( "." );
    for ( size_t x=0; x<W; x++ ) {
      double v = local_menger_curvature_relaxation( alpha, G, x,y, p, radius );
      if ( values != NULL ) {
	values->push_back( v );
      }
      tc += v;
    }
  }

  gsl_matrix_free( G );
  gsl_matrix_free( dxp );
  gsl_matrix_free( dyp );

  return tc / double( W*H );
}



double coco::total_variation( gsl_matrix *M, vector<double> *values )
{
  size_t W = M->size2;
  size_t H = M->size1;
  gsl_matrix *dxp = gsl_matrix_alloc( H,W );
  gsl_matrix *dyp = gsl_matrix_alloc( H,W );
  gsl_matrix *G = gsl_matrix_alloc( H,W );
  gsl_matrix_dx_forward( 1.0, M, dxp );
  gsl_matrix_dy_forward( 1.0, M, dyp );
  gsl_matrix_norm_grad( dxp, dyp, G );

  double tv = 0.0;
  size_t i=0;
  for ( size_t y=0; y<H; y++ ) {
    for ( size_t x=0; x<W; x++ ) {
      if ( values != NULL ) {
	values->push_back( G->data[i] );
      }
      tv += G->data[i];
      i++;
    }
  }

  gsl_matrix_free( G );
  gsl_matrix_free( dxp );
  gsl_matrix_free( dyp );
  return tv / double( W*H );
}


double coco::total_mean_curvature( gsl_matrix *M, double p, vector<double> *values )
{
  size_t W = M->size2;
  size_t H = M->size1;
  gsl_matrix *dxp = gsl_matrix_alloc( H,W );
  gsl_matrix *dyp = gsl_matrix_alloc( H,W );
  gsl_matrix *N = gsl_matrix_alloc( H,W );
  gsl_matrix *G = gsl_matrix_alloc( H,W );
  gsl_matrix_dx_forward( 1.0, M, dxp );
  gsl_matrix_dy_forward( 1.0, M, dyp );
  gsl_matrix_norm_grad( dxp, dyp, N );
  gsl_matrix_divergence( 1.0, 1.0, dxp, dyp, G );

  double tmc = 0.0;
  size_t i=0;
  for ( size_t y=0; y<H; y++ ) {
    for ( size_t x=0; x<W; x++ ) {
      double v = pow( fabs(G->data[i]), p ) * N->data[i];
      if ( values != NULL ) {
	values->push_back( v );
      }
      tmc += v;
      i++;
    }
  }

  gsl_matrix_free( G );
  gsl_matrix_free( dxp );
  gsl_matrix_free( dyp );
  return tmc / double( W*H );
}




double coco::local_cauchy_curvature( gsl_matrix *g, int x, int y, int radius )
{
  int W = (int)g->size2;
  int H = (int)g->size1;
  double w1 = gsl_matrix_get( g, y,x );
  if ( w1==0.0 ) {
    return 0.0;
  }

  // Outer loop for second point
  complex<double> cauchy_integral( 0.0, 0.0 );
  for ( int i2=-radius; i2<=radius; i2++ ) {
    for ( int j2=-radius; j2<=radius; j2++ ) {
      int x2 = x+i2;
      int y2 = y+j2;
      if ( x2<0 || x2>= W || y2<0 || y2>=H ) {
	continue;
      }
      double w2 = gsl_matrix_get( g, y2,x2 );
      if ( w2 == 0.0 ) {
	continue;
      }

      // Compute Cauchy kernel and add to integral
      cauchy_integral += w1*w2 * cauchy_curvature_weight( i2,j2 );

    }
  }

  return abs( cauchy_integral );
}


double coco::local_stc_curvature( gsl_matrix *g, int x, int y, int radius )
{
  int W = (int)g->size2;
  int H = (int)g->size1;
  double w1 = gsl_matrix_get( g, y,x );
  if ( w1==0.0 ) {
    return 0.0;
  }

  // Outer loop for second point
  double integral = 0.0;
  for ( int i2=-radius; i2<=radius; i2++ ) {
    for ( int j2=-radius; j2<=radius; j2++ ) {
      int x2 = x+i2;
      int y2 = y+j2;
      if ( x2<0 || x2>= W || y2<0 || y2>=H ) {
	continue;
      }
      if ( hypot( i2,j2 ) > double(radius) ) {
	continue;
      }

      double w2 = gsl_matrix_get( g, y2,x2 );
      if ( w2 == 0.0 ) {
	continue;
      }

      // Compute Cauchy kernel and add to integral
      double y0 = double(i2); // / double(radius);
      double y1 = double(j2); // / double(radius);
      double sw = stc_curvature_weight( y0,y1 );

      if ( (x==38 && y==82) ) {
	TRACE( "P0: " << i2 << " " << j2 << "   -   " << w1 << " " << w2 << " " << sw << endl );
      }
      else if ( (x==73 && y==65) ) {
	TRACE( "P1: " << i2 << " " << j2 << "   -   " << w1 << " " << w2 << " " << sw << endl );
      }
      integral += w1*w2 * sw;

    }
  }

  return integral;
}



double coco::local_menger_curvature_frob( gsl_matrix *dx, gsl_matrix *dy, int x, int y, int radius )
{
  int W = (int)dx->size2;
  int H = (int)dy->size1;
  double w1x = gsl_matrix_get( dx, y,x );
  double w1y = gsl_matrix_get( dy, y,x );
  if ( w1x==0.0 && w1y == 0.0 ) {
    return 0.0;
  }

  // Outer loop for second point
  double mc = 0.0;
  for ( int i2=-radius; i2<=radius; i2++ ) {
    for ( int j2=-radius; j2<=radius; j2++ ) {
      int x2 = x+i2;
      int y2 = y+j2;
      if ( x2<0 || x2>= W || y2<0 || y2>=H ) {
	continue;
      }
      double w2x = gsl_matrix_get( dx, y2,x2 );
      double w2y = gsl_matrix_get( dy, y2,x2 );
      if ( w2x==0.0 && w2y==0.0 ) {
	continue;
      }

      // Inner loop for third point
      for ( int i3=-radius; i3<=radius; i3++ ) {
	for ( int j3=-radius; j3<=radius; j3++ ) {
	  int x3 = x+i3;
	  int y3 = y+j3;
	  if ( x3<0 || x3>= W || y3<0 || y3>=H ) {
	    continue;
	  }
	  double w3x = gsl_matrix_get( dx, y3,x3 );
	  double w3y = gsl_matrix_get( dy, y3,x3 );
	  if ( w3x==0.0 && w3y==0.0 ) {
	    continue;
	  }

	  // Compute curvature weight (homogenous coords)
	  //double cv = menger_curvature_weight( i2/double(W), j2/double(H), i3/double(W), j3/double(H) );
	  double cv = menger_curvature_weight( i2, j2, i3, j3 );

	  // Add to total
	  TRACE8( "nonzero weight at " << i2 << " " << j2 << " : " << i3 << " " << j3 << endl );
	  TRACE8( "  menger radius " << cv << endl );
	  
	  double fnorm = 0.0;
	  fnorm += pow( w1x * w2x * w3x, 2.0 );
	  fnorm += pow( w1x * w2x * w3y, 2.0 );
	  fnorm += pow( w1x * w2y * w3x, 2.0 );
	  fnorm += pow( w1x * w2y * w3y, 2.0 );
	  fnorm += pow( w1y * w2x * w3x, 2.0 );
	  fnorm += pow( w1y * w2x * w3y, 2.0 );
	  fnorm += pow( w1y * w2y * w3x, 2.0 );
	  fnorm += pow( w1y * w2y * w3y, 2.0 );

	  mc += cv * sqrt( fnorm );
	}
      }
      // End inner loop

    }
  }

  return mc;
}


