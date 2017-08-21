/** \file gsl_matrix_helper.cpp

    File imported from "common" lib, use if this library is not available.

    Additional functions to help with handling of gsl_matrix objects.
    
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

#include "debug.h"
#include "gsl_matrix_helper.h"
#include "gsl_matrix_convolutions.h"

#include "../defs.h"
#include "linalg3d.h"

#include <float.h>
#include <math.h>
#include <algorithm>

//#include <gsl/gsl_blas.h>

#include <string>
#include <string.h>
#include <zlib.h>

using namespace std;
using namespace coco;


/// Create from float (image) buffer
gsl_matrix* coco::gsl_matrix_from_buffer( size_t W, size_t H, float *buffer )
{
  size_t N = W*H;
  assert( N>0 );
  gsl_matrix *M = gsl_matrix_alloc( H,W );
  for ( size_t i=0; i<N; i++ ) {
    M->data[i] = buffer[i];
  }
  return M;
}


/// Initialize float array from matrix (must be same size)
bool coco::gsl_matrix_copy_to_buffer( gsl_matrix *m,  float *array )
{
  size_t N = m->size1 * m->size2;
  assert( N>0 );
  for ( size_t i=0; i<N; i++ ) {
    array[i] = m->data[i];
  }
  return true;
}




gsl_matrix* coco::gsl_matrix_from_vector( size_t rows, size_t cols, const gsl_vector *v )
{
  size_t N = rows * cols;
  if ( N==0 || N != v->size ) {
    assert( false );
    return NULL;
  }
  gsl_matrix *m = gsl_matrix_alloc( rows, cols );
  memcpy( m->data, v->data, sizeof(double)*N );
  return m;
}


/// Initialize matrix from a float array (same size)
void coco::gsl_matrix_copy_from_float( gsl_matrix *m,  float *array )
{
  if ( m==NULL ) {
    assert( false );
    return;
  }

  double *d = m->data;
  size_t N = m->size1 * m->size2;
  for ( size_t i=0; i<N; i++ ) {
    *(d++) = double( *(array++) );
  }
}


/// Normalize a matrix to a range
bool coco::gsl_matrix_normalize( gsl_matrix *m, double vmin, double vmax )
{
  return normalize_vector( m->data, m->size2 * m->size1, vmin, vmax );
}

/// Similar vector function
bool coco::normalize_vector( double *pv, size_t n, double vmin, double vmax )
{
  if ( n==0 ) {
    return false;
  }

  double dmin = pv[0];
  double dmax = pv[0];
  double *d = pv;
  for ( size_t i=1; i<n; i++ ) {
    if ( *d > dmax ) dmax = *d;
    if ( *d < dmin ) dmin = *d;
    d++;
  }

  double r = dmax - dmin;
  if ( r==0.0 ) {
    double v = (vmax + vmin) / 2.0;
    d = pv;
    for ( size_t i=0; i<n; i++ ) {
      *(d++) = v;
    }    
    return true;
  }

  r = (vmax - vmin) / r;
  d = pv;
  for ( size_t i=0; i<n; i++ ) {
    *d = (*d - dmin) * r + vmin;
    d++;
  }    
  return true;
}


/// Similar vector function
bool coco::normalize_vector( float *pv, size_t n, float vmin, float vmax )
{
  if ( n==0 ) {
    return false;
  }

  float dmin = pv[0];
  float dmax = pv[0];
  float *d = pv;
  for ( size_t i=1; i<n; i++ ) {
    if ( *d > dmax ) dmax = *d;
    if ( *d < dmin ) dmin = *d;
    d++;
  }

  float r = dmax - dmin;
  if ( r==0.0 ) {
    float v = (vmax + vmin) / 2.0;
    d = pv;
    for ( size_t i=0; i<n; i++ ) {
      *(d++) = v;
    }    
    return true;
  }

  r = (vmax - vmin) / r;
  d = pv;
  for ( size_t i=0; i<n; i++ ) {
    *d = (*d - dmin) * r + vmin;
    d++;
  }    
  return true;
}






/// Copy in to out. If out is larger than in, replicate values at the boundary.
bool coco::gsl_matrix_pad_replicate( const gsl_matrix *in, gsl_matrix *out )
{
  // Rows/Cols to copy
  size_t ch = min( in->size1, out->size1 );
  size_t cw = min( in->size2, out->size2 );
  // Padding
  size_t pow = out->size1 - cw;

  // Loop over rows
  for ( size_t h=0; h<ch; h++ ) {
    // Copy one row
    memcpy( out->data + h*out->size2, in->data + h*in->size2, sizeof(double) * cw );
    // Pad row replicating last value
    double *d = out->data + h*out->size2 + cw;
    double v = *(d-1);
    for ( size_t pw=0; pw<pow; pw++ ) {
      *(d++) = v;
    }
  }
  // Pad end of columns replicating last row
  double *src = out->data + out->size1 * (ch-1);
  for ( size_t ph=ch; ph<out->size2; ph++ ) {
    memcpy( out->data + ph*out->size1, src, sizeof(double) * out->size1 );
  }

  return true;
}


/// Threshold a matrix to 0-1
bool coco::gsl_matrix_threshold( gsl_matrix *out, double t )
{
  size_t H = out->size1;
  size_t W = out->size2;
  size_t N = W*H;

  double *o = out->data;
  for ( size_t i=0; i<N; i++ ) {
    double v = (*o >= t) ? 1.0 : 0.0;
    *(o++) = v;
  }

  return true;
}


/// Copy in to out. If out is larger than in or in larger than out, ignore values outside.
bool coco::gsl_matrix_copy_to( const gsl_matrix *in, gsl_matrix *out )
{
  // Rows/Cols to copy
  size_t ch = min( in->size1, out->size1 );
  size_t cw = min( in->size2, out->size2 );
  // Loop over rows
  for ( size_t h=0; h<ch; h++ ) {
    // Copy one row
    memcpy( out->data + h*out->size2, in->data + h*in->size2, sizeof(double) * cw );
  }
  return true;
}

/// Downsampling of in into the smaller matrix out.
/** Currently, only integers are supported for the size factor.
 */
bool coco::gsl_matrix_downsample( const gsl_matrix *in, gsl_matrix *out )
{
  size_t fw = in->size2 / out->size2;
  size_t fh = in->size1 / out->size1;
  if ( fw * out->size2 != in->size2 || fh * out->size1 != in->size1 ) {
    ERROR( "Downsampling ratio not integer." << endl );
    assert( false );
    return false;
  }

  double *d = out->data;
  for ( size_t h=0; h<out->size1; h++ ) {
    for ( size_t w=0; w<out->size2; w++ ) {
      // Sample rectangle in input matrix
      double v = 0.0;
      for ( size_t hh=0; hh<fh; hh++ ) {
	double *s = in->data + (h*fh+hh)*in->size2 + w*fw;
	for ( size_t ww=0; ww<fw; ww++ ) {
	  v += *(s++);
	}
      }

      // Normalize and write out
      *(d++) = v / double(fw*fh);
    }
  }

  return false;
}


/// Interpolation using Dirichlet boundary conditions, i.e. outside region is zero
double coco::gsl_matrix_interpolate_dirichlet( const gsl_matrix *m, double x, double y )
{
  int W = m->size2;
  int H = m->size1;
  int px = int( floor(x) );
  int py = int( floor(y) );
  double cxmym = 0.0;
  double cxMym = 0.0;
  double cxmyM = 0.0;
  double cxMyM = 0.0;
  const int pxm = px;
  const int pxM = px+1;
  const int pym = py;
  const int pyM = py+1;

  // Get matrix values and interpolate using remainders as weights
  if ( pxm>=0 && pxm<W && pym>=0 && pym<H ) {
    cxmym = *( m->data + pym*W + pxm );
  }
  if ( pxM>=0 && pxM<W && pym>=0 && pym<H ) {
    cxMym = *( m->data + pym*W + pxM );
  }
  if ( pxm>=0 && pxm<W && pyM>=0 && pyM<H ) {
    cxmym = *( m->data + pyM*W + pxm );
  }
  if ( pxM>=0 && pxM<W && pyM>=0 && pyM<H ) {
    cxMym = *( m->data + pyM*W + pxM );
  }

  double xr = x - double(px);
  double yr = y - double(py);
  const double cym = (1.0f-xr)*cxmym + xr*cxMym;
  const double cyM = (1.0f-xr)*cxmyM + xr*cxMyM;
  return (1.0-yr)*cym + yr*cyM;
}

/// Interpolation using Neumann boundary conditions, i.e. outside region is zero
double coco::gsl_matrix_interpolate_neumann( const gsl_matrix *m, double x, double y )
{
  int W = m->size2;
  int H = m->size1;
  int px = int( floor(x) );
  int py = int( floor(y) );
  int pxm = std::max( 0, std::min( px,W-1 ));
  int pxM = std::max( 0, std::min( px+1, W-1 ));
  int pym = std::max( 0, std::min( py,H-1 ));
  int pyM = std::max( 0, std::min( py+1, H-1 ));

  // Get matrix values and interpolate using remainders as weights
  const double cxmym( *(m->data + pym*W + pxm) );
  const double cxMym( *(m->data + pym*W + pxM) );
  const double cxmyM( *(m->data + pyM*W + pxm) );
  const double cxMyM( *(m->data + pyM*W + pxM) );

  double xr = x - double(px);
  double yr = y - double(py);
  const double cym = (1.0-xr)*cxmym + xr*cxMym;
  const double cyM = (1.0-xr)*cxmyM + xr*cxMyM;
  return (1.0-yr)*cym + yr*cyM;
}




/// Upsampling of in into the larger matrix out.
/** Interpolation uses Neumann boundary conditions suitable for scalar upsampling.
 */
bool coco::gsl_matrix_upsample_neumann( const gsl_matrix *in, gsl_matrix *out )
{
  size_t H = out->size1;
  size_t W = out->size2;
  //double hf = float( in->size1 ) / float( out->size1 );
  //double wf = float( in->size2 ) / float( out->size2 );
  double *d = out->data;
  for ( size_t h=0; h<H; h++ ) {
    for ( size_t w=0; w<W; w++ ) {
      // Evaluation coordinates are in the corners of the grid
      double v = 0.0;
      double y = double(h/2);
      double x = double(w/2);
      v += gsl_matrix_interpolate_neumann( in, x-0.5, y-0.5 );
      v += gsl_matrix_interpolate_neumann( in, x+0.5, y-0.5 );
      v += gsl_matrix_interpolate_neumann( in, x-0.5, y+0.5 );
      v += gsl_matrix_interpolate_neumann( in, x+0.5, y+0.5 );
      *(d++) = v/4.0;
    }
  }
  return true;
}

/// Upsampling of in into the larger matrix out.
/** Interpolation uses Dirichlet boundary conditions suitable for vector upsampling.
 */
bool coco::gsl_matrix_upsample_dirichlet( const gsl_matrix *in, gsl_matrix *out )
{
  size_t H = out->size1;
  size_t W = out->size2;
  double *d = out->data;
  for ( size_t h=0; h<H; h++ ) {
    for ( size_t w=0; w<W; w++ ) {
      // Evaluation coordinates are in the corners of the source grid
      double v = 0.0;
      double y = double(h/2);
      double x = double(w/2);
      v += gsl_matrix_interpolate_dirichlet( in, x-0.5, y-0.5 );
      v += gsl_matrix_interpolate_dirichlet( in, x+0.5, y-0.5 );
      v += gsl_matrix_interpolate_dirichlet( in, x-0.5, y+0.5 );
      v += gsl_matrix_interpolate_dirichlet( in, x+0.5, y+0.5 );
      *(d++) = v/4.0;
    }
  }
  return true;
}

/// Add two matrices and store result in third
bool coco::gsl_matrix_add( const gsl_matrix *in0, const gsl_matrix *in1, gsl_matrix *out )
{
  size_t H = out->size1;
  size_t W = out->size2;
  if ( H != in0->size1 || H != in1->size1 || W != in0->size2 || W != in1->size2 ) {
    return false;
  }
  size_t N = W*H;

  double *s0 = in0->data;
  double *s1 = in1->data;
  double *o  = out->data;
  for ( size_t i=0; i<N; i++ ) {
    *(o++) = *(s0++) + *(s1++);
  }

  return true;
}

/// Multiply first with second
bool coco::gsl_matrix_mul_with( gsl_matrix *out, const gsl_matrix *in )
{
  size_t H = out->size1;
  size_t W = out->size2;
  if ( H != in->size1 || H != in->size1 ) {
    return false;
  }
  size_t N = W*H;

  double *s = in->data;
  double *o = out->data;
  for ( size_t i=0; i<N; i++ ) {
    *(o++) *= *(s++);
  }

  return true;
}


/// Divide first by second
bool coco::gsl_matrix_div_by( gsl_matrix *out, const gsl_matrix *in )
{
  size_t H = out->size1;
  size_t W = out->size2;
  if ( H != in->size1 || H != in->size1 ) {
    return false;
  }
  size_t N = W*H;

  double *s = in->data;
  double *o = out->data;
  for ( size_t i=0; i<N; i++ ) {
    double v = *(s++);
    if ( v != 0.0 ) {
      *o /= v;
    }
    o++;
  }

  return true;
}



bool coco::gsl_matrix_add_and_scale( const gsl_matrix *in0, const gsl_matrix *in1, const double f, gsl_matrix *out )
{
  size_t H = out->size1;
  size_t W = out->size2;
  if ( H != in0->size1 || H != in1->size1 || W != in0->size2 || W != in1->size2 ) {
    return false;
  }
  size_t N = W*H;
    
  double *s0 = in0->data;
  double *s1 = in1->data;
  double *o  = out->data;
  for ( size_t i=0; i<N; i++ ) {
    *(o++) = f * (*(s0++) + *(s1++));
  }

  return true;
}


/// Compute C = sA + tB
bool coco::gsl_matrix_add_scaled( const double s, const gsl_matrix *A, const double t, const gsl_matrix *B,
				 gsl_matrix *C )
{
  size_t H = C->size1;
  size_t W = C->size2;
  if ( H != A->size1 || H != B->size1 || W != A->size2 || W != B->size2 ) {
    return false;
  }
  size_t N = W*H;
    
  double *s0 = A->data;
  double *s1 = B->data;
  double *o  = C->data;
  for ( size_t i=0; i<N; i++ ) {
    *(o++) = s * (*(s0++) ) + + t * (*(s1++));
  }

  return true;
}



bool coco::gsl_matrix_warp( const gsl_matrix *in, const gsl_matrix *dx, const gsl_matrix *dy,
                      gsl_matrix *out )
{
  size_t H = out->size1;
  size_t W = out->size2;  
  double *d = out->data;
  double *dxp = dx->data;
  double *dyp = dy->data;
  for ( size_t h=0; h<H; h++ ) {
    for ( size_t w=0; w<W; w++ ) {
      double x = double(w) + *dxp;
      double y = double(h) + *dyp;
      *(d++) = gsl_matrix_interpolate_neumann( in, x,y );
      dxp++;
      dyp++;
    }
  }
  return true;
}



bool coco::gsl_matrix_flip_y( gsl_matrix *M )
{
  size_t W = M->size2;
  size_t H = M->size1;
  for ( size_t y=0; y<H/2; y++ ) {
    for ( size_t x=0; x<W; x++ ) {
      swap( M->data[ x + y*W ], M->data[ x + (H-1-y) * W ] );
    }
  }
  return true;
}

static float linear_to_sRGB(float linear) {
  if (linear <= 0.0031308) {
    return 12.92 * linear;
  } else {
    return 1.055* pow(linear, 1./2.4) - 0.055;
  }
}

void coco::gsl_matrix_delinearize( gsl_matrix *M)
{
  size_t W = M->size2;
   size_t H = M->size1;
   for ( size_t y=0; y<H; y++ ) {
     for ( size_t x=0; x<W; x++ ) {
       M->data[ x + y*W ] = linear_to_sRGB(M->data[x+ y * W]);
     }
   }
}

// Compute some statistics for a matrix
coco::gsl_matrix_stats coco::gsl_matrix_get_stats( gsl_matrix *M )
{
  gsl_matrix_stats stats;
  stats._sum = 0.0;
  stats._average = 0.0;
  stats._max = DOUBLE_MIN;
  stats._absmax = DOUBLE_MIN;
  stats._min = DOUBLE_MAX;
  stats._absmin = DOUBLE_MAX;

  assert( M != NULL );
  size_t N = M->size1 * M->size2;
  if ( N==0 ) {
    return stats;
  }

  double *m = M->data;
  for ( size_t n=0; n<N; n++ ) {
    double v = *(m++);
    double av = fabs( v );
    stats._sum += v;

    if ( v > stats._max ) stats._max = v;
    if ( av > stats._absmax ) stats._absmax = av;

    if ( v < stats._min ) stats._min = v;
    if ( av < stats._absmin ) stats._absmin = av;
  }

  stats._average = stats._sum / double( N );
  return stats;
}



std::ostream &coco::operator<< ( std::ostream &o, gsl_matrix_stats &stats )
{
  o << "avg: " << stats._average << "min: " << stats._min << " max: " << stats._max;
  return o;
}



struct matrix_header {
  size_t _version;
  size_t _size1;
  size_t _size2;
};


// Save image with full precision
bool coco::gsl_matrix_save( const char *filename, const gsl_matrix *M )
{
  gzFile f = gzopen( filename, "wb9" );
  if ( f==NULL ) {
    return false;
  }

  // Write header
  matrix_header hdr;
  hdr._version = 1;
  hdr._size1 = M->size1;
  hdr._size2 = M->size2;
  gzwrite( f, &hdr, sizeof( matrix_header ));

  // Write data
  size_t N = M->size1 * M->size2 * sizeof( double );
  gzwrite( f, M->data, N );
  gzclose( f );
  return true;
}


// Load from lossless save file
gsl_matrix* coco::gsl_matrix_load( const char *filename )
{
  gzFile f = gzopen( filename, "rb" );
  if ( f==NULL ) {
    return NULL;
  }

  // read header
  matrix_header hdr;
  gzread( f, &hdr, sizeof( matrix_header ));
  if ( hdr._version != 1 ) {
    assert( false );
    return NULL;
  }
	  
  gsl_matrix *M = gsl_matrix_alloc( hdr._size1, hdr._size2 );
  if ( M==NULL ) {
    return NULL;
  }

  // Write data
  size_t N = M->size1 * M->size2 * sizeof( double );
  gzread( f, M->data, N );
  gzclose( f );
  return M;
}




/*
bool coco::gsl_matrix_product( gsl_matrix *out, gsl_matrix *A, gsl_matrix *B )
{
  //size_t M = A->size1;
  size_t N = A->size2;
  //size_t L = B->size2;
  if ( N != B->size1 ) {
    assert( false );
    return false;
  }
  gsl_blas_dgemm( CblasNoTrans, CblasNoTrans,
		  1.0, A, B,
		  0.0, out );
  return true;
}

bool coco::gsl_matrix_AtA( gsl_matrix *out, gsl_matrix *A )
{
  gsl_blas_dgemm( CblasTrans, CblasNoTrans,
		  1.0, A, A,
		  0.0, out );
  return true;
}


bool coco::gsl_matrix_AAt( gsl_matrix *out, gsl_matrix *A )
{
  gsl_blas_dgemm( CblasNoTrans, CblasTrans,
		  1.0, A, A,
		  0.0, out );
  return true;
}
*/



bool coco::gsl_matrix_set_random_values( gsl_matrix *out, double min, double max )
{
  size_t N = out->size1 * out->size2;
  for ( size_t i=0; i<N; i++ ) {
    out->data[i] = (rand() % 1000) * ( max-min ) / 1000.0f + min;
  }
  return true;
}




double coco::gsl_vector_norm( gsl_vector *v )
{
  double n = 0.0;
  double *d = v->data;
  for ( size_t i=0; i<v->size; i++ ) {
    n += square( *(d++) );
  }
  return sqrt( n );
}

void coco::gsl_vector_reproject( gsl_vector *v )
{
  double n = gsl_vector_norm( v );
  if ( n > 1.0 ) {
    double *d = v->data;
    for ( size_t i=0; i<v->size; i++ ) {
      *(d++) /= n;
    }
  }
}



void coco::gsl_vector_out( gsl_vector *v )
{
  cout << "( ";
  for ( size_t i=0; i<v->size; i++ ) {
    cout << v->data[i];
    cout << " ";
  }
  cout << ")";
}


void coco::gsl_matrix_out( gsl_matrix *A )
{
  cout << "( ";
  for ( size_t i=0; i<A->size1; i++ ) {
    cout << "  ( ";
    for ( size_t j=0; j<A->size2; j++ ) {
      cout << gsl_matrix_get( A, i,j ) << " ";
    }
    cout << " )" << endl;
  }
  cout << ")";
}


double coco::l2_distance( gsl_matrix *a, gsl_matrix *b )
{
  size_t h = a->size2;
  size_t w = a->size1;
  assert( b->size2 == h );
  assert( b->size1 == w );
  double d = 0.0;
  for ( size_t i=0; i<h*w; i++ ) {
    d += pow( a->data[i] - b->data[i], 2.0 );
  }
  return sqrt( d / double(h*w) );
}

double coco::l2_distance( vector<gsl_matrix*> A, vector<gsl_matrix*> B )
{
  size_t k = A.size();
  assert( k==B.size() );
  double d = 0.0;
  for ( size_t i=0; i<k; i++ ) {
    gsl_matrix *a = A[i];
    gsl_matrix *b = B[i];
    d += pow( l2_distance( a,b ), 2.0 );
  }
  return d/k;
}


/// Pointwise addition of scalar
bool coco::gsl_matrix_add_scalar( gsl_matrix *out, double s )
{
  size_t H = out->size1;
  size_t W = out->size2;
  size_t N = W*H;
    
  double *o  = out->data;
  for ( size_t i=0; i<N; i++ ) {
    *(o++) *= s;
  }

  return true;
}



double coco::gsl_matrix_ssim( gsl_matrix *A, gsl_matrix *B, double dynamic_range )
{
  // default settings
  //const double C1 = 6.5025, C2 = 58.5225; // for 0-255
  const double C1 = pow( 0.01 * dynamic_range, 2.0 );
  const double C2 = pow( 0.03 * dynamic_range, 2.0 );

  size_t H = A->size1;
  size_t W = A->size2;
  assert( B->size1 == H );
  assert( B->size2 == W );
  // compute element-wise squares and product
  gsl_matrix *A_sq = gsl_matrix_alloc( H,W );
  gsl_matrix *B_sq = gsl_matrix_alloc( H,W );
  gsl_matrix_copy_to( A, A_sq );
  gsl_matrix_mul_with( A_sq, A );
  gsl_matrix_copy_to( B, B_sq );
  gsl_matrix_mul_with( B_sq, B );
  gsl_matrix *A_B = gsl_matrix_alloc( H,W );
  gsl_matrix_copy_to( A, A_B );
  gsl_matrix_mul_with( A_B, B );

  // first convolution, squares and products
  gsl_matrix *mu1 = gsl_matrix_alloc( H,W );
  gsl_matrix *mu2 = gsl_matrix_alloc( H,W );
  gsl_matrix *mu1_sq = gsl_matrix_alloc( H,W );
  gsl_matrix *mu2_sq = gsl_matrix_alloc( H,W );
  gsl_matrix *mu1_mu2 = gsl_matrix_alloc( H,W );

  // second convolution, squares and products
  gsl_matrix *sigma1 = gsl_matrix_alloc( H,W );
  gsl_matrix *sigma2 = gsl_matrix_alloc( H,W );
  gsl_matrix *sigma1_sq = gsl_matrix_alloc( H,W );
  gsl_matrix *sigma2_sq = gsl_matrix_alloc( H,W );
  gsl_matrix *sigma12 = gsl_matrix_alloc( H,W );
  
  // temporary variables
  gsl_matrix *temp1 = gsl_matrix_alloc( H,W );
  gsl_matrix *temp2 = gsl_matrix_alloc( H,W );
  gsl_matrix *temp3 = gsl_matrix_alloc( H,W );
  /*************************** END INITS **********************************/



  //////////////////////////////////////////////////////////////////////////
  // PRELIMINARY COMPUTING

  // Gaussian 1 (mu)
  gsl_matrix_gauss_filter( A, mu1, 1.5, 11 );
  gsl_matrix_gauss_filter( B, mu2, 1.5, 11 );
  // squares and products
  gsl_matrix_copy_to( mu1, mu1_sq );
  gsl_matrix_mul_with( mu1_sq, mu1 );
  gsl_matrix_copy_to( mu2, mu2_sq );
  gsl_matrix_mul_with( mu2_sq, mu2 );
  gsl_matrix_copy_to( mu1, mu1_mu2 );
  gsl_matrix_mul_with( mu1_mu2, mu2 );

  // Gaussian 2 (sigma), convolution of squares
  gsl_matrix_gauss_filter( A_sq, sigma1_sq, 1.5, 11 );
  gsl_matrix_gauss_filter( B_sq, sigma2_sq, 1.5, 11 );
  gsl_matrix_gauss_filter( A_B, sigma12, 1.5, 11 );

  gsl_matrix_add_scaled( 1.0, sigma1_sq, -1.0, mu1_sq, sigma1_sq );
  gsl_matrix_add_scaled( 1.0, sigma2_sq, -1.0, mu2_sq, sigma2_sq );
  gsl_matrix_add_scaled( 1.0, sigma12, -1.0, mu1_mu2, sigma12 );


  //////////////////////////////////////////////////////////////////////////
  // FORMULA
  
  // (2*mu1_mu2 + C1)
  gsl_matrix_copy_to( mu1_mu2, temp1 );
  gsl_matrix_scale( temp1, 2.0 );
  gsl_matrix_add_scalar( temp1, C1 );

  // (2*sigma12 + C2)
  gsl_matrix_copy_to( sigma12, temp2 );
  gsl_matrix_scale( temp2, 2.0 );
  gsl_matrix_add_scalar( temp2, C2 );
  
  // ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
  gsl_matrix_add_scaled( 1.0, temp1, 1.0, temp2, temp3 );
  
  // (mu1_sq + mu2_sq + C1)
  gsl_matrix_add( mu1_sq, mu2_sq, temp1 );
  gsl_matrix_add_scalar( temp1, C1 );
  
  // (sigma1_sq + sigma2_sq + C2)
  gsl_matrix_add( sigma1_sq, sigma2_sq, temp2 );
  gsl_matrix_add_scalar( temp2, C2 );

  // ((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2))
  gsl_matrix_add_scaled( 1.0, temp1, 1.0, temp2, temp1 );
  
  // ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2))
  gsl_matrix_div_by( temp3, temp1 );

  // result is average of temp3 (=ssim_map)
  double ssim = gsl_matrix_get_stats( temp3 )._average;

  // free up temporary matrices
  gsl_matrix_free( A_sq );
  gsl_matrix_free( B_sq );
  gsl_matrix_free( A_B );

  gsl_matrix_free( mu1 );
  gsl_matrix_free( mu2 );
  gsl_matrix_free( mu1_sq );
  gsl_matrix_free( mu2_sq );
  gsl_matrix_free( mu1_mu2 );

  gsl_matrix_free( sigma1 );
  gsl_matrix_free( sigma2 );
  gsl_matrix_free( sigma12 );
  gsl_matrix_free( sigma1_sq );
  gsl_matrix_free( sigma2_sq );

  gsl_matrix_free( temp1 );
  gsl_matrix_free( temp2 );
  gsl_matrix_free( temp3 );

  return ssim;
}


// Median filter
bool coco::gsl_matrix_median( gsl_matrix *A, gsl_matrix *B, int radius )
{
  int H = A->size1;
  int W = A->size2;
  for ( int x=0; x<W; x++ ) {
    for ( int y=0; y<H; y++ ) {
      vector<double> entries;
      for ( int xx=x-radius; xx<x+radius; xx++ ) {
	for ( int yy=y-radius; yy<y+radius; yy++ ) {
	  if ( xx>=0 && xx<W && yy>=0 && yy<H ) {
	    entries.push_back( gsl_matrix_get( A, yy,xx ));
	  }
	}
      }
      sort( entries.begin(), entries.end() );
      double v = entries[ entries.size() / 2 ];
      gsl_matrix_set( B, y,x, v );
    }
  }
  return true;
}



// Replacements for GSL functions
coco::gsl_matrix *coco::gsl_matrix_alloc( size_t size1, size_t size2 )
{
  coco::gsl_matrix *M = new coco::gsl_matrix;
  assert( size1 > 0 );
  assert( size2 > 0 );
  M->size1 = size1;
  M->size2 = size2;
  M->tda = 0;
  M->data = new double[ size1*size2 ];
  M->block = NULL;
  M->owner = 0;
  return M;
}

void coco::gsl_matrix_free( gsl_matrix *M )
{
  if ( M==NULL ) {
    return;
  }
  delete[] M->data;
  M->data = NULL;
  delete M;
}

double coco::gsl_matrix_get( const gsl_matrix *M, size_t s1, size_t s2 )
{
  assert( M != NULL );
  assert( s1 < M->size1 );
  assert( s2 < M->size2 );
  return M->data[ s1 * M->size2 + s2 ];
}

void coco::gsl_matrix_set( gsl_matrix *M , size_t s1, size_t s2, double v )
{
  assert( M != NULL );
  assert( s1 < M->size1 );
  assert( s2 < M->size2 );
  M->data[ s1 * M->size2 + s2 ] = v;
}

void coco::gsl_matrix_scale( gsl_matrix *M, double v )
{
  assert( M != NULL );
  assert( M->data != NULL );
  for ( size_t i=0; i<M->size1 * M->size2; i++ ) {
    M->data[i] *= v;
  }
}

void coco::gsl_matrix_transpose_memcpy( gsl_matrix *dest, const gsl_matrix *src )
{
  assert( dest != NULL );
  assert( dest->data != NULL );
  assert( src != NULL );
  assert( src->data != NULL );
  assert( src->size1 == dest->size2 );
  assert( src->size2 == dest->size1 );
  size_t i=0;
  for ( size_t x=0; x<dest->size2; x++ ) {
    for ( size_t y=0; y<dest->size1; y++ ) {
      dest->data[i++] += dest->data[ x*src->size2 + y ];
    }
  }
}

void coco::gsl_matrix_add( gsl_matrix *dest, const gsl_matrix *src )
{
  assert( dest != NULL );
  assert( dest->data != NULL );
  assert( src != NULL );
  assert( src->data != NULL );
  assert( src->size1 == dest->size1 );
  assert( src->size2 == dest->size2 );
  for ( size_t i=0; i<src->size1 * src->size2; i++ ) {
    dest->data[i] += src->data[i];
  }
}


void coco::gsl_matrix_set_all( gsl_matrix *M, double v )
{
  assert( M != NULL );
  assert( M->data != NULL );
  for ( size_t i=0; i<M->size1 * M->size2; i++ ) {
    M->data[i] = v;
  }
}



// Alloc/free replacements for GSL functions
coco::gsl_vector *coco::gsl_vector_alloc( size_t size )
{
  coco::gsl_vector *v = new coco::gsl_vector;
  assert( size>0 );
  v->size = size;
  v->stride = 0;
  v->data = new double[size];
  v->block = NULL;
  v->owner = 0;
  return v;
}

void coco::gsl_vector_free( gsl_vector *v )
{
  assert( v != NULL );
  assert( v->data != NULL );
  delete[] v->data;
  delete v;
}
