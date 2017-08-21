/** \file gsl_matrix_convolutions.h

    File imported from "common" lib, use if this library is not available.
    Uses "gov" namespace (Graphics-optics-vision, MPI).

    Compute convolutions for gsl matrices
    
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

#include <math.h>
#include <assert.h>

#include "gsl_matrix_convolutions.h"
#include "gsl_matrix_helper.h"
#include "gsl_image.h"

#include "linalg3d.h"
#include "defs.h"

using namespace std;


/********************************************************
  Create some typical kernels (GSL objects)
*********************************************************/

coco::gsl_vector *coco::gsl_kernel_gauss_1xn( size_t n, double sigma )
{
  gsl_vector *v = gsl_vector_alloc( n );
  int o = n/2;

  // Only odd sizes supported
  assert( o*2+1 == (int)n );
  double s = 0.0;
  double *kd = v->data;
  for ( int x=-o; x<=o; x++ ) {
    double ex = exp( -0.5 * x*x / (sigma*sigma) );
    *kd = ex;
    s += *(kd++);
  }

  assert( s > 0.0 );
  for ( size_t i=0; i<n; i++ ) {
    v->data[i] /= s;
  }

  return v;
}

coco::gsl_matrix *coco::gsl_kernel_gauss_3x3( double sigmax, double sigmay )
{
  gsl_matrix *k = gsl_matrix_alloc( 3,3 );

  double s = 0.0;
  double *kd = k->data;
  for ( int y=-1; y<=1; y+=1 ) {
    for ( int x=-1; x<=1; x+=1 ) {

      double ex = exp( -0.5*x*x / pow(sigmax, 2.0) );
      double ey = exp( -0.5*y*y / pow(sigmay, 2.0) );
      *kd = ex*ey;
      s += *(kd++);

    }
  }

  kd = k->data;
  for ( int i=0; i<9; i++ ) {
    *(kd++) /= s;
  }

  return k;
}



coco::gsl_matrix *coco::gsl_kernel_gauss_nxn( size_t n, double sigmax, double sigmay, double angle )
{
  gsl_matrix *k = gsl_matrix_alloc( n,n );
  int n2 = (n-1) / 2;
  assert( n2*2 + 1 == (int)n );

  double s = 0.0;
  double *kd = k->data;
  for ( int y=-n2; y<=n2; y+=1 ) {
    for ( int x=-n2; x<=n2; x+=1 ) {

      double xs = cos( angle )*x + sin(angle) * y;
      double ys = -sin( angle ) * x + cos(angle) * y; 
      double ex, ey;
      if ( sigmax != 0.0 ) {
	ex = exp( -0.5*xs*xs / pow(sigmax, 2.0) );
      }
      else {
	ex = (x==0) ? 1.0 : 0.0;
      }
      if ( sigmay != 0.0 ) {
	ey = exp( -0.5*ys*ys / pow(sigmay, 2.0) );
      }
      else {
	ey = (y==0) ? 1.0 : 0.0;
      }
      *kd = ex*ey;
      s += *(kd++);

    }
  }

  kd = k->data;
  for ( int i=0; i<n*n; i++ ) {
    *(kd++) /= s;
  }

  return k;
}


coco::gsl_matrix *coco::gsl_kernel_box( size_t w, size_t h )
{
  gsl_matrix *k = gsl_matrix_alloc( h,w );
  double *s = k->data;
  const double v = 1.0 / double(w*h);
  for ( size_t y=0; y<h; y++ ) {
    for ( size_t x=0; x<w; x++ ) {
      *(s++) = v;
    }
  }
  return k;
}


// Compute product filter
coco::gsl_matrix *coco::gsl_kernel_combine( gsl_matrix *a, gsl_matrix *b )
{
  // Alloc product matrix
  assert( a != NULL );
  assert( b != NULL );
  size_t aw = a->size2;
  size_t ah = a->size1;
  size_t bw = b->size2;
  size_t bh = b->size1;
  assert( (bw % 2) == 1 );
  assert( (bh % 2) == 1 );
  size_t pw = bw + aw-1;
  size_t ph = bh + ah-1;
  gsl_matrix *p = gsl_matrix_alloc( ph,pw );
  memset( p->data, 0, sizeof(double) * pw * ph );

  // Compute entries
  // Loop over A, then loop over B to update inner entries
  for ( size_t by=0; by<bh; by++ ) {
    for ( size_t bx=0; bx<bw; bx++ ) {

      double bv = gsl_matrix_get( b, by,bx );
      for ( size_t ay=0; ay<ah; ay++ ) {
	for ( size_t ax=0; ax<aw; ax++ ) {

	  size_t px = bx + ax;
	  size_t py = by + ay;
	  double v = bv * gsl_matrix_get( a, ay,ax ) + gsl_matrix_get( p,py,px );
	  gsl_matrix_set( p, py,px, v );

	}
      }

    }
  }

  return p;
}


// Compute product filter
coco::gsl_vector *coco::gsl_kernel_combine( gsl_vector *a, gsl_vector *b )
{
  gsl_matrix *ma = gsl_matrix_alloc( 1, a->size );
  memcpy( ma->data, a->data, sizeof(double) * a->size );
  gsl_matrix *mb = gsl_matrix_alloc( 1, b->size );
  memcpy( mb->data, b->data, sizeof(double) * b->size );
  gsl_matrix *m = gsl_kernel_combine( ma,mb );

  assert( m->size1 == 1 );
  gsl_vector *v = gsl_vector_alloc( m->size2 );
  memcpy( v->data, m->data, sizeof(double) * m->size2 );

  gsl_matrix_free( m );
  gsl_matrix_free( ma );
  gsl_matrix_free( mb );

  return v;
}



bool coco::gsl_matrix_convolution_3x1( const gsl_matrix *ms, const gsl_matrix *k, gsl_matrix* md )
{
  // Probably bugged, review (boundary case not Neumann)
  assert( false );

  if ( k->size1 != 1 || k->size2 != 3 ) {
    assert( false );
    return false;
  }

  size_t h = ms->size1;
  size_t w = ms->size2;
  if ( h<2 || w<2 ) {
    assert( false );
    return false;
  }
  if ( md->size2 != w || md->size1 != h ) {
    assert( false );
    return false;
  }

  const double k0 = k->data[0];
  const double k1 = k->data[1];
  const double k2 = k->data[2];
  double *s = ms->data;
  double *d = md->data;
  for ( size_t y=0; y<h; y++ ) {
    // First entry: One-sided derivative
    *d = (*(s+1)*k2 + *s*k1) / 2.0;
    s++; d++;

    // Next entries until last: Central difference
    for ( size_t x=1; x<w-1; x++ ) {
      *d = (*(s+1)*k2 + *s*k1 + *(s-1)*k0) / 3.0;
      s++; d++;
    }

    // Last entry: One-sided derivative
    *d = (*s*k1 + *(s-1)*k0) / 2.0;
    s++; d++;
  }
  
  return true;
}

bool coco::gsl_matrix_convolution_1x3( const gsl_matrix *ms, const gsl_matrix* k, gsl_matrix *md )
{
  // Probably bugged, review (boundary case not Neumann)
  assert( false );

  if ( k->size1 != 3 || k->size2 != 1 ) {
    assert( false );
    return false;
  }

  size_t h = ms->size1;
  size_t w = ms->size2;
  if ( h<2 || w<2 ) {
    assert( false );
    return false;
  }
  if ( md->size2 != w || md->size1 != h ) {
    assert( false );
    return false;
  }

  const double k0 = k->data[0];
  const double k1 = k->data[1];
  const double k2 = k->data[2];
  double *s = ms->data;
  double *d = md->data;

  // First row: Ignore first filter entry
  for ( size_t x=0; x<w; x++ ) {
    *d = (*(s+w)*k2 + *s*k1) / 2.0;
    s++; d++;
  }

  // Next rows until last: Full filter kernel
  for ( size_t y=1; y<h-1; y++ ) {
    for ( size_t x=0; x<w; x++ ) {
      *d = (*(s+w)*k2 + *s*k1 + *(s-w)*k0) / 3.0;
      s++; d++;
    }
  }
 
  // Last row: Ignore last filter entry
  for ( size_t x=0; x<w; x++ ) {
    *d = (*s*k1 + *(s-w)*k0) / 2.0;
    s++; d++;
  }

  return true;
}

bool coco::gsl_matrix_convolution_3x3( const gsl_matrix *ms, const gsl_matrix* k, gsl_matrix *md )
{
  if ( k->size1 != 3 || k->size2 != 3 ) {
    assert( false );
    return false;
  }

  size_t h = ms->size1;
  size_t w = ms->size2;
  if ( h<3 || w<3 ) {
    assert( false );
    return false;
  }
  if ( md->size2 != w || md->size1 != h ) {
    assert( false );
    return false;
  }

  const double k00 = k->data[0];
  const double k01 = k->data[1];
  const double k02 = k->data[2];
  const double k10 = k->data[3];
  const double k11 = k->data[4];
  const double k12 = k->data[5];
  const double k20 = k->data[6];
  const double k21 = k->data[7];
  const double k22 = k->data[8];
  double *s = ms->data;
  double *d = md->data;

#define COMPUTE_KERNEL( a,b,c,d,e,f,g,h,i ) ( *(a)*k00 + *(b)*k01 + *(c)*k02 \
					      + *(d)*k10 + *(e)*k11 + *(f)*k12 \
					      + *(g)*k20 + *(h)*k21 + *(i)*k22 )
  // First row: ignore first kernel row
  *(d++) = COMPUTE_KERNEL( s+0, s+0, s+1,
			   s+0, s+0, s+1,
			   s+w, s+w, s+w+1 );
  s++;
  for ( size_t x=1; x<w-1; x++ ) {
    *(d++) = COMPUTE_KERNEL( s-1, s+0, s+1,
			     s-1, s+0, s+1,
			     s+w-1, s+w+0, s+w+1 );
    s++;
  }
  *(d++) = COMPUTE_KERNEL( s-1, s+0, s+0,
			   s-1, s+0, s+0,
			   s+w-1, s+w+0, s+w+0 );
  s++;

  // Next rows: Full y range of kernel
  for ( size_t y=1; y<h-1; y++ ) {
    *(d++) = COMPUTE_KERNEL( s-w+0, s-w+0, s-w+1,
			     s+0, s+0, s+1,
			     s+w+0, s+w+0, s+w+1 );
    s++;

    // Next entries until last: Full kernel
    for ( size_t x=1; x<w-1; x++ ) {
      *(d++) = COMPUTE_KERNEL( s-w-1, s-w+0, s-w+1,
			       s-1, s+0, s+1,
			       s+w-1, s+w+0, s+w+1 );
      s++;
    }

    // Last entry: Ignore last column
    *(d++) = COMPUTE_KERNEL( s-w-1, s-w+0, s-w+0,
			     s-1, s+0, s+0,
			     s+w-1, s+w+0, s+w+0 );
    s++;
  }

  // Last row: Ignore last kernel row
  *(d++) = COMPUTE_KERNEL( s-w+0, s-w+0, s-w+1,
			   s+0, s+0, s+1,
			   s+0, s+0, s+1 );
  s++;
  for ( size_t x=1; x<w-1; x++ ) {
    *(d++) = COMPUTE_KERNEL( s-w-1, s-w+0, s-w+1,
			     s-1, s+0, s+1,
			     s-1, s+0, s+1 );
    s++;
  }
  *(d++) = COMPUTE_KERNEL( s-w-1, s-w+0, s-w+0,
			   s-1, s+0, s+0,
			   s-1, s+0, s+0 );
  s++;
  return true;
}


// Convolutions in a single direction, arbitrary odd kernel size
bool coco::gsl_matrix_convolution_1xn( const gsl_matrix *s, const gsl_vector* kernel, gsl_matrix *d )
{
  // Input check
  assert( s != NULL );
  assert( kernel != NULL );
  assert( d != NULL );
  // Size check
  int W = s->size2;
  int H = s->size1;
  assert( W == (int)d->size2 );
  assert( H == (int)d->size1 );
  double *sd = s->data;
  double *dd = d->data;
  int offset = kernel->size / 2;
  // Odd kernel size only
  assert( offset*2+1 == (int)kernel->size );
  for ( int y=0; y<H; y++ ) {
    for ( int x=0; x<W; x++ ) {
      int om = min( offset, x );
      int oM = min( offset, W-x-1 );
      int N = (oM+om) + 1;
      double weight = 0.0;
      double *sr = sd-om;
      double *kr = kernel->data + max( 0, offset-x );
      double sum = 0.0;
      //cout << "cv " << x << ":" << y << " " << N << endl;

      for ( int i=0; i<N; i++ ) {
	double k = *(kr++);
	weight += k;
	sum += k * *(sr++);
      }
      *dd = sum / weight;
      dd++;
      sd++;
    }
  }

  return true;
}

bool coco::gsl_matrix_convolution_nx1( const gsl_matrix *s, const gsl_vector* kernel, gsl_matrix *d )
{
  // Input check
  assert( s != NULL );
  assert( kernel != NULL );
  assert( d != NULL );
  // Size check
  int W = s->size2;
  int H = s->size1;
  assert( W == (int)d->size2 );
  assert( H == (int)d->size1 );
  double *sd = s->data;
  double *dd = d->data;
  int offset = kernel->size / 2;
  // Odd kernel size only
  assert( offset*2+1 == (int)kernel->size );
  for ( int y=0; y<H; y++ ) {
    for ( int x=0; x<W; x++ ) {
      int om = min( offset, y );
      int oM = min( offset, H-y-1 );
      int N = (oM+om) + 1;
      double weight = 0.0;
      double *sr = sd-om*W;
      double *kr = kernel->data + max( 0, offset-y );
      double sum = 0.0;
      for ( int i=0; i<N; i++ ) {
	double k = *(kr++);
	weight += k;
	sum += k * (*sr);
	sr += W;
      }
      *dd = sum / weight;
      dd++;
      sd++;
    }
  }

  return true;
}


bool coco::gsl_matrix_convolution_nxn( const gsl_matrix *s, const gsl_matrix* kernel, gsl_matrix *d )
{
  // This is slow.
  // Input check
  assert( s != NULL );
  assert( kernel != NULL );
  assert( d != NULL );
  // Size check
  int W = s->size2;
  int H = s->size1;
  assert( W == (int)d->size2 );
  assert( H == (int)d->size1 );

  // Slow loop.
  int khh = (kernel->size1 - 1) / 2;
  int kwh = (kernel->size2 - 1) / 2;

  for ( int y=0; y<H; y++ ) {
    for ( int x=0; x<W; x++ ) {

      double v = 0.0;
      double w = 0.0;
      size_t index=0;
      for ( int j=0; j<(int)kernel->size2; j++ ) {
	for ( int i=0; i<(int)kernel->size1; i++ ) {
	  
	  int xx = x - kwh + i;
	  int yy = y - khh + j;
	  if ( xx>=0 && xx<W && yy>=0 && yy<H ) {
	    double k = kernel->data[index];
	    w += k;
	    v += k * gsl_matrix_get( s, yy,xx );
	  }

	  index++;
	}
      }

      if ( w>0.0 ) {
	v /= w;
      }
      gsl_matrix_set( d, y,x, v );

    }
  }

  return true;
}

bool coco::gsl_matrix_gauss_filter( gsl_matrix *s, gsl_matrix *d, double sigma, size_t kernel_size )
{
  size_t W = s->size2;
  size_t H = s->size1;
  assert( d->size2 == W );
  assert( d->size1 == H );
  if ( sigma==0.0 ) {
    gsl_matrix_copy_to( s,d );
    return true;
  }

  gsl_vector *kd = gsl_kernel_gauss_1xn( kernel_size, sigma );
  gsl_matrix *tmp = gsl_matrix_alloc( H,W );
  gsl_matrix_convolution_1xn( s,kd, tmp );
  gsl_matrix_convolution_nx1( tmp,kd, d );
  gsl_matrix_free( tmp );
  gsl_vector_free( kd );

  return true;
}


