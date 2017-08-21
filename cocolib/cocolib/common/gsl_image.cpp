/** \file gsl_image.cpp

    File imported from "common" lib, use if this library is not available.
    Uses "gov" namespace (Graphics-optics-vision, MPI).

    Image structure and helper functions. Built on gsl_matrix and QImage.
    
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

#include <qimage.h>
#include <assert.h>
#include <algorithm>
#include <math.h>
#include <float.h>

#include <zlib.h>
#include <qimage.h>

#include "../common/debug.h"
#include "../common/linalg3d.h"

/*
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
*/

#include "gsl_image.h"
#include "gsl_matrix_helper.h"
#include "gsl_matrix_convolutions.h"
#include "../defs.h"

using namespace std;
using namespace coco;

#if QT_VERSION >= 0x40000
#define QIMAGE_FORMAT_ARGB32 QImage::Format_ARGB32
#else
#define QIMAGE_FORMAT_ARGB32 32
#endif


// Allocate image
gsl_image *coco::gsl_image_alloc( int w, int h )
{
  if ( w<=0 || h<=0 ) {
    assert( false );
    return NULL;
  }

  gsl_image *I = new gsl_image;
  if ( I==NULL ) {
    return NULL;
  }
  memset( I,0, sizeof( gsl_image ));

  I->_r = gsl_matrix_alloc( h,w );
  I->_g = gsl_matrix_alloc( h,w );
  I->_b = gsl_matrix_alloc( h,w );
  I->_a = gsl_matrix_alloc( h,w );
  if ( I->_r == NULL || I->_g == NULL || I->_b == NULL || I->_a == NULL ) {
    gsl_image_free( I );
  }

  size_t N = w*h*sizeof(double);
  memset( I->_r->data, 0, N );
  memset( I->_g->data, 0, N );
  memset( I->_b->data, 0, N );
  memset( I->_a->data, 0, N );

  I->_w = w;
  I->_h = h;
  I->_cs = RGB;

  return I;
}

// Copy matrix into image, weighted color channels
bool coco::gsl_image_from_matrix( gsl_image *I,
			    const gsl_matrix *M,
			    double rs, double gs, double bs )
{
  double *s = M->data;
  double *rd = I->_r->data;
  double *gd = I->_g->data;
  double *bd = I->_b->data;
  if ( I->_h != M->size1 || I->_w != M->size2 ) {
    assert( false );
    return false;
  }

  size_t N = I->_w * I->_h;
  for ( size_t i=0; i<N; i++ ) {
    *(rd++) = *s * rs;
    *(gd++) = *s * gs;
    *(bd++) = *s * bs;
    s++;
  }
  
  return true;
}

// Copy buffer into each image channel
bool coco::gsl_image_from_buffer( gsl_image *I, float *buffer )
{
  size_t N = I->_w * I->_h;
  for ( size_t i=0; i<N; i++ ) {
    I->_r->data[i] = buffer[i];
    I->_g->data[i] = buffer[i];
    I->_b->data[i] = buffer[i];
  }
  return true;
}



// Get image channels
vector<gsl_matrix*> coco::gsl_image_get_channels( gsl_image *I )
{
  vector<gsl_matrix*> Cs;
  Cs.push_back( I->_r );
  Cs.push_back( I->_g );
  Cs.push_back( I->_b );
  return Cs;
}


gsl_image coco::gsl_image_from_channels( const std::vector<gsl_matrix*> &u )
{
  assert( u.size() > 2 );
  gsl_image I;
  I._r = u[0];
  I._g = u[1];
  I._b = u[2];
  if ( u.size() > 3 ) {
    I._a = u[3];
  }
  else {
    I._a = NULL;
  }

  I._w = u[0]->size2;
  I._h = u[0]->size1;

  assert( I._w == u[1]->size2 );
  assert( I._h == u[1]->size1 );
  assert( I._w == u[2]->size2 );
  assert( I._h == u[2]->size1 );

  return I;
}




// Create single greyscale value matrix from image
gsl_matrix *coco::gsl_image_to_greyscale( const gsl_image *I )
{
  gsl_matrix *M = gsl_matrix_alloc( I->_h, I->_w );
  double *s = M->data;
  double *rd = I->_r->data;
  double *gd = I->_g->data;
  double *bd = I->_b->data;

  size_t N = I->_w * I->_h;
  for ( size_t i=0; i<N; i++ ) {
    *(s++) = (*(rd++) + *(gd++) + *(bd++)) / 3.0;
  }
  
  return M;
}


// Create single greyscale value matrix from image
/*
bool coco::gsl_image_transform( gsl_image *I, const Mat44d &M )
{
  double *rd = I->_r->data;
  double *gd = I->_g->data;
  double *bd = I->_b->data;

  size_t N = I->_w * I->_h;
  for ( size_t i=0; i<N; i++ ) {
    Vec4d v( *rd, *gd, *bd, 1.0 );
    Vec4d w = M*v;

    *(rd++) = w.x;
    *(gd++) = w.y;
    *(bd++) = w.z;
  }
  
  return true;
}
*/



// Get image channel
gsl_matrix *coco::gsl_image_get_channel( gsl_image *I, gsl_image_channel n )
{
  switch( n ) {
  case GSL_IMAGE_RED:
    return I->_r;
  case GSL_IMAGE_GREEN:
    return I->_g;
  case GSL_IMAGE_BLUE:
    return I->_b;
  case GSL_IMAGE_ALPHA:
    return I->_a;
  default:
    ERROR( "Unknown image channel " << n << " requested." << endl );
    assert( false );
  }
  return NULL;
}



// Copy an image at a specific location into another one
bool coco::gsl_image_copy_to( const gsl_image *src, gsl_image *dst, size_t X, size_t Y )
{
  size_t W = src->_w;
  size_t H = src->_h;
  if ( X>= dst->_w ) return false;
  if ( Y>= dst->_h ) return false;
  size_t wc = min( W, dst->_w - X );
  size_t hc = min( H, dst->_h - Y );
  for ( size_t y=0; y<hc; y++ ) {
    size_t pos_src = y*W;
    size_t pos_dst = (y+Y) * dst->_w + X;
    for ( size_t x=0; x<wc; x++ ) {
      dst->_r->data[pos_dst] = src->_r->data[pos_src];
      dst->_g->data[pos_dst] = src->_g->data[pos_src];
      dst->_b->data[pos_dst] = src->_b->data[pos_src];
      pos_src++;
      pos_dst++;
    }
  }
  return true;
}


// Copy an image at a specific location into another one
bool coco::gsl_image_copy_to( const gsl_image *src, gsl_image *dst,
			     gsl_matrix *stencil, size_t X, size_t Y )
{
  size_t W = src->_w;
  size_t H = src->_h;
  if ( X>= dst->_w ) return false;
  if ( Y>= dst->_h ) return false;
  assert( src->_w == stencil->size2 );
  assert( src->_h == stencil->size1 );
  size_t wc = min( W, dst->_w - X );
  size_t hc = min( H, dst->_h - Y );
  for ( size_t y=0; y<hc; y++ ) {
    size_t pos_src = y*W;
    size_t pos_dst = (y+Y) * dst->_w + X;
    for ( size_t x=0; x<wc; x++ ) {
      double ap = stencil->data[pos_src];
      double am = 1.0 - ap;
      dst->_r->data[pos_dst] = ap*src->_r->data[pos_src] + am*dst->_r->data[pos_dst];
      dst->_g->data[pos_dst] = ap*src->_g->data[pos_src] + am*dst->_g->data[pos_dst];
      dst->_b->data[pos_dst] = ap*src->_b->data[pos_src] + am*dst->_b->data[pos_dst];
      pos_src++;
      pos_dst++;
    }
  }
  return true;
}




// Copy image into another where matrix has positive entries
bool coco::gsl_image_copy_with_stencil( const gsl_image *in, gsl_matrix *stencil, gsl_image *out )
{
  if ( in->_h != out->_h || in->_w != out->_w ) {
    assert( false );
    return false;
  }
  if ( in->_h != stencil->size1 || in->_w != stencil->size2 ) {
    assert( false );
    return false;
  }

  size_t N = in->_w * in->_h;
  for ( size_t i=0; i<N; i++ ) {
    if ( stencil->data[i] > 0.0 ) {
      out->_r->data[i] = in->_r->data[i];
      out->_g->data[i] = in->_g->data[i];
      out->_b->data[i] = in->_b->data[i];
    }
  }

  return true;
}

// Copy image into another where matrix has positive entries
bool coco::gsl_image_copy_matrix_to_channel( const gsl_matrix *in, gsl_image *out,
					    gsl_image_channel channel )
{
  if ( in==NULL || out==NULL ) {
    assert( false );
    return false;
  }

  if ( out->_h != in->size1 || out->_w != in->size2 ) {
    assert( false );
    return false;
  }

  double *dst = NULL;
  switch( channel ) {
  case GSL_IMAGE_RED:
    dst = out->_r->data;
    break;
  case GSL_IMAGE_GREEN:
    dst = out->_g->data;
    break;
  case GSL_IMAGE_BLUE:
    dst = out->_b->data;
    break;
  case GSL_IMAGE_ALPHA:
    dst = out->_a->data;
    break;
  }

  if ( dst==NULL ) {
    assert( false );
    return false;
  }

  size_t N = out->_w * out->_h;
  double *src = in->data;
  if ( src==NULL ) {
    assert( false );
    return false;
  }

  for ( size_t i=0; i<N; i++ ) {
    *(dst++) = *(src++);
  }

  return true;
}


// Copy matrix into image, weighted color channels
bool coco::gsl_image_from_signed_matrix( gsl_image *I,
					const gsl_matrix *M,
					double rs_neg, double gs_neg, double bs_neg,
					double rs_pos, double gs_pos, double bs_pos )
{
  double *s = M->data;
  double *rd = I->_r->data;
  double *gd = I->_g->data;
  double *bd = I->_b->data;
  if ( I->_h != M->size1 || I->_w != M->size2 ) {
    assert( false );
    return false;
  }

  size_t N = I->_w * I->_h;
  for ( size_t i=0; i<N; i++ ) {
    if ( *s < 0.0 ) {
      *(rd++) = -*s * rs_neg;
      *(gd++) = -*s * gs_neg;
      *(bd++) = -*s * bs_neg;
    }
    else {
      *(rd++) =  *s * rs_pos;
      *(gd++) =  *s * gs_pos;
      *(bd++) =  *s * bs_pos;
    }
    s++;
  }
  
  return true;
}



// Load image
gsl_image *coco::gsl_image_load( const std::string &filename )
{
  QImage I( filename.c_str() );
  return gsl_image_from_qimage( I );
}

// Load image with specific resolution
gsl_image *coco::gsl_image_load_scaled( const std::string &filename, size_t w, size_t h )
{
  QImage I( filename.c_str() );
  if ( w*h != 0 ) {
    QImage S = I.scaled( w,h, Qt::IgnoreAspectRatio, Qt::SmoothTransformation );
    return gsl_image_from_qimage( S );
  }
  return gsl_image_from_qimage( I );
}


gsl_image *coco::gsl_image_load_pfm( const std::string &filename )
{
  std::FILE *const file = std::fopen(filename.c_str(),"rb");
  if(!file) {
    TRACE("gsl_image_load_pfm() : PFM file not found" << filename.c_str() << endl );
    return 0;
  }

  char pfm_type, item[1024] = { 0 };
  int W = 0, H = 0, err = 0;
  double scale = 0;
  // scan first line
  while ((err=std::fscanf(file,"%1023[^\n]",item))!=EOF && (*item=='#' || !err)) {
    std::fgetc(file);
  }

  if (std::sscanf(item," P%c",&pfm_type)!=1) {
    TRACE("gsl_image_load_pfm() : PFM header not found in file " << filename.c_str() << endl );
    return 0;
  }

  // scan second line
  while ((err=std::fscanf(file," %1023[^\n]",item))!=EOF && (*item=='#' || !err)) {
    std::fgetc(file);
  }

  if ((err=std::sscanf(item," %d %d",&W,&H))<2) {
    TRACE("gsl_image_load_pfm() : W and H not defined in header of file " << filename.c_str() << endl );
    return 0;
  }

  if (err==2) {
    while ((err=std::fscanf(file," %1023[^\n]",item))!=EOF && (*item=='#' || !err)) {
      std::fgetc(file);
    }
    if (std::sscanf(item,"%lf",&scale)!=1) {
      TRACE("gsl_image_load_pfm() : WARNING : SCALE field is undefined in file " << filename.c_str() << endl );
    }
  }

  std::fgetc(file);
  const bool is_color = (pfm_type=='F'), is_inverted = (scale>0)!=endianness();

  gsl_image *G = gsl_image_alloc(W,H);

  if (is_color) {
    float* buf = (float*) malloc(3*W*sizeof(float));
    double *ptr_r = G->_r->data, *ptr_g = G->_g->data, *ptr_b = G->_b->data;
    for(int y=0; y<H; ++y) {
      std::fread(buf, sizeof(float), 3*W, file);
      if (is_inverted) {
        invert_endianness(buf,3*W);
      }

      const float *ptrs = buf;
      for( int x=0; x<W; ++x) {
        *(ptr_r++) = (double)*(ptrs++);
        *(ptr_g++) = (double)*(ptrs++);
        *(ptr_b++) = (double)*(ptrs++);
      }
    }
  } else {
    float* buf = (float*) malloc(W*sizeof(float));
    double *ptrd = G->_r->data;
    for(int y=0; y<H; ++y) {
      std::fread(buf, sizeof(float), W, file);
      if (is_inverted) {
        invert_endianness(buf,W);
      }
      const float *ptrs = buf;
      for( int x=0; x<W; ++x) {
        *(ptrd++) = (double)*(ptrs++);
      }
    }

    // copy the other channels for the grayscale image
    memcpy(G->_g->data, G->_r->data, W*H*sizeof(double));
    memcpy(G->_b->data, G->_r->data, W*H*sizeof(double));
  }
  std::fclose(file);

  // flip image: pfm has the 0,0 coordinate at the BOTTOM left corner of the image
  // gsl_image (and matrix) has the 0,0 coordinate at the TOP left corner of the image
  gsl_image_flip_y( G );

  return G;
}


// Copy image
gsl_image *coco::gsl_image_copy( const gsl_image *I )
{
  if ( I==NULL ) {
    assert( false );
    return NULL;
  }
  size_t N = I->_w * I->_h * sizeof(double);
  if ( N==0 ) {
    assert( false );
    return NULL;
  }

  gsl_image *G = gsl_image_alloc( I->_w,I->_h );
  if ( G==NULL ) {
    assert( false );
    return NULL;
  }

  memcpy( G->_r->data, I->_r->data, N );
  memcpy( G->_g->data, I->_g->data, N );
  memcpy( G->_b->data, I->_b->data, N );
  if ( I->_a != NULL ) {
    memcpy( G->_a->data, I->_a->data, N );
  }
  else {
    memset( G->_a->data, 0, N );
  }

  return G;
}


// Initialize image from QImage
gsl_image *coco::gsl_image_from_qimage( const QImage &I )
{
  int w = I.width();
  int h = I.height();
  if ( w*h==0 ) {
    return NULL;
  }

  gsl_image *G = gsl_image_alloc( w,h );
  if ( G==NULL ) {
    return G;
  }

  QImage J;
  if ( I.depth() != 32) {
#if QT_VERSION >= 0x40000
    J = I.convertToFormat( QIMAGE_FORMAT_ARGB32 );
#else
    J = I;
    J.convertDepth( 32 );
#endif
  }
  else {
    J = I;
  }
  
  // Transfer data from QImage to gsl image
  double *ad = G->_a->data;
  double *rd = G->_r->data;
  double *gd = G->_g->data;
  double *bd = G->_b->data;
  for ( int y=0; y<h; y++ ) {
    QRgb *c = (QRgb*)J.scanLine( y );
    for ( int x=0; x<w; x++ ) {
      int a = qAlpha( *c );
      int r = qRed( *c );
      int g = qGreen( *c );
      int b = qBlue( *c );
      c++;
      
      *(ad++) = a / 255.0;
      *(rd++) = r / 255.0;
      *(gd++) = g / 255.0;
      *(bd++) = b / 255.0;
    }
  }

  return G;
}


bool coco::gsl_image_to_qimage( const gsl_image *G, QImage &I )
{
  if ( G==NULL ) {
    return false;
  }
  size_t w = G->_w;
  size_t h = G->_h;
  if ( w*h==0 ) {
    return false;
  }
  I = QImage( w,h, QIMAGE_FORMAT_ARGB32 );

  // Transfer data from gsl image to QImage
  //double *ad = G->_a->data;
  double *rd = G->_r->data;
  double *gd = G->_g->data;
  double *bd = G->_b->data;
  for ( size_t y=0; y<h; y++ ) {
    QRgb *c = (QRgb*)I.scanLine( y );
    for ( size_t x=0; x<w; x++ ) {
      //int a = clamp( int( *(ad++)*255.0 ), 0, 255 );
      int r = clamp( int( *(rd++)*255.0 ), 0, 255 );
      int g = clamp( int( *(gd++)*255.0 ), 0, 255 );
      int b = clamp( int( *(bd++)*255.0 ), 0, 255 );
      *(c++) = qRgb( r,g,b );
    }
  }

  return true;
}

// Warp domain using given homography
bool coco::gsl_image_warp_to( gsl_image *in, gsl_image *out, const Mat33f &M )
{
  int i=0;
  for ( size_t y=0; y<out->_h; y++ ) {
    for ( size_t x=0; x<out->_w; x++ ) {
      Vec2f p = M * Vec2f( x,y );
      gsl_image_get_color_interpolate( in, p.x, p.y,
				       out->_r->data[i],
				       out->_g->data[i],
				       out->_b->data[i] );
      i++;
    }
  }
  return true;
}

// Save image
bool coco::gsl_image_save( const string& filename, const gsl_image *I, const char *format )
{
  QImage Q( I->_w, I->_h, QIMAGE_FORMAT_ARGB32 );
  if ( !gsl_image_to_qimage( I,Q )) {
    return false;
  }
  return Q.save( filename.c_str(), format );
}


// Save image
bool coco::gsl_image_save_scaled( const string& filename, const gsl_image *I,
				 size_t W, size_t H, const char *format )
{
  QImage Q( I->_w, I->_h, QIMAGE_FORMAT_ARGB32 );
  if ( !gsl_image_to_qimage( I,Q )) {
    return false;
  }
  QImage R = Q.scaled( W,H );
  return R.save( filename.c_str(), format );
}


// Free image
bool coco::gsl_image_free( gsl_image* I )
{
  if ( I==NULL ) {
    return false;
  }
  if ( I->_r != NULL ) gsl_matrix_free( I->_r );
  if ( I->_g != NULL ) gsl_matrix_free( I->_g );
  if ( I->_b != NULL ) gsl_matrix_free( I->_b );
  if ( I->_a != NULL ) gsl_matrix_free( I->_a );
  delete I;
  return true;
}


// Normalize an image
bool coco::gsl_image_normalize( gsl_image* I, bool invert )
{
  // Compute Min/Max
  double *r = I->_r->data;
  double *g = I->_g->data;
  double *b = I->_b->data;
  assert( r != NULL );
  assert( g != NULL );
  assert( b != NULL );

  double m =  FLT_MAX;
  double M = -FLT_MAX;
  size_t N = I->_w * I->_h;
  for ( size_t i=0; i<N; i++ ) {
    m = std::min( m, std::min(*r, std::min(*g,*b)) );
    M = std::max( M, std::max(*r, std::max(*g,*b)) );
    r++; g++; b++;
  }

  // Normalize to range (0.0 - 1.0)
  r = I->_r->data;
  g = I->_g->data;
  b = I->_b->data;
  float D = M-m;
  if ( D==0.0f ) D=1.0f;
  if ( invert ) {
    for ( size_t i=0; i<N; i++ ) {
      *r = (M - *r) / D;
      *g = (M - *g) / D;
      *b = (M - *b) / D;
      r++; g++; b++;
    }
  }
  else {
    for ( size_t i=0; i<N; i++ ) {
      *r = (*r - m) / D;
      *g = (*g - m) / D;
      *b = (*b - m) / D;
      r++; g++; b++;
    }
  }

  return true;
}


// Normalize image range, i.e. transform range to 0-1, clamp everything else
bool coco::gsl_image_normalize_range( gsl_image *I, double m, double M )
{
  // Compute Min/Max
  double *r = I->_r->data;
  double *g = I->_g->data;
  double *b = I->_b->data;
  assert( r != NULL );
  assert( g != NULL );
  assert( b != NULL );
  size_t N = I->_w * I->_h;

  // Normalize to range (0.0 - 1.0)
  float D = M-m;
  if ( D==0.0f ) D=1.0f;
  for ( size_t i=0; i<N; i++ ) {
    *r = clamp( (*r - m) / D, 0.0, 1.0 );
    *g = clamp( (*g - m) / D, 0.0, 1.0 );
    *b = clamp( (*b - m) / D, 0.0, 1.0 );
    r++; g++; b++;
  }

  return true;
}



// Normalize image, treat color channels separately
bool coco::gsl_image_normalize_separate_channels( gsl_image *I )
{
  // Compute Min/Max
  double *r = I->_r->data;
  double *g = I->_g->data;
  double *b = I->_b->data;
  double *a = I->_a->data;
  assert( r != NULL );
  assert( g != NULL );
  assert( b != NULL );
  assert( a != NULL );

  double mr =  coco::FLOAT_MAX;
  double Mr = -coco::FLOAT_MAX;
  double mg =  coco::FLOAT_MAX;
  double Mg = -coco::FLOAT_MAX;
  double mb =  coco::FLOAT_MAX;
  double Mb = -coco::FLOAT_MAX;
  double ma =  coco::FLOAT_MAX;
  double Ma = -coco::FLOAT_MAX;

  size_t N = I->_w * I->_h;
  for ( size_t i=0; i<N; i++ ) {
    mr = std::min( mr, *r );
    Mr = std::max( Mr, *r );
    mg = std::min( mg, *g );
    Mg = std::max( Mg, *g );
    mb = std::min( mb, *b );
    Mb = std::max( Mb, *b );
    ma = std::min( mb, *a );
    Ma = std::max( Mb, *a );
    r++; g++; b++; a++;
  }

  // Normalize to range (0.0 - 1.0)
  r = I->_r->data;
  g = I->_g->data;
  b = I->_b->data;
  a = I->_a->data;
  float Dr = Mr-mr;
  if ( Dr==0.0f ) Dr=1.0f;
  float Dg = Mg-mg;
  if ( Dg==0.0f ) Dg=1.0f;
  float Db = Mb-mb;
  if ( Db==0.0f ) Db=1.0f;
  float Da = Ma-ma;
  if ( Da==0.0f ) Da=1.0f;
  for ( size_t i=0; i<N; i++ ) {
    *r = (*r - mr) / Dr;
    *g = (*g - mg) / Dg;
    *b = (*b - mb) / Db;
    *a = (*a - ma) / Da;
    r++; g++; b++; a++;
  }

  return true;
}


struct image_header {
  size_t _version;
  size_t _w;
  size_t _h;
  bool _alpha;
};

// Save image with full precision
// Use zlib to stream data
bool coco::gsl_image_save_lossless( const string &filename, const gsl_image *I )
{
  gzFile f = gzopen( filename.c_str(), "wb9" );
  if ( f==NULL ) {
    return false;
  }

  // Write header
  image_header hdr;
  hdr._version = 1;
  hdr._w = I->_w;
  hdr._h = I->_h;
  hdr._alpha = (I->_a != NULL) ? true : false;
  gzwrite( f, &hdr, sizeof( image_header ));

  // Write data
  size_t N = I->_w * I->_h * sizeof( double );
  gzwrite( f, I->_r->data, N );
  gzwrite( f, I->_g->data, N );
  gzwrite( f, I->_b->data, N );
  if ( hdr._alpha ) {
    gzwrite( f, I->_a->data, N );
  }
  gzclose( f );
  return true;
}

// Load from lossless save file
// Stream via zlib
gsl_image* coco::gsl_image_load_lossless( const string &filename )
{
  gzFile f = gzopen( filename.c_str(), "rb" );
  if ( f==NULL ) {
    return NULL;
  }

  // Write header
  image_header hdr;
  gzread( f, &hdr, sizeof( image_header ));
  if ( hdr._version != 1 ) {
    assert( false );
    return NULL;
  }
	  
  gsl_image *I = gsl_image_alloc( hdr._w, hdr._h );
  if ( I==NULL ) {
    return I;
  }

  // Write data
  size_t N = I->_w * I->_h * sizeof( double );
  gzread( f, I->_r->data, N );
  gzread( f, I->_g->data, N );
  gzread( f, I->_b->data, N );
  if ( hdr._alpha && I->_a != NULL ) {
    gzread( f, I->_a->data, N );
  }
  gzclose( f );
  return I;
}


static const double TWO_PI = 6.2831853071795864769252866;
 
static double gauss_rand(const double &variance)
{
  static bool hasSpare = false;
  static double rand1, rand2;
  
  if (hasSpare) {
    hasSpare = false;
    return sqrt(variance * rand1) * sin(rand2);
  }
  
  hasSpare = true;
  
  rand1 = rand() / ((double) RAND_MAX);
  if(rand1 < 1e-100) rand1 = 1e-100;
  rand1 = -2 * log(rand1);
  rand2 = (rand() / ((double) RAND_MAX)) * TWO_PI;
  
  return sqrt(variance * rand1) * cos(rand2);
}

void coco::gsl_image_add_gaussian_noise( gsl_image *I, double sigma )
{
  size_t W = I->_w;
  size_t H = I->_h;
  size_t N = W*H;
  double variance = sigma * sigma;
  for ( size_t i=0; i<N; i++ ) {
    I->_r->data[i] = clamp( I->_r->data[i] + gauss_rand( variance ), 0.0, 1.0 );
    I->_g->data[i] = clamp( I->_g->data[i] + gauss_rand( variance ), 0.0, 1.0 );
    I->_b->data[i] = clamp( I->_b->data[i] + gauss_rand( variance ), 0.0, 1.0 );
  }
}


bool coco::gsl_image_gauss_filter( gsl_image *s, gsl_image *d, double sigma )
{
  size_t W = s->_w;
  size_t H = s->_h;
  assert( d->_w == W );
  assert( d->_h == H );
  if ( sigma==0.0 ) {
    gsl_image_copy_to( s,d, 0,0 );
    return true;
  }

  gsl_vector *kd = gsl_kernel_gauss_1xn( 11, sigma );
  vector<gsl_matrix*> CIs = gsl_image_get_channels( s );
  vector<gsl_matrix*> CId = gsl_image_get_channels( d );
  gsl_matrix *tmp = gsl_matrix_alloc( H,W );
  for ( size_t i=0; i<3; i++ ) {
    gsl_matrix *s = CIs[i];
    gsl_matrix *d = CId[i];
    gsl_matrix_convolution_1xn( s,kd, tmp );
    gsl_matrix_convolution_nx1( tmp,kd, d );
  }

  gsl_matrix_free( tmp );
  gsl_vector_free( kd );
  return true;
}



#define ZRAND (double(rand() % 10000) / double(10000.0))
#define ORAND (double(rand() % 10000) / double(10000.0))

void coco::gsl_image_add_salt_and_pepper_noise( gsl_image *I, double sigma )
{
  size_t W = I->_w;
  size_t H = I->_h;
  size_t N = W*H;
  for ( size_t i=0; i<N; i++ ) {
    if ( double(rand() % 10000) / 10000.0 < sigma ) {
      if ( ( rand() % 10000 ) > 5000 ) {
	I->_r->data[i] = ZRAND;
	I->_g->data[i] = ZRAND;
	I->_b->data[i] = ZRAND;
      }
      else {
	I->_r->data[i] = ORAND;
	I->_g->data[i] = ORAND;
	I->_b->data[i] = ORAND;
      }
    }
  }
}


double coco::mse( gsl_matrix* A, gsl_matrix *B )
{
  size_t W = A->size2;
  size_t H = A->size1;
  assert( W==B->size2 );
  assert( H==B->size1 );
  size_t N = W*H;
  if ( N==0 ) {
    return 0.0;
  }

  double mse = 0.0;
  for ( size_t i=0; i<N; i++ ) {
    mse += square( A->data[i] - B->data[i] );
  }
  return mse / double(N);
}

// No-boundary version
static double mse_nb( gsl_matrix* A, gsl_matrix *B )
{
  size_t W = A->size2;
  size_t H = A->size1;
  assert( W==B->size2 );
  assert( H==B->size1 );
  assert( W>4 );
  assert( H>4 );

  double mse = 0.0;
  for ( size_t x=2; x<W-2; x++ ) {
    for ( size_t y=2; y<H-2; y++ ) {
      mse += square( gsl_matrix_get( A, y,x ) - gsl_matrix_get( B, y,x ));
    }
  }
  return mse / double( (W-4) * (H-4) );
}

double coco::psnr( double vmax, double mse )
{
  if ( mse == 0.0 ) {
    return DBL_MAX;
  }
  return 10.0 * log( vmax*vmax / mse ) / log( 10.0 );
}


double coco::gsl_image_psnr( gsl_image *A, gsl_image *B )
{
  double mse_r = mse_nb( A->_r, B->_r );
  double mse_g = mse_nb( A->_g, B->_g );
  double mse_b = mse_nb( A->_b, B->_b );
  double mse_t = (mse_r + mse_g + mse_b) / 3.0;
  return psnr( 1.0, mse_t );
}

// Structural similarity between two images
double coco::gsl_image_ssim( gsl_image *A, gsl_image *B, double dynamic_range )
{
  double ssim = 0.0;
  assert( A->_w == B->_w );
  assert( A->_h == B->_h );
  ssim += gsl_matrix_ssim( A->_r, B->_r, dynamic_range );
  ssim += gsl_matrix_ssim( A->_g, B->_g, dynamic_range );
  ssim += gsl_matrix_ssim( A->_b, B->_b, dynamic_range );
  return ssim / 3.0;
}




void coco::gsl_image_pointwise_norm( gsl_image *I )
{
  size_t index = 0;
  for ( size_t y=0; y<I->_h; y++ ) {
    for ( size_t x=0; x<I->_w; x++ ) {
      double r = I->_r->data[index];
      double g = I->_g->data[index];
      double b = I->_b->data[index];
      double v = sqrt( r*r + g*g + b*b );
      I->_r->data[index] = v;
      I->_g->data[index] = v;
      I->_b->data[index] = v;
      index++;
    }
  }
}


gsl_image *coco::create_test_image_rgb_wheel( size_t W, size_t H, size_t asteps, size_t rsteps )
{
  gsl_image *I = gsl_image_alloc( W,H );
  size_t R = min( W/2, H/2 );
  double rmod = double(R) / double(rsteps);
  double amod = 2.0 * M_PI  / double(asteps);
  double a2mod = M_PI  / double(asteps);
  size_t index = 0;
  for ( size_t y=0; y<H; y++ ) {
    for ( size_t x=0; x<W; x++ ) {
      // Transform to polar coords
      double xx = double(x)-W/2;
      double yy = double(y)-H/2;
      double rr = sqrt( xx*xx + yy*yy );
      double r = (rr - fmod( rr,rmod )) / R;
      double aa = atan2( yy, xx );
      double a = (aa + M_PI - fmod( aa+M_PI, amod )) / (2.0 * M_PI);
      double a2 = fabs(aa) - fmod( fabs(aa),a2mod );
      a2 /= M_PI;

      double cr = a2;
      double cg = r;
      double cb = a;

      I->_r->data[index] = cr;
      I->_g->data[index] = cg;
      I->_b->data[index] = cb;
      index++;
    }
  }

  return I;
}




// Compute normalized cross correlation between two rows of data
double coco::compute_ncc( double *x, double *y, size_t n )
{
  // first compute means
  assert( n>1 );
  double mx = 0.0;
  double my = 0.0;
  for ( size_t i=0; i<n; i++ ) {
    mx += x[i]; my += y[i];
  }
  mx /= double(n); my /= double(n);

  // then compute covariance and standard deviations
  double cov = 0.0;
  double sx = 0.0;
  double sy = 0.0;
  for ( size_t i=0; i<n; i++ ) {
    sx += pow( x[i] - mx, 2.0 );
    sy += pow( y[i] - my, 2.0 );
    cov += ( x[i] - mx ) * ( y[i] - my );
  }
  if ( sx==0.0f || sy==0.0f ) {
    return 0.0f;
  }
  return cov / sqrt( sx * sy );
}


// Compute normalized cross correlation between two image patches
double coco::gsl_image_ncc( gsl_image *I, int xi, int yi, gsl_image *J, int xj, int yj, int d )
{
  size_t nmax = 3*square( double(d*2 + 1) );
  double *vi = new double[ nmax ];
  double *vj = new double[ nmax ];
  double ncc = 0.0;
  if ( d>xi ) return ncc;
  if ( d>yi ) return ncc;
  if ( d>xj ) return ncc;
  if ( d>yj ) return ncc;
  if ( d+xi >= int(I->_w) ) return ncc;
  if ( d+yi >= int(I->_h) ) return ncc;
  if ( d+xj >= int(J->_w) ) return ncc;
  if ( d+yj >= int(J->_h) ) return ncc;
  // Store values in arrays
  size_t n=0;
  for ( int i=-d; i<=d; i++ ) {
    for ( int j=-d; j<=d; j++ ) {
      size_t oi = (xi+i) + (yi+j) * I->_w;
      size_t oj = (xj+i) + (yj+j) * J->_w;
      vi[n] = I->_r->data[ oi ];
      vj[n] = J->_r->data[ oj ];
      n++;
      vi[n] = I->_g->data[ oi ];
      vj[n] = J->_g->data[ oj ];
      n++;
      vi[n] = I->_b->data[ oi ];
      vj[n] = J->_b->data[ oj ];
      n++;
    }
  }
  assert( n==nmax );

  // Compute ncc
  ncc = compute_ncc( vi, vj, nmax );
  if ( ncc < -1.001 || ncc > 1.001 ) {
    TRACE( "ncc error " << ncc << endl );
  }
  //  TRACE( ncc << " " << endl );
  delete[] vi;
  delete[] vj;
  return ncc;
}


// Compute sum of absolute differences between two image patche
double coco::gsl_image_sad( gsl_image *I, int xi, int yi, gsl_image *J, int xj, int yj, int d )
{
  if ( xi<0 || yi<0 || xj<0 || yj<0 ||
				  xi>=(int)I->_w || yi >(int)I->_h || xj >= (int)J->_w || yj >= (int)J->_h ) {
    return 0.5;
  }

  // Lower bounds
  int dxm = -min( d, min( xi,xj ));
  int dym = -min( d, min( yi,yj ));
  // Upper bounds
  int dxM = min( d, min( int(I->_w)-xi-1, int(J->_w)-xj-1 ));
  int dyM = min( d, min( int(I->_h)-yi-1, int(J->_h)-yj-1 ));
  // Sum differences
  double err = 0.0;
  size_t n = 0;
  for ( int dy=dym; dy<= dyM; dy++ ) {
    for ( int dx=dxm; dx<= dxM; dx++ ) {
      size_t oi = (xi+dx) + (yi+dy) * I->_w;
      size_t oj = (xj+dx) + (yj+dy) * J->_w;
      err += max( fabs( I->_r->data[oi] - J->_r->data[oj] ),
		  max( fabs( I->_g->data[oi] - J->_g->data[oj] ),
		       fabs( I->_b->data[oi] - J->_b->data[oj] ) ));
      n++;
    }
  }

  return err / double(n);
}


// Compute sum of squared differences between two image patche
double coco::gsl_image_ssd( gsl_image *I, int xi, int yi, gsl_image *J, int xj, int yj, int d )
{
  if ( xi<0 || yi<0 || xj<0 || yj<0 ||
				  xi>=(int)I->_w || yi >(int)I->_h || xj >= (int)J->_w || yj >= (int)J->_h ) {
    return 0.5;
  }

  // Lower bounds
  int dxm = -min( d, min( xi,xj ));
  int dym = -min( d, min( yi,yj ));
  // Upper bounds
  int dxM = min( d, min( int(I->_w)-xi-1, int(J->_w)-xj-1 ));
  int dyM = min( d, min( int(I->_h)-yi-1, int(J->_h)-yj-1 ));
  // Sum squared differences
  double err = 0.0;
  size_t n = 0;
  for ( int dy=dym; dy<= dyM; dy++ ) {
    for ( int dx=dxm; dx<= dxM; dx++ ) {
      size_t oi = (xi+dx) + (yi+dy) * I->_w;
      size_t oj = (xj+dx) + (yj+dy) * J->_w;
      err += pow( I->_r->data[oi] - J->_r->data[oj], 2.0 );
      err += pow( I->_g->data[oi] - J->_g->data[oj], 2.0 );
      err += pow( I->_b->data[oi] - J->_b->data[oj], 2.0 );
      n++;
    }
  }
  if ( isnan( err ) || n==0 ) {
    TRACE( xi << " " << yi << "  -  " << xj << " " << yj << "   -   " << dxm << " " << dym << "   -   " << dxM << " " << dyM << endl );
  }
  return err / (3.0 * n);
}


// Convert image color space, normalize all channels to range [0,1]
bool coco::gsl_image_color_space_convert( gsl_image *I, const color_space &target_cs )
{
  size_t W = I->_w;
  size_t H = I->_h;
  size_t N = W*H;

  // First transform [0,1] range to color space range
  float css_c0_min = 0.0; float css_c0_max = 1.0;
  float css_c1_min = 0.0; float css_c1_max = 1.0;
  float css_c2_min = 0.0; float css_c2_max = 1.0;
  color_space_range( I->_cs,
		     css_c0_min, css_c0_max,
		     css_c1_min, css_c1_max,
		     css_c2_min, css_c2_max );
  double c0s = css_c0_max - css_c0_min;
  double c1s = css_c1_max - css_c1_min;
  double c2s = css_c2_max - css_c2_min;
  for ( size_t i=0; i<N; i++ ) {
    I->_r->data[i] = I->_r->data[i] * c0s + css_c0_min;
    I->_g->data[i] = I->_g->data[i] * c1s + css_c1_min;
    I->_b->data[i] = I->_b->data[i] * c2s + css_c2_min;
  }

  // Then transform colors
  for ( size_t i=0; i<N; i++ ) {
    color c( I->_cs, I->_r->data[i], I->_g->data[i], I->_b->data[i] );
    color_convert( I->_cs, c, target_cs, c );
    I->_r->data[i] = c._c0;
    I->_g->data[i] = c._c1;
    I->_b->data[i] = c._c2;
  }

  // Finally transform color space range to [0,1]
  float cst_c0_min = 0.0; float cst_c0_max = 1.0;
  float cst_c1_min = 0.0; float cst_c1_max = 1.0;
  float cst_c2_min = 0.0; float cst_c2_max = 1.0;
  color_space_range( target_cs,
		     cst_c0_min, cst_c0_max,
		     cst_c1_min, cst_c1_max,
		     cst_c2_min, cst_c2_max );
  double c0t = cst_c0_max - cst_c0_min;
  double c1t = cst_c1_max - cst_c1_min;
  double c2t = cst_c2_max - cst_c2_min;
  for ( size_t i=0; i<N; i++ ) {
    I->_r->data[i] = ( I->_r->data[i] - cst_c0_min ) / c0t;
    I->_g->data[i] = ( I->_g->data[i] - cst_c1_min ) / c1t;
    I->_b->data[i] = ( I->_b->data[i] - cst_c2_min ) / c2t;
  }
  I->_cs = target_cs;
  return true;
}



// Get color value (Neumann boundary conditions)
bool coco::gsl_image_get_color( const gsl_image *I, int x, int y, double &r, double &g, double &b )
{
  bool ret = true;
  int w = I->_w;
  int h = I->_h;
  if ( w*h==0 ) {
    return false;
  }

  if ( x>=w ) {x=w-1; ret=false;}
  if ( x<0 )  {x=0; ret=false;} 
  if ( y>=w ) {y=h-1; ret=false;}
  if ( y<0 )  {y=0; ret=false;}

  int offset = x + y*I->_w;
  r = I->_r->data[offset];
  g = I->_g->data[offset];
  b = I->_b->data[offset];

  return ret;
}


// Image color in RGB (interpolated)
bool coco::gsl_image_get_color_interpolate( const gsl_image *I, float x, float y, double &r, double &g, double &b )
{
  assert( I != NULL );
  assert( I->_r != NULL );
  assert( I->_g != NULL );
  assert( I->_b != NULL );
  r = gsl_matrix_interpolate_neumann( I->_r, x,y );
  g = gsl_matrix_interpolate_neumann( I->_g, x,y );
  b = gsl_matrix_interpolate_neumann( I->_b, x,y );
  return true;
}

double coco::gsl_image_matching_cost( matching_score m, size_t region,
				     gsl_image *image1, int x1, int y1,
				     gsl_image *image2, int x2, int y2 )
{
  int W = image1->_w;
  int H = image1->_h;
  float v = 0.5;

  if ( m == MS_NCC ) {
    float ncc = 0.0;
    ncc = gsl_image_ncc( image1, x1,y1, image2, x2,y2, region );
    v = min( 1.0f - ncc, 1.0f );
  }
  else if ( m == MS_PAD ) {
    if ( x2>=0 && x2<W && y2>=0 && y2<H ) {
      double rl,gl,bl;
      gsl_image_get_color( image1, x1,y1, rl,gl,bl );
      double rr,gr,br;
      gsl_image_get_color( image2, x2,y2, rr,gr,br );	  
      double diff = max( fabs( rr-rl ), max( fabs( gr-gl ), fabs( br-bl )) );
      v = min( 1.0, diff*4.0 );
    }
  }
  else if ( m == MS_SAD ) {
    v = gsl_image_sad( image1, x1,y1, image2, x2,y2, region );
    v = min( 1.0, v*4.0 );
  }
  else if ( m == MS_SSD ) {
    v = gsl_image_ssd( image1, x1,y1, image2, x2,y2, region );
    v = min( 1.0, v*4.0 );
  }
  else {
    // Not supported
    assert( false );
  }
  return v;
}

// Flip image vertically
bool coco::gsl_image_flip_y( gsl_image *I )
{
  gsl_matrix_flip_y( I->_r );
  gsl_matrix_flip_y( I->_g );
  gsl_matrix_flip_y( I->_b );
  gsl_matrix_flip_y( I->_a );
  return true;
}

void coco::gsl_image_delinearize( gsl_image *I ){
  gsl_matrix_delinearize( I->_r );
  gsl_matrix_delinearize( I->_g );
  gsl_matrix_delinearize( I->_b );
}

