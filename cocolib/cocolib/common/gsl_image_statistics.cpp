/** \file gsl_image_statistics.cpp

    Image statistics and segmentation data terms
    
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

#include <math.h>
#include <map>

#include "gsl_image_statistics.h"
#include "gsl_matrix_helper.h"
#include "../cuda/cuda_interface.h"

#ifdef LIB_ANN
#include <ANN/ANN.h>
#endif

using namespace coco;
using namespace std;


// Checks mask if image pixel lies in background
// Background color is saturated blue
bool coco::is_in_bg( gsl_image *I, gsl_image *mask, size_t x, size_t y )
{
  size_t factorX = mask->_w / I->_w;
  assert( factorX * I->_w == mask->_w );
  size_t factorY = mask->_h / I->_h;
  assert( factorY * I->_h == mask->_h );
  size_t index = x*factorX + y*factorY*mask->_w;
  if ( mask->_r->data[index] == 0.0 && mask->_g->data[index] == 0.0 && mask->_b->data[index] == 1.0 ) {
    return true;
  }
  return false;
}

// Checks mask if image pixel lies in foreground
// Foreground color is saturated red
bool coco::is_in_fg( gsl_image *I, gsl_image *mask, size_t x, size_t y )
{
  size_t factorX = mask->_w / I->_w;
  assert( factorX * I->_w == mask->_w );
  size_t factorY = mask->_h / I->_h;
  assert( factorY * I->_h == mask->_h );
  size_t index = x*factorX + y*factorY*mask->_w;
  if ( mask->_r->data[index] == 1.0 && mask->_g->data[index] == 0.0 && mask->_b->data[index] == 0.0 ) {
    return true;
  }
  return false;
}

void coco::stats_init( stats &S, size_t bins )
{
  coco::histogram_init( S._r, 0.0, 1.0, bins );
  coco::histogram_init( S._g, 0.0, 1.0, bins );
  coco::histogram_init( S._b, 0.0, 1.0, bins );
}
void coco::stats_add( stats &S, double r, double g, double b )
{
  coco::histogram_add( S._r, r );
  coco::histogram_add( S._g, g );
  coco::histogram_add( S._b, b );
}
void coco::stats_normalize( stats &S )
{
  coco::histogram_normalize( S._r );
  coco::histogram_normalize( S._g );
  coco::histogram_normalize( S._b );
}
double coco::stats_prob( const stats &S, double r, double g, double b )
{
  return coco::histogram_bin_value( S._r, r )
    * coco::histogram_bin_value( S._g, g )
    * coco::histogram_bin_value( S._b, b );
}



// Create segmentation data term
coco::binary_classifier *coco::gsl_image_classifier_histogram( gsl_image *I, gsl_image *mask, size_t bins )
{
  assert( mask->_h == I->_h );
  assert( mask->_w == I->_w );
  size_t H = I->_h;
  size_t W = I->_w;

  // Build up statistics of image (RGB, grayscale histogram)
  binary_classifier_histogram *C = new binary_classifier_histogram;
  C->_dim = 3;

  // Create histograms
  stats_init( C->_stats_r1, bins );
  stats_init( C->_stats_r0, bins );

  // Accumulate statistics
  size_t n = 0;
  for ( size_t y=0; y<H; y++ ) {
    for ( size_t x=0; x<W; x++ ) {
      if ( is_in_bg( I, mask, x, y )) {
	stats_add( C->_stats_r0, I->_r->data[n], I->_g->data[n], I->_b->data[n] );
      }
      if ( is_in_fg( I, mask, x, y )) {
	stats_add( C->_stats_r1, I->_r->data[n], I->_g->data[n], I->_b->data[n] );
      }
      n++;
    }
  }
  stats_normalize( C->_stats_r0 );
  stats_normalize( C->_stats_r1 );
  return C;
}


// classifier function, implemented by derived structs
// returns probability for class 0 and class 1, respectively
void binary_classifier_histogram::classify( double *v, double &prob_0, double &prob_1 )
{
  prob_0 = stats_prob( _stats_r0, v[0], v[1], v[2] );
  prob_1 = stats_prob( _stats_r1, v[0], v[1], v[2] );
}





// Create segmentation classifier based on RGB-histograms
coco::binary_classifier *coco::gsl_image_classifier_histogram_rgb( gsl_image *I, gsl_image *mask, size_t bins )
{
  assert( mask->_h == I->_h );
  assert( mask->_w == I->_w );
  size_t H = I->_h;
  size_t W = I->_w;

  // Build up statistics of image
  double sigma = 0.0;

  // Build up statistics of image (RGB, grayscale histogram)
  binary_classifier_histogram_rgb *C = new binary_classifier_histogram_rgb;
  C->_dim = 3;
  md_histogram_init( C->_H_r0, 3 );
  md_histogram_set_range( C->_H_r0, 0, bins, 0.0, 1.0 );
  md_histogram_set_range( C->_H_r0, 1, bins, 0.0, 1.0 );
  md_histogram_set_range( C->_H_r0, 2, bins, 0.0, 1.0 );
  C->_H_r0._sigma[0] = sigma;
  C->_H_r0._sigma[1] = sigma;
  C->_H_r0._sigma[2] = sigma;

  md_histogram_init( C->_H_r1, 3 );
  md_histogram_set_range( C->_H_r1, 0, bins, 0.0, 1.0 );
  md_histogram_set_range( C->_H_r1, 1, bins, 0.0, 1.0 );
  md_histogram_set_range( C->_H_r1, 2, bins, 0.0, 1.0 );
  C->_H_r1._sigma[0] = sigma;
  C->_H_r1._sigma[1] = sigma;
  C->_H_r1._sigma[2] = sigma;

  // Accumulate statistics
  TRACE( "Building stats [" );
  size_t n = 0;
  for ( size_t y=0; y<H; y++ ) {
    if ( (y%(H/20)) == 0 ) {
      TRACE( "." );
    }
    for ( size_t x=0; x<W; x++ ) {
      vector<double> c;
      c.push_back( I->_r->data[n] );
      c.push_back( I->_g->data[n] );
      c.push_back( I->_b->data[n] );
      if ( is_in_bg( I, mask, x, y )) {
	md_histogram_insert( C->_H_r0, c );
      }
      if ( is_in_fg( I, mask, x, y )) {
	md_histogram_insert( C->_H_r1, c );
      }
      n++;
    }
  }
  TRACE( "] done." << endl );

  md_histogram_normalize( C->_H_r0 );
  md_histogram_normalize( C->_H_r1 );
  return C;
}


// classifier function, implemented by derived structs
// returns probability for class 0 and class 1, respectively
void binary_classifier_histogram_rgb::classify( double *v, double &prob_0, double &prob_1 )
{
  vector<double> vs;
  vs.push_back( v[0] );
  vs.push_back( v[1] );
  vs.push_back( v[2] );
  prob_0 = md_histogram_bin_value( _H_r0, vs );
  prob_1 = md_histogram_bin_value( _H_r1, vs );
}



#ifdef LIB_ANN

struct rgb {
  double _r;
  double _g;
  double _b;
};
bool operator< ( const rgb &r1, const rgb &r2 )
{
  if (r1._r < r2._r) {
    return true;
  }
  else if ( r1._r > r2._r ) {
    return false;
  }

  if (r1._g < r2._g) {
    return true;
  }
  else if ( r1._g > r2._g ) {
    return false;
  }

  if (r1._b < r2._b) {
    return true;
  }

  return false;
}

struct coco_count {
  size_t _fg;
  size_t _bg;
};

struct color_point {
  double _r;
  double _g;
  double _b;
  double _fg;
  double _bg;
};




// Create segmentation data term based on k-nearest neighbours algorithm
  // Create segmentation data term based on k-nearest neighbours algorithm
coco::binary_classifier* coco::gsl_image_classifier_knn( gsl_image *I, gsl_image *mask,
							 size_t k )
{
  assert( mask->_h == I->_h );
  assert( mask->_w == I->_w );
  size_t H = I->_h;
  size_t W = I->_w;

  // build statistics
  binary_classifier_knn *C = new binary_classifier_knn;
  C->_dim = 3;
  C->_k = k;
  C->_neighbours = new int[ k ];
  C->_dists = new double[ k ];

  // Create a map which stores all the pixel color values which occur in I, together
  // with a count how often they appear in FG or BG, respectively.
  cout << "  creating image color table ..." << endl;
  map<rgb,coco_count> cm;
  C->_total_bg = 0;
  C->_total_fg = 0;
  for ( size_t y=0; y<H; y++ ) {
    for ( size_t x=0; x<W; x++ ) {
      int i = x + y*W;
      rgb c;
      c._r = I->_r->data[i];
      c._g = I->_g->data[i];
      c._b = I->_b->data[i];

      // Check for fg/bg, ignore unmarked pixels
      bool fg = is_in_fg( I, mask, x,y );
      bool bg = is_in_bg( I, mask, x,y );
      if ( fg ) {
	C->_total_fg++;
      }
      else if ( bg ) {
	C->_total_bg++;
      }
      else {
	continue;
      }

      // Search RGB color in map
      map<rgb,coco_count>::iterator it = cm.find( c );
      if ( it != cm.end() ) {
	if ( fg ) {
	  (*it).second._fg ++;
	}
	else {
	  (*it).second._bg ++;
	}
      }
      else {
	if ( fg ) {
	  coco_count ct;
	  ct._fg = 1;
	  ct._bg = 0;
	  cm[ c ] = ct;
	}
	else {
	  coco_count ct;
	  ct._fg = 0;
	  ct._bg = 1;
	  cm[ c ] = ct;
	}
      }

    }
  }

    
  // Push the map into an array.
  TRACE( "  creating search tree ..." );
  C->_N = cm.size();
  C->_pts = new double*[ C->_N ];
  size_t i=0;
  for ( map<rgb,coco_count>::const_iterator it = cm.begin();
	it != cm.end(); it++ ) {

    color_point* pt = new color_point;
    pt->_r = (*it).first._r;
    pt->_g = (*it).first._g;
    pt->_b = (*it).first._b;
    pt->_fg = (*it).second._fg;
    pt->_bg = (*it).second._bg;

    C->_pts[i++] = (double*)pt;

  }
  assert( (int)i == C->_N );

  // Initialize k-nearest-neighbours search structure
  C->_kdTree = new ANNbd_tree( C->_pts, C->_N, 3 );
  TRACE( " done." << endl );

  return C;
}


// cleanup
binary_classifier_knn::~binary_classifier_knn()
{
  delete _kdTree;
  for ( int i=0; i<_N; i++ ) {
    delete (color_point*)_pts[i];
  }
  delete[] _pts;
  delete[] _neighbours;
  delete[] _dists;
}




// classifier function, implemented by derived structs
// returns probability for class 0 and class 1, respectively
void binary_classifier_knn::classify( double *v, double &prob_0, double &prob_1 )
{
  // find nearest neighbours
  int kk = _k;
  _kdTree->annkSearch( v, kk, _neighbours, _dists, 0 );
  
  // compute probabilities for fg/bg from distribution
  prob_1 = 0.0;
  prob_0 = 0.0;
  double w_total = 0.0;
  for ( int j=0; j<kk; j++ ) {
    color_point *p = (color_point*)_pts[ _neighbours[j] ];
    double n = p->_fg + p->_bg;
    double w = n / (1.0 + sqrt( _dists[j] ));
    double fg = p->_fg / n;
    double bg = p->_bg / n;
    
    prob_1 += fg * w;
    prob_0 += bg * w;
    w_total += w;
    //TRACE( "NN " << j << "  fg: " << fg << "  bg: " << bg << endl );
  }
  prob_1 /= w_total * double(_total_fg);
  prob_0 /= w_total * double(_total_bg);
  //TRACE( "p0: " << prob_0 << " p1: " << prob_1 << "  t " << _total_fg << ":" << _total_bg << endl );
}

#endif




// Compute segmentation data term using a binary classifier
bool coco::gsl_image_segmentation_data_term( binary_classifier *C,
					     gsl_image *I, 
					     gsl_image *mask,
					     gsl_matrix *out,
					     double factor,                      // factor for log-probability scaling
					     double offset,
					     double clamp_min, double clamp_max,
					     double stddev_scale ) // data term range
{
  if ( mask != NULL ) {
    assert( mask->_h == I->_h );
    assert( mask->_w == I->_w );
  }

  size_t H = I->_h;
  size_t W = I->_w;
  gsl_matrix *a = gsl_matrix_alloc( H,W );
  gsl_matrix *fg = gsl_matrix_alloc( H,W );
  gsl_matrix *bg = gsl_matrix_alloc( H,W );

  // Create fg/bg log-probability table
  size_t count = 0;
  size_t n = 0;
  double stddev = 0.0;
  double v[3];
  for ( size_t y=0; y<H; y++ ) {
    for ( size_t x=0; x<W; x++ ) {
      v[0] = I->_r->data[n];
      v[1] = I->_g->data[n];
      v[2] = I->_b->data[n];
      double prob_fg, prob_bg;
      C->classify( v, prob_bg, prob_fg );

      double qlog = 0.0;
      if ( prob_bg != 0.0 && prob_fg != 0.0 ) {
	qlog = log( prob_bg / prob_fg );
	//TRACE( "pfg " << prob_fg << " pbg " << prob_bg << "  qlog " << qlog << " " << endl );
	count ++;
      }

      a->data[n] = qlog;
      fg->data[n] = prob_fg;
      bg->data[n] = prob_bg;
      stddev += pow( qlog, 2.0 );
      //TRACE( "(" << x << "," << y << ") " << prob_fg << " " << prob_bg << " " << qlog << endl );
      n++;
    }
  }

  TRACE6( "  sum " << stddev << "  count " << count << endl );
  TRACE6( "  avg " << stddev / double(count) << endl );
  stddev = sqrt( stddev / double(count) );

  // Scale 2 standard deviations to [-1,1]
  TRACE6( "stddev " << stddev << endl );
  if ( stddev != 0.0 && stddev_scale != 0.0 ) {
    gsl_matrix_scale( a, stddev_scale / stddev );
  }
  //TRACE6( "  range " << gsl_matrix_min( a ) << "  :  " << gsl_matrix_max( a ) << endl );

  // Clamp
  n = 0;
  for ( size_t y=0; y<H; y++ ) {
    for ( size_t x=0; x<W; x++ ) {
      double av = a->data[n];
      double prob_fg = fg->data[n];
      double prob_bg = bg->data[n];
      
      bool reset = false;
      if ( prob_bg == 0.0 && prob_fg != 0.0 ) {
	reset = true;
	av = -1.0;
      }
      if ( prob_fg == 0.0 && prob_bg != 0.0 ) {
	reset = true;
	av = 1.0;
      }
      if ( mask != NULL ) {
	if ( is_in_bg( I, mask, x,y )) {
	  reset = true;
	  av = 1.0;
	}
	if ( is_in_fg( I, mask, x,y )) {
	  reset = true;
	  av = -1.0;
	}
      }

      if ( !reset ) {
	//TRACE( "qlog " << av << " " );
      }

      a->data[n++] = max( clamp_min, min( clamp_max, factor * av - offset ));
    }
  }

  // Create result
  gsl_matrix_downsample( a,out );
  gsl_matrix_free( a );
  gsl_matrix_free( fg );
  gsl_matrix_free( bg );
  return true;
}



// Create segmentation data term
bool coco::gsl_image_segmentation_data_term_ms( gsl_image *I, gsl_matrix *out,
						bool use_minmax,
						double mean1, double mean2 )
{
  assert( I->_w % out->size2 == 0 );
  assert( I->_h % out->size1 == 0 );
  size_t H = I->_h;
  size_t W = I->_w;
  gsl_matrix *a = gsl_matrix_alloc( I->_h, I->_w );
  gsl_matrix *m = coco::gsl_image_to_greyscale( I );

  // Find min-max
  if ( use_minmax ) {
    gsl_matrix_stats stats = gsl_matrix_get_stats( m );
    mean2 = stats._min + (stats._max - stats._min) * 0.25;
    mean1 = stats._max - (stats._max - stats._min) * 0.25;
  }

  for ( size_t i=0; i<W*H; i++ ) {
    double v = m->data[i];
    double av = pow( v-mean1, 2.0 ) - pow( v-mean2, 2.0 );
    //  TRACE( v << " " << vmax << " " << vmin << " " << av << " " << endl );
    a->data[i] = max( -1.0, min( 1.0, av ));
  }

  // Create result
  gsl_matrix_downsample( a,out );
  gsl_matrix_free( a );
  gsl_matrix_free( m );
  return true;
}



bool coco::gsl_image_inpainting_mask( gsl_image *I, gsl_matrix *mask )
{
  size_t fw = I->_w / mask->size2;
  size_t fh = I->_h / mask->size1;
  if ( fw * mask->size2 != I->_w || fh * mask->size1 != I->_h ) {
    ERROR( "Downsampling ratio not integer." << endl );
    assert( false );
    return false;
  }

  double *d = mask->data;
  for ( size_t h=0; h<mask->size1; h++ ) {
    for ( size_t w=0; w<mask->size2; w++ ) {
      // Sample rectangle in input matrix
      double v = 1.0;
      for ( size_t hh=0; hh<fh; hh++ ) {
	size_t ob = (h*fh+hh)*I->_w + w*fw;
	for ( size_t ww=0; ww<fw; ww++ ) {
	  size_t o = ob + ww;
	  if ( I->_r->data[o] == 0.0 &&
	       I->_g->data[o] == 1.0 &&
	       I->_b->data[o] == 0.0 ) {
	    // Damaged region.
	    v = 0.0;
	  }
	}
      }

      // Normalize and write out
      *(d++) = v;
    }
  }

  return false;
}




