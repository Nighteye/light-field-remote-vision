/** \file gsl_image_statistics.h

    Compute image statistics and segmentation probabilities
    
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

#ifndef __COCO_COMMON_IMAGE_STATISTICS_H
#define __COCO_COMMON_IMAGE_STATISTICS_H

#include <string.h>
#include <vector>

#include "../common/debug.h"
#include "../defs.h"

#include "histogram.h"
#include "gsl_image.h"

#ifdef LIB_ANN
#include <ANN/ANN.h>
#endif


namespace coco {

  // Image statistics
  struct stats {
    histogram _r;
    histogram _b;
    histogram _g;
  };

  // Initialize statistics
  void stats_init( stats &S, size_t bins );
  // Add color to statistics table
  void stats_add( stats &S, double r, double g, double b );
  // Normalize tables
  void stats_normalize( stats &S );
  // Probability that color fits
  double stats_prob( const stats &S, double r, double g, double b );

  // Binary (segmentation) classifier
  struct binary_classifier
  {
    // dimension of classified vectors
    size_t _dim;

    // classifier function, implemented by derived structs
    // returns probability for class 0 and class 1, respectively
    virtual void classify( double *v, double &prob_0, double &prob_1 )=0;

    // cleanup
    virtual ~binary_classifier() {};
  };

  // specialized classifier based on grayscale histograms
  struct binary_classifier_histogram : public binary_classifier
  {
    // Histogram statistics for both regions
    stats _stats_r0;
    stats _stats_r1;

    // classifier function, implemented by derived structs
    // returns probability for class 0 and class 1, respectively
    virtual void classify( double *v, double &prob_0, double &prob_1 );

    // cleanup
    virtual ~binary_classifier_histogram() {};
  };

  // specialized classifier based on color histograms
  struct binary_classifier_histogram_rgb : public binary_classifier
  {
    // Histogram statistics for both regions
    md_histogram _H_r0;
    md_histogram _H_r1;

    // classifier function, implemented by derived structs
    // returns probability for class 0 and class 1, respectively
    virtual void classify( double *v, double &prob_0, double &prob_1 );

    // cleanup
    virtual ~binary_classifier_histogram_rgb() {};
  };

  // Create segmentation classifier based on grayscale histograms
  binary_classifier *gsl_image_classifier_histogram( gsl_image *I, gsl_image *mask, size_t bins=16 );
  // Create segmentation classifier based on RGB-histograms
  binary_classifier *gsl_image_classifier_histogram_rgb( gsl_image *I, gsl_image *mask, size_t bins=16 );


#ifdef LIB_ANN
  // specialized classifier based on k-nearest neighbours and trees
  struct binary_classifier_knn : public binary_classifier
  {
    // classifier function, implemented by derived structs
    // returns probability for class 0 and class 1, respectively
    virtual void classify( double *v, double &prob_0, double &prob_1 );

    // cleanup
    virtual ~binary_classifier_knn();


    // helper structures
    int _k;
    size_t _total_bg;
    size_t _total_fg;
    int *_neighbours;
    double *_dists;
    int _N;
    double **_pts;
    ANNbd_tree *_kdTree;
  };

  // Create segmentation data term based on k-nearest neighbours algorithm
  binary_classifier* gsl_image_classifier_knn( gsl_image *I, gsl_image *mask,
					       size_t k );
#endif

  // Compute segmentation data term using a binary classifier
  bool gsl_image_segmentation_data_term( binary_classifier *C,
					 gsl_image *I, 
					 gsl_image *mask,
					 gsl_matrix *out,
					 double factor=1.0,                            // factor for log-probability scaling
					 double offset=0.0,
					 double clamp_min=-1.0, double clamp_max=1.0,
					 double stddev_scale=0.0 ); // data term range

  // Create segmentation data term (Mumford-Shah based on minmax or given intensity)
  bool gsl_image_segmentation_data_term_ms( gsl_image *I, gsl_matrix *out,
					    bool use_minmax,
					    double mean1, double mean2 );

  // Create inpainting mask
  bool gsl_image_inpainting_mask( gsl_image *I, gsl_matrix *mask );

  // Checks mask if image pixel lies in background
  // Default background color is saturated blue
  bool is_in_bg( gsl_image *I, gsl_image *mask, size_t x, size_t y );
  // Checks mask if image pixel lies in foreground
  // Default foreground color is saturated red
  bool is_in_fg( gsl_image *I, gsl_image *mask, size_t x, size_t y );

}


#endif
