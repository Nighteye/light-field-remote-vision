/** \file gsl_image.h

    File imported from "common" lib, use if this library is not available.
    Uses "gov" namespace (Graphics-optics-vision, MPI).

    Image structure and helper functions, built on gsl matrix objects.
    
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

#ifndef __GSL_IMAGE_H
#define __GSL_IMAGE_H

#include <vector>
#include <string>

#include "linalg3d.h"
#include "gsl_matrix_helper.h"
#include "color_spaces.h"


class QImage;


/********************************************************************
  IMPORTANT NOTE: Some functions need reworking such that they can
  cope with an undefined (NULL) alpha channel.

  However, for time reasons, this will be done for each function
  only when needed.
*********************************************************************/

namespace coco {

  struct gsl_image
  {
    size_t _w;
    size_t _h;
    color_space _cs;
    gsl_matrix *_r;
    gsl_matrix *_g;
    gsl_matrix *_b;
    gsl_matrix *_a;
  };

  enum gsl_image_channel
  {
    GSL_IMAGE_RED,
    GSL_IMAGE_GREEN,
    GSL_IMAGE_BLUE,
    GSL_IMAGE_ALPHA,
  };

  enum matching_score {
    MS_NCC,   // Normalized cross-correlation
    MS_PAD,   // Pixel-wise absolute difference
    MS_SAD,   // Sum of absolute difference (in window)
    MS_SSD,   // Sum of squared difference (in window)
    MS_DAISY, // Daisy features
  };


  // Allocate image
  gsl_image *gsl_image_alloc( int w, int h );
  // Initialize image from QImage
  gsl_image *gsl_image_from_qimage( const QImage &I );
  // Initialize QImage from image
  bool gsl_image_to_qimage( const gsl_image *G, QImage &I );
  // Initialize image structure from channels
  // Can't (and does not need to) be deallocated 
  gsl_image gsl_image_from_channels( const std::vector<gsl_matrix*> &u );
  // Create a test image
  gsl_image *create_test_image_rgb_wheel( size_t W, size_t H, size_t asteps, size_t rsteps );

  // Get image channels
  std::vector<gsl_matrix*> gsl_image_get_channels( gsl_image *I );

  // Create a copy of an image
  gsl_image *gsl_image_copy( const gsl_image *I );

  // Copy an image at a specific location into another one
  bool gsl_image_copy_to( const gsl_image *src, gsl_image *dst, size_t X, size_t Y );
  // Copy an image at a specific location into another one, using alpha stencil
  bool gsl_image_copy_to( const gsl_image *src, gsl_image *dst, gsl_matrix *stencil, size_t X, size_t Y );

  // Create single greyscale value matrix from image
  gsl_matrix *gsl_image_to_greyscale( const gsl_image *I );
  // Inplace grayscale conversion (replace each channel with norm)
  void gsl_image_pointwise_norm( gsl_image *I );

  // Transform image RGB values by the given matrix
  //bool gsl_image_transform( gsl_image *I, const gov::Mat44d &M );

  // Warp domain using given homography
  bool gsl_image_warp_to( gsl_image *in, gsl_image *out, const Mat33f &M );

  // Copy matrix into image, weighted color channels
  bool gsl_image_from_matrix( gsl_image *I,
			      const gsl_matrix *M,
			      double rs=1.0, double gs=1.0, double bs=1.0 );
  // Copy matrix into image, weighted and signed color channels
  bool gsl_image_from_signed_matrix( gsl_image *I,
				     const gsl_matrix *M,
				     double rs_neg=0, double gs_neg=0, double bs_neg=1,
				     double rs_pos=1, double gs_pos=0, double bs_pos=0 );
  // Copy image into another where matrix has positive entries
  bool gsl_image_copy_with_stencil( const gsl_image *in, gsl_matrix *stencil, gsl_image *out );
  // Copy image into another where matrix has positive entries
  bool gsl_image_copy_matrix_to_channel( const gsl_matrix *in, gsl_image *out, gsl_image_channel channel );

  // Copy buffer into each image channel
  bool gsl_image_from_buffer( gsl_image *I, float *buffer );

  // Get image channel
  gsl_matrix *gsl_image_get_channel( gsl_image *I, gsl_image_channel n );


  // Load image
  gsl_image *gsl_image_load( const std::string &filename );
  // Load image with specific resolution
  gsl_image *gsl_image_load_scaled( const std::string &filename, size_t w, size_t h );
  // Load image in PFM format (not provided by Qt)
  gsl_image *gsl_image_load_pfm( const std::string &filename );
  // Save image
  //bool gsl_image_save( const std::string &filename, const gsl_image *I, const std::string &format="PNG" );
  // Wrapper for NVCC
  bool gsl_image_save( const std::string &filename, const gsl_image *I, const char *format="PNG" );
  // Wrapper for NVCC
  bool gsl_image_save_scaled( const std::string &filename, const gsl_image *I,
			      size_t W, size_t H, const char *format="PNG" );
  // Free image
  bool gsl_image_free( gsl_image* I );

  // Save image with full precision
  bool gsl_image_save_lossless( const std::string &filename, const gsl_image *I );
  // Load from lossless save file
  gsl_image* gsl_image_load_lossless( const std::string &filename );

  // Normalize image, i.e. make largest color channel value to 1.0
  bool gsl_image_normalize( gsl_image *I, bool invert=false );
  // Normalize image range, i.e. transform range to 0-1, clamp everything else
  bool gsl_image_normalize_range( gsl_image *I, double rmin, double rmax );
  // Normalize image, treat color channels separately
  bool gsl_image_normalize_separate_channels( gsl_image *I );
  
  // Get pixel color
  bool gsl_image_get_color( const gsl_image *I, int x, int y, double &r, double &g, double &b );
  // Image color in RGB (interpolated)
  bool gsl_image_get_color_interpolate( const gsl_image *I, float x, float y, double &r, double &g, double &b ); 

  // Convert image color space, normalize all channels to range [0,1]
  bool gsl_image_color_space_convert( gsl_image *I, const color_space &target_cs );

  // Error between image colors at certain offset
  //  double gsl_color_error( const gsl_image *I, const gsl_image *J, size_t x, size_t y );

  // Add noise to an image
  void gsl_image_add_gaussian_noise( gsl_image *I, double sigma );
  void gsl_image_add_salt_and_pepper_noise( gsl_image *I, double sigma );

  // Gauss filter image
  bool gsl_image_gauss_filter( gsl_image *s, gsl_image *d, double sigma );

  // Mean squared error between two matrices
  double mse( gsl_matrix* A, gsl_matrix *B );
  // PSNR from max value/mse
  double psnr( double vmax, double mse );
  // PSNR between two images
  double gsl_image_psnr( gsl_image *A, gsl_image *B );

  // Structural similarity between two images
  double gsl_image_ssim( gsl_image *A, gsl_image *B, double dynamic_range=1.0 );

  // Compute normalized cross correlation between two rows of data
  double compute_ncc( double *x, double *y, size_t n );

  // Compute normalized cross correlation between two image patches
  double gsl_image_ncc( gsl_image *I, int xi, int yi, gsl_image *J, int xj, int yj, int d );
  // Compute sum of absolute differences between two image patche
  double gsl_image_sad( gsl_image *I, int xi, int yi, gsl_image *J, int xj, int yj, int d );
  // Compute sum of squared differences between two image patche
  double gsl_image_ssd( gsl_image *I, int xi, int yi, gsl_image *J, int xj, int yj, int d );

  // Compute matching cost using a matching score
  double gsl_image_matching_cost( matching_score m, size_t region,
				  gsl_image *image1, int x1, int y1,
				  gsl_image *image2, int x2, int y2 );

  // Flip image vertically
  bool gsl_image_flip_y( gsl_image *I );

  // convert from linear to sRGB
  void gsl_image_delinearize( gsl_image *I );
}



#endif

