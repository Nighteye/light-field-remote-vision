/** \file gsl_matrix_helper.h

    File imported from "common" lib, use if this library is not available.

    Additional code to help handling gsl matrices.
    
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

#ifndef __GSL_MATRIX_HELPER_H
#define __GSL_MATRIX_HELPER_H

//#include <gsl/gsl_matrix.h>
#include <iostream>
#include <vector>

//#include "linalg3d.h"


namespace coco {

  // Replacements for gsl matrix and gsl vector, to avoid linking against GSL
  // Binary compatible, so can be cast to a real gsl_matrix* directly.

  // gsl_matrix
  typedef struct
  {
    size_t size1;
    size_t size2;
    size_t tda;
    double * data;
    void * block;
    int owner;
  } gsl_matrix;
  // Replacements for GSL functions
  gsl_matrix *gsl_matrix_alloc( size_t size1, size_t size2 );
  void gsl_matrix_free( gsl_matrix * );
  double gsl_matrix_get( const gsl_matrix *, size_t, size_t );
  void gsl_matrix_set( gsl_matrix *, size_t, size_t, double );
  void gsl_matrix_scale( gsl_matrix *, double );
  void gsl_matrix_transpose_memcpy( gsl_matrix *dest, const gsl_matrix *src );
  void gsl_matrix_add( gsl_matrix *dest, const gsl_matrix *src );
  void gsl_matrix_set_all( gsl_matrix *M, double v );

  // gsl_vector
  typedef struct
  {
    size_t size;
    size_t stride;
    double * data;
    void * block;
    int owner;
  } gsl_vector;
  // Alloc/free replacements for GSL functions
  gsl_vector *gsl_vector_alloc( size_t size1 );
  void gsl_vector_free( gsl_vector * );


  /// Create from float (image) buffer
  gsl_matrix* gsl_matrix_from_buffer( size_t W, size_t H, float *buffer );

  /// Normalize a matrix to a range
  bool gsl_matrix_normalize( gsl_matrix *m, double vmin=0.0, double vmax=1.0 );
  
  /// Initialize matrix with random values
  bool gsl_matrix_set_random_values( gsl_matrix *out, double min=0.0, double max=1.0 );

  /// Threshold a matrix to 0-1
  bool gsl_matrix_threshold( gsl_matrix *m, double t=0.5 );
  
  /// Initialize matrix from a vector
  gsl_matrix* gsl_matrix_from_vector( size_t rows, size_t cols, const gsl_vector *v );
  
  /// Initialize matrix from a float array (same size)
  void gsl_matrix_copy_from_float( gsl_matrix *m,  float *array );
  /// Initialize float array from matrix (must be same size)
  bool gsl_matrix_copy_to_buffer( gsl_matrix *m,  float *array );
  
  /// Copy in to out. If out is larger than in, replicate values at the boundary.
  bool gsl_matrix_pad_replicate( const gsl_matrix *in, gsl_matrix *out );
  
  /// Copy in to out. If out is larger than in or in larger than out, ignore values outside.
  bool gsl_matrix_copy_to( const gsl_matrix *in, gsl_matrix *out );
  

  /// Downsampling of in into the smaller matrix out.
  /** Currently, only integers are supported for the size factor.
   */
  bool gsl_matrix_downsample( const gsl_matrix *in, gsl_matrix *out );
  

  /// Interpolation using Dirichlet boundary conditions, i.e. outside region is zero
  double gsl_matrix_interpolate_dirichlet( const gsl_matrix *m, double x, double y );
  
  /// Interpolation using Neumann boundary conditions, i.e. outside region is zero
  /** Essentially the same as gsl_matrix_interpolate
   */
  double gsl_matrix_interpolate_neumann( const gsl_matrix *m, double x, double y );
  
  /// Upsampling of in into the larger matrix out.
  /** Interpolation uses Neumann boundary conditions suitable for scalar upsampling.
   */
  bool gsl_matrix_upsample_neumann( const gsl_matrix *in, gsl_matrix *out );
  
  /// Upsampling of in into the larger matrix out.
  /** Interpolation uses Dirichlet boundary conditions suitable for vector upsampling.
   */
  bool gsl_matrix_upsample_dirichlet( const gsl_matrix *in, gsl_matrix *out );
 
  
  /// Add two matrices and store result in third
  bool gsl_matrix_add( const gsl_matrix *in0, const gsl_matrix *in1, gsl_matrix *out );

  /// Pointwise multiply first matrix with second
  bool gsl_matrix_mul_with( gsl_matrix *out, const gsl_matrix *in );
  /// Pointwise division first matrix with second. Ignores zero fields.
  bool gsl_matrix_div_by( gsl_matrix *out, const gsl_matrix *in );

  /// Pointwise addition of scalar
  bool gsl_matrix_add_scalar( gsl_matrix *out, double s );

  /// Add two matrices, multiply result by scalar, and store in third
  bool gsl_matrix_add_and_scale( const gsl_matrix *in0, const gsl_matrix *in1, const double s,
				 gsl_matrix *out );

  /// Compute C = sA + tB
  bool gsl_matrix_add_scaled( const double s, const gsl_matrix *A, const double t, const gsl_matrix *B,
			      gsl_matrix *C );


  // removed, require gsl
  /*
  /// Multiply two matrices (matrix product)
  bool gsl_matrix_product( gsl_matrix *out, gsl_matrix *A, gsl_matrix *B );

  /// Compute A^T A
  bool gsl_matrix_AtA( gsl_matrix *out, gsl_matrix *A );

  /// Compute A A^T
  bool gsl_matrix_AAt( gsl_matrix *out, gsl_matrix *A );
  */


  /// Output matrix is constructed by adding the given offsets to the pixel positions,
  /// and interpolating input matrix at this point.
  bool gsl_matrix_warp( const gsl_matrix *in,
			const gsl_matrix *dx, const gsl_matrix *dy,
			gsl_matrix *out );
  

  // Median filter
  bool gsl_matrix_median( gsl_matrix *in, gsl_matrix *out, int radius );

  // Flip along X-axis (in y direction)
  bool gsl_matrix_flip_y( gsl_matrix *M );

  void gsl_matrix_delinearize( gsl_matrix *M);

  // Compute some statistics for a matrix
  struct gsl_matrix_stats {
    double _sum;
    double _average;
    double _min;
    double _max;
    double _absmin;
    double _absmax;
  };

  gsl_matrix_stats gsl_matrix_get_stats( gsl_matrix *M );
  std::ostream &operator<< ( std::ostream &o, gsl_matrix_stats &stats );


  // Save image with full precision
  bool gsl_matrix_save( const char *filename, const gsl_matrix *M );
  // Load from lossless save file
  gsl_matrix* gsl_matrix_load( const char *filename );


  // Vector norm
  double gsl_vector_norm( gsl_vector *v );
  // Vector reprojection to unit ball
  void gsl_vector_reproject( gsl_vector *v );

  // Vector output to stdio
  void gsl_vector_out( gsl_vector *v );
  // Matrix output to stdio
  void gsl_matrix_out( gsl_matrix *A );

  // Distance between matrices
  double l2_distance( gsl_matrix *a, gsl_matrix *b );
  // Distance between matrix arrays
  double l2_distance( std::vector<gsl_matrix*> A, std::vector<gsl_matrix*> B );

  // Structural similarity measure
  double gsl_matrix_ssim( gsl_matrix *A, gsl_matrix *B, double dynamic_range=1.0 );

  // Flip array along x-axis
  template<class T> bool flip_array_Y( size_t W, size_t H, T* v ) {
    T* tmp = new T[W];
    for ( size_t y=0; y<H/2; y++ ) {
      memcpy( tmp, v + y*W, sizeof(T) * W );
      size_t ys = H-1-y;
      memcpy( v+y*W, v+ys*W, sizeof(T) * W );
      memcpy( v+ys*W, tmp, sizeof(T)*W );
    }
    delete[] tmp;
    return true;
  }

  /// Some vector helper functions
  bool normalize_vector( double *v, size_t n, double vmin=0.0, double vmax=1.0 );
  bool normalize_vector( float *v, size_t n, float vmin=0.0f, float vmax=1.0f );

}





#endif
