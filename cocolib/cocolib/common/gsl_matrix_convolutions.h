/** \file gsl_matrix_convolutions.h

    File imported from "common" lib, use if this library is not available.
    Uses "gov" namespace (Graphics-optics-vision, MPI).

    Compute convolutions for matrices
    
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

#ifndef __GSL_MATRIX_CONVOLUTIONS
#define __GSL_MATRIX_CONVOLUTIONS

#include "gsl_image.h"

namespace coco {

  // Special convolution functions for certain kernel sizes
  bool gsl_matrix_convolution_1x3( const gsl_matrix *s, const gsl_matrix* kernel, gsl_matrix *d );
  bool gsl_matrix_convolution_3x1( const gsl_matrix *s, const gsl_matrix* kernel, gsl_matrix *d );
  bool gsl_matrix_convolution_3x3( const gsl_matrix *s, const gsl_matrix* kernel, gsl_matrix *d );

  // Convolutions in a single direction, arbitrary odd kernel size
  bool gsl_matrix_convolution_1xn( const gsl_matrix *s, const gsl_vector* kernel, gsl_matrix *d );
  bool gsl_matrix_convolution_nx1( const gsl_matrix *s, const gsl_vector* kernel, gsl_matrix *d );

  // Arbitrary convolution
  bool gsl_matrix_convolution_nxn( const gsl_matrix *s, const gsl_matrix* kernel, gsl_matrix *d );

  // Gauss filter a matrix
  bool gsl_matrix_gauss_filter( gsl_matrix *s, gsl_matrix *d, double sigma, size_t kernel_size=11 );



  /********************************************************
  Create some typical kernels (GSL objects)
  *********************************************************/

  // Gaussian 3x3
  gsl_matrix *gsl_kernel_gauss_3x3( double sigmax, double sigmay );
  // Gaussian nxn
  gsl_matrix *gsl_kernel_gauss_nxn( size_t n, double sigmax, double sigmay, double angle );
  // Gaussian 1xn
  gsl_vector *gsl_kernel_gauss_1xn( size_t n, double sigma );
  // Box kernel wxh
  gsl_matrix *gsl_kernel_box( size_t w, size_t h );
  // Compute product kernel (matrix version)
  gsl_matrix *gsl_kernel_combine( gsl_matrix *a, gsl_matrix *b );
  // Compute product kernel (vector version)
  gsl_vector *gsl_kernel_combine( gsl_vector *a, gsl_vector *b );


}

    
#endif
