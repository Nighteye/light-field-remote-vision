/** \file menger_curvature.h

    Computes Menger curvature radius and local values
    
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

#ifndef __COCO_COMMON_MENGER_CURVATURE_H
#define __COCO_COMMON_MENGER_CURVATURE_H

#include <string.h>
#include <vector>
#include <math.h>
#include <complex>

#include "gsl_matrix_helper.h"
#include "../common/debug.h"


namespace coco {

  // Compute Menger curvature weight
  // Equals 1/radius of circle through (0,0), (x2,y2), (x3,y3)
  double menger_curvature_weight( double x2, double y2, double x3, double y3 );

  // Compute local Menger curvature at location (x,y),
  // for a given window radius
  double local_menger_curvature( gsl_matrix *g, int x, int y, double p, int radius );
  
  // Compute total Menger curvature
  double total_menger_curvature( gsl_matrix *M, double p, int radius, std::vector<double> *values );

  // Compute local Menger curvature relaxation at location (x,y),
  // for a given window radius
  double local_menger_curvature_relaxation( double alpha, gsl_matrix *g, int x, int y, double p, int radius );
  
  // Compute total Menger curvature
  double total_menger_curvature_relaxation ( double alpha, gsl_matrix *M, double p, int radius, std::vector<double> *values );

  // Compute total variation
  double total_variation( gsl_matrix *M, std::vector<double> *values );

  // Compute total mean curvature
  double total_mean_curvature( gsl_matrix *M, double p, std::vector<double> *values );

  // Compute Cauchy curvature weight
  // Equals Cauchy kernel 1/z
  std::complex<double> cauchy_curvature_weight( double z_re, double z_im );

  // Compute local Cauchy curvature at location (x,y),
  // for a given window radius
  double local_cauchy_curvature( gsl_matrix *g, int x, int y, int radius );

  // Compute simplified curvature weight
  double stc_curvature_weight( double y0, double y1 );

  // Compute local simplified total curvature at location (x,y),
  // for a given window radius
  double local_stc_curvature( gsl_matrix *g, int x, int y, int radius );

  // Compute local Menger curvature at location (x,y),
  // for a given window radius
  // This version uses the Frobenius norm
  // Proved to be the same as the other !
  double local_menger_curvature_frob( gsl_matrix *dx, gsl_matrix *dy, int x, int y, int radius );
  
}

    
#endif
