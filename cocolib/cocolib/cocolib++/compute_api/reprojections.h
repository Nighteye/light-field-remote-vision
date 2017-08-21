/** \file reprojections.h

    Reprojection functions for compute arrays.

    Copyright (C) 2014 Bastian Goldluecke.

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

#ifndef __COCO_REPROJECTIONS_H
#define __COCO_REPROJECTIONS_H

#include "compute_array.h"


namespace coco {

  // Array arithmetics

  // L1-Norm
  float l1_norm( vector_valued_function_2D &V );
  // L2-Norm
  float l2_norm( vector_valued_function_2D &V );


  // Array reprojections
  // The reprojection functions project a consecutive subset of
  // 2D vector fields within a vector-valued function
  // to a circle with given radius according to various norms.
  // For nfields==1, all projections are the same

  // Reprojection to maximum norm unit ball
  // (in effect, each vector field projected separately)
  bool reprojection_max_norm( vector_valued_function_2D *U, float radius, int start_index, int nfields );
  bool reprojection_weighted_max_norm( vector_valued_function_2D *U, const compute_buffer &weight, int start_index, int nfields );


  // Reprojection according to Frobenius norm
  // (in effect, treated as one long vector with length 2*nfields)
  bool reprojection_frobenius_norm( vector_valued_function_2D *U, float radius, int start_index, int nfields );
  bool reprojection_weighted_frobenius_norm( vector_valued_function_2D *U, const compute_buffer &weight, int start_index, int nfields );


  // Reprojection according to Nuclear norm
  // Only supports nfields==1 or nfields==3
  bool reprojection_nuclear_norm( vector_valued_function_2D *U, float radius, int start_index, int nfields );
  bool reprojection_weighted_nuclear_norm( vector_valued_function_2D *U, const compute_buffer &weight, int start_index, int nfields );

};



#endif
