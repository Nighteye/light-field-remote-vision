/* -*-c++-*- */
/** \file kernels_reprojections.h

    VTV kernels on grids
    Reprojections according to various norms

    Copyright (C) 2011-2014 Bastian Goldluecke,
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

#ifndef __COCO_COMPUTE_API_KERNELS_REPROJECTIONS_H
#define __COCO_COMPUTE_API_KERNELS_REPROJECTIONS_H

#include "compute_grid.h"

namespace coco {

  // Reprojection to Euclidean unit ball, 1D
  void kernel_reproject_euclidean_1D( const compute_grid *G,
				      const float radius,
				      compute_buffer &px1 );

  // Reprojection to Euclidean unit ball, 2D
  void kernel_reproject_euclidean_2D( const compute_grid *G,
				      const float radius,
				      compute_buffer &px1, compute_buffer &py1 );

  // Reprojection to Euclidean unit ball, 4D
  void kernel_reproject_euclidean_4D( const compute_grid *G,
				      const float radius,
				      compute_buffer &px1, compute_buffer &py1,
				      compute_buffer &px2, compute_buffer &py2 );

  // Reprojection to Euclidean unit ball, 6D
  void kernel_reproject_euclidean_6D( const compute_grid *G,
				      const float radius,
				      compute_buffer &px1, compute_buffer &py1,
				      compute_buffer &px2, compute_buffer &py2,
				      compute_buffer &px3, compute_buffer &py3 );

  // Reprojection to Nuclear norm unit ball, 6D
  void kernel_reproject_nuclear_6D( const compute_grid *G,
				    const float radius,
				    compute_buffer &px1, compute_buffer &py1,
				    compute_buffer &px2, compute_buffer &py2,
				    compute_buffer &px3, compute_buffer &py3 );


} // namespace
#endif
