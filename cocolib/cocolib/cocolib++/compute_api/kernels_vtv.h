/* -*-c++-*- */
/** \file kernels_vtv.h

    VTV kernels on grids

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

#ifndef __COCO_COMPUTE_API_KERNELS_VTV_H
#define __COCO_COMPUTE_API_KERNELS_VTV_H

#include "compute_grid.h"

namespace coco {

  /// Kernel for extragradient step
  void kernel_extragradient_step( const compute_grid *G,
				  const float theta,
				  compute_buffer &uq,
				  compute_buffer &u );

  /// Kernel for ROF functional, compute exact solution for primal variable
  /*
  void kernel_rof_functional_primal_exact_solution( const compute_grid *G,
						    const float lambda,
						    const compute_buffer &u,
						    const compute_buffer &f,
						    compute_buffer &px, compute_buffer &py );
  */

  /// Kernel for ROF functional, compute primal prox operator
  void kernel_rof_primal_prox( const compute_grid *G,
			       const float tau,
			       compute_buffer &u,
			       const float lambda,
			       const compute_buffer &f );

  // Gradient operator step kernels
  void kernel_gradient_operator_primal_step( const compute_grid *G,
					     const float tau,
					     compute_buffer &u,
					     const compute_buffer &px, const compute_buffer &py );

  void kernel_gradient_operator_dual_step( const compute_grid *G, 
					   const float tstep,
					   const compute_buffer &u,
					   compute_buffer &px, compute_buffer &py );

} // namespace
#endif

