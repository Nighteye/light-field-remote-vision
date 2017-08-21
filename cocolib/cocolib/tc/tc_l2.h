/** \file tc_l2.h
    Solver for Total Curvature - L2 data term
    with product relaxed via sum term of dual variables
    (symmetric formulation)

   Copyright (C) 2011 Bastian Goldluecke,
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

#ifndef __COCO_TC_L2_H
#define __COCO_TC_L2_H

#include "tc.h"


namespace coco {

  /*****************************************************************************
       TC-L2 algorithm I: Experimental, Bermudez-Moreno
                          Based on relaxation of total curvature via
                          a sum of dual variables
  *****************************************************************************/

  // Main iteration functions
  bool tc_l2_init( tc_data *data );
  bool tc_l2_iteration( tc_data *data );
  bool tc_l2_primal_prox( tc_data *data );
  bool tc_l2_dual_prox( tc_data *data );
  bool tc_l2_dual_reprojection( tc_data *data );
  bool tc_l2_overrelaxation( tc_data *data );

  // Aux functions
  bool tc_get_dual_x( tc_data *data,
		      stcflt* &p1, stcflt* &p2,
		      int y0, int y1, int z0, int z1 );
  bool tc_get_dual_y( tc_data *data,
		      stcflt* &p1, stcflt* &p2,
		      int x0, int x1, int z0, int z1 );
  bool tc_get_dual_z( tc_data *data,
		      stcflt* &p1, stcflt* &p2,
		      int x0, int x1, int y0, int y1 );
    
}


#endif
