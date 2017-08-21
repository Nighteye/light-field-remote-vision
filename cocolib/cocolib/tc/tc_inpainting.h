/** \file curvature_linear.h
    Solver for anisotropic TV - linear data term

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

#ifndef __COCO_CURVATURE_LINEAR_INPAINTING_FISTA_H
#define __COCO_CURVATURE_LINEAR_INPAINTING_FISTA_H

#include "tc.h"


namespace coco {

  /*****************************************************************************
       TC-Inpainting algorithm I: Experimental, FISTA with Bermudez-Moreno
  *****************************************************************************/

  // Perform one iteration (outer loop)
  bool tc_inpainting_fista_init( tc_data *data );
  bool tc_inpainting_fista_iteration( tc_data *data );
  bool tc_inpainting_fista_product_iteration( tc_data *data );
}


#endif
