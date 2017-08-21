/* -*-c++-*- */
/** \file curvature_linear.cuh

   Total mean curvature - linear data term.
   Locally used functions - energy computations

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

namespace coco {

  double compute_primal_tv_energy_u( coco::tc_data *data, stcflt *gpu_u );
  double compute_primal_rof_energy( tc_data *data );
  double compute_primal_inpainting_energy( tc_data *data );

}
