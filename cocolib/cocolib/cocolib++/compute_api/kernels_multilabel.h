/* -*-c++-*- */
/** \file kernels_multilabel.h

    MULTILABEL kernels on grids

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

#ifndef __COCO_COMPUTE_API_KERNELS_MULTILABEL_H
#define __COCO_COMPUTE_API_KERNELS_MULTILABEL_H

#include "compute_grid.h"

namespace coco {

  void kernel_assign_optimum_label( const compute_grid *G,
				    int N,
				    const compute_buffer &a,
				    compute_buffer &u );

} // namespace
#endif

