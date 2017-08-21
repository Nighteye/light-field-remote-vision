/** \file gsl_image.h

    Visualization tools (vector fields etc).
    
    Copyright (C) 2013 Bastian Goldluecke,
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

#ifndef __VISUALIZE_H
#define __VISUALIZE_H

#include <vector>
#include <string>

#include "gsl_matrix_helper.h"

class QImage;


namespace coco {

  // Draw outline of level set using marching squares algorithm
  bool draw_vector_field_to_image( gsl_matrix *dx, gsl_matrix *dy,
				   double scale,
				   QImage &I,
				   int xbase, int xstep,
				   int ybase, int ystep,
				   QRgb colorBase, QRgb colorArrow,
				   double baseCircleRadius, double lineWidth );
}


#endif

