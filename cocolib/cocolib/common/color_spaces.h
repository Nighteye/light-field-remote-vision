/** \file color_spaces.h

    Color space conversions
    
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

#ifndef __COLOR_SPACES_H
#define __COLOR_SPACES_H

//#include <gsl/gsl_matrix.h>
#include <vector>
#include <string>
#include "gsl_matrix_helper.h"

namespace coco {

  // Color space IDs
  enum color_space {
    GRAYSCALE,
    RGB,
    HSV,
    HSL,
    XYZ,
    CIELAB,
    N_COLOR_SPACES,
  };

  // Color data structure
  struct color {
    // Constructor (note: does not automatically clip)
    color( const color_space &cs, float c0, float c1, float c2 );
    // castable to float*
    float* operator() ( color &c );

    // Data, each component is a union for name overloading
    union {
      float _c0;
      float _r;
      float _h;
      float _X;
      float _L;
    };
    union {
      float _c1;
      float _g;
      float _s;
      float _Y;
      float _a;
    };
    union {
      float _c2;
      float _v;
      float _l;
      float _Z;
      float _b;
    };

    // Color space identifier
    color_space _cs;
  };


  // Get color space name, empty if undefined
  std::string color_space_name( const color_space &cs );
  // Get color space id
  bool color_space_id( const std::string &cs_name, color_space &cs );

  // Get color space range
  bool color_space_range( const color_space &cs, 
			  float &c0_min, float &c0_max,
			  float &c1_min, float &c1_max,
			  float &c2_min, float &c2_max );

  // Clip color to allowed range
  bool color_clip( color &c );

  // Color conversion function (note: no automatic clipping)
  bool color_convert( const color_space cs_in, const color &c_in,
		      const color_space cs_out, color &c_out );
}



#endif

