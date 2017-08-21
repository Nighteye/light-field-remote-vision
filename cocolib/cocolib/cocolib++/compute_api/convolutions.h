/** \file cuda_convolutions.h

    Convolution functions supported by compute API.

    Copyright (C) 2010-2014 Bastian Goldluecke,
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

#ifndef __COMPUTE_API_CONVOLUTIONS_H
#define __COMPUTE_API_CONVOLUTIONS_H

#include "../../defs.h"
#include "../../common/gsl_matrix_helper.h"
#include "compute_grid.h"

namespace coco {

  /********************************************************
  Convolution kernel structure
  *********************************************************/
  
  // Convolution kernel
  struct convolution_kernel {
    // Init arbitrary convolution kernel
    convolution_kernel( compute_engine*, const gsl_matrix *m );
    // Init separable convolution kernel
    convolution_kernel( compute_engine*, const gsl_vector *vx, const gsl_vector *vy );
    // Release convolution kernel
    ~convolution_kernel();

    // Kernel size
    int _w;
    int _h;
    // Separable kernel?
    bool _separable;
    // Kernel data, non-separable
    compute_buffer* _data;
    // Kernel data, separable
    compute_buffer* _data_x;
    compute_buffer* _data_y;
    // Compute engine
    compute_engine *_engine;
  };



  /********************************************************
  Convolution functions
  *********************************************************/

  // Convolve array with kernel
  bool convolution( const compute_grid *grid, 
		    const convolution_kernel *kernel, 
		    const compute_buffer &in, compute_buffer &out );

}
  
#endif
