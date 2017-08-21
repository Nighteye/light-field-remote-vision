/* -*-c++-*- */
/** \file convolutions.cpp

    Convolution implementation, generic (non API) part.

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

#include <stdio.h>

#include "convolutions.h"

using namespace std;


/********************************************************
  Convolution kernel structure
*********************************************************/

// Init arbitrary convolution kernel
coco::convolution_kernel::convolution_kernel( compute_engine *CE, const gsl_matrix *m )
{
  _w = m->size2;
  _h = m->size1;
  _separable = false;

  _data = new compute_buffer( CE, _w*_h * sizeof(float) );
  float *buffer = new float[ _w*_h ];
  for ( int i=0; i<_w*_h; i++ ) {
    buffer[i] = m->data[i];
  }
  _data->memcpy_from_cpu( buffer );

  _data_x = NULL;
  _data_y = NULL;
  _engine = CE;

  delete[] buffer;
}


// Init separable convolution kernel
coco::convolution_kernel::convolution_kernel( compute_engine *CE, const gsl_vector *vx, const gsl_vector *vy )
{
  _w = vx->size;
  _h = vy->size;
  _separable = true;
  _data = NULL;
  _engine = CE;

  _data_x = new compute_buffer( CE, _w * sizeof(float) );
  float *buffer_x = new float[ _w ];
  for ( int i=0; i<_w; i++ ) {
    buffer_x[i] = vx->data[i];
  }
  _data_x->memcpy_from_cpu( buffer_x );
  delete[] buffer_x;

  _data_y = new compute_buffer( CE, _h * sizeof(float) );
  float *buffer_y = new float[ _h ];
  for ( int i=0; i<_h; i++ ) {
    buffer_y[i] = vy->data[i];
  }
  _data_y->memcpy_from_cpu( buffer_x );
  delete[] buffer_y;

  _data = new compute_buffer( CE, _w*_h * sizeof(float) );
  float *buffer = new float[ _w*_h ];
  int i=0;
  for ( int y=0; y<_h; y++ ) {
    for ( int x=0; x<_w; x++ ) {
      buffer[i++] = vx->data[x] * vy->data[y];
    }
  }
  _data->memcpy_from_cpu( buffer );
  delete[] buffer;
}

// Release convolution kernel
coco::convolution_kernel::~convolution_kernel()
{
  delete _data;
  delete _data_x;
  delete _data_y;
}
