/* -*-c++-*- */
/** \file cuda_inline_device_functions.cu

    Some standard CUDA device functions.

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


///////////////////////////////////////////////////////////////////////////////////////////
// Interpolation and resampling
///////////////////////////////////////////////////////////////////////////////////////////

// interpolates a matrix at px,py
// array coordinates, center of the pixel
static __device__ float bilinear_interpolation( int W, int H, float *u, float px, float py )
{
  int ux = (int)floor(px);
  int uy = (int)floor(py);
  float ax = px - (float)ux;
  float ay = py - (float)uy;

  int o = ux + uy*W;
  float uxmym, uxpym, uxpyp, uxmyp;
  if ( ux >= 0 ) {
    if ( ux<W-2 ) {
      // x ok.
      if ( uy>=0 ) {
	if ( uy<H-1 ) {
	  // all valid.
	  uxmym = u[o];
	  uxpym = u[o+1];
	  uxmyp = u[o+W];
	  uxpyp = u[o+W+1];
	}
	else if ( uy<H ) {
	  // y+1 invalid.
	  uxmym = u[o];
	  uxpym = u[o+1];
	  uxmyp = uxmym;
	  uxpyp = uxpym;
	}
	else {
	  // invalid.
	  uxmym = 0.0f;
	  uxpym = 0.0f;
	  uxmyp = 0.0f;
	  uxpyp = 0.0f;
	}
      }
      else if ( uy==-1 ) {
	// only y+1 valid.
	uxmyp = u[o+W];
	uxpyp = u[o+W+1];
	uxmym = uxmyp;
	uxpym = uxpyp;
      }
      else {
	// invalid.
	uxmym = 0.0f;
	uxpym = 0.0f;
	uxmyp = 0.0f;
	uxpyp = 0.0f;
      }
    }
    else if ( ux < W ) {
      // only x valid
      if ( uy>=0 ) {
	if ( uy<H-1 ) {
	  // y valid.
	  uxmym = u[o];
	  uxpym = uxmym;
	  uxmyp = u[o+W];
	  uxpyp = uxmyp;
	}
	else if ( uy<H ) {
	  // y+1 invalid.
	  uxmym = u[o];
	  uxpym = uxmym;
	  uxmyp = uxmym;
	  uxpyp = uxpym;
	}
	else {
	  // invalid.
	  uxmym = 0.0f;
	  uxpym = 0.0f;
	  uxmyp = 0.0f;
	  uxpyp = 0.0f;
	}
      }
      else if ( uy==-1 ) {
	// only y+1 valid.
	uxmyp = u[o+W];
	uxpyp = uxmyp;
	uxmym = uxmyp;
	uxpym = uxpyp;
      }
      else {
	// invalid.
	uxmym = 0.0f;
	uxpym = 0.0f;
	uxmyp = 0.0f;
	uxpyp = 0.0f;
      }
    }
    else {  // ux >= W
      // invalid.
      uxmym = 0.0f;
      uxpym = 0.0f;
      uxmyp = 0.0f;
      uxpyp = 0.0f;
    }
  }
  else if ( ux == -1 ) {
    // only x+1 ok.
    if ( uy>=0 ) {
      if ( uy<H-1 ) {
	// all valid.
	uxpym = u[o+1];
	uxmym = uxpym;
	uxpyp = u[o+W+1];
	uxmyp = uxpyp;
      }
      else if ( uy<H ) {
	// y+1 invalid.
	uxpym = u[o+1];
	uxmym = uxpym;
	uxmyp = uxmym;
	uxpyp = uxpym;
      }
      else {
	// invalid.
	uxmym = 0.0f;
	uxpym = 0.0f;
	uxmyp = 0.0f;
	uxpyp = 0.0f;
      }
    }
    else if ( uy==-1 ) {
      // only y+1 valid.
      uxpyp = u[o+W+1];
      uxmyp = uxpyp;
      uxmym = uxmyp;
      uxpym = uxpyp;
    }
    else {
      // invalid.
      uxmym = 0.0f;
      uxpym = 0.0f;
      uxmyp = 0.0f;
      uxpyp = 0.0f;
    }
  }
  else {
    // invalid.
    uxmym = 0.0f;
    uxpym = 0.0f;
    uxmyp = 0.0f;
    uxpyp = 0.0f;
  }

  float uym = ax * uxpym + (1.0f - ax) * uxmym;
  float uyp = ax * uxpyp + (1.0f - ax) * uxmyp;
  return ay * uyp + (1.0f - ay) * uym;
}
