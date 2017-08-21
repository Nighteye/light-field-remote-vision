/** \file color_spaces.cpp

    Color space conversions
    conversion formulas taken from http://easyrgb.com 
    (thanks for making them available)

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

#include "color_spaces.h"
#include "debug.h"

#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <iostream>

using namespace std;


// color constructor
coco::color::color( const color_space &cs, float c0, float c1, float c2 )
{
  if ( cs>=N_COLOR_SPACES ) {
    // invalid color space
    _cs = RGB;
    assert( false );
  }
  _r = c0;
  _g = c1;
  _b = c2;
}


// castable to float
float* coco::color::operator() ( color &c )
{
  return &(c._c0);
}


// Get color space name, empty if undefined
string coco::color_space_name( const color_space &cs )
{
  switch (cs) {
  case GRAYSCALE:
    return "grayscale";
  case RGB:
    return "rgb";
  case HSV:
    return "hsv";
  case HSL:
    return "hsl";
  case XYZ:
    return "xyz";
  case CIELAB:
    return "cielab";
  default:
    return "";
  }
}

// Get color space id
bool coco::color_space_id( const string &cs_name, color_space &cs )
{
  for ( int i=0; i<N_COLOR_SPACES; i++ ) {
    if ( color_space_name( (color_space)i ) == cs_name ) {
      cs = (color_space)i;
      return true;
    }
  }
  return false;
}


// Get color space range
bool coco::color_space_range( const color_space &cs, 
			      float &c0_min, float &c0_max,
			      float &c1_min, float &c1_max,
			      float &c2_min, float &c2_max )
{
  switch (cs) {
  case GRAYSCALE:
    c0_min = 0.0f; c0_max = 1.0f;
    c1_min = 0.0f; c1_max = 1.0f;
    c2_min = 0.0f; c2_max = 1.0f;
    return true;
  case RGB:
    c0_min = 0.0f; c0_max = 1.0f;
    c1_min = 0.0f; c1_max = 1.0f;
    c2_min = 0.0f; c2_max = 1.0f;
    return true;
  case HSV:
    c0_min = 0.0f; c0_max = 1.0f;
    c1_min = 0.0f; c1_max = 1.0f;
    c2_min = 0.0f; c2_max = 1.0f;
    return true;
  case HSL:
    c0_min = 0.0f; c0_max = 1.0f;
    c1_min = 0.0f; c1_max = 1.0f;
    c2_min = 0.0f; c2_max = 1.0f;
    return true;
  case XYZ:
    c0_min = 0.0f; c0_max = 95.047f;
    c1_min = 0.0f; c1_max = 100.0f;
    c2_min = 0.0f; c2_max = 108.883f;
    return true;
  case CIELAB:
    c0_min = 0.0f; c0_max = 100.0f;
    c1_min = -110.0f; c1_max = 110.0f;
    c2_min = -110.0f; c2_max = 110.0f;
    return true;
  default:
    c0_min = 0.0f; c0_max = 1.0f;
    c1_min = 0.0f; c1_max = 1.0f;
    c2_min = 0.0f; c2_max = 1.0f;
    return false;
  }
}


// Clip color to allowed range
bool coco::color_clip( color &c )
{
  float c0_min, c0_max;
  float c1_min, c1_max;
  float c2_min, c2_max;
  bool ok = color_space_range( c._cs, c0_min, c0_max, c1_min, c1_max, c2_min, c2_max );
  c._c0 = max( c0_min, min( c._c0, c0_max ));
  c._c1 = max( c1_min, min( c._c1, c1_max ));
  c._c2 = max( c2_min, min( c._c2, c2_max ));
  return ok;
}


// Color conversion function
bool coco::color_convert( const color_space cs_in, const color &c_in,
			  const color_space cs_out, color &c_out )
{
  const double ref_X =  95.047;
  const double ref_Y = 100.000;
  const double ref_Z = 108.883;


  // Conversions from other color spaces to RGB
  switch (cs_in) {
  case GRAYSCALE:
    {
      c_out._r = c_in._r;
      c_out._g = c_in._r;
      c_out._b = c_in._r;
    }
    break;

  case RGB:
    {
      c_out._r = c_in._r;
      c_out._g = c_in._g;
      c_out._b = c_in._b;
    }
    break;

  case HSV:
    assert( false );
    break;
  
  case HSL:
    {
      double h = c_in._h;
      double s = c_in._s;
      double l = c_in._l;
      double r,g,b;
      double chroma = 0.0;
      if ( l <= 0.5 ) {
	chroma = 2.0*l*s;
      }
      else {
	chroma = (2.0 - 2.0*l) * s;
      }
      
      double H = h * 360.0 / 60.0;
      double x = chroma * (1.0 - fabs( fmod( H, 2.0 ) - 1.0));
      double m = l - 0.5*chroma;
      if ( 0.0 <= H && H <= 1.0 ) {
	r = chroma + m;
	g = x + m;
	b = m;
      }
      else if ( H <= 2.0 ) {
	r = x + m;
	g = chroma + m;
	b = m;
      }
      else if ( H <= 3.0 ) {
	r = m;
	g = chroma + m;
	b = x + m;
      }
      else if ( H <= 4.0 ) {
	r = m;
	g = x + m;
	b = chroma + m;
      }
      else if ( H <= 5.0 ) {
	r = x + m;
	g = m;
	b = chroma + m;
      }
      else if ( H <= 6.0 ) {
	r = chroma + m;
	g = m;
	b = x + m;
      }
      else {
	r = g = b = 0.0;
      }

      c_out._r = r;
      c_out._g = g;
      c_out._b = b;
    }
    break;

  case XYZ:
    {
      float var_X = c_in._X / 100.0f;
      float var_Y = c_in._Y / 100.0f;
      float var_Z = c_in._Z / 100.0f;

      float var_R = var_X *  3.2406f + var_Y * -1.5372f + var_Z * -0.4986f;
      float var_G = var_X * -0.9689f + var_Y *  1.8758f + var_Z *  0.0415f;
      float var_B = var_X *  0.0557f + var_Y * -0.2040f + var_Z *  1.0570f;

      if ( var_R > 0.0031308f ) var_R = 1.055f * ( pow( var_R, 1.0f / 2.4f ) - 0.055f );
      else                      var_R = 12.92f * var_R;
      if ( var_G > 0.0031308f ) var_G = 1.055f * ( pow( var_G, 1.0f / 2.4f ) - 0.055f );
      else                      var_G = 12.92f * var_G;
      if ( var_B > 0.0031308f ) var_B = 1.055f * ( pow( var_B, 1.0f / 2.4f ) - 0.055f );
      else                      var_B = 12.92f * var_B;

      c_out._r = var_R;
      c_out._g = var_G;
      c_out._b = var_B;
    }
    break;

  case CIELAB:
    {
      float var_Y = ( c_in._L + 16.0f ) / 116.0f;
      float var_X = c_in._a / 500.0f + var_Y;
      float var_Z = var_Y - c_in._b / 200.0f;

      if ( pow( var_Y, 3.0f ) > 0.008856 ) var_Y = pow( var_Y, 3.0f );
      else                      var_Y = ( var_Y - 16.0f / 116.0f ) / 7.787f;
      if ( pow( var_X, 3.0f ) > 0.008856 ) var_X = pow( var_X, 3.0f );
      else                      var_X = ( var_X - 16.0f / 116.0f ) / 7.787f;
      if ( pow( var_Z, 3.0f ) > 0.008856 ) var_Z = pow( var_Z, 3.0f );
      else                      var_Z = ( var_Z - 16.0f / 116.0f ) / 7.787f;

      c_out._X = ref_X * var_X;
      c_out._Y = ref_Y * var_Y;
      c_out._Z = ref_Z * var_Z;
      color_convert( XYZ, c_out, RGB, c_out );
    }
    break;

  default:
    ERROR( "Unknown input color space ID or invalid conversion." << endl );
    assert( false );
    break;
  } 
  c_out._cs = RGB;


  // Conversions from RGB to other color spaces
  switch (cs_out) {
  case GRAYSCALE:
    {
      double a = (c_in._r + c_in._g + c_in._b) / 3.0;
      c_out._r = a;
      c_out._g = a;
      c_out._b = a;
    }
    return true;
  case RGB:
    // already done.
    return true;
  case HSV:
    assert( false );
    return true;
  case HSL:
    {
      double r = c_in._r;
      double g = c_in._g;
      double b = c_in._b;
      double M = std::max( r, std::max( g,b ));
      double m = std::min( r, std::min( g,b ));
      double h,s,l;
      
      if ( M==m ) {
	h = 0.0;
      }
      else if ( M==r ) {
	h = (g-b) / (6.0 * (M-m));
	if ( h<0.0 ) h += 1.0;
      }
      else if ( M==g ) {
	h = (b-r) / (6.0 * (M-m)) + 1.0 / 3.0;
      }
      else {
	h = (r-g) / (6.0 * (M-m)) + 2.0 / 3.0;
      }

      l = M+m;
      if ( M==m ) {
	s = 0.0;
      }
      else if ( l>1.0 ) {
	s = (M-m) / (2.0 - l);
      }
      else {
	s = (M-m) / l;
      }
      
      l = l*0.5;

      c_out._h = h;
      c_out._s = s;
      c_out._l = l;
      c_out._cs = HSL;
    }
    break;

  case XYZ:
    {
      double var_R = c_in._r;
      double var_G = c_in._g;
      double var_B = c_in._b;
      
      if ( var_R > 0.04045 ) var_R = pow( ( var_R + 0.055 ) / 1.055, 2.4 );
      else                   var_R = var_R / 12.92;
      if ( var_G > 0.04045 ) var_G = pow( ( var_G + 0.055 ) / 1.055, 2.4 );
      else                   var_G = var_G / 12.92;
      if ( var_B > 0.04045 ) var_B = pow( ( var_B + 0.055 ) / 1.055, 2.4 );
      else                   var_B = var_B / 12.92;

      var_R = var_R * 100;
      var_G = var_G * 100;
      var_B = var_B * 100;
      
      //Observer. = 2°, Illuminant = D65
      double X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805;
      double Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722;
      double Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505;
      c_out._X = X;
      c_out._Y = Y;
      c_out._Z = Z;
      c_out._cs = XYZ;
    }
    return true;

  case CIELAB:
    {
      color_convert( RGB, c_out, XYZ, c_out );

      // Observer= 2°, Illuminant= D65
      double var_X = c_in._X / ref_X;
      double var_Y = c_in._Y / ref_Y;
      double var_Z = c_in._Z / ref_Z;

      if ( var_X > 0.008856 ) var_X = pow( var_X, 1.0 / 3.0 );
      else                    var_X = ( 7.787 * var_X ) + ( 16.0 / 116.0 );
      if ( var_Y > 0.008856 ) var_Y = pow( var_Y, 1.0 / 3.0 );
      else                    var_Y = ( 7.787 * var_Y ) + ( 16.0 / 116.0 );
      if ( var_Z > 0.008856 ) var_Z = pow( var_Z, 1.0 / 3.0 );
      else                    var_Z = ( 7.787 * var_Z ) + ( 16.0 / 116.0 );
      
      double L = ( 116.0 * var_Y ) - 16.0;
      double a = 500.0 * ( var_X - var_Y );
      double b = 200.0 * ( var_Y - var_Z );
      c_out._L = L;
      c_out._a = a;      
      c_out._b = b;
      c_out._cs = CIELAB;
    }
    return true;

  default:
    ERROR( "Unknown color space ID or invalid conversion." << endl );
    assert( false );
    return false;
  }


  return false;
}


