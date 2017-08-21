/** \file linalg3d_quat.h

    File imported from "common" lib, use if this library is not available.
    Uses "gov" namespace (Graphics-optics-vision, MPI).

    Basic linear algebra up to dimension 4: Vectors, matrices, quaternions.
    Quaternion code not included in standard header.
    
    Copyright (C) 2002 Bastian Goldluecke,
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

#ifndef __GOV_MATH_LINALG_QUAT_H
#define __GOV_MATH_LINALG_QUAT_H


#include "linalg3d.h"


// Quaternion class
namespace coco
{

  /// Quaternion class
  /** \note The Quaternions here are meant to always represent
      pure rotations. To ensure this, they are normalized after
      every operation.
      \ingroup mathlinal
  */
  template<class T>
  struct Quat
  {
    /// Components
    T _x, _y, _z, _w;

    /// By value constructor
    Quat( const T& x, const T& y, const T& z, const T& w ) {
      _x = x; _y = y; _z = z; _w = w;
      normalize();
    }

    /// By (rotation) matrix constructor
    Quat( const Mat44<T> &m ) {

      // 0 = 00   1 = 01   2 = 02   3 = 03
      // 4 = 10   5 = 11   6 = 12   7 = 13
      // 8 = 20   9 = 21  10 = 22  11 = 23

      // Calculate the trace of the matrix T from the equation:
      float d0 = m[0][0];
      float d1 = m[1][1];
      float d2 = m[2][2];
      float t = 1.0f + d0+d1+d2;


      // If the trace of the matrix is greater than zero, then
      // perform an "instant" calculation.
      // Important note wrt. rouning errors:

      float S,X,Y,Z,W;
      if ( t > 0.00000001 ) {
	S = sqrt(t) * 2;
	X = ( m[2][1] - m[1][2] ) / S;
	Y = ( m[0][2] - m[2][0] ) / S;
	Z = ( m[1][0] - m[0][1] ) / S;
	W = 0.25 * S;
      }

      // If the trace of the matrix is equal to zero then identify
      // which major diagonal element has the greatest value.
      // Depending on this, calculate the following:

      if ( d0>d1 && d0>d2 ) {       // Column 0: 
        S  = sqrt( 1.0 + d0-d1-d2 ) * 2;
        X = 0.25 * S;
        Y = (m[1][0] + m[0][1] ) / S;
        Z = (m[0][2] + m[2][0] ) / S;
        W = (m[2][1] - m[1][2] ) / S;
      }
      else if ( d1>d2 ) {                    // Column 1: 
        S  = sqrt( 1.0 + d1 - d0 - d2 ) * 2;
        X = (m[1][0] + m[0][1] ) / S;
        Y = 0.25 * S;
        Z = (m[2][1] + m[1][2] ) / S;
        W = (m[0][2] - m[2][0] ) / S;
      } 
      else {                                            // Column 2:
        S  = sqrt( 1.0 + d2 - d0 - d1 ) * 2;
        X = (m[0][2] + m[2][0] ) / S;
        Y = (m[2][1] + m[1][2] ) / S;
        Z = 0.25 * S;
        W = (m[1][0] - m[0][1] ) / S;
      }

      _x = X; _y = Y; _z = Z; _w = W;
      normalize();
    }

    /// By rotation parameters constructor
    Quat( const Vec3<T> axis=Vec3<T>(T(1),T(0),T(0)), const T angle=T(0) ) {
      T angle2 = angle / T(2);
      T s = sin( angle2 );
      _x = axis.x*s; _y = axis.y*s; _z = axis.z*s;
      _w = cos( angle2 );
      normalize();
    }

    /// Convert to rotation matrix
    operator Mat44<T>() const {
      Mat44<T> m;
      T zz = _z*_z; T yy = _y*_y; T xx=_x*_x;
      m[0][0] = T(1)-T(2)*(yy+zz);
      m[1][1] = T(1)-T(2)*(xx+zz);
      m[2][2] = T(1)-T(2)*(xx+yy);
      m[3][3] = T(1);
      T xy = _x*_y; T wz=_w*_z;
      m[1][0] = T(2)*(xy-wz); m[0][1] = T(2)*(xy+wz);
      T wy = _w*_y; T xz = _x*_z;
      m[2][0] = T(2)*(wy+xz); m[0][2] = T(2)*(xz-wy);
      T yz = _y*_z; T wx=_w*_x;
      m[2][1] = T(2)*(yz-wx); m[1][2] = T(2)*(yz+wx);
      m.transpose();
      return m;
    }

    /// Return the axis of rotation in 3-Space
    Vec3<T> axisOfRotation() const {
      T s = sqrt( T(1) - _w*_w );
      return Vec3<T> ( _x/s, _y/s, _z/s );
    }

    /// Return the angle of the rotation around the axis
    T angleOfRotation() const {
      return T(2)*arccos( _w );
    }

    /// Length of quaternion
    inline double length() const {
      return sqrt( _x*_x + _y*_y + _z*_z + _w*_w );
    }

    /// Normalize quaternion
    inline void normalize() {
      double len = length();
      _x /= len; _y /= len; _z /= len; _w /= len;
    }

    /// Multiply by another quaternion
    Quat &operator*= ( const Quat& q ) {
      T x = _y*q._z - _z*q._y + _w*q._x + _x*q._w;
      T y = _z*q._x - _x*q._z + _w*q._y + _y*q._w;
      T z = _x*q._y - _y*q._x + _w*q._z + _z*q._w;
      T w = _w*q._w - _x*q._x - _y*q._y - _z*q._z;
      _x=x; _y=y; _z=z; _w=w;
      normalize();
      return (*this);
    }
    /// Multiply by another quaternion
    Quat operator* ( const Quat& q ) {
      Quat r;
      r._x = _y*q._z - _z*q._y + _w*q._x + _x*q._w;
      r._y = _z*q._x - _x*q._z + _w*q._y + _y*q._w;
      r._z = _x*q._y - _y*q._x + _w*q._z + _z*q._w;
      r._w = _w*q._w - _x*q._x - _y*q._y - _z*q._z;
      r.normalize();
      return r;
    }
  };


  /// Quaternion with float entries
  /** \ingroup mathlinal */
  typedef Quat<float> Quatf;
  /// Quaternion with double entries
  /** \ingroup mathlinal */
  typedef Quat<double> Quatd;


  /// Interpolation between two rotations
  /** \ingroup mathlinal */
  template<class T>
  Quat<T> interpolate( const Quat<T>& q1, const Quat<T>& q2, const T& v );

  /// Return the rotation leading from v0 to v1
  /** \ingroup mathlinal */
  template<class T>
  Quat<T> rotationArc( Vec3<T> v0, Vec3<T> v1 );






  // Convert a matrix into a quaternion
  template<class T>
    void mat2quat( Mat44<T> &a, Quat<T>& q ) {
    T trace = a._11 + a._22 + a._33 + 1.0f;
 
    if( trace > 1e-6 ) {
      T s = 0.5f / sqrtf(trace);
      q._w = 0.25f / s;
      q._x = ( a._32 - a._23 ) * s;
      q._y = ( a._31 - a._13 ) * s;
      q._z = ( a._21 - a._12 ) * s;
    } else {
      if ( a._11 > a._22 && a._11 > a._33 ) {
        T s = 2.0f * sqrtf( 1.0f + a._11 - a._22 - a._33);
        q._x = 0.25f * s;
        q._y = (a._21 + a._12 ) / s;
        q._z = (a._31 + a._13 ) / s;
        q._w = (a._32 - a._23 ) / s;
      
      } else if (a[1][1] > a[2][2]) {
        T s = 2.0f * sqrtf( 1.0f + a._22 - a._11 - a._33);
        q._x = (a._21 + a._12 ) / s;
        q._y = 0.25f * s;
        q._z = (a._32 + a._23 ) / s;
        q._w = (a._31 - a._13 ) / s;
      } else {
        T s = 2.0f * sqrtf( 1.0f + a._33 - a._11 - a._22 );
        q._x = (a._31 + a._13 ) / s;
        q._y = (a._32 + a._23 ) / s;
        q._z = 0.25f * s;
        q._w = (a._12 - a._21 ) / s;
      }
    }
    q.normalize();
  }


  // Return the rotation leading from v0 to v1
  template<class T>
    Quat<T> rotationArc( Vec3<T> v0, Vec3<T> v1 )
    {
      v0.normalize();
      v1.normalize();
      Vec3<T> c = cross( v0,v1 );
      T d = v0*v1;
      T s = sqrt( T(2)*( T(1) + d ) );
      if (s != T(0)) {
        return Quat<T> ( c.x/s, c.y/s, c.z/s, s/T(2) );
      }
      else {
	// d==-1 -> Rotate by 180 degrees around arbitrary axis perp. to v0
	if (v0.x != T(0)) {
	  return Quat<T> ( Vec3<T>( v0.y, -v0.x, T(0)), M_PI );
	}
	else {
	  return Quat<T> ( Vec3<T>( T(0), v0.z, -v0.y ), M_PI );
        }
      }
    }


  // Interpolation between two Quaternions
  template<class T> Quat<T> interpolate( const Quat<T> &qr1, const Quat<T> &qr2, const T& lambda )
    {
      Quat<T> _qrot;
      _qrot._x = qr2._x * lambda + qr1._x * (1.0f - lambda);
      _qrot._y = qr2._y * lambda + qr1._y * (1.0f - lambda);
      _qrot._z = qr2._z * lambda + qr1._z * (1.0f - lambda);
      _qrot._w = qr2._w * lambda + qr1._w * (1.0f - lambda);
      _qrot.normalize();
      return _qrot;
    }








}

#endif

