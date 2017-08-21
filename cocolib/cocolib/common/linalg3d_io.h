/** \file linalg3d_io.h

    File imported from "common" lib, use if this library is not available.
    Uses "gov" namespace (Graphics-optics-vision, MPI).

    Basic linear algebra up to dimension 4: Vectors, matrices, quaternions.
    Extensions for IO not compiled included in the standard header.
    
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

#ifndef __GOV_MATH_LINALG_IO_H
#define __GOV_MATH_LINALG_IO_H

#include <iostream>

#include "linalg3d.h"
#include "linalg3d_quat.h"


// IO-Stream operations for matrices and vectors
namespace coco
{

  /// Output to a stream
  /** \ingroup mathlinal */
  template<class T>
  std::ostream &operator<< (std::ostream &o, const coco::Vec2<T> &v)
  {
    return o << "(" << v.x << "," << v.y << ")";
  }

  /// Output to a stream
  /** \ingroup mathlinal */
  template<class T>
  std::ostream &operator<< (std::ostream &o, const coco::Vec3<T> &v)
  {
    return o << "(" << v.x << "," << v.y << "," << v.z << ")";
  }

  /// Output to a stream
  /** \ingroup mathlinal */
  template<class T>
  std::ostream &operator<< (std::ostream &o, const coco::Vec4<T> &v)
  {
    return o << "(" << v.x << "," << v.y << "," << v.z << "," << v.w << ")";
  }


  /// Output to a stream
  /** \ingroup mathlinal */
  template<class T>
  std::ostream &operator<< (std::ostream &o, const coco::Quat<T> &q)
  {
    return o << "[" << q._x << "," << q._y << "," << q._z << "," << q._w << "]";
  }

  /// Output to a stream
  /** \ingroup mathlinal */
  template<class T>
  std::ostream &operator<< (std::ostream &o, const coco::Mat44<T> &m)
  {
    o << "{{" << m[0][0] << "," << m[0][1] << "," << m[0][2] << "," << m[0][3] << "}," << std::endl;
    o << " {" << m[1][0] << "," << m[1][1] << "," << m[1][2] << "," << m[1][3] << "}," << std::endl;
    o << " {" << m[2][0] << "," << m[2][1] << "," << m[2][2] << "," << m[2][3] << "}," << std::endl;
    o << " {" << m[3][0] << "," << m[3][1] << "," << m[3][2] << "," << m[3][3] << "}}" << std::endl;
    return o;
  }


  /// Input from a stream
  /** \ingroup mathlinal */
  template<class T>
  std::istream &operator>> (std::istream &i, coco::Mat44<T> &m)
  {
    char c;
    i >> c >> c >> m[0][0] >> c >> m[0][1] >> c >> m[0][2] >> c >> m[0][3] >> c >> c;
    i >> c >> m[1][0] >> c >> m[1][1] >> c >> m[1][2] >> c >> m[1][3] >> c >> c;
    i >> c >> m[2][0] >> c >> m[2][1] >> c >> m[2][2] >> c >> m[2][3] >> c >> c;
    i >> c >> m[3][0] >> c >> m[3][1] >> c >> m[3][2] >> c >> m[3][3] >> c >> c;
    return i;
  }


  /// Output to a stream
  /** \ingroup mathlinal */
  template<class T>
  std::ostream &operator<< (std::ostream &o, const coco::Mat33<T> &m)
  {
    o << "{{" << m._11 << "," << m._12 << "," << m._13 << "}," << std::endl;
    o << "{" << m._21 << "," << m._22 << "," << m._23 << "}," << std::endl;
    o << "{" << m._31 << "," << m._32 << "," << m._33 << "}}";
    return o;
  }

  /// Output to a stream
  /** \ingroup mathlinal */
  template<class T>
  std::ostream &operator<< (std::ostream &o, const coco::Mat22<T> &m)
  {
    o << "{{" << m._11 << "," << m._12 << "}," << std::endl;
    o << "{" << m._21 << "," << m._22 << "}}";
    return o;
  }

  /// Input from a stream
  /** \ingroup mathlinal */
  template<class T>
  std::istream &operator>> (std::istream &i, coco::Mat33<T> &m)
  {
    char c;
    i >> c >> c >> m._11 >> c >> m._12 >> c >> m._13 >> c >> std::endl;
    i >> c >> m._21 >> c >> m._22 >> c >> m._23 >> c >> std::endl;
    i >> c >> m._31 >> c >> m._32 >> c >> m._33 >> c >> c >> std::endl;
    return i;
  }


  /// Input from a stream
  /** \ingroup mathlinal */
  template<class T>
  std::istream &operator>> (std::istream &o, coco::Vec2<T> &v)
  {
    char c;
    return o >> c >> v.x >> c >> v.y >> c;
  }

  /// Input from a stream
  /** \ingroup mathlinal */
  template<class T>
  std::istream &operator>> (std::istream &o, coco::Vec3<T> &v)
  {
    char c;
    return o >> c >> v.x >> c >> v.y >> c >> v.z >> c;
  }

  /// Input from a stream
  /** \ingroup mathlinal */
  template<class T>
  std::istream &operator>> (std::istream &o, coco::Vec4<T> &v) 
  {
    char c;
    return o >> c >> v.x >> c >> v.y >> c >> v.z >> c >> v.w >> c;
  }





}

#endif

