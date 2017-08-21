/** \file linalg3d.h

    File imported from "common" lib, use if this library is not available.
    Uses "gov" namespace (Graphics-optics-vision, MPI).

    Basic linear algebra up to dimension 4: Vectors, matrices, quaternions.
    Optimized for speed and ease of use.
    
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

#ifndef __GOV3_VEC_H
#define __GOV3_VEC_H

#include <math.h>
#include <assert.h>
#include <string.h>

#include <vector>
#include <iostream>

#include "debug.h"
#include "../defs.h"

namespace coco
{
  
  // some useful inline functions
  /// Clamps a value to a given range
  /** \ingroup mathutil */
  template<class T> inline T clamp( const T& v, const T& min, const T& max ) {
    if (v<min) return min;
    if (v>max) return max;
    return v;
  };
  
  /// Square of a number
  /** \ingroup mathutil */
  template<class T> T square( const T& v ) {
    return (T)pow( v,2 );
  };
  
  /// Convert degrees to radian
  /** \ingroup mathutil */
  template<class T> T d2r( const T& x ) {
    return x*M_PI / 180.0f;
  };
  
  /// Interpolation from a to b at t
  /** \ingroup mathutil */
  template<class T> T interpolate( const T& a, const T& b, const T& t ) {
    return t*b + (1-t)*a;
  };
  
  /// Convert radian to degrees
  /** \ingroup mathutil */
  template<class T> T r2d( const T& x ) {
    return x*180.0f/M_PI;
  };



  extern double __mathbase_resolution;
  extern double __mathbase_epsilon;


  /// Vector with integer entries
  /** \ingroup mathlinal */
  typedef std::vector<int> intv;

  // Forward declaration
  template<class T> struct Vec3;

  /// Vertex in 2D space
  /** \ingroup mathlinal */
  template<class T>
  struct Vec2 {
    /// Coordinates
    T x,y;
    /// Default constructor
    Vec2() {
      x=T(0); y=T(0);
    };
    /// Construct by coordinates
    Vec2( T xx, T yy ) {
      x=xx; y=yy;
    };
    /// Construct by float pointer
    Vec2( const T *pv ) {
      assert( pv );
      x=*pv; y=*(pv+1);
    };
    // Construct by 3-Vector (homogenous divide)
    Vec2( const Vec3<T>& v ) {
      x=v.x/v.z; y=v.y/v.z;
    }
    /// Sum of vectors
    Vec2 operator+ ( const Vec2&v ) const {
      return Vec2( x+v.x, y+v.y );
    };
    /// Difference of vectors
    Vec2 operator- ( const Vec2&v ) const {
      return Vec2( x-v.x, y-v.y );
    };
    /// Unary minus
    Vec2 operator- () const {
      return Vec2( -x, -y );
    };
    /// Sum of vectors
    Vec2 &operator+= ( const Vec2&v ) {
      x+=v.x; y+=v.y;
      return *this;
    };
    /// Difference of vectors
    Vec2 &operator-= ( const Vec2&v ) {
      x-=v.x; y-=v.y;
      return *this;
    };
    /// Multiplication by scalar
    Vec2 &operator*= ( T f ) {
      x*=f; y*=f;
      return *this;
    };
    /// Division by scalar
    Vec2 &operator/= ( T f ) {
      assert( f != T(0) );
      x/=f; y/=f;
      return *this;
    };
    /// Multiplication by scalar
    Vec2 operator* ( const T s ) const {
      return Vec2( x*s, y*s );
    };
    /// Division by scalar
    Vec2 operator/ ( const T s ) const {
      assert( s != T(0) );
      return Vec2( x/s, y/s );
    };
    /// Dot product
    T operator*( const Vec2 &v ) const {
      return x*v.x + y*v.y;
    };
    /// Length of the vector
    double length() const {
      return sqrt( double(x*x + y*y) );
    }
    /// Minimum component
    T min() const {
      return std::min( x,y );
    }
    /// Maximum component
    T max() const {
      return std::max( x,y );
    }
    /// Normalize the vector
    Vec2 &normalize() {
      T len = sqrt(x*x + y*y);
      if (len>T(0)) {
	x /= len; y/= len;
      };
      return *this;
    };
    /// Cast the vector to a T* used by OpenGL
    operator T*() {
      return &x;
    };
    /// Cast the vector to a T* used by OpenGL
    operator const T*() const {
      return &x;
    };
    /// Comparison
    bool operator== ( const Vec2& v ) const {
      return x==v.x && y==v.y;
    };
  };

  // Forward declaration
  template<class T> struct Vec4;

  /// Vertex in 3D space
  /** \ingroup mathlinal */
  template<class T>
  struct Vec3 {
    /// Coordinates
    T x,y,z;
    /// Default constructor
    Vec3() {
      x=T(0); y=T(0); z=T(0);
    };
    /// Construct by coordinates
    Vec3( T xx, T yy, T zz ) {
      x=xx; y=yy; z=zz;
    };
#ifdef LIB_GTS
    /// Construct from Gts vertex
    Vec3( const GtsVertex *v ) {
      x = T(v->p.x);
      y = T(v->p.y);
      z = T(v->p.z);
    };
#endif
    /// Construct by float pointer
    Vec3( const T *pv ) {
      if (pv==NULL) {
	x=y=z=T(0);
	return;
      }
      x=*pv; y=*(pv+1); z=*(pv+2);
    };
    /// Construct by 4D-vertex (divide by w)
    Vec3( const Vec4<T>& );
    /// Construct by 2D-point (append 1)
    Vec3( const Vec2<T>&v ) {
      x = v.x; y = v.y; z = T(1);
    }
    /// Difference of vectors
    Vec3 operator- ( const Vec3&v ) const {
      return Vec3( x-v.x, y-v.y, z-v.z );
    };
    /// Unary minus
    Vec3 operator- () const {
      return Vec3( -x, -y, -z );
    };
    /// Sum of vectors
    Vec3 operator+ ( const Vec3&v ) const {
      return Vec3( x+v.x, y+v.y, z+v.z );
    };
    /// Sum of vectors
    Vec3 &operator+= ( const Vec3&v ) {
      x+=v.x; y+=v.y; z+=v.z;
      return *this;
    };
    /// Difference of vectors
    Vec3 &operator-= ( const Vec3&v ) {
      x-=v.x; y-=v.y; z-=v.z;
      return *this;
    };
    /// Multiplication by scalar
    Vec3 &operator*= ( const T &f ) {
      x*=f; y*=f; z*=f;
      return *this;
    };
    /// Multiplication by scalar
    Vec3 &operator/= ( const T &f ) {
      assert( f != T(0) );
      x/=f; y/=f; z/=f;
      return *this;
    };
    /// Multiplication by scalar
    Vec3 operator* ( const T &s ) const {
      return Vec3( x*s, y*s, z*s );
    };
    /// Multiplication by scalar
    Vec3 operator/ ( const T &s ) const {
      assert( s != T(0) );
      return Vec3( x/s, y/s, z/s );
    };
    /// Dot product
    T operator*( const Vec3 &v ) const {
      return x*v.x + y*v.y + z*v.z;
    };
    /// Length of the vector
    double length() const {
      return sqrt( x*x + y*y + z*z );
    }
    /// Normalize the vector
    Vec3 &normalize() {
      T len = sqrt(x*x + y*y + z*z);
      if (len>T(0)) {
	x /= len; y/= len; z/= len;
      };
      return *this;
    };
    /// Minimum component
    T min() const {
      return std::min( x,std::min(y,z) );
    }
    /// Maximum component
    T max() const {
      return std::max( x,std::max(y,z) );
    }
    /// Cast the vector to a T* used by OpenGL
    operator T*() {
      return &x;
    };
    /// Cast the vector to a T* used by OpenGL
    operator const T*() const {
      return &x;
    };
    /// Comparison
    bool operator== ( const Vec3& v ) const {
      return x==v.x && y==v.y && z==v.z;
    };
  };
  
  
  /// Vertex in 4D space
  /** \ingroup mathlinal */
  template<class T>
  struct Vec4 {
    /// Coordinates
    T x,y,z,w;
    /// Default constructor
    Vec4() {
      x=T(0); y=T(0); z=T(0); w=T(1);
    };
    /// Construct by coordinates
    Vec4( T xx, T yy, T zz, T ww ) {
      x=xx; y=yy; z=zz; w=ww;
    };
    /// Construct by float pointer
    Vec4( const T *pv ) {
      assert( pv );
      x=*pv; y=*(pv+1); z=*(pv+2); w=*(pv+3);
    };
    /// Construct by 3D-Vector. w will be set to 1.0.
    Vec4( const Vec3<T> &v ) {
      x=v.x; y=v.y; z=v.z; w=T(1);
    };
    /// Difference of vectors
    Vec4 operator- ( const Vec4&v ) const {
      return Vec4( x-v.x, y-v.y, z-v.z, w-v.w);
    };
    /// Sum of vectors
    Vec4 operator+ ( const Vec4&v ) const {
      return Vec4( x+v.x, y+v.y, z+v.z, w+v.w );
    };
    /// Sum of vectors
    Vec4 &operator+= ( const Vec4&v ) {
      x+=v.x; y+=v.y; z+=v.z; w+=v.w;
      return *this;
    };
    /// Multiplication by scalar
    Vec4 &operator*= ( T f ) {
      x*=f; y*=f; z*=f; w*= f;
      return *this;
    };
    /// Multiplication by scalar
    Vec4 operator* ( const T s ) const {
      return Vec4( x*s, y*s, z*s, w*s );
    };
    /// Dot product
    T operator*( const Vec4 &v ) const {
      return x*v.x + y*v.y + z*v.z + w*v.w;
    };
    /// Length of the vector (in 3-space)
    double length() const {
      return sqrt( x*x + y*y + z*z );
    }
    /// Normalize the vector
    Vec4 &normalize() {
      double len = length();
      if (len>T(0)) {
	x /= len; y/= len; z/= len; w/= len;
      };
      return *this;
    };
    /// Minimum component
    T min() const {
      return std::min( std::min(x,y),std::min(z,w) );
    }
    /// Maximum component
    T max() const {
      return std::max( std::max(x,y),std::max(z,w) );
    }
    /// Cast the vector to a T* used by OpenGL
    operator T*() {
      return &x;
    };
    /// Cast the vector to a T* used by OpenGL
    operator const T*() const {
      return &x;
    };
    /// Comparison
    bool operator== ( const Vec4& v ) const {
      return x==v.x && y==v.y && z==v.z && w==v.w;
    };
  };
  
  

  /// Four by four transformation matrix
  /** \note BEWARE: For some historic and bloody stupid reason, 
      the names of the entries in the matrix are misleading.
      _23 actually means the second element in the third row, and
      <b>NOT</b> the third element in the second row, as one would
      expect. I hope that I will have time to change it to normal
      at some time. In order to avoid further pitfalls, I have
      made the elements private and restricted access to the
      access operator[], which works as expected.
      \ingroup mathlinal */
  template<class T>
  struct Mat44 
  {
    /// Constructor
    Mat44();

    /// Get a single row
    T* operator[] ( int nRow );
    /// Get a single row (constant)
    const T* operator[] ( int nRow ) const;
    /// Get pointer to first entry
    operator T*();
    /// Get const pointer to first entry
    operator const T*() const;

    /// Get inverse matrix
    Mat44 inverse() const;
    /// Get simple inverse matrix
    /** This function assumes that matrix has a fourth row [0 0 0 1]
	much faster than full inverse.
     */
    Mat44 inverse_simple() const;
    /// Get normal transformation
    Mat44 normalTransform() const;


    /// Transform a vertex
    Vec4<T> operator* ( const Vec4<T>& ) const;
    /// Multiply by another matrix, return new one.
    /** \note Using this function probably results in inefficient code, so dont. */
    Mat44 operator* ( const Mat44 & ) const;

    /// Set matrix to identity
    void setIdentity( void );
    /// Transpose matrix
    void transpose();

    /// Pre multiplication of this matrix by another
    void preMultiply( const Mat44 &M );
    /// Post multiplication of this matrix by another
    void operator*=( const Mat44 &M );

    /// Pre-multiply matrix by rotation around X-Axis
    void rotateX( T fAngle );
    /// Pre-multiply matrix by rotation around Y-Axis
    void rotateY( T fAngle );
    /// Pre-multiply matrix by rotation around Z-Axis
    void rotateZ( T fAngle );
    /// Pre-multiply matrix by rotation around arbitrary axis
    void rotate( const Vec3<T>& vAxis, T fAngle );

    /// Pre-multiply matrix by translation
    void translate( T fX, T fY, T fZ );
    /// Pre-multiply matrix by scaling
    void scale( T fSX, T fSY, T fSZ );

    /// Set translation components of matrix to zero.
    void zeroTranslation();


  private:
    // Matrix data. Private because of stupid naming convention, see note above.
    T _11,_21,_31,_41;
    T _12,_22,_32,_42;
    T _13,_23,_33,_43;
    T _14,_24,_34,_44;
  };


  /// 2x2 Matrix template class
  /** \ingroup mathlinal */
  template<class T> struct Mat22
  {
    /// Constructor 
    Mat22() {
      memset( this,0,sizeof(Mat22<T>));
    }
    /// Transform a vertex
    Vec2<T> operator* ( const Vec2<T>&v ) const {
      return Vec2<T> ( _11*v.x+ _12*v.y,
		       _21*v.x+ _22*v.y );
    }

    /// Set matrix to identity
    void setIdentity( void ) {
      memset( this,0,sizeof(Mat22<T>) );
      _11=_22=T(1);
    }

    /// Transpose matrix
    void transpose() {
      std::swap( _21,_12 );
    }

    /// Matrix multiplication
    void preMultiply( const Mat22<T> &M ) {
      T n11 = M._11*_11 + M._12*_21;
      T n12 = M._11*_12 + M._12*_22;
      T n21 = M._21*_11 + M._22*_21;
      T n22 = M._21*_12 + M._22*_22;
      _11 = n11; _12 = n12; _21 = n21; _22 = n22;
    }
    Mat22<T> operator*( const Mat22<T> &M ) {
      Mat22<T> r;
      r._11 = _11*M._11 + _12*M._21;
      r._12 = _11*M._12 + _12*M._22;
      r._21 = _21*M._11 + _22*M._21;
      r._22 = _21*M._12 + _22*M._22;
      return r;
    }

    // Matrix data
    T _11,_12;
    T _21,_22;
  };



  /// 3x3 Matrix template class
  /** \ingroup mathlinal */
  template<class T> struct Mat33
  {
    /// Constructor 
    Mat33() {
      memset( this,0,sizeof(Mat33<T>));
    }
    /// Transform a vertex
    Vec3<T> operator* ( const Vec3<T>&v ) const {
      return Vec3<T> ( _11*v.x+ _12*v.y+ _13*v.z,
		       _21*v.x+ _22*v.y+ _23*v.z,
		       _31*v.x+ _32*v.y+ _33*v.z );
    }

    /// Set matrix to identity
    void setIdentity( void ) {
      memset( this,0,sizeof(Mat33<T>) );
      _11=_22=_33=T(1);
    }

    /// Transpose matrix
    void transpose() {
      swap( _21,_12 ); swap( _31,_13 ); swap( _32,_23 );
    }

    /// Matrix multiplication
    void preMultiply( const Mat33<T> &M ) {
      T n11 = M._11*_11 + M._12*_21 + M._13*_31;
      T n12 = M._11*_12 + M._12*_22 + M._13*_32;
      T n13 = M._11*_13 + M._12*_23 + M._13*_33;

      T n21 = M._21*_11 + M._22*_21 + M._23*_31;
      T n22 = M._21*_12 + M._22*_22 + M._23*_32;
      T n23 = M._21*_13 + M._22*_23 + M._23*_33;

      T n31 = M._31*_11 + M._32*_21 + M._33*_31;
      T n32 = M._31*_12 + M._32*_22 + M._33*_32;
      T n33 = M._31*_13 + M._32*_23 + M._33*_33;

      _11 = n11; _12 = n12; _13 = n13;
      _21 = n21; _22 = n22; _23 = n23;
      _31 = n31; _32 = n32; _33 = n33;
    }

    /// Matrix multiplication
    void preMultiply( const Mat22<T> &M ) {
      T n11 = M._11*_11 + M._12*_21;
      T n12 = M._11*_12 + M._12*_22;
      T n13 = M._11*_13 + M._12*_23 + _33;

      T n21 = M._21*_11 + M._22*_21;
      T n22 = M._21*_12 + M._22*_22;
      T n23 = M._21*_13 + M._22*_23 + _33;

      T n31 = _31;
      T n32 = _32;
      T n33 = _33;

      _11 = n11; _12 = n12; _13 = n13;
      _21 = n21; _22 = n22; _23 = n23;
      _31 = n31; _32 = n32; _33 = n33;
    }

    /// Pre-multiply matrix by translation
    void translate( T fX, T fY ) {
      Mat33<T> M;
      M.setIdentity();
      M._13 = fX;
      M._23 = fY;
      M._33 = T(1);
      preMultiply(M);
    }
    /// Pre-multiply matrix by scaling
    void scale( T fSX, T fSY ) {
      Mat33<T> M;
      M._13 = fSX;
      M._23 = fSY;
      M._33 = T(1);
      preMultiply(M);
    }

    // Matrix data
    T _11,_12,_13;
    T _21,_22,_23;
    T _31,_32,_33;
  };


  /// 2D Vector with double entries
  /** \ingroup mathlinal */
  typedef Vec2<double> Vec2d;
  /// 2D Vector with float entries
  /** \ingroup mathlinal */
  typedef Vec2<float> Vec2f;
  /// 2D Vector with integer entries
  /** \ingroup mathlinal */
  typedef Vec2<int> Vec2i;

  /// 3D Vector with double entries
  /** \ingroup mathlinal */
  typedef Vec3<double> Vec3d;
  /// 3D Vector with float entries
  /** \ingroup mathlinal */
  typedef Vec3<float> Vec3f;
  /// 3D Vector with integer entries
  /** \ingroup mathlinal */
  typedef Vec3<int> Vec3i;

  /// 4D Vector with double entries
  /** \ingroup mathlinal */
  typedef Vec4<double> Vec4d;
  /// 4D Vector with float entries
  /** \ingroup mathlinal */
  typedef Vec4<float> Vec4f;
  /// 4D Vector with integer entries
  /** \ingroup mathlinal */
  typedef Vec4<int> Vec4i;

  /// 4x4 Matrix with double entries
  /** \ingroup mathlinal */
  typedef Mat44<double> Mat44d;
  /// 4x4 Matrix with float entries
  /** \ingroup mathlinal */
  typedef Mat44<float> Mat44f;

  /// 3x3 Matrix with double entries
  /** \ingroup mathlinal */
  typedef Mat33<double> Mat33d;
  /// 3x3 Matrix with float entries
  /** \ingroup mathlinal */
  typedef Mat33<float> Mat33f;

  /// 2x2 Matrix with double entries
  /** \ingroup mathlinal */
  typedef Mat22<double> Mat22d;
  /// 2x2 Matrix with float entries
  /** \ingroup mathlinal */
  typedef Mat22<float> Mat22f;




  /*****************************************
   *** VECTOR CONVERSIONS                ***
   *****************************************/
  // Parallel projection onto xy plane
  inline Vec2d v3p( const Vec3d &v ) {
    return Vec2d( v.x, v.y );
  }
  // Extension by setting z to zero
  inline Vec3d v2x( const Vec2d &v ) {
    return Vec3d( v.x, v.y, 0.0 );
  }


  /*****************************************
   *** MATRIX ALGEBRA IMPLEMENTATION     ***
   *****************************************/

  // Calculate the cross product
  template<class T>
  Vec3<T> cross( const Vec3<T> &v1, const Vec3<T> &v2 )
  {
    return Vec3<T>( v1.y*v2.z - v1.z*v2.y,
                    v1.z*v2.x - v1.x*v2.z,
                    v1.x*v2.y - v1.y*v2.x );
  }

  // Construct by 4-Vector (homogenous divide)
  template<class T>
    Vec3<T>::Vec3( const Vec4<T>& v ) {
    x=v.x/v.w; y=v.y/v.w; z=v.z/v.w;
  }



  // -----------------------------------------------------------------------
  template<class T>
  Mat44<T>::Mat44()
  {
    setIdentity();
  }
  

  // -----------------------------------------------------------------------
  template<class T>
  void Mat44<T>::setIdentity()
  {
    memset( &_11, 0, sizeof(T)*16 );
    _11 = 1.0f;
    _22 = 1.0f;
    _33 = 1.0f;
    _44 = 1.0f;
  }

  // -----------------------------------------------------------------------
  template<class T>
  void Mat44<T>::zeroTranslation()
  {
    _41 = _42 = _43 = _14 = _24 = _34 = 0;
  }


  // -----------------------------------------------------------------------
  template<class T>
  T* Mat44<T>::operator[] (int nRow)
  {
    assert( nRow>=0 && nRow<4 );
    return (&_11) + nRow*4;
  }



  // -----------------------------------------------------------------------
  template<class T>
  const T* Mat44<T>::operator[] (int nRow) const
  {
    assert( nRow>=0 && nRow<4 );
    return (&_11) + nRow*4;
  }



  // -----------------------------------------------------------------------
  template<class T>
  Mat44<T>::operator T*()
  {
    return &_11;
  }

  template<class T>
  Mat44<T>::operator const T*() const
  {
    return &_11;
  }



  // -----------------------------------------------------------------------
  template<class T>
  void Mat44<T>::transpose()
  {
    swap( _12, _21 );
    swap( _13, _31 );
    swap( _14, _41 );

    swap( _23, _32 );
    swap( _24, _42 );

    swap( _34, _43 );
  }



  // -----------------------------------------------------------------------
  template<class T>
  void Mat44<T>::preMultiply( const Mat44<T> &M ) 
  {
    int i, j, k;
    Mat44<T> MTemp( *this );
    memset( &_11, 0, sizeof( T ) * 16 );
    for( i=0; i<4; i++ ) {
      const T* pfLine = M[i];
      for( j=0; j<4; j++ ) {
        T &entry = (*this)[i][j];
        for( k=0; k<4; k++ ) {
          entry += pfLine[k] * MTemp[k][j];
        }
      }
    }
  }

  // -----------------------------------------------------------------------
  template<class T>
  void Mat44<T>::operator*=( const Mat44<T> &M )
  {
    int i, j, k;
    Mat44<T> MTemp( *this );
    memset( &_11, 0, sizeof( T ) * 16 );
    for( i=0; i<4; i++ ) {
      const T* pfLine = MTemp[i];
      for( j=0; j<4; j++ ) {
        T &entry = (*this)[i][j];
        for( k=0; k<4; k++ ) {
          entry += pfLine[k] * M[k][j];
        }
      }
    }
  }


  // -----------------------------------------------------------------------
  template<class T>
  Mat44<T> Mat44<T>::operator*( const Mat44<T> &M ) const
  {
    Mat44<T> ret(*this);
    ret *= M;
    return ret;
  }

  // -----------------------------------------------------------------------
  template<class T>
  Vec4<T> Mat44<T>::operator*(const Vec4<T> &v) const
  {
    Vec4<T> ret;
    const T* f = (*this)[0];
    ret.x = f[0]*v.x + f[1]*v.y + f[2]*v.z + f[3]*v.w;
    f = (*this)[1];
    ret.y = f[0]*v.x + f[1]*v.y + f[2]*v.z + f[3]*v.w;
    f = (*this)[2];
    ret.z = f[0]*v.x + f[1]*v.y + f[2]*v.z + f[3]*v.w;
    f = (*this)[3];
    ret.w = f[0]*v.x + f[1]*v.y + f[2]*v.z + f[3]*v.w;
    return ret;
  }


  // -----------------------------------------------------------------------
  template<class T>
  Mat44<T> Mat44<T>::inverse_simple( void ) const
  {
    T fInvDeterm = 1.0f / ( _11 * ( _22 * _33 - _23 * _32 ) -
                            _12 * ( _21 * _33 - _23 * _31 ) +
                            _13 * ( _21 * _32 - _22 * _31 ) );
  
    T mx11 =  fInvDeterm * ( _22 * _33 - _23 * _32 );
    T mx12 = -fInvDeterm * ( _12 * _33 - _13 * _32 );
    T mx13 =  fInvDeterm * ( _12 * _23 - _13 * _22 );
    T mx21 = -fInvDeterm * ( _21 * _33 - _23 * _31 );
    T mx22 =  fInvDeterm * ( _11 * _33 - _13 * _31 );
    T mx23 = -fInvDeterm * ( _11 * _23 - _13 * _21 );
    T mx31 =  fInvDeterm * ( _21 * _32 - _22 * _31 );
    T mx32 = -fInvDeterm * ( _11 * _32 - _12 * _31 );
    T mx33 =  fInvDeterm * ( _11 * _22 - _12 * _21 );
    T mx41 = -( _41 * mx11 + _42 * mx21 + _43 * mx31 );
    T mx42 = -( _41 * mx12 + _42 * mx22 + _43 * mx32 );
    T mx43 = -( _41 * mx13 + _42 * mx23 + _43 * mx33 );
	
    Mat44<T> ret;
    ret._11 = mx11; ret._12 = mx12; ret._13 = mx13; ret._14 = 0.0f;
    ret._21 = mx21; ret._22 = mx22; ret._23 = mx23; ret._24 = 0.0f;
    ret._31 = mx31; ret._32 = mx32; ret._33 = mx33; ret._34 = 0.0f;
    ret._41 = mx41; ret._42 = mx42; ret._43 = mx43; ret._44 = 1.0f;
    return ret;
  }


  template<class T>
  Mat44<T> Mat44<T>::normalTransform( void ) const
  {
    Mat44<T> dest( *this );
    dest._14 = 0.0;
    dest._24 = 0.0;
    dest._34 = 0.0;
    dest._41 = 0.0;
    dest._42 = 0.0;
    dest._43 = 0.0;
    dest.transpose();
    return dest.inverse_simple();
  }



  /********************************************************************
   *
   * input:
   * b - pointer to array of 16 single floats (source matrix)
   * output:
   * a - pointer to array of 16 single floats (invert matrix)
   *
   ********************************************************************/
  template<class T>
  void Invert( const T *pb, T *pa)
  {
    T a[4][4];
    T b[4][4];
    memcpy( a, pa, sizeof( T ) * 16);
    memcpy( b, pb, sizeof( T ) * 16);
	
    long indxc[4], indxr[4], ipiv[4];
    long i, icol=0, irow=0, j, ir, ic;
    float big, dum, pivinv, temp, bb;
    ipiv[0] = -1; ipiv[1] = -1; ipiv[2] = -1; ipiv[3] = -1;
    a[0][0] = b[0][0];
    a[1][0] = b[1][0];
    a[2][0] = b[2][0];
    a[3][0] = b[3][0];
    a[0][1] = b[0][1];
    a[1][1] = b[1][1];
    a[2][1] = b[2][1];
    a[3][1] = b[3][1];
    a[0][2] = b[0][2];
    a[1][2] = b[1][2];
    a[2][2] = b[2][2];
    a[3][2] = b[3][2];
    a[0][3] = b[0][3];
    a[1][3] = b[1][3];
    a[2][3] = b[2][3];
    a[3][3] = b[3][3];
    for (i = 0; i < 4; i++) {
      big = 0.0f;
      for (j = 0; j < 4; j++) {
        if (ipiv[j] != 0) {
          if (ipiv[0] == -1) {
            if ((bb = ( float) fabs(a[j][0])) > big) {
              big = bb; irow = j; icol = 0;
            }
          }
          else if (ipiv[0] > 0) {
            memcpy( pa, a, 16*sizeof( T ));
            return;
          }

          if (ipiv[1] == -1) {
            if ((bb = ( float) fabs(( float) a[j][1])) > big) {
              big = bb; irow = j; icol = 1;
            }
          }
          else if (ipiv[1] > 0) {
            memcpy( pa, a, 16*sizeof( T ));
            return;
          }

          if (ipiv[2] == -1) {
            if ((bb = ( float) fabs(( float) a[j][2])) > big) {
              big = bb; irow = j; icol = 2;
            }
          }
          else if (ipiv[2] > 0) {
            memcpy( pa, a, 16*sizeof( T ));
            return;
          }

          if (ipiv[3] == -1) {
            if ((bb = ( float) fabs(( float) a[j][3])) > big) {
              big = bb; irow = j; icol = 3;
            }
          }
          else if (ipiv[3] > 0) {
            memcpy( pa, a, 16*sizeof( T ));
            return;
          }
        }
      }
      ++(ipiv[icol]);

      if (irow != icol) {
        temp = a[irow][0]; a[irow][0] = a[icol][0]; a[icol][0] = temp;
        temp = a[irow][1]; a[irow][1] = a[icol][1]; a[icol][1] = temp;
        temp = a[irow][2]; a[irow][2] = a[icol][2]; a[icol][2] = temp;
        temp = a[irow][3]; a[irow][3] = a[icol][3]; a[icol][3] = temp;
      }

      indxr[i] = irow; indxc[i] = icol;
      if (a[icol][icol] == 0.0) {
        memcpy( pa, a, 16*sizeof( T ));
        return;
      }

      pivinv = 1.0f / a[icol][icol];
      a[icol][icol] = 1.0f;
      a[icol][0] *= pivinv;
      a[icol][1] *= pivinv;
      a[icol][2] *= pivinv;
      a[icol][3] *= pivinv;

      if (icol != 0) {
        dum = a[0][icol];
        a[0][icol] = 0.0f;
        a[0][0] -= a[icol][0] * dum;
        a[0][1] -= a[icol][1] * dum;
        a[0][2] -= a[icol][2] * dum; 
        a[0][3] -= a[icol][3] * dum;
      }

      if (icol != 1) {
        dum = a[1][icol]; 
        a[1][icol] = 0.0f; 
        a[1][0] -= a[icol][0] * dum; 
        a[1][1] -= a[icol][1] * dum; 
        a[1][2] -= a[icol][2] * dum; 
        a[1][3] -= a[icol][3] * dum; 
      }

      if (icol != 2) {
        dum = a[2][icol];
        a[2][icol] = 0.0f;
        a[2][0] -= a[icol][0] * dum;
        a[2][1] -= a[icol][1] * dum;
        a[2][2] -= a[icol][2] * dum;
        a[2][3] -= a[icol][3] * dum;
      }

      if (icol != 3) {
        dum = a[3][icol];
        a[3][icol] = 0.0f;
        a[3][0] -= a[icol][0] * dum;
        a[3][1] -= a[icol][1] * dum;
        a[3][2] -= a[icol][2] * dum;
        a[3][3] -= a[icol][3] * dum;
      }
    }

    if (indxr[3] != indxc[3]) {
      ir = indxr[3]; ic = indxc[3];
      temp = a[0][ir]; a[0][ir] = a[0][ic]; a[0][ic] = temp;
      temp = a[1][ir]; a[1][ir] = a[1][ic]; a[1][ic] = temp;
      temp = a[2][ir]; a[2][ir] = a[2][ic]; a[2][ic] = temp; 
      temp = a[3][ir]; a[3][ir] = a[3][ic]; a[3][ic] = temp;
    }

    if (indxr[2] != indxc[2]) {
      ir = indxr[2]; ic = indxc[2];
      temp = a[0][ir]; a[0][ir] = a[0][ic]; a[0][ic] = temp;
      temp = a[1][ir]; a[1][ir] = a[1][ic]; a[1][ic] = temp;
      temp = a[2][ir]; a[2][ir] = a[2][ic]; a[2][ic] = temp;
      temp = a[3][ir]; a[3][ir] = a[3][ic]; a[3][ic] = temp;
    }
	
    if (indxr[1] != indxc[1]) {
      ir = indxr[1]; ic = indxc[1];
      temp = a[0][ir]; a[0][ir] = a[0][ic]; a[0][ic] = temp;
      temp = a[1][ir]; a[1][ir] = a[1][ic]; a[1][ic] = temp;
      temp = a[2][ir]; a[2][ir] = a[2][ic]; a[2][ic] = temp;
      temp = a[3][ir]; a[3][ir] = a[3][ic]; a[3][ic] = temp;
    }
	
    if (indxr[0] != indxc[0]) {
      ir = indxr[0]; ic = indxc[0];
      temp = a[0][ir]; a[0][ir] = a[0][ic]; a[0][ic] = temp;
      temp = a[1][ir]; a[1][ir] = a[1][ic]; a[1][ic] = temp;
      temp = a[2][ir]; a[2][ir] = a[2][ic]; a[2][ic] = temp;
      temp = a[3][ir]; a[3][ir] = a[3][ic]; a[3][ic] = temp;
    }

    memcpy( pa, a, 16*sizeof( T ));
  } 



  // -----------------------------------------------------------------------
  template<class T>
  Mat44<T> Mat44<T>::inverse( void ) const
  {
    Mat44<T> dest;
    Invert<T> ( &_11, &dest._11 );
    return dest;
  }






  /****************************
   *** VIEWS AND PROJECTIONS ***
   *****************************/

  // -----------------------------------------------------------------------
  template<class T>
  void Mat44<T>::rotateX( T fAngle )
  {
    Mat44<T> mRotate;
    mRotate._22 = mRotate._33 = (T)cos( fAngle );
    mRotate._23 = (T)sin( fAngle );
    mRotate._32 = -mRotate._23;
    preMultiply( mRotate );
  }

  // -----------------------------------------------------------------------
  template<class T>
  void Mat44<T>::rotateY( T fAngle )
  {
    Mat44<T> mRotate;
    mRotate._11 = mRotate._33 = (T)cos( fAngle );
    mRotate._13 = (T)sin( -fAngle );
    mRotate._31 = - mRotate._13;
    preMultiply( mRotate );
  }

  // -----------------------------------------------------------------------
  template<class T>
  void Mat44<T>::rotateZ( T fAngle )
  {
    Mat44<T> mRotate;
    mRotate._11 = mRotate._22 = (T)cos( fAngle );
    mRotate._12 = (T)sin( fAngle );
    mRotate._21 = - mRotate._12;
    preMultiply( mRotate );
  }

  // -----------------------------------------------------------------------
  template<class T>
  void Mat44<T>::rotate( const Vec3<T> &va, T fAngle )
  {
    T fCos = (T)cos( fAngle );
    T fSin = (T)sin( fAngle );
    T fX, fY, fZ;
    Mat44<T> mRotate;

    //
    // Normalize the vector
    //
    Vec3<T> vAxis( va );
    vAxis.normalize();
    fX = vAxis.x;
    fY = vAxis.y;
    fZ = vAxis.z;

    mRotate._11 = (fX * fX) * (1.0f - fCos) + fCos;
    mRotate._12 = (fX * fY) * (1.0f - fCos) - (fZ * fSin);
    mRotate._13 = (fX * fZ) * (1.0f - fCos) + (fY * fSin);

    mRotate._21 = (fY * fX) * (1.0f - fCos) + (fZ * fSin);
    mRotate._22 = (fY * fY) * (1.0f - fCos) + fCos ;
    mRotate._23 = (fY * fZ) * (1.0f - fCos) - (fX * fSin);
  
    mRotate._31 = (fZ * fX) * (1.0f - fCos) - (fY * fSin);
    mRotate._32 = (fZ * fY) * (1.0f - fCos) + (fX * fSin);
    mRotate._33 = (fZ * fZ) * (1.0f - fCos) + fCos;
  
    mRotate._14 = mRotate._24 = mRotate._34 = 0.0f;
    mRotate._41 = mRotate._42 = mRotate._43 = 0.0f;
    mRotate._44 = 1.0f;
  
    preMultiply( mRotate );
  }

  // -----------------------------------------------------------------------
  template<class T>
  void Mat44<T>::translate( T fX, T fY, T fZ )
  {
    Mat44<T> mTranslate;
    mTranslate._41 = fX;
    mTranslate._42 = fY;
    mTranslate._43 = fZ;
    preMultiply( mTranslate );
  }

  // -----------------------------------------------------------------------
  template<class T>
  void Mat44<T>::scale( T fSX, T fSY, T fSZ )
  {
    Mat44<T> mScale;
    mScale._11 = fSX;
    mScale._22 = fSY;
    mScale._33 = fSZ;
    preMultiply( mScale );
  }



  typedef Mat44<float> Mat44f;
}


#include "linalg3d_io.h"

#endif
