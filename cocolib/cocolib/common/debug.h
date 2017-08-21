/** \file debug.h

    File imported from "common" lib, use if this library is not available.

    Implements some system dependent and debugging code.
    Ancient, but should work reasonably well.

    Copyright (C) 2001 Bastian Goldluecke,
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

#ifndef __DEBUG_H_STDEXT
#define __DEBUG_H_STDEXT

#include <assert.h>
#include <stdio.h>

#include <string>
#include <vector>
#include <fstream>

#include "../modules.h"
#include "../defs.h"

namespace coco
{

  /// Double maximum value
  /** \ingroup sysmisc */
  extern const double DOUBLE_MAX;
  /// Double minimum value
  /** \ingroup sysmisc */
  extern const double DOUBLE_MIN;

  /// Float maximum value
  /** \ingroup sysmisc */
  extern const double FLOAT_MAX;
  /// Float minimum value
  /** \ingroup sysmisc */
  extern const double FLOAT_MIN;

  /// This function returns the path to the resource directory.
  /** For more details, see the resource management group.
      \ingroup resource */
  std::string resourceDir();
  /// Set resource directory
  /** Call this function before any library objects are constructed.
      \ingroup resource */
  void setResourceDir( const std::string & );


  /// This function returns the path to the debug directory
  /** If library functions write debug information files, then those
      files are written to this directory. It should be set as the
      first instruction in main() using setDebugDir().
  */
  std::string debugDir();

  /// This function sets the path to the debug directory
  /** If library functions write debug information files, then those
      files are written to this directory. It should be set as the
      first instruction in main().
  */
  void setDebugDir( const std::string & );



  /// Get current debug stream
  /** All output written using the TRACE macro is appended to this stream.
      Overwrite the standard output stream using setDebugStream. That way,
      you can write all library output into a file.
      \ingroup debug
  */
  std::ostream &debugStream();
  /// Set new debug stream
  /** \ingroup sysdebug */
  void setDebugStream( std::ostream & );
  /// Write prefix to debug stream output
  /** For internal use.
      \ingroup sysdebug */
  void debugStreamMsg();

  /// Get current error stream
  /** All output written using the ERROR macro is appended to this stream.
      Overwrite the standard error stream using setErrorStream. That way,
      you can write all library output into a file.
      \ingroup sysdebug
  */
  std::ostream &errorStream();
  /// Set new error stream
  /** \ingroup sysdebug */
  void setErrorStream( std::ostream & );
  /// Write prefix to error stream output
  /** For internal use.
      \ingroup sysdebug */
  void errorStreamMsg();



  /// Set trace level.
  /** Only traces of level smaller than the current trace level are printed.
   */
  void setTraceLevel( int nLevel );

  /// Get current trace level
  int traceLevel();


  /// Get total memory in use by the current process
  unsigned long memory_usage();



  /****************************************************
                    FILE HANDLING TOOLS
  *****************************************************/

  /// Breaks a path name into the base directory and the file name
  void breakupFileName( const std::string &path, std::string &dir, std::string &file );
  /// Breaks a path name into the base directory, file name and extension
  void breakupFileName( const std::string &path, std::string &dir, std::string &file, std::string &extension );

  /// Test for existence of a file
  bool fexist( const char *filename );

  /// Helper function: Read a file into a char buffer
  /** Returns the buffer size in \a nLength.
      \note The buffer has to be freed with delete[] after use.
      \ingroup sysfile
  */
  char *readBinary( const std::string &name, int *nLength );


  // Check for endianness of the system
  inline bool endianness()
  {
    const int x = 1;
    return ((unsigned char *)&x)[0] ? false : true;
  }
  
  // Invert endian
  inline void invert_endianness( float* const buffer, const unsigned long size )
  {
    for (unsigned int *ptr = (unsigned int*)buffer+size; ptr>(unsigned int*)buffer; ) {
      const unsigned int val = *(--ptr);
      *ptr = (val>>24)|((val>>8)&0xff00)|((val<<8)&0xff0000)|(val<<24);
    }
  }



  /// Class maintaining information about the current directory
  /** \note All functions in this class are static, so there is no need
      to ever construct one (it is more like a namespace).
      A constructor is therefore not provided.
     \ingroup sysfile
   */
  class Directory
  {
    public:
    
    /// Change current directory
    static bool				cd( const std::string &strNewDir );
    /// Create a directory
    static bool                         create( const std::string &strNewDir );
    /// Get current directory
    static std::string			current();

    /// Get files in current directory matching the given wildcards
    static std::vector<std::string> files( const std::string &strWildcard );
    
    /// Get base directory process was started in
    static std::string			base();
    /// Get symbol between directory names in a path
    static std::string			getDelimiter();

  private:
    /// Constructor is private to prevent creation of objects.
    Directory() {};
  };



  /****************************************************
                    MISC TOOLS
  *****************************************************/

  /// Upper case a string
  void toUpper( std::string &str );
  /// Make RGB value
  inline unsigned int make_rgb(int r, int g, int b)
  {
    return (0xffu << 24) | ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff);
  }


};


/// Write output to debug output stream. The stream is automatically flushed afterwards.
/** \ingroup sysdebug */
#define TRACE(s) TRACE0(s)
#define TRACE0(s) if (coco::traceLevel() >= 0) {coco::debugStreamMsg(); coco::debugStream() << s; coco::debugStream().flush();}
#define TRACE1(s) if (coco::traceLevel() >= 1) {coco::debugStreamMsg(); coco::debugStream() << s; coco::debugStream().flush();}
#define TRACE2(s) if (coco::traceLevel() >= 2) {coco::debugStreamMsg(); coco::debugStream() << s; coco::debugStream().flush();}
#define TRACE3(s) if (coco::traceLevel() >= 3) {coco::debugStreamMsg(); coco::debugStream() << s; coco::debugStream().flush();}
#define TRACE4(s) if (coco::traceLevel() >= 4) {coco::debugStreamMsg(); coco::debugStream() << s; coco::debugStream().flush();}
#define TRACE5(s) if (coco::traceLevel() >= 5) {coco::debugStreamMsg(); coco::debugStream() << s; coco::debugStream().flush();}
#define TRACE6(s) if (coco::traceLevel() >= 6) {coco::debugStreamMsg(); coco::debugStream() << s; coco::debugStream().flush();}
#define TRACE7(s) if (coco::traceLevel() >= 7) {coco::debugStreamMsg(); coco::debugStream() << s; coco::debugStream().flush();}
#define TRACE8(s) if (coco::traceLevel() >= 8) {coco::debugStreamMsg(); coco::debugStream() << s; coco::debugStream().flush();}
// Final trace level to output just about everything
#define TRACE9(s) if (coco::traceLevel() >= 9) {coco::debugStreamMsg(); coco::debugStream() << s; coco::debugStream().flush();}

/// Write output to error output stream. The stream is automatically flushed afterwards.
/** \ingroup sysdebug */
#define ERROR(s) coco::errorStreamMsg(); coco::errorStream() << s; coco::errorStream().flush();


#endif //__DEBUG_H_STDEXT
