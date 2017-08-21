/** \file debug.cpp

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

#include "debug.h"

#include <float.h>
#include <string.h>

#include <algorithm>
#include <iostream>

#include <stddef.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <dirent.h>
#include <fnmatch.h>
#include <unistd.h>

#define _chdir chdir
#define _getcwd getcwd
#define _finddata_t finddata_t
#define _findfirst findfirst
#define _findclose findclose

using namespace coco;
using namespace std;

const double coco::DOUBLE_MAX = DBL_MAX;
const double coco::DOUBLE_MIN = DBL_MIN;
const double coco::FLOAT_MAX = FLT_MAX;
const double coco::FLOAT_MIN = FLT_MIN;


namespace coco {

  static string __resourceDir = "../resource/";
  static string __debugDir = "./";


  // Get resource directory
  string resourceDir()
  {
    char *pres = getenv( "RESOURCE" );
    if (pres != NULL) {
      static char buf[200];
      strcpy( buf, pres );
      int l = strlen(buf);
      if (l>0 && buf[l-1] != '/') {
	strcat( buf, "/" );
      }
      return buf;
    }
    return __resourceDir;
  }

  // Set resource directory
  void setResourceDir( const string &dir )
  {
    __resourceDir = dir;
    if (__resourceDir.length()>0 && __resourceDir[ __resourceDir.length()-1 ] != '/') {
      __resourceDir += "/";
    }
  }


  // This function returns the path to the debug directory
  string debugDir()
  {
    char *pres = getenv( "DEBUGDIR" );
    if (pres != NULL) {
      static char buf[200];
      strcpy( buf, pres );
      int l = strlen(buf);
      if (l>0 && buf[l-1] != '/') {
	strcat( buf, "/" );
      }
      return buf;
    }
    return __debugDir;
  }

  // This function sets the path to the debug directory
  void setDebugDir( const string &dir )
  {
    __debugDir = dir;
    if (__debugDir.length()>0 && __debugDir[ __debugDir.length()-1 ] != '/') {
      __debugDir += "/";
    }
  }



  /// Breaks a path name into the base directory and the file name
  void breakupFileName( const string &path, string &dir, string &file )
  {
    size_t n = path.rfind( '/' );
    if (n != string::npos) {
      file = path.substr(n+1,path.length()-n-1);
      dir = path.substr(0,n+1);
    }
  }

  /// Test for existence of a file
  bool fexist( const char *filename )
  {
    // following code has problems (no idea why).
    // better not use it.
    assert( false );

    struct stat buffer;
    // return value of zero means success
    if ( 0 == stat( filename, &buffer ) ) {
      return true;
    }
    return false;
  }

  /// Upper case a string
  void toUpper( string &str )
  {
    std::transform(str.begin(), str.end(),str.begin(), ::toupper);
  }

  /// Breaks a path name into the base directory and the file name
  void breakupFileName( const string &path, string &dir, string &file, string &extension )
  {
    size_t n = path.rfind( '/' );
    if (n != string::npos) {
      file = path.substr(n+1,path.length()-n-1);
      dir = path.substr(0,n+1);
    }
    else {
      file = path;
      dir = "";
    }
    
    n = file.find( '.' );
    if (n != string::npos) {
      extension = file.substr(n+1,file.length()-n-1);
      file = file.substr(0,n);
    }
    else {
      extension = "";
    }
  }




  static ostream *__pDebugStream = &cout;
  static ostream *__pErrorStream = &cerr;

  // Get current debug stream
  std::ostream &debugStream()
  {
    return *__pDebugStream;
  }

  // Set new debug stream
  void setDebugStream( std::ostream &o )
  {
    __pDebugStream = &o;
  }

  // Write prefix to debug stream output
  void debugStreamMsg() {
  }

  // Get current error stream
  std::ostream &errorStream()
  {
    return *__pErrorStream;
  }

  // Set new error stream
  void setErrorStream( std::ostream &o )
  {
    __pErrorStream = &o;
  }

  // Write prefix to error stream output
  void errorStreamMsg()
  {
#ifdef HAVE_MPI
    if (g_MPI != NULL) {
      (*__pErrorStream) << "FATAL(rank " << g_MPI->rank() << "): ";
    }
    else {
#else
      (*__pErrorStream) << "FATAL: ";
#endif

#ifdef HAVE_MPI
    }
#endif
  }


  /// Current trace level
  int g_traceLevel = 0;

  /// Set trace level.
  /** Only traces of level smaller than the current trace level are printed.
   */
  void setTraceLevel( int nLevel ) {
    g_traceLevel = nLevel;
    TRACE5( "Setting new trace level: " << nLevel << endl );
  }

  /// Get current trace level
  int traceLevel() {
      return g_traceLevel;
  }   


char *readFile( const string &name, int *nLength )
{
  ifstream ifile( name.c_str() );
  if (!ifile.is_open()) {
    ERROR( "Error opening file " << name << " !" << endl );
    assert( false );
  }

  // get file size using buffer's members
  TRACE2( "Reading file " << name << "," );
  filebuf* pbuf=ifile.rdbuf();
  size_t len = pbuf->pubseekoff (0,ios::end,ios::in) * 2;
  pbuf->pubseekpos (0,ios::in);
  TRACE2( " size " << len << endl );

  if (nLength != NULL) {
    *nLength = len;
  }
  char *buf = new char[len*2];
  buf[0] = char(0);

  char endline[2] = {char(10),char(0)};
  while (!ifile.eof()) {
    unsigned len2 = 10000;
    char buf2[len2+1];
    ifile.getline( buf2, len2 );
    if (strlen(buf) + strlen(buf2) < len-10) {
      strcat( buf, buf2 );
      strcat( buf, endline );
    }
  }

  strcat( buf, "***END***" );
  strcat( buf, endline );
  return buf;
}


char *readBinary( const string &name, int *nLength )
{
  ifstream ifile( name.c_str() );
  if (!ifile.is_open()) {
    ERROR( "Error opening file " << name << " !" << endl );
    assert( false );
  }

  // get file size using buffer's members
  TRACE2( "Reading file " << name << "," );
  filebuf* pbuf=ifile.rdbuf();
  int len = pbuf->pubseekoff(0,ios::end,ios::in);
  pbuf->pubseekpos (0,ios::in);
  TRACE2( " size " << len << endl );

  if (nLength != NULL) {
    *nLength = len;
  }
  char *buf = new char[len];
  pbuf->sgetn (buf,len);
  return buf;
}



/// Change current directory
bool		Directory::cd( const string &strNewDir )
{
  if (_chdir( strNewDir.c_str() ) != 0)
    {
      return false;
    }
  return true;
}


/// Create a directory
bool    Directory::create( const std::string &strNewDir )
{
  // Create directory tree from the beginning
  size_t p=0;
  unsigned flags = S_IRWXU | S_IXGRP | S_IRGRP | S_IROTH | S_IXOTH;

  while ( (p=strNewDir.find( "/", p+1 )) != string::npos ) {
    string strCurrentDir = strNewDir.substr( 0,p );
    mkdir( strCurrentDir.c_str(), flags );
  }

  if ( mkdir( strNewDir.c_str(), flags ) ) {
    return false;
  }

  return true;
}

/// Get current directory
string	Directory::current()
{
  char buf[5000];
  _getcwd( buf, 4999 );
  return string(buf);
}

/// Get files in current directory matching the given wildcards
vector<string>	Directory::files( const string &strWildcard )
{
  vector<string> ret;
  
  DIR *dp;
  struct dirent *ep;

  dp = opendir ("./");
  if (dp != NULL) {
    while ( (ep = readdir (dp)) != NULL ) {
      if ( 0==fnmatch( strWildcard.c_str(), ep->d_name, FNM_PATHNAME )) {
	ret.push_back( ep->d_name );
      }
    }
    closedir (dp);
  }

  return ret;
}


class FileSysInitializer
{
public:
  char BaseDirectory[1000];
  
  FileSysInitializer()
  {
    _getcwd( BaseDirectory, 999 );
  };
} FileSysInitializer;


string Directory::base()
{
  string ret( std::string( FileSysInitializer.BaseDirectory ));
  return ret;
}


string Directory::getDelimiter()
{
  return string("/");
}


}

