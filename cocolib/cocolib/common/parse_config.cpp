/** \file debug.cpp

    File imported from "common" lib, use if this library is not available.
    Uses "gov" namespace (Graphics-optics-vision, MPI).

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
#include <sstream>
#include <locale>

#include <algorithm>
#include <iostream>

#include "parse_config.h"
#include "debug.h"

using namespace std;

namespace coco {

  // read from file
  bool config::parse_config_file( const string &filename )
  {
    FILE *f = fopen( filename.c_str(), "r" );
    if ( f==NULL ) {
      ERROR( "could not read config file " << filename << endl );
      return false;
    }

    TRACE9( "parsing config " << filename << endl );
    char line[600];
    while (!feof( f )) {
      if ( fgets( line, 500, f ) == NULL ) {
	TRACE9( "eof during string read" << endl );
	break;
      }
      int L = strlen(line);
      if ( L>0 && line[L-1] == '\n' ) {
	line[L-1] = char(0);
      }

      char name[600];
      char value[600];
      int count = sscanf( line, "%s %s\n", &name, &value );
      if ( count == 0 ) {
	continue;
      }

      string svalue = "";
      if ( count > 1 && value != NULL ) {
	char *loc = strstr( line, value );
	if ( loc == NULL ) {
	  ERROR( "parsing error:  expected to find '" << value << "' in '" << line << "'." << endl );
	  continue;
	}
	svalue = loc;
      }

      if ( name != NULL ) {
	if ( name[0] == '#' ) {
	  continue;
	}

	string sname = name;
	if ( svalue != "" ) {
	  set_switch( sname, svalue );
	  TRACE9( "set switch " << sname << " to " << svalue << endl );
	}
	else {
	  add_string( sname );
	  TRACE9( "add string " << sname << endl );
	}
      }
    }

    TRACE9( "done." << endl );
    return true;
  }

  // read from command line
  bool config::parse_command_line( int argn, char **argv )
  {
    TRACE9( "PARSING COMMAND LINE" << endl );

    int pos = 1;
    while ( pos < argn ) {

      TRACE9( "  TESTING ARGUMENT " << pos << " : " << argv[pos] << endl );
      char *name = NULL;
      char *value = NULL;
      string svalue = "1";
      if ( strlen( argv[pos] ) > 0 ) {
	TRACE9( "    STRLEN ok" << endl );

	if ( argv[pos][0] == '-' ) {
	  // look for argument value
	  name = argv[pos] + 1;
	  TRACE9( "    ARGUMENT " << name << " found. " );
	  if ( pos+1 < argn ) {
	    value = argv[pos+1];
	    svalue = value;
	    TRACE9( " Value: " << svalue << endl );
	  }
	  else {
	    TRACE9( " Value undefined." << endl );
	  }

	  // set argument switch
	  string sname( name );
	  set_switch( sname, svalue );
	  pos ++;
	}
	else if ( argv[pos][0] == '+' ) {
	  // flag argument.
	  name = argv[pos] + 1;
	  TRACE9( "    ARGUMENT FLAG " << name << " set." << endl );
	  // set argument switch
	  string sname( name );
	  set_switch( sname, svalue );
	}
	else {
	  name = argv[pos];
	  string sname = name;
	  if ( sname != "" ) {
	    TRACE9( "    STRING " << sname );
	    add_string( argv[pos] );
	  }
	}

      }
      pos++;
    }

    return true;
  }


  // dump to file
  bool config::dump( const string &filename ) const
  {
    ofstream out( filename.c_str() );
    map<string, string>::const_iterator it = _switches.begin();
    while( it != _switches.end() ) {
      out << (*it).first << " " << (*it).second << endl;
      it++;
    }
    return true;
  }

  // add global string
  void config::add_string( const string &str )
  {
    _strings.push_back( str );
  }

  // set switch
  void config::set_switch( const string &name, const string &value )
  {
    // special cases
    if ( name == "config" ) {
      parse_config_file( value );
    }
    else if ( name == "tracelevel" ) {
      int level = atoi( value.c_str() );
      coco::setTraceLevel( atoi( value.c_str() ) );
      TRACE5( "     Setting trace level " << level << endl ); 
    }
    _switches[name] = value;
    _multi_switches.insert( pair<string,string> (name,value) );
  }


  // global string list
  const vector<string> &config::get_strings() const
  {
    return _strings;
  }


  // just test for switch presence
  bool config::get_switch( const string &name ) const
  {
    map<string,string>::const_iterator it = _switches.find( name );
    if ( it != _switches.end() ) {
      return true;
    }
    return false;
  }

  // single switch (return string)
  bool config::get_switch( const string &name, string &value ) const
  {
    map<string,string>::const_iterator it = _switches.find( name );
    if ( it != _switches.end() ) {
      value = (*it).second;
      return true;
    }
    return false;
  }

  // multiswitch (string list)
  bool config::get_switch( const string &name, vector<string> &values ) const
  {
    values.clear();
    multimap<string,string>::const_iterator it = _multi_switches.find( name );
    while ( it != _multi_switches.end() ) {
      if ( (*it).first == name ) {
	    values.push_back( (*it).second );
      }
      it++;
    }
    if ( values.size() == 0 ) {
      return false;
    }
    return true;
  }


  // siconfig::ngle switch (return integer)
  bool config::get_switch( const string &name, int &value ) const
  {
    map<string,string>::const_iterator it = _switches.find( name );
    if ( it != _switches.end() ) {
      value = atoi( (*it).second.c_str() );
      return true;
    }
    return false;
  }

  // single switch (return uint)
  bool config::get_switch( const string &name, uint &value ) const
  {
    map<string,string>::const_iterator it = _switches.find( name );
    if ( it != _switches.end() ) {
      // fix for locales which use "," delimiter
      // always force "."
      std::istringstream istr( (*it).second );
      istr.imbue(std::locale("C"));
      istr >> value;
      return true;
    }
    return false;
  }

  // single switch (return float)
  bool config::get_switch( const string &name, float &value ) const
  {
    map<string,string>::const_iterator it = _switches.find( name );
    if ( it != _switches.end() ) {
      // fix for locales which use "," delimiter
      // always force "."
      std::istringstream istr( (*it).second );
      istr.imbue(std::locale("C"));
      istr >> value;
      return true;
    }
    return false;
  }

  // single switch (return double)
  bool config::get_switch( const string &name, double &value ) const
  {
    map<string,string>::const_iterator it = _switches.find( name );
    if ( it != _switches.end() ) {
      // fix for locales which use "," delimiter
      // always force "."
      std::istringstream istr( (*it).second );
      istr.imbue(std::locale("C"));
      istr >> value;
      return true;
    }
    return false;
  }


  
  
}
