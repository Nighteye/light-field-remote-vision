/** \file parse_config.h

    File imported from "common" lib, use if this library is not available.

    Simple config file and command line parser

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

#ifndef __PARSE_CONFIG_H
#define __PARSE_CONFIG_H

#include <assert.h>
#include <stdio.h>

#include <string>
#include <vector>
#include <fstream>
#include <map>

namespace coco
{

  // configuration data
  class config
  {
  public:

    // SETUP CONFIG

    // read from file
    bool parse_config_file( const std::string &filename );
    // read from command line
    bool parse_command_line( int argn, char **argv );

    // add global string
    void add_string( const std::string &str );
    // set switch
    void set_switch( const std::string &name, const std::string &value );

    // READ CONFIG FLAGS

    // dump content to file
    bool dump( const std::string &filename ) const;

    // global std::string list
    const std::vector<std::string> &get_strings() const;

    // just test for switch presence
    bool get_switch( const std::string &name ) const;

    // single switch (return string)
    bool get_switch( const std::string &name, std::string &value ) const;
    // single switch (return integer)
    bool get_switch( const std::string &name, int &value ) const;
    // single switch (return uint)
    bool get_switch( const std::string &name, unsigned int &value ) const;
    // single switch (return float)
    bool get_switch( const std::string &name, float &value ) const;
    // single switch (return double)
    bool get_switch( const std::string &name, double &value ) const;

    // multiswitch (string list)
    bool get_switch( const std::string &name, std::vector<std::string> &values ) const;


  private:
    // global config strings
    std::vector<std::string> _strings;
    // switch map
    std::map<std::string,std::string> _switches;
    // switch map
    std::multimap<std::string,std::string> _multi_switches;
  };

}



#endif
