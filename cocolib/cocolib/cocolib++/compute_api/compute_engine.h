/** \file compute_engine.h

    Structure for parallel grid computation engine,
    i.e. as implemented by CUDA or OpenCL

    Copyright (C) 2014 Bastian Goldluecke.

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

#ifndef __COCO_COMPUTE_ENGINE_H
#define __COCO_COMPUTE_ENGINE_H

#include <map>
#include <string>

#include "compute_buffer.h"

namespace coco {

  /// Grid compute engine
  /** The idea is to put one layer of abstraction between anything
      related to CUDA and the computation algorithms. This way,
      a later reimplementation of the engine in e.g. OpenCL
      might become possible without too much effort.
  */
  struct compute_engine
  {
    // Construction and destruction
    compute_engine( const std::map<std::string,std::string> *param_list = NULL );
    virtual ~compute_engine();

    // Activation
    // Necessary if multiple engines are in use by a program
    bool set_active() const;

    // Queries
    const void* internal_data() const;


  private:
    void* _implementation_data;
  };

};



#endif
