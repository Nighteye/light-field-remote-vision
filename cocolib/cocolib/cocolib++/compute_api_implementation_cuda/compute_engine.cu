/* -*-c++-*- */
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

#include "../../cuda/cuda_helper.h"
#include "../compute_api/compute_engine.h"

using namespace coco;

// Construction and destruction
compute_engine::compute_engine( const std::map<std::string,std::string> *param_list )
{
}

compute_engine::~compute_engine()
{
}

// Activation
// Necessary if multiple engines are in use by a program
bool compute_engine::set_active() const
{
  assert( false );
  return false;
}
