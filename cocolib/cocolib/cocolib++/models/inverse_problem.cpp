/* -*-c++-*- */
/** \file inverse_problem.cpp

    Base data structure for inverse problem solvers,
    consisting of regularizer and data term.

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

#include "inverse_problem.h"

using namespace coco;


// Construction and destruction
inverse_problem::inverse_problem( regularizer *J, data_term *F )
  : variational_model( J,F )
{
}

inverse_problem::~inverse_problem()
{
}

