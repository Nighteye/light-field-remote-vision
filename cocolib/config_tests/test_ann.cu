/* -*-c++-*- */
/** \file test_ann.cu

    Test for ANN library availability.

    Copyright (C) 2012 Bastian Goldluecke,
    bastian.goldluecke@iwr.uni-heidelberg.de
    
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

#include <ANN/ANN.h>

int main()
{
  double pts[3];
  pts[0] = 0.0;
  pts[1] = 0.0;
  pts[2] = 0.0;
  ANNbd_tree *kdTree = new ANNbd_tree( (double**)pts, 1, 3 );
  if ( kdTree == NULL ) {
    return 1;
  }
  delete kdTree;
  return 0;
}
