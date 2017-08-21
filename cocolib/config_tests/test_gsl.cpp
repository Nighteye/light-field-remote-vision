/* -*-c++-*- */
/** \file test_gsl.cu

    Test for GSL library availability.

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

#include <gsl/gsl_matrix.h>

int main()
{
  gsl_matrix *M = gsl_matrix_alloc( 10,10 );
  if ( M == NULL ) {
    return 1;
  }
  gsl_matrix_free( M );
  return 0;
}
