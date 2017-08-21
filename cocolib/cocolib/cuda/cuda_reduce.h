/** \file cuda_reduce.h

    Reduction functions for CUDA -
    e.g. sum of an array

    Copyright (C) 2012 Bastian Goldluecke,
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

#ifndef __CUDA_REDUCE_H
#define __CUDA_REDUCE_H

#include "../defs.h"
#include "cuda_interface.h"
#include "cuda_helper.h"

namespace coco {

  /********************************************************
  Reduction functions
  *********************************************************/

  // Reduce an array to a single float using addition.
  // First element of second array contains result, one element is sufficient
  bool cuda_sum_reduce( size_t W, size_t H,
			cuflt *in, cuflt *out,
			float *cpu_result = NULL );

  // Reduce an array to a single float using addition.
  // First element of second array contains result, one element is sufficient
  bool cuda_sum_reduce( size_t W, size_t H,
			int *in, int *out,
			int *cpu_result = NULL );


  // Reduce an array to a single float using the max function
  // First element of second array contains result, one element is sufficient
  bool cuda_max_reduce( size_t W, size_t H,
			cuflt *in, cuflt *out,
			float *cpu_result = NULL );

  // Reduce an array to a single float using the min function
  // First element of second array contains result, one element is sufficient
  bool cuda_min_reduce( size_t W, size_t H,
			cuflt *in, cuflt *out,
			float *cpu_result = NULL );

}
  
#endif
