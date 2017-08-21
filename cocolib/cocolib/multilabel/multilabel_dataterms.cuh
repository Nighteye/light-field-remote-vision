/* -*-c++-*- */
/** \file multilabel_dataterms.cuh
   Helper functions: compute various dataterms for multilabel problems
   
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

#ifndef __COCO_MULTILABEL_DATATERMS_CUH
#define __COCO_MULTILABEL_DATATERMS_CUH


////////////////////////////////////////////////////////////////////////////////
// Stereo dataterm from two images, 1 layer
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_stereo_dataterm_rgb_layer( int W, int H,
	   					cuflt lambda,
						cuflt dx, cuflt dy,
						cuflt *r0, cuflt *g0, cuflt* b0,
						cuflt *r1, cuflt *g1, cuflt* b1,
						cuflt *rho,
						cuflt *count );

#endif
