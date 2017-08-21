/* -*-c++-*- */
/** \of_tv_l1_image.cuh

    Example code for COCOLIB: call solvers for the optical flow models
    with TV, VTV_J or VTV_F regularizer term.

    Copyright (C) 2012 Ole Johannsen,
    <first name>.<last name>ATberlin.de

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
#include "../common/gsl_image.h"

void computeColor(float fx, float fy, float *pix);
void image_to_grayscale(size_t W, size_t H, coco::gsl_image *im, coco::gsl_matrix *mim);

