/* -*-c++-*- */
/** \file tv_multilabel.cuh

   CUDA-Only includes for tv_multilabel solvers

   Copyright (C) 2010 Bastian Goldluecke,
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

#include "multilabel.cuh"

namespace coco {

  struct tv_multilabel_workspace : public multilabel_workspace
  {
    // Dual variables
    float *_p1;
    float *_p2;
    float *_p3;

    // Overrelaxation paramter
    float _theta;
    // Primal step size
    float _tau_p;
    // Dual step size
    float _tau_d;
  };

  // Init/free
  bool tv_multilabel_workspace_init( tv_multilabel_data *data, tv_multilabel_workspace *w );
  bool tv_multilabel_workspace_free( tv_multilabel_data *data, tv_multilabel_workspace *w );
}
