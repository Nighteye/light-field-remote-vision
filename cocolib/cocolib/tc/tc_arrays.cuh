/* -*-c++-*- */
/** \file curvature_linear.cuh
   Total mean curvature - linear data term.
   Local CUDA workspace structure definition - array helper functions

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

namespace coco {

  // Array handling functions
  bool cvl_alloc_array( tc_workspace*, std::vector<stcflt*> & );
  bool cvl_copy_array( tc_workspace*, std::vector<stcflt*> &, std::vector<stcflt*> & );
  bool cvl_copy_array_to_cpu( tc_workspace*, std::vector<stcflt*> &, std::vector<stcflt*> & );
  bool cvl_copy_array_to_gpu( tc_workspace*, std::vector<stcflt*> &, std::vector<stcflt*> & );
  bool cvl_copy_array_pointers( tc_workspace*, std::vector<stcflt*> &, std::vector<stcflt*> & );
  bool cvl_clear_array( tc_workspace*, std::vector<stcflt*> & );
  bool cvl_free_array( tc_workspace*, std::vector<stcflt*> & );

  // Array handling (CPU)
  bool cvl_cpu_alloc_array( tc_workspace*, std::vector<stcflt*> & );
  bool cvl_cpu_free_array( tc_workspace*, std::vector<stcflt*> & );
  bool cvl_cpu_clear_array( tc_workspace*, std::vector<stcflt*> & );

  // Access functions
  // y,z index the offset from the center ( -(N-1)/2 .... (N-1)/2 )
  stcflt *cvl_get_3d_variable( tc_workspace*, std::vector<stcflt*> &, int y0, int y1, int z0, int z1 );

}
