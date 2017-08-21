/* -*-c++-*- */
/** \file volume_4d.cuh
   Algorithms to solve the TV-segmentation model in 4D

   Local CUDA workspace structure definition.

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


  struct cuda_volume_4d_workspace
  {
    // Data in the grid
    float *_data;
    // Total number of bytes
    size_t _nbytes;
    // Number of bytes per layer
    size_t _nbytes_layer;

    // CUDA call layout
    dim3 _dimGrid;
    dim3 _dimBlock;
  };


  // The CUDA volume uses the same data structure as the
  // CPU volume.
  // The difference is that all float* point to allocated GPU
  // memory.
  /*
  // Allocate CUDA 4D volume
  volume_4d *cuda_volume_4d_alloc( volume_4d *src );
  // Zero out volume
  bool cuda_volume_4d_set_zero( volume_4d* gpu_dest );
  // Copy volume on GPU
  bool cuda_volume_4d_copy_on_device( volume_4d* gpu_dest, volume_4d* gpu_source );
  // Copy memory from CPU to GPU
  bool cuda_volume_4d_set( volume_4d* gpu_dest, volume_4d *cpu_src );
  // Copy memory from GPU to GPU
  bool cuda_volume_4d_get( volume_4d* gpu_src, volume_4d *cpu_dest );
  // Copy memory from CPU to GPU, one slice
  bool cuda_volume_4d_set_slice( volume_4d* gpu_dest, size_t t, float *dest );
  // Copy memory from GPU to CPU, one slice
  bool cuda_volume_4d_get_slice( volume_4d *gpu_src, size_t t, float *dest );
  // Release CUDA 4D volume
  bool cuda_volume_4d_free( volume_4d *v4d );
  */
}
