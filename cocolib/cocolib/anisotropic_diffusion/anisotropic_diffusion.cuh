/* -*-c++-*- */
/** \file anisotropic_diffusion.cuh
    Perona-Malik isotropic and Weickert's coherence-enhancing diffusion,
    different discretizations,
    inpainting models.

    CUDA-specific header

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
#include <vector>
#include "../cuda/cuda_convolutions.h"

namespace coco {
 
  // diffusion workspace, vectorial image
  struct coco_diffusion_workspace
  {
    // Solution components
    std::vector<float*> _U;
    // Initialization components (for inpainting etc.)
    std::vector<float*> _F;

    // Stencil (e.g. inpainting)
    float *_stencil;
    // Auxiliary variables
    float* _aux;

    // Diffusion tensor
    float *_a;
    float *_b;
    float *_c;

    // Constants
    size_t _nfbytes;

    // current algorithm iteration
    size_t _iteration;

    // CUDA block dimensions
    dim3 _dimBlock;
    dim3 _dimGrid;
  };




  // Auxiliary algorithm functions

  // Compute diffusion tensor from structure tensor (inplace)
  // Variant 1: Weickert coherence-enhancing anisotropic diffusion
  bool cuda_coherence_enhancing_diffusion_tensor( size_t W, size_t H,
						  float c1, float c2,
						  gpu_float_array a,
						  gpu_float_array b,
						  gpu_float_array c );

  // Compute diffusion tensor from structure tensor (inplace)
  // Variant 2: Perona-Malik edge-enhancing isotropic diffusion
  bool cuda_perona_malik_diffusion_tensor( size_t W, size_t H,
					   float K_sq,
					   gpu_float_array a,
					   gpu_float_array b,
					   gpu_float_array c );


  // Compute single diffusion step for non-negativity scheme using a diffusion tensor
  // Rotation invariant scheme
  // Required workspace size is 5*W*H floats
  bool cuda_anisotropic_diffusion_roi( size_t W, size_t H,
				       float tau,
				       gpu_float_array a, gpu_float_array b, gpu_float_array c,
				       gpu_float_array u,
				       gpu_float_array workspace );

  // Compute single diffusion step for non-negativity scheme using a diffusion tensor
  // Non-negativity scheme
  // Required workspace size is 5*W*H floats
  bool cuda_anisotropic_diffusion_nonneg( size_t W, size_t H,
					  float tau,
					  gpu_float_array a, gpu_float_array b, gpu_float_array c,
					  gpu_float_array u,
					  gpu_float_array workspace );

  // Compute single diffusion step for non-negativity scheme using a diffusion tensor
  // Simple scheme
  // Required workspace size is 5*W*H floats
  bool cuda_anisotropic_diffusion_simple( size_t W, size_t H,
					  float tau,
					  gpu_float_array a, gpu_float_array b, gpu_float_array c,
					  gpu_float_array u,
					  gpu_float_array workspace );
  
}
