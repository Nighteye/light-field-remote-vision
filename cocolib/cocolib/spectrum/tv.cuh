/* -*-c++-*- */
#include <vector>
#include "../cuda/cuda_arrays.h"

namespace coco {
 
  // Workspace for spectral decomposition using VTV
  struct coco_vtv_spectrum_workspace
  {
    // Input image
    gpu_2D_float_array_vector _F;

    // Ring buffer for iterates
    gpu_2D_float_array_vector _U[3];
    // Indices for current, previous and next iterate
    int _current;
    int _next;
    int _previous;

    // Generated sequence of spectral vectors
    vector<gpu_2D_float_array_vector*> _phi;
    // Spectrum
    vector<double> _spectrum;
    // Residual
    gpu_2D_float_array_vector _residual;

    // Constants
    size_t _nfbytes;

    // current algorithm iteration
    size_t _iteration;

    // VTV solver for inner ROF models
    coco_vtv_data *_vtv;

    // CUDA block dimensions
    dim3 _dimBlock;
    dim3 _dimGrid;
  };


}
