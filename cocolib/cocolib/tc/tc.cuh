/* -*-c++-*- */
/** \file curvature_linear.cuh
   Total mean curvature - linear data term.
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

#include "../cuda/cuda_convolutions.h"

namespace coco {
  enum DUAL_VARIABLES {
    PX1, PX2,
    PY1, PY2,
    PZ1, PZ2,
    LAMBDA_X,
    LAMBDA_Y,
    LAMBDA_Z,
    NUM_DUAL,
  };
  enum {
    NUM_DUAL_ROF = 2,
  };

  // Workspace structure, stores allocated GPU data
  struct tc_workspace
  {
    // Redundant data
    size_t _W;
    size_t _H;
    size_t _num_dual;

    // Smoothness constants
    stcflt _lambda;
    stcflt _tv_lambda;
    stcflt _rof_lambda;

    // Flag if curvature is active (for debugging)
    bool _curvature;

    // Flag if xi or lambda is a CPU variable
    bool _cpu_xi;

    // Primal variable
    stcflt *_u;
    stcflt *_uq;
    stcflt *_u_star;

    // Derivative of dual
    stcflt *_D;

    // Dual variables
    // Temp computations
    stcflt *_px;
    stcflt *_py;
    // For TV of u
    stcflt *_pxq;
    stcflt *_pyq;
    // For TC: same ordering as _vq for each variable
    std::vector<stcflt*> _p[NUM_DUAL];

    // Data term
    stcflt *_a;
    // Data term (ROF)
    stcflt *_f;

    // Data term mask (e.g. for inpainting)
    stcflt *_mask;

    // Data term kernels (e.g. deblurring)
    coco::cuda_kernel *_b;
    coco::cuda_kernel *_bq;

    // Precomputed curvature weight
    stcflt *_cp;
    stcflt *_cp_cpu;

    // Saves last computed stats about energy
    double _energy;
    double _energy_data_term;
    double _energy_curvature;
    double _energy_tv;
    double _energy_lambda;
    double _energy_cv;
    // Energy computation states
    bool _compute_energy_rof;
    bool _compute_energy_inpainting;

    // Number of bytes in each layer
    size_t _Nf;
    // Number of bytes per layer array
    size_t _Nv;
    // Curvature window
    size_t _N;
    int _N2;
    // Energy integration scales (TODO: does it make sense?)
    double _energy_scale_v;
    double _energy_scale_u;

    // ALGORITHM INTERNALS
    // Step sizes (primal/dual)
    stcflt _sigma;
    stcflt _tv_sigma;

    // FISTA
    // Lipschitz constant
    stcflt _L;
    // Relaxation factor
    stcflt _t;

    // Computation parameters
    dim3 _dimGrid;
    dim3 _dimBlock;
  };

  // Reset norm of xi
  bool clear_xi_norm( coco::tc_workspace *w );
  // Initialize ROF step sizes
  bool cv_init_rof_stepsize( tc_data *data, double lambda );
  // Perform one ROF iteration
  bool cv_rof_iteration( tc_data *data );

  // Offset macros
  // Offset to field with y/z coordinates 
#define AOFF( y0,y1, z0,z1 ) ( (y0 + N*( y1 + N*( z0 + N*z1 ))) * W*H )
  // Index of field with y/z coordinates
#define AIND( y0,y1, z0,z1 ) ( (y0) + N*( (y1) + N*( (z0) + N*(z1) )))

}
