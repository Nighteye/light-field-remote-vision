/* -*-c++-*- */
/** \file vectorial_multilabel.cuh

   CUDA-Only includes for vectorial multilabel solvers
   Experimental code for kD label space

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

namespace coco {

  // Workspace for one of the label space dimensions
  struct vml_dim_workspace
  {
    // array for final integer solution (integer label number)
    int *_ur;
    // number of bytes in array
    int _urbytes;

    // Primal variable
    float *_u;
    // Primal variable lead
    float *_uq;

    // Dual variables
    float *_px;
    float *_py;
    // Primal variables for Lagrangian in case of more complex regularizers
    // (Linear, truncated linear)
    // Required for each PAIR of variables, so scales with local G^2
    float *_eta_x;
    float *_eta_y;

    // Lagrange multiplier for simplex constraint
    float *_sigma;

    // Dual variable for data term
    float *_q;

    // Spatial smoothness weight
    float *_g;

    // Step sizes
    float _tau_u;
    float _tau_eta;
    float _sigma_q;
    float _sigma_s;
    float _sigma_p;

    // Memory alloc sizes
    size_t _nfbytes;
    size_t _nfbytes_sigma;
    size_t _nfbytes_eta;

    // Total required memory
    size_t _total_mem;
  };


  // Workspace for complete vectorial multilabel problem
  struct vml_workspace
  {
    // Overrelaxation constant
    float _theta;
    // Step sizes
    float _tau_mu;

    // Current iteration
    size_t _iteration;

    // Precomputed data term
    float *_rho;

    // Dual variables for global projection implementation
    // (not memory efficient, but fast)
    float *_mu;

    // Chunk layout (if direct projection for q)
    size_t _chunk_width;
    size_t _nchunks;
    size_t _nfbytes_chunk;
    // Chunk grid (smaller)
    dim3 _dimGridChunk;
    // Data term in chunks
    std::vector<float*> _chunk_rho;

    // On-the-fly computation for dataterm
    bool _dataterm_on_the_fly;
    bool _dataterm_on_the_fly_segmentation;
    float* _dataterm_segmentation_r;
    float* _dataterm_segmentation_g;
    float* _dataterm_segmentation_b;

    // Sizes
    size_t _nfbytes;
    size_t _total_mem;

    // Block/grid sizes
    dim3 _dimBlock;
    dim3 _dimGrid;
  };

  
  // Allocation helper functions
  bool vml_dimension_data_alloc_workspace( vml_data *data, vml_dimension_data *ddata );
  // Some helper function for index arrays etc.
  bool vml_index_to_label( vml_data *, size_t idx, int* L );
  size_t vml_label_index( vml_data *, int* L );

  // Total label count
  size_t vml_total_label_count( vml_data *data );
  // Offset into label array
  size_t vml_dim_label_offset( vml_dimension_data *ddata, size_t g, size_t x=0, size_t y=0 );
  // Offset into regularizer array
  size_t vml_dim_regularizer_offset( vml_dimension_data *ddata, size_t g1, size_t g2, size_t x=0, size_t y=0 );

  // Project current relaxed solution onto integer values
  // Unnecessary, always combined projections
  //bool vml_project_q( vml_data *D );


  /*****************************************************************************
       Vectorial multilabel nD algorithm:
       Strekalovskiy / Goldluecke / Cremers ICCV 2011
  *****************************************************************************/

  // Compute primal prox operator
  bool vml_primal_prox( vml_data *data, bool final_iteration );
  // Compute dual prox operator
  bool vml_dual_prox( vml_data *data );
  // Update overrelaxation
  bool vml_update_overrelaxation( vml_data *data );

  // Data term relaxation, global Lagrange multipliers
  bool vml_update_primal_relaxation( vml_data *D, bool final_iteration );
  // Primal projection, global Lagrange multipliers
  bool vml_update_primal_relaxation_on_the_fly_segmentation( vml_data *D, bool final_iteration );
  // Data term relaxation, projection method
  bool vml_primal_prox_projection( vml_data *D, bool final_iteration );
  // Primal projection, direct projection chunk-wise
  bool vml_primal_prox_projection_on_the_fly_segmentation( vml_data *D, bool final_iteration );
  // Perform one primal step for regularizer
  bool vml_update_primal_regularizer( vml_data *D, vml_dimension_data *ddata, bool final_iteration );

}
