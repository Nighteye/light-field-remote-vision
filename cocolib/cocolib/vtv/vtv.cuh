/* -*-c++-*- */
#include <vector>
#include "../cuda/cuda_convolutions.h"

namespace coco {
 
  // Manifold structure, stored on GPU
  struct coco_vtv_workspace
  {
    // Primal components
    std::vector<cuflt*> _U;
    // Primal lead
    std::vector<cuflt*> _Uq;
    // Function components
    std::vector<cuflt*> _F;

    // Aux: prox gradient
    std::vector<cuflt*> _G;
    // Aux: temp
    std::vector<cuflt*> _temp;

    // Stencil (e.g. inpainting)
    cuflt *_stencil;
    // Regularizer weight (testing, currently only used internally)
    std::vector<cuflt*> _g;

    // Dual variables XI
    std::vector<cuflt*> _X1;
    std::vector<cuflt*> _X2;
    // Dual variables ETA
    std::vector<cuflt*> _E1;
    std::vector<cuflt*> _E2;

    // Lead dual variables XI
    std::vector<cuflt*> _X1q;
    std::vector<cuflt*> _X2q;
    // Lead dual variables ETA
    std::vector<cuflt*> _E1q;
    std::vector<cuflt*> _E2q;

    // Temp dual variables
    std::vector<cuflt*> _X1t;
    std::vector<cuflt*> _X2t;

    // Kernels for deblurring
    coco::cuda_kernel *_b;
    coco::cuda_kernel *_bq;

    // For ROF-SUM
    bool _delete_rof_sum_data;
    std::vector<cuflt*> _sum_Fs;
    std::vector<cuflt*> _sum_weights;

    // Constraints
    cuflt *_constraints;
    // Constants
    size_t _nfbytes;

    // current algorithm iteration
    size_t _iteration;

    // CUDA block dimensions
    dim3 _dimBlock;
    dim3 _dimGrid;
  };




  // Auxiliary algorithm functions
  
  // Perform one primal step (with finite step size)
  bool coco_vtv_rof_primal_step( coco_vtv_data *data );
  // Perform one primal step ("infinite" step size)
  bool coco_vtv_rof_primal_infinite( coco_vtv_data *data );
  // Perform one dual step, no reprojection
  bool coco_vtv_rof_dual_step( coco_vtv_data *data );
  // Perform reprojection
  bool coco_vtv_rof_reproject( coco_vtv_data *data );
  // Overrelaxation step
  bool coco_vtv_rof_overrelaxation( coco_vtv_data *data, cuflt theta );
  // Overrelaxation step (FGP dual)
  bool coco_vtv_rof_fgp_overrelaxation( coco_vtv_data *data, cuflt theta );
  // ISTA step
  bool coco_vtv_rof_ista_step( coco_vtv_data *data ); 



  /*****************************************************************************
       TV_x ROF-SUM, INITIALIZATION WITH CUDA ARRAYS
       no local copies, highly efficient
       sum of ROF terms, each with point-wise weight scaled by 1/lambda
  *****************************************************************************/

  // *** MTV-ROF base algorithms ***
  // Init functional
  bool coco_vtv_rof_sum_init( coco_vtv_data *data,
      std::vector<float*> Fs,
      std::vector<float*> weights );

  // Helper function: ISTA step
  bool coco_vtv_rof_sum_ista_step( coco_vtv_data *data );

  // Cleanup for ROF-sum
  bool coco_vtv_rof_sum_data_free( coco_vtv_data *data );


}
