#ifndef __COCO_VTV_H
#define __COCO_VTV_H

#include <map>
#include <vector>
#include <assert.h>

#include "../cuda/cuda_interface.h"
#include "../defs.h"
#include "../modules.h"
#include "../common/gsl_image.h"



namespace coco {

  /*** CORE PDE METHODS ***/
  struct coco_vtv_workspace;
  struct coco_vtv_sr_data;
  struct coco_vtv_sr_data_unstructured;
  struct Data;

  // Helper structure to set all parameters
  struct coco_vtv_data
  {
    // Field size
    size_t _W;
    size_t _H;
    size_t _N;
    // Number of channels
    size_t _nchannels;

    // Smoothness parameter
    double _lambda;
    // Smoothness multiplicator for adaptive algorithms
    double _lambda_max_factor;

    // Step sizes (primal/dual)
    double _tau;
    double _sigma;
    // Step sizes (fista)
    double _alpha;
    // Number of inner iterations (fista)
    size_t _inner_iterations;
    // Threshold to decide if pixels in interval
    // (dips - _disp_threshold, dips + _disp_threshold) must be merged
    double _disp_threshold;

    // Lipschitz constant
    double _L;
    // Uniform convexity constant
    double _gamma;

    // Regularizer
    // 0: TV_S
    // 1: TV_F
    // 2: TV_J
    size_t _regularizer;

    // Data term regularizer
    // 1: L^1 - norm
    // 2: L^2 - norm
    // Works only together with Arrow-Hurwicz or Chambolle-Pock,
    // and with general inverse problem
    size_t _data_term_p;

    // Super-resolution data (if initialized)
    coco_vtv_sr_data *_sr_data;
    // Super-resolution unstructured data (if initialized)
    coco_vtv_sr_data_unstructured *_sr_data_unstructured;

    // Base directory for test output
    std::string _basedir;

    // Number of bytes per image double layer
    size_t _nfbytes;
    // Workspace data
    coco_vtv_workspace* _workspace;
  };




  // Alloc PDE data with sensible defaults
  coco_vtv_data* coco_vtv_alloc( std::vector<gsl_matrix*> F );
  // Free up PDE data
  bool coco_vtv_free( coco_vtv_data *data );

  // alloc rarely needed variables
  bool coco_vtv_alloc_aux_fields( coco_vtv_data *data );

  // Init local regularizer weights
  bool coco_vtv_set_regularizer_weight( coco_vtv_data *data, gsl_matrix *g );
  // Init local regularizer weights
  bool coco_vtv_set_regularizer_weight( coco_vtv_data *data, std::vector<gsl_matrix*> g );

  // Initialize workspace with current solution
  bool coco_vtv_initialize( coco_vtv_data *data,
      std::vector<gsl_matrix*> &U );

  // Get current solution
  bool coco_vtv_get_solution( coco_vtv_data *data,
      std::vector<gsl_matrix*> &U );

  // Get dual variable XI (vector of dimension 2)
  bool coco_vtv_get_dual_xi( coco_vtv_data *data,
      std::vector<gsl_matrix*> &XI,
			     size_t channel=0 );
  
  // Get dual variable ETA (vector of dimension equal to channel number)
  bool coco_vtv_get_dual_eta( coco_vtv_data *data,
      std::vector<gsl_matrix*> &ETA );
  



  /*****************************************************************************
       TV_x ROF
  *****************************************************************************/

  // Compute primal energy
  double coco_vtv_rof_primal_energy( coco_vtv_data *data );

  // *** MTV-ROF base algorithms ***

  // Perform one iteration of Bermudez-Morena scheme
  bool coco_vtv_rof_iteration_bermudez_morena( coco_vtv_data *data );
  // Perform one iteration of Arrow-Hurwicz scheme
  bool coco_vtv_rof_iteration_arrow_hurwicz( coco_vtv_data *data );
  // Perform one iteration of Algorithm 1, Chambolle-Pock
  bool coco_vtv_rof_iteration_chambolle_pock_1( coco_vtv_data *data );
  // Perform one iteration of Algorithm 2, Chambolle-Pock
  bool coco_vtv_rof_iteration_chambolle_pock_2( coco_vtv_data *data );
  // Perform one iteration of FISTA
  bool coco_vtv_rof_iteration_fista( coco_vtv_data *data );


  /*****************************************************************************
       TV_x ROF-SUM
       sum of ROF terms, each with point-wise weight scaled by 1/lambda
       interface currently only supports 1D functions
  *****************************************************************************/

  // *** MTV-ROF base algorithms ***
  // Init functional
  bool coco_vtv_rof_sum_init( coco_vtv_data *data,
      std::vector<gsl_matrix*> Fs,
      std::vector<gsl_matrix*> weights );

  // Perform one iteration of FISTA
  bool coco_vtv_rof_sum_iteration_fista( coco_vtv_data *data );



  /*****************************************************************************
       TV_x Deblurring
  *****************************************************************************/

  // Init kernel
  bool coco_vtv_set_separable_kernel( coco_vtv_data *data, gsl_vector *kernel_x, gsl_vector *kernel_y );
  bool coco_vtv_set_kernel( coco_vtv_data *data, gsl_matrix *kernel );

  // Compute primal energy
  double coco_vtv_deblurring_primal_energy( coco_vtv_data *data );

  // Perform one primal step (several iterations of gradient descent for the prox operator)
  bool coco_vtv_deblurring_primal_step( coco_vtv_data *data );
  // Perform one primal step (several iterations of gradient descent for the prox operator)
  bool coco_vtv_deblurring_dual_step( coco_vtv_data *data );
  // Single primal descent step (using functional gradient only)
  bool coco_vtv_deblurring_primal_descent_step( coco_vtv_data *data );
  // Perform one iteration of Algorithm 1, Chambolle-Pock
  bool coco_vtv_deblurring_iteration_chambolle_pock_1( coco_vtv_data *data );
  // Perform one iteration of Algorithm 2, Chambolle-Pock
  bool coco_vtv_deblurring_iteration_chambolle_pock_2( coco_vtv_data *data );
  // Perform one iteration of Arrow-Hurwicz
  bool coco_vtv_deblurring_iteration_arrow_hurwicz( coco_vtv_data *data );

  // Perform one single shrinkage step (ISTA)
  bool coco_vtv_deblurring_ista_step( coco_vtv_data *data );
  // Perform one full iteration of FISTA
  bool coco_vtv_deblurring_iteration_fista( coco_vtv_data *data );



  /*****************************************************************************
       TV_x Zooming (essentially SR with a single image)
       CURRENTLY USES SR IMPLEMENTATION, SO fx must be equal to fy
  *****************************************************************************/

  // Init functional
  bool coco_vtv_zooming_init( coco_vtv_data *data, gsl_image *source );
  // Perform one full iteration of FISTA
  bool coco_vtv_zooming_iteration_fista( coco_vtv_data *data );



  /*****************************************************************************
       TV_x Superresolution
  *****************************************************************************/

  // Setup SR algorithm: init view and resolution data
  bool coco_vtv_sr_init( coco_vtv_data *data, size_t nviews, size_t ds_factor );
  // Free up data for SR algorithm
  bool coco_vtv_sr_free( coco_vtv_data *data );

  // set the ugrad threshold
  void coco_vtv_sr_set_ugrad_threshold( coco_vtv_data *data, float ugrad_threshold);

  // set the sigma value of the sensor noise
  void coco_vtv_sr_set_sigma_sensor( coco_vtv_data *data, float sigma_sensor );
  void coco_vtv_sr_set_sigma_disp( coco_vtv_data *data, float sigma_disp );

  // Method used to remove last view : decreases nview in one element
  // Used to "skip" the input view of the synthesised one
  void coco_vtv_sr_remove_last_view( coco_vtv_data *data );

  // End view creation, finalize data structures
  bool coco_vtv_sr_end_view_creation( coco_vtv_data *data );

  // Setup a single test view with globally constant displacement for testing.
  // displacement is measured in percent of an original pixel.
  // part 1: dmap and structure init
  bool coco_vtv_sr_create_test_view( coco_vtv_data *data, size_t nview, double dx, double dy );
  // part 2: image init (after "end_view_creation")
  bool coco_vtv_sr_render_test_view( coco_vtv_data *data, size_t nview );

  // Setup a single test view given a disparity map and source image
  // displacement is measured in disparity map units
  bool coco_vtv_sr_create_view( coco_vtv_data *data, size_t nview,
				double dx, double dy,
				gsl_image *view, float *disparity, float *disp_sigma = 0 );

  // get view image, lores version
  gsl_image *coco_vtv_sr_get_view_lores( coco_vtv_data *data, size_t nview );

  // Compute new weights
  bool coco_vtv_sr_compute_weights( coco_vtv_data *data );
  // Compute interpolated image (just average of all inputs)
  bool coco_vtv_sr_compute_averaged_forward_warp( coco_vtv_data *data );
  // Initialize regularizer weight _g using sr->_target_mask
  bool coco_vtv_sr_init_regularizer_weight( coco_vtv_data *data );

  // Compute interpolated image from a single view
  bool coco_vtv_sr_compute_forward_warp( coco_vtv_data *data, size_t nview );

  // Compute primal energy
  double coco_vtv_sr_primal_energy( coco_vtv_data *data );

  // Perform one full iteration of FISTA
  bool coco_vtv_sr_iteration_fista( coco_vtv_data *data );
  // Perform one iteration of Algorithm 1, Chambolle-Pock
  bool coco_vtv_sr_iteration_chambolle_pock_1( coco_vtv_data *data );
  // Perform one iteration of Algorithm 1, Chambolle-Pock, experimental disparity optimization
  bool coco_vtv_sr_dmap_iteration_chambolle_pock_1( coco_vtv_data *data );

  // Experimental: optimize the disparity maps using the current solution, then
  // re-initialize algorithm with new ones
  bool vtv_sr_optimize_disparity_maps( coco_vtv_data *data );



  /*****************************************************************************
       TV_x Inpainting
  *****************************************************************************/

  // Init inpainting stencil
  bool coco_vtv_set_stencil( coco_vtv_data *data, gsl_matrix *stencil );

  // Compute primal energy
  double coco_vtv_inpainting_primal_energy( coco_vtv_data *data );

  // Perform one single shrinkage step (ISTA)
  bool coco_vtv_inpainting_ista_step( coco_vtv_data *data );
  // Perform one full iteration of FISTA
  bool coco_vtv_inpainting_iteration_fista( coco_vtv_data *data );

}


#endif
