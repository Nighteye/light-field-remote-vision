#ifndef __COCO_VTV_SR_H
#define __COCO_VTV_SR_H

#include <vector>
#include "../cuda/cuda_convolutions.h" //cuda_kernel
#include <vector_types.h> //dim3

namespace coco {

typedef unsigned char byte;

// Extra workspace data per view
struct coco_vtv_sr_view_data {

    // warp displacement for the optical center of the view
    // all in high_resolution coordinates
    float _dx_vu;
    float _dy_vu;

    // image data in float to avoid rounding errors
    // densely packed in plannar form RRRR...GGGG...BBBB...
    float *_image_f;

    // mask telling if a pixel is visible or not
    // in high-resolution
    bool *_visibility_mask;

    // visibility weight on v computed with the gradient
    // of the disparity (denotes valid umap)
    float* _vmask;

    // depth map on v (with u resolution)
    float *_dmap_v;

    // variance of the disparity measure
    float *_dmap_sigma;

    // depth vote for u (CPU)
    float *_dmap_u;

    cuflt* warp_beta_x;
    cuflt* warp_beta_y;
    cuflt* warp_tau_x;
    cuflt* warp_tau_y;
    cuflt* dpart_x; // partial derivative sigma_dmap * dtau/dz
    cuflt* dpart_y;
    cuflt* deform_weight_tau; // deformation weights |det D tau|^(-1)
    cuflt* deform_weight_beta; // deformation weights |det D beta|^(-1)
    cuflt* lumi_weights; // weights of lumigraph rendering (in Gamma)

    // visibility weight on lores-v
    // its the sum of the weights of each pixel
    // corresponding high resolution pixel
    float* _vmask_lo;

    // target cell array in u
    // this is an array of index of non overlapping pixel groups
    // used to parallellize the forward mapping tasks
    int *_cells_tau;
    int *_cells_beta;
    std::vector<int> _seg_end_tau;
    std::vector<int> _seg_end_beta;

    // warp block configuration
    /*
    int _step_x;
    int _step_y;
    dim3 _dimBlockWarp;
    dim3 _dimGridWarp;
    */
};

// Extra workspace data for Superresolution model
struct coco_vtv_sr_data {

    // number of input views
    size_t _nviews;

    // input view downscale factor
    size_t _dsf;
    // input view downscaled resolution
    size_t _w;
    size_t _h;

    // sigma of the sensor noise, in the same units as _image
    // could be moved to view_data and different for each input image
    float _sigma_sensor;

    float _aux_dmap_sigma;

    // threshold for the u gradient.
    // Values bigger than this threshold will be set to the threshold
    // This is to avoid too low weights
    float _ugrad_threshold;

    // sparse warp matrix max size (larger = better, but might cost too much mem)
    //int _nwarp; // NOT USED

    // mem sizes
    size_t _nfbytes_lo; // _w * _h * sizeof(float);
    size_t _nfbytes_hi; // W * H * sizeof(float);
    size_t _vmask_bytes; // W * H * sizeof(float);

    // input view data
    std::vector<coco_vtv_sr_view_data*> _views;

    // disparity map created using vtv_sr_init_target_disparity_map
    float *_dmap_u;
    float _disp_max;

    // Target mask (in hires)
    float* _target_mask;

    // Kernel for visibility map smoothing
    cuda_kernel *_vmask_filter;

    // grid data
    dim3 _dimGrid;
    dim3 _dimBlock;

    bool _lumigraph;
};

typedef enum  {
  Gamma_from_Omegai,
  Omegai_from_Gamma
} t_warp_direction;

// Setup SR algorithm: init view and resolution data (Greg)
bool coco_vtv_sr_init_unstructured( coco_vtv_data *data, size_t nviews, size_t ds_factor, bool lumigraph );

// Free up data for SR algorithm (Greg)
bool coco_vtv_sr_free_unstructured( coco_vtv_data *data );

// If the view to synthsize is one the input, remove it (Greg)
void coco_vtv_sr_remove_last_view_unstructured( coco_vtv_data *data );

// Setup a single view (Greg)
bool coco_vtv_sr_create_view_unstructured( coco_vtv_data *data, size_t nview, gsl_image *I);

// Load disparities to test unstructured sr (Greg)
bool coco_vtv_sr_load_disparities_unstructured( coco_vtv_data *data, size_t nview,
                                                double dx, double dy,
                                                float *disparity, float *disp_sigma);

// compute the target disparity map using the votes dmap_u[i] (Goldluecke)
bool vtv_sr_init_target_disparity_map_unstructured( coco_vtv_data *data );

// Init forward warp for a view : uses warps (make sure they are computed) (Greg)
// warp=0: tau, warp=1:beta
// Currently completely on host, TODO: try to parallelize (hard)
bool vtv_sr_init_forward_warp_structure_unstructured( coco_vtv_data *data, size_t nview, bool warp );

// Currently completely on host, TODO: try to parallelize (hard)
// fills _dmap_u with the depth of the warped pixels (Greg)
bool vtv_sr_init_u_dmap_unstructured( coco_vtv_data *data, size_t nview );

// Compute the beta warped image u for a single view vi to test only (Greg)
bool coco_vtv_sr_compute_warped_image( coco_vtv_data *data, size_t nview );

// Compute tau and beta warps and dpart of a single view  (Greg)
bool coco_vtv_sr_warps_from_disp( coco_vtv_data *data );

// Compute the tau warp of a view (Greg)
bool coco_vtv_sr_compute_tau_warp( coco_vtv_data *data, size_t nview );
// Compute the beta warp of a view (Greg)
bool coco_vtv_sr_compute_beta_warp( coco_vtv_data *data, size_t nview );
// Compute the dpart of a view (sigma_dmap * tau partial derivative) (Greg)
bool coco_vtv_sr_compute_dpart( coco_vtv_data *data, size_t nview );

// compute visibility mask from the tau warps
bool coco_vtv_visibility_from_tau( coco_vtv_data *data );

// Update the weights (Greg)
bool coco_vtv_sr_compute_weights_unstructured( coco_vtv_data *data );

// warp input images into GAMMA
bool coco_warp_input_images(coco_vtv_data *data);

// Compute the starting image by using beta warps (Greg)
bool coco_vtv_sr_compute_averaged_beta_warp(coco_vtv_data *data);

// Compute primal energy (Greg)
double coco_vtv_sr_primal_energy_unstructured( coco_vtv_data *data );

// Perform one iteration of Algorithm 1, Chambolle-Pock (Goldluecke)
bool coco_vtv_sr_iteration_fista_unstructured( coco_vtv_data *data );

// Perform one single shrinkage step (ISTA) (Greg)
bool vtv_sr_ista_step_unstructured( coco_vtv_data *data );

// Compute gradient of data term (Greg)
bool vtv_sr_dataterm_gradient_unstructured( coco_vtv_data *data );

// Read the beta warps: from gsl_image to device float* (Greg)
bool coco_vtv_sr_read_beta( coco_vtv_data *data, gsl_image** beta_warps );

// Read the tau warps and deformation weights: from gsl_image to device float* (Greg)
bool coco_vtv_sr_read_tau( coco_vtv_data *data, gsl_image** tau_warps );

// Read the partial tau: from gsl_image to device float* (Greg)
bool coco_vtv_sr_read_partial_tau( coco_vtv_data *data, gsl_image** partial_tau, float sigma_const = -1 );

// Read the lumigraph weights: from gsl_image to device float* (Greg)
bool coco_vtv_sr_read_lumi_weights( coco_vtv_data *data, gsl_image** lumi_weights );

// Write current solution in pfm format
bool coco_vtv_sr_write_pfm_solution( coco_vtv_data *data );

// Initialize regularizer weight _g using sr->_target_mask
bool coco_vtv_sr_init_regularizer_weight_unstructured( coco_vtv_data *data );

// compute non-overlapping segments (cell structure) for all views and both for tau and beta splatting
bool coco_vtv_sr_init_cells_unstructured( coco_vtv_data *data );

// warp (in high res) the input image given the warping direction
// omega_i to gamma: 0, gamma to omega_i: 1
// input and output can be in omega_i or gamma
bool coco_vtv_sr_warp_unstructured( coco_vtv_data *data, size_t nview, t_warp_direction direction, float* input, float* output );

// Forward splatting with warp (either beta or tau) of a single input view (high res)
bool coco_vtv_sr_fw_splatting_unstructured( coco_vtv_data *data,
                                            cuflt *input,
                                            cuflt *fw_warp_x,
                                            cuflt *fw_warp_y,
                                            cuflt *bw_warp_x,
                                            cuflt *bw_warp_y,
                                            int *cells,
                                            std::vector<int> &seg_ends,
                                            cuflt *output );

bool check_warp_coherence( coco_vtv_data *data);

// Get current target mask
bool coco_vtv_get_target_mask( coco_vtv_data *data, gsl_matrix* U );


}

#endif // #ifndef __COCO_VTV_SR_H
