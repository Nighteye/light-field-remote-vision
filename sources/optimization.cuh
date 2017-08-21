#ifndef __COCO_OPTIMIZATION_H
#define __COCO_OPTIMIZATION_H

#include <vector>
#include <cuda/cuda_convolutions.h> //cuda_kernel
#include <vector_types.h> //dim3

class Config_data;

namespace coco {

struct coco_vtv_data;

typedef unsigned char byte;

// Extra workspace data per view
struct coco_vtv_sr_view_data_unstructured {

    // image data in float to avoid rounding errors
    // densely packed in plannar form RRRR...GGGG...BBBB...
    cuflt *_image_f; // v_i in low resolution, in device

    // weights omega_i in "novel view synthesis and IBR principles", in device
    cuflt* _weights_omega_i;

    // warps, in device. only omega_i-to-gamma warp is needed
    // in low res, values in high res
    // visibility is determined by invalid warp value (<0)
    cuflt* _warp_tau_x;
    cuflt* _warp_tau_y;

    // partial derivative sigma_dmap * dtau/dz, in device, low res
    cuflt* dpart_x;
    cuflt* dpart_y;

    // contains the element of the sparse matrix A_i such that v_i = A_i*u
    cuflt* _A;

    // target cell array in u
    // this is an array of index of non overlapping pixel groups
    // used to parallellize the forward mapping tasks
    int *_cells;
    std::vector<int> _seg_end;
};

// Extra workspace data for Superresolution model
struct coco_vtv_sr_data_unstructured {

    // number of input views
    size_t _nviews;

    // input view downscale factor
    size_t _dsf;
    // input view downscaled resolution
    size_t _w;
    size_t _h;

    // size of the kernels in A
    size_t _ks;

    // sigma of the sensor noise, in the same units as _image
    // could be moved to view_data and different for each input image
    cuflt _sigma_sensor;

    // threshold for the u gradient.
    // Values bigger than this threshold will be set to the threshold
    // This is to avoid too low weights
    cuflt _ugrad_threshold;

    // 0: Wanner's deformation weights
    // 1: Sergi's deformation weights
    // 2: Nieto's deformation weights
    int _dw_type;

    // 0: no weights for gradient term
    // 1: Pujades's weights for gradient term
    // 2: Nieto's weights for gradient term
    int _gw_type;

    // Dataterm parameters. alpha: intensity term. beta: gradient term.
    float _dt_alpha;
    float _dt_beta;

    float _gradient_step;

    // Dataterm gradients
    std::vector<cuflt*> _G_gradients;
    std::vector<cuflt*> _G_intensities;

    // mem sizes
    size_t _nfbytes_lo; // _w * _h * sizeof(cuflt);
    size_t _nfbytes_hi; // W * H * sizeof(cuflt);

    // input view data
    std::vector<coco_vtv_sr_view_data_unstructured*> _views;

    // normalization mask, in low res, device
    cuflt* _norm_mask;

    // backward visibility mask (gamma domain)
    bool* _visibility_mask;

    // Kernel for visibility map smoothing
    cuda_kernel *_vmask_filter;

    // grid data
    dim3 _dimGrid;
    dim3 _dimBlock;
};

// Compute backward visibility (gamma domain)
void coco_vtv_setup_visibility_mask( coco_vtv_data *data );

// Perform TV on init image to fill holes
//void coco_vtv_hole_filling( coco_vtv_data *data );

// Perform push pull Laplacian hole-filling
void coco_vtv_push_pull( coco_vtv_data *data );

// Setup unstructured SR algorithm: init view and resolution data
bool coco_vtv_sr_init_unstructured( coco_vtv_data *data, Config_data *config_data );

// Free up data for unstructured SR algorithm
bool coco_vtv_sr_free_unstructured( coco_vtv_data *data );

// Compute the sparse matrix A
bool coco_vtv_sr_compute_sparse_matrix( coco_vtv_data *data );

// Init forward warp for a view : uses warps (make sure they are computed)
// warp=0: tau, warp=1:beta
// Currently completely on host, TODO: try to parallelize (hard)
bool vtv_sr_init_forward_warp_structure_unstructured( coco_vtv_data *data, size_t nview );

// Setup a single view
bool coco_vtv_sr_create_view_unstructured( coco_vtv_data *data, size_t nview, gsl_image *I);

// Update weight_omega_i
bool coco_vtv_sr_compute_weights_unstructured( coco_vtv_data *data );

// Read the tau warps and deformation weights: from gsl_image to device cuflt*
bool coco_vtv_sr_read_tau( coco_vtv_data *data, gsl_image** tau_warps );

// Read the partial tau: from gsl_image to device cuflt*
bool coco_vtv_sr_read_partial_tau( coco_vtv_data *data, gsl_image** partial_tau );

// Compute the initial image, starting point of the algorithm
void coco_vtv_sr_compute_initial_image( coco_vtv_data *data, Config_data *config_data );

// warp each input view separately by splatting
bool coco_vtv_sr_test_warps( coco_vtv_data *data );

// Blur high res image to test the kernels
bool coco_vtv_sr_downsample( coco_vtv_data *data );

// Compute primal energy
double coco_vtv_sr_primal_energy_unstructured( coco_vtv_data *data, float previous_energy = 0.0 );

// Write current solution in pfm format
bool coco_vtv_sr_write_pfm_solution( coco_vtv_data *data );

// Perform one iteration of Algorithm 1, Chambolle-Pock
bool coco_vtv_sr_iteration_fista_unstructured( coco_vtv_data *data, bool data_term = true );

// Perform one single shrinkage step (ISTA)
bool vtv_sr_ista_step_unstructured( coco_vtv_data *data, bool data_term = true );

// Compute gradient of dataterm (difference of intensities)
bool vtv_sr_dataterm_intensities( coco_vtv_data *data );

// Compute gradient of dataterm (difference of gradients)
bool vtv_sr_dataterm_gradients( coco_vtv_data *data );

bool vtv_sr_dataterm_gradients2( coco_vtv_data *data );

bool coco_vtv_sr_init_regularizer_weight_unstructured( coco_vtv_data *data );

}

#endif // #ifndef __COCO_OPTIMIZATION_H
