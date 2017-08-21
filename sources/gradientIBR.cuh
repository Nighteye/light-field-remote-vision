#ifndef GRADIENT_IBR_CUH
#define GRADIENT_IBR_CUH

#include <vector>
#include <vector_types.h> //dim3
#include <string>
#include <common/gsl_image.h>

class Config_data;

typedef unsigned char byte;

// Extra workspace data per view
struct ViewData {

    // image data in float to avoid rounding errors
    // densely packed in plannar form RRRR...GGGG...BBBB...
    float *_image_f; // v_i in low resolution, in device

    // weights omega_i in "novel view synthesis and IBR principles", in device
    float* _weights_omega_i;

    // warps, in device. only omega_i-to-gamma warp is needed
    // in low res, values in high res
    // visibility is determined by invalid warp value (<0)
    float* _warp_tau_x;
    float* _warp_tau_y;

    // partial derivative sigma_dmap * dtau/dz, in device, low res
    float* dpart_x;
    float* dpart_y;

    // contains the element of the sparse matrix B_i such that v_i = B_i*u
    // B is the linear operation that consists in warping then blurring
    float* _B;

    // target cell array in u
    // this is an array of index of non overlapping pixel groups
    // used to parallellize the forward mapping tasks
    int *_cells;
    std::vector<int> _seg_end;
};

// Extra workspace data for Superresolution model
struct Data {

    // number of input views
    size_t _nviews;

    // input view downscale factor
    size_t _dsf;
    // input view downscaled resolution
    size_t _w;
    size_t _h;
    // output view resolution
    size_t _W;
    size_t _H;

    // size of the kernels in B
    size_t _ks;

    // sigma of the sensor noise, in the same units as _image
    // could be moved to view_data and different for each input image
    float _sigma_sensor;

    // threshold for the u gradient.
    // Values bigger than this threshold will be set to the threshold
    // This is to avoid too low weights
    float _ugrad_threshold;

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

    // mem sizes
    size_t _nfbytes_lo; // _w * _h * sizeof(float);
    size_t _nfbytes_hi; // _W * _H * sizeof(float);

    // input view data
    std::vector<ViewData*> _views;

    // normalization mask, in low res, device
    float* _norm_mask;

    // gradient of the target view, gamma domain, device
    float* _u_grad_x;
    float* _u_grad_y;

    // target view, current solution
    std::vector<float*> _U;

    // target mask, gamma domain, device
    float* _target_mask;

    // backward visibility mask (gamma domain)
    bool* _visibility_mask;

    // temporary vectors
    std::vector<float*> _temp;

    // CUDA block dimensions
    // low res
    dim3 _dimGrid;
    dim3 _dimBlock;
    // high res
    dim3 _DimGrid;
    dim3 _DimBlock;

    // output directory
    std::string _outdir;

    // image initialization
    std::string _init_name;
};

// Get current solution
void get_solution( Data* data, std::vector<coco::gsl_matrix*> &U );

// Compute backward visibility (gamma domain)
void setup_target_visibility( Data* data );

// Perform TV on init image to fill holes
void hole_filling( Data* data );

// Setup unstructured data algorithm: init view and resolution data
Data* init_data( Config_data *config_data );

// Free up data for unstructured SR algorithm
void free_data( Data* data );

// Compute the sparse matrix B
void compute_sparse_matrix( Data* data );

// Init forward warp for a view : uses warps (make sure they are computed)
// warp=0: tau, warp=1:beta
// Currently completely on host, TODO: try to parallelize (hard)
void init_forward_warp_structure( Data* data, size_t nview );

// Setup a single view
void create_view( Data* data, size_t nview, coco::gsl_image *I);

// Update weight_omega_k
void compute_weights( Data* data );

// Read the tau warps and deformation weights: from gsl_image to device float*
void read_tau( Data* data, coco::gsl_image** tau_warps );

// Read the partial tau: from gsl_image to device float*
void read_partial_tau( Data* data, coco::gsl_image** partial_tau );

// Compute the initial image, starting point of the algorithm
void compute_initial_image( Data* data );

// Write current solution in pfm format
void write_pfm_solution( Data* data );

// initialize the gradient of the target view by splatting
void init_u_gradient( Data* data );

// Perform Poisson integration with Jacobi method
void poisson_jacobi( Data* data );

#endif // #ifndef GRADIENT_IBR_CUH
