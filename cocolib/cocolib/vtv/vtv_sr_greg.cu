/* -*-c++-*- */
#include <iostream>
#include <algorithm>

#include "vtv.h"
#include "vtv.cuh"
#include "vtv_sr_greg.cuh"

#include "../common/gsl_matrix_helper.h"
#include "../common/gsl_matrix_convolutions.h"

#include "../cuda/cuda_helper.h"
#include "../cuda/cuda_reduce.h"
#include "../cuda/cuda_kernels.cuh"
#include "../cuda/cuda_convolutions.h"
#include "../defs.h"
#include "../common/profiler.h"
#include "../common/linalg3d.h"

#include "../cuda/cuda_inline_device_functions.cu"
#include "vtv_sr_kernels.cu"
#include "vtv_sr_kernels_greg.cu"

using namespace std;

// 0: backward warping, 1: splatting, 2: hybrid
typedef enum  {
    backwards_warp,
    forward_splatting,
    hybrid_warp
} t_warp_method;

const t_warp_method warping = backwards_warp;//forward_splatting;// backwards_warp;

typedef enum {
  t_nearest_neighbor,
  t_bilinear_interpolation,
  t_bicubic_interpolation
} t_sampling_method;

// 0: nearest neighbor, 1: bilinear interpolation, 2: bicubic interpolation
const t_sampling_method sampling = t_bilinear_interpolation;

const float INVALID_DEPTH = -1e30;

/*****************************************************************************
       TV_x Superresolution
*****************************************************************************/

// Setup SR algorithm: init view and resolution data (Greg)
bool coco::coco_vtv_sr_init_unstructured( coco_vtv_data *data, size_t nviews, size_t ds_factor, bool lumigraph )
{
    // can only be initialized once.
    assert( data->_sr_data == NULL );
    coco_vtv_sr_data *sr = new coco_vtv_sr_data;
    size_t W = data->_W;
    size_t H = data->_H;
    sr->_nviews = nviews;
    sr->_dsf = ds_factor;
    sr->_w = W / sr->_dsf;
    sr->_h = H / sr->_dsf;
    // validate downscale factor (exact multiple of size)
    assert( sr->_w * sr->_dsf == W );
    assert( sr->_h * sr->_dsf == H );

    sr->_lumigraph = lumigraph;

    // default for 8-bit normalized
    sr->_sigma_sensor = 1./255.0f;
    sr->_aux_dmap_sigma = 0.1;
    sr->_ugrad_threshold = 0.01;

    // compute mem layout
    sr->_nfbytes_lo = sr->_w * sr->_h * sizeof(float);
    sr->_nfbytes_hi = W*H*sizeof(float);
    sr->_vmask_bytes = W*H*sizeof(float);
    sr->_dimBlock = dim3( cuda_default_block_size_x(),
                          cuda_default_block_size_y() );
    size_t blocks_w = sr->_w / sr->_dimBlock.x;
    if ( sr->_w % sr->_dimBlock.x != 0 ) {
        blocks_w += 1;
    }
    size_t blocks_h = sr->_h / sr->_dimBlock.y;
    if ( sr->_h % sr->_dimBlock.y != 0 ) {
        blocks_h += 1;
    }
    sr->_dimGrid = dim3(blocks_w, blocks_h);

    // create views
    //int nwarp = 4;
    //sr->_nwarp = nwarp;
    sr->_disp_max = 0.0f;

    size_t MB = 1048576;
    size_t bytes_per_view = data->_nchannels *  sr->_nfbytes_lo // image_f
            + sr->_nfbytes_lo // _vmask_lo
            + sr->_vmask_bytes  // _vmask
            + 8 * sr->_nfbytes_hi;  // warps (beta x/y, tau x/y) and dparts (dparts x/y) and deformation weights (tau and beta)
    size_t bytes_view_total = nviews * bytes_per_view;
    if ( sr->_lumigraph ) { // we add the lumigraph weights
        bytes_per_view += sr->_nfbytes_hi;
    }
    TRACE( "Allocating mem:" << endl );
    TRACE( "  " << bytes_per_view / MB << " Mb per view, " << bytes_view_total/MB << " total." << endl );

    for ( size_t i=0; i<nviews; i++ ) {
        coco_vtv_sr_view_data *view = new coco_vtv_sr_view_data;

        // disparities are allocated in host
        view->_dmap_v = (cuflt*) malloc(W*H * sizeof(cuflt));
        memset(view->_dmap_v, 0.0, W*H * sizeof(cuflt));
        view->_dmap_sigma = (cuflt*) malloc(W*H * sizeof(cuflt));
        memset(view->_dmap_sigma, 0.0, W*H * sizeof(cuflt));

        CUDA_SAFE_CALL( cudaMalloc( &view->_image_f, data->_nchannels * sr->_nfbytes_lo ));
        CUDA_SAFE_CALL( cudaMemset( view->_image_f, 0, data->_nchannels * sr->_nfbytes_lo ));

        CUDA_SAFE_CALL( cudaMalloc( &view->_visibility_mask, W*H * sizeof(bool) ));
        CUDA_SAFE_CALL( cudaMemset( view->_visibility_mask, 0, W*H * sizeof(bool) ));

        CUDA_SAFE_CALL( cudaMalloc( &view->warp_beta_x, W*H * sizeof(cuflt) ));
        CUDA_SAFE_CALL( cudaMemset( view->warp_beta_x, 0.0, W*H * sizeof(cuflt) ));
        CUDA_SAFE_CALL( cudaMalloc( &view->warp_beta_y, W*H * sizeof(cuflt) ));
        CUDA_SAFE_CALL( cudaMemset( view->warp_beta_y, 0.0, W*H * sizeof(cuflt) ));
        CUDA_SAFE_CALL( cudaMalloc( &view->warp_tau_x, W*H * sizeof(cuflt) ));
        CUDA_SAFE_CALL( cudaMemset( view->warp_tau_x, 0.0, W*H * sizeof(cuflt) ));
        CUDA_SAFE_CALL( cudaMalloc( &view->warp_tau_y, W*H * sizeof(cuflt) ));
        CUDA_SAFE_CALL( cudaMemset( view->warp_tau_y, 0.0, W*H * sizeof(cuflt) ));
        CUDA_SAFE_CALL( cudaMalloc( &view->dpart_x, W*H * sizeof(cuflt) ));
        CUDA_SAFE_CALL( cudaMemset( view->dpart_x, 0.0, W*H * sizeof(cuflt) ));
        CUDA_SAFE_CALL( cudaMalloc( &view->dpart_y, W*H * sizeof(cuflt) ));
        CUDA_SAFE_CALL( cudaMemset( view->dpart_y, 0.0, W*H * sizeof(cuflt) ));
        CUDA_SAFE_CALL( cudaMalloc( &view->deform_weight_tau, W*H * sizeof(cuflt) ));
        CUDA_SAFE_CALL( cudaMemset( view->deform_weight_tau, 0.0, W*H * sizeof(cuflt) ));
        CUDA_SAFE_CALL( cudaMalloc( &view->deform_weight_beta, W*H * sizeof(cuflt) ));
        CUDA_SAFE_CALL( cudaMemset( view->deform_weight_beta, 0.0, W*H * sizeof(cuflt) ));

        if ( sr->_lumigraph ) { // we add the lumigraph weights

            CUDA_SAFE_CALL( cudaMalloc( &view->lumi_weights, W*H * sizeof(cuflt) ));
            CUDA_SAFE_CALL( cudaMemset( view->lumi_weights, 0.0, W*H * sizeof(cuflt) ));
        }

        CUDA_SAFE_CALL( cudaMalloc( &(view->_vmask_lo), sr->_nfbytes_lo ));
        CUDA_SAFE_CALL( cudaMemset( view->_vmask_lo, 0, sr->_nfbytes_lo ));

        CUDA_SAFE_CALL( cudaMalloc( &(view->_vmask), sr->_vmask_bytes ));
        CUDA_SAFE_CALL( cudaMemset( view->_vmask, 0, sr->_vmask_bytes ));

        view->_cells_tau = NULL;
        view->_cells_beta = NULL;
        view->_dmap_u = NULL;

        // done
        sr->_views.push_back( view );
    }

    // Additional work mem (TODO: reduce, use temp buffers w->F[...])
    size_t srbytes = sr->_nfbytes_hi
            + sr->_nfbytes_hi;
    TRACE( "  " << srbytes/MB << " Mb for additional work structures." << endl );

    // Target coverage
    CUDA_SAFE_CALL( cudaMalloc( &sr->_dmap_u, sr->_nfbytes_hi ));
    CUDA_SAFE_CALL( cudaMemset( sr->_dmap_u, 0, sr->_nfbytes_hi ));
    CUDA_SAFE_CALL( cudaMalloc( &(sr->_target_mask), sr->_nfbytes_hi ));
    CUDA_SAFE_CALL( cudaMemset( sr->_target_mask, 0, sr->_nfbytes_hi ));

    // Check for grayscale and add temp buffers if necessary
    coco_vtv_workspace *w = data->_workspace;
    while ( w->_temp.size() < data->_nchannels+2 ) {
        float *tmp = NULL;
        CUDA_SAFE_CALL( cudaMalloc( &tmp, sr->_nfbytes_hi ));
        CUDA_SAFE_CALL( cudaMemset( tmp, 0, sr->_nfbytes_hi ));
        w->_temp.push_back( tmp );
    }

    // Filter for visibility masks
    gsl_vector *gaussian = gsl_kernel_gauss_1xn( 11, 2.0f );
    sr->_vmask_filter = cuda_kernel_alloc_separable( gaussian, gaussian );
    gsl_vector_free( gaussian );

    // Finalize
    data->_sr_data = sr;
    return true;
}

// Free up data for SR algorithm (Greg)
bool coco::coco_vtv_sr_free_unstructured( coco_vtv_data *data )
{
    assert( data != NULL );
    coco_vtv_sr_data *sr = data->_sr_data;
    assert( sr != NULL );

    for ( size_t i=0; i<sr->_nviews; i++ ) {
        coco_vtv_sr_view_data *view = sr->_views[i];
        CUDA_SAFE_CALL( cudaFree( view->_image_f ));
        CUDA_SAFE_CALL( cudaFree( view->_visibility_mask ));
        CUDA_SAFE_CALL( cudaFree( view->_cells_tau ));
        CUDA_SAFE_CALL( cudaFree( view->_cells_beta ));
        CUDA_SAFE_CALL( cudaFree( view->warp_beta_x ));
        CUDA_SAFE_CALL( cudaFree( view->warp_beta_y ));
        CUDA_SAFE_CALL( cudaFree( view->warp_tau_x ));
        CUDA_SAFE_CALL( cudaFree( view->warp_tau_y ));
        CUDA_SAFE_CALL( cudaFree( view->dpart_x ));
        CUDA_SAFE_CALL( cudaFree( view->dpart_y ));
        CUDA_SAFE_CALL( cudaFree( view->deform_weight_tau ));
        CUDA_SAFE_CALL( cudaFree( view->deform_weight_beta ));
        if ( sr->_lumigraph ) {
            CUDA_SAFE_CALL( cudaFree( view->lumi_weights ));
        }

        CUDA_SAFE_CALL( cudaFree( view->_vmask ));
        CUDA_SAFE_CALL( cudaFree( view->_vmask_lo ));

        delete view->_dmap_u;
        delete view->_dmap_v;
        delete view->_dmap_sigma;

        delete view;
    }

    write_pfm_image_signed(data->_W, data->_H, sr->_target_mask, data->_basedir + "target_mask.pfm", 0);

    CUDA_SAFE_CALL( cudaFree( sr->_target_mask ));

    CUDA_SAFE_CALL( cudaFree( sr->_dmap_u ));
    cuda_kernel_free( sr->_vmask_filter );

    // finalize
    delete data->_sr_data;
    data->_sr_data = NULL;
    return true;
}

// If the view to synthsize is one the input, remove it (Greg)
void coco::coco_vtv_sr_remove_last_view_unstructured( coco_vtv_data *data )
{
    assert( data != NULL );
    coco_vtv_sr_data *sr = data->_sr_data;
    assert( sr != NULL );
    coco_vtv_sr_view_data *view = sr->_views.back();
    CUDA_SAFE_CALL( cudaFree( view->_image_f ));
    CUDA_SAFE_CALL( cudaFree( view->_vmask ));
    CUDA_SAFE_CALL( cudaFree( view->_vmask_lo ));
    CUDA_SAFE_CALL( cudaFree( view->_visibility_mask ));
    CUDA_SAFE_CALL( cudaFree( view->_cells_tau ));
    CUDA_SAFE_CALL( cudaFree( view->_cells_beta ));
    CUDA_SAFE_CALL( cudaFree( view->dpart_x ));
    CUDA_SAFE_CALL( cudaFree( view->dpart_y ));
    CUDA_SAFE_CALL( cudaFree( view->warp_beta_x ));
    CUDA_SAFE_CALL( cudaFree( view->warp_beta_y ));
    CUDA_SAFE_CALL( cudaFree( view->warp_tau_x ));
    CUDA_SAFE_CALL( cudaFree( view->warp_tau_y ));
    CUDA_SAFE_CALL( cudaFree( view->deform_weight_tau ));
    CUDA_SAFE_CALL( cudaFree( view->deform_weight_beta ));
    if ( sr->_lumigraph ) {
        CUDA_SAFE_CALL( cudaFree( view->lumi_weights ));
    }
    delete view->_dmap_v;
    delete view->_dmap_sigma;
    delete[] view->_dmap_u;
    delete view;
    sr->_views.pop_back();
    sr->_nviews = data->_sr_data->_views.size();
}

// Setup a single view (Greg)
bool coco::coco_vtv_sr_create_view_unstructured( coco_vtv_data *data, size_t nview,
                                                 gsl_image *I)
{
    // check for required data
    assert( data != NULL );
    coco_vtv_sr_data *sr = data->_sr_data;
    assert( sr != NULL );
    size_t W = data->_W;
    size_t H = data->_H;

    assert( W*H > 0 );
    assert( nview < sr->_nviews );
    coco_vtv_sr_view_data *view = sr->_views[nview];

    // view image should be equal to downsampled size
    assert( I->_w == W / sr->_dsf );
    assert( I->_h == H / sr->_dsf );

    // Image
    size_t N = sr->_w * sr->_h;
    float *buffer_f = new float[ N*data->_nchannels ];

    for ( size_t n=0; n<data->_nchannels; n++ ) {
        // load view to device
        gsl_matrix *channel = gsl_image_get_channel( I, (coco::gsl_image_channel)n );

        for ( size_t i=0; i<N; i++ ) {
            buffer_f[N*n+i] = (float)channel->data[i];
        }
    }

    CUDA_SAFE_CALL( cudaMemcpy( view->_image_f, buffer_f, data->_nchannels*N*sizeof(float), cudaMemcpyHostToDevice ));
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    delete[] buffer_f;

    return true;
}


// Load disparities to test unstructured sr (Greg)
bool coco::coco_vtv_sr_load_disparities_unstructured( coco_vtv_data *data, size_t nview,
                                                      double dx, double dy,
                                                      float *disparity, float *disp_sigma)
{
    // check for required data
    assert( data != NULL );
    coco_vtv_sr_data *sr = data->_sr_data;
    assert( sr != NULL );
    size_t W = data->_W;
    size_t H = data->_H;
    assert( W*H > 0 );
    assert( nview < sr->_nviews );
    coco_vtv_sr_view_data *view = sr->_views[nview];

    size_t dsf = sr->_dsf;
    size_t w = W / dsf;
    size_t h = H / dsf;

    // Warp geometry
    view->_dx_vu = dx*dsf;
    view->_dy_vu = -dy*dsf;

    float *dmap_lo = (float*) malloc(w*h*sizeof(float)); //low res dmap
    memcpy(dmap_lo, disparity, w*h*sizeof(float));
    // upsample to hires DMAP; values are not scaled
    for ( int oy=0; oy<H; oy++ ) {

        for ( int ox=0; ox<W; ox++ ) {

            int px = int(float(ox) / sr->_dsf); // coordinate truncation
            int py = int(float(oy) / sr->_dsf);
            if ( px>=w || py>=h ) {
                continue;
            }

            view->_dmap_v[ox+oy*W] = dmap_lo[px+py*w];
        }
    }

    // Disparity confidence -- reuse dmap_lo temp var
    // if it's NULL set it to 0
    bool free_it = false;
    if ( disp_sigma == 0 ) {
        disp_sigma = (float*) malloc( w*h*sizeof(float));
        memset(disp_sigma, 0.0, w*h*sizeof(float));
        free_it = true;
    }

    memcpy(dmap_lo, disp_sigma, w*h*sizeof(float));

    if ( free_it ) {
        free (disp_sigma);
    }
    // upsample to hires; values are not scaled
    for ( int oy=0; oy<H; oy++ ) {

        for ( int ox=0; ox<W; ox++ ) {

            int px = int(float(ox) / sr->_dsf); // coordinate truncation
            int py = int(float(oy) / sr->_dsf);
            if ( px>=w || py>=h ) {
                continue;
            }

            view->_dmap_sigma[ox+oy*W] = dmap_lo[px+py*w];
        }
    }

    delete dmap_lo;

    // Vote for u_dmap
    vtv_sr_init_u_dmap_unstructured( data, nview );

    // max disparity
    for ( size_t i=0; i<w*h; i++ ) {
        float d = fabs( disparity[i] );
        if ( d > sr->_disp_max ) {
            sr->_disp_max = d;
        }
    }

    return true;
}

// compute the target disparity map using the votes dmap_u[i] (Goldluecke)
bool coco::vtv_sr_init_target_disparity_map_unstructured( coco_vtv_data *data )
{
    // check for required data
    assert( data != NULL );
    coco_vtv_sr_data *sr = data->_sr_data;
    assert( sr != NULL );
    size_t W = data->_W;
    size_t H = data->_H;
    assert( W*H > 0 );
    size_t N = W*H;

    // Temp buffers
    float *dmap_u = new float[N];

    // Median filter of all depth votes
    TRACE( "init target depth map ..." );
    for ( size_t i=0; i<N; i++ ) {
        vector<float> votes;
        for ( size_t nview=0; nview<sr->_nviews; nview++ ) {
            coco_vtv_sr_view_data *view = sr->_views[nview];
            float D = view->_dmap_u[i];
            if ( D != INVALID_DEPTH ) {
                votes.push_back( D );
            }
        }

        // if nobody votes, value is invalid (INVALID_DEPTH)
        if ( votes.size() == 0 ) {
            dmap_u[i] = INVALID_DEPTH;
        }
        else {
            // median filter? max? test what works best here.
            sort( votes.begin(), votes.end() );
            dmap_u[i] = votes[ votes.size() - 1 ];
            //dmap_u[i] = votes[ votes.size() / 2 ];
        }
    }
    TRACE( " done." << endl );

    // copy to device and cleanup
    CUDA_SAFE_CALL( cudaMemcpy( sr->_dmap_u, dmap_u, sizeof(float) * N, cudaMemcpyHostToDevice ));
    delete[] dmap_u;
    return true;
}

// Init forward warp for a view : uses warps (make sure they are computed) (Greg)
// warp=0: tau, warp=1:beta
// Currently completely on host, TODO: try to parallelize (hard)
bool coco::vtv_sr_init_forward_warp_structure_unstructured( coco_vtv_data *data, size_t nview, bool direction )
{
    // check for required data
    assert( data != NULL );
    coco_vtv_sr_data *sr = data->_sr_data;
    assert( sr != NULL );
    int W = data->_W;
    int H = data->_H;
    assert( W*H > 0 );
    int N = W*H;
    coco_vtv_workspace *ws = data->_workspace;
    assert( ws != NULL );
    assert( nview < sr->_nviews );
    coco_vtv_sr_view_data *view = sr->_views[nview];

    // Need warps from GPU
    cuflt *tmp_warp_x = new cuflt[N];
    cuflt *tmp_warp_y = new cuflt[N];
    if ( direction == 0 ) { // forward warp is tau
        CUDA_SAFE_CALL( cudaMemcpy( tmp_warp_x, view->warp_tau_x, sizeof(cuflt) * N, cudaMemcpyDeviceToHost ));
        CUDA_SAFE_CALL( cudaMemcpy( tmp_warp_y, view->warp_tau_y, sizeof(cuflt) * N, cudaMemcpyDeviceToHost ));
        view->_seg_end_tau.clear();
    } else { // forward warp is beta
        CUDA_SAFE_CALL( cudaMemcpy( tmp_warp_x, view->warp_beta_x, sizeof(cuflt) * N, cudaMemcpyDeviceToHost ));
        CUDA_SAFE_CALL( cudaMemcpy( tmp_warp_y, view->warp_beta_y, sizeof(cuflt) * N, cudaMemcpyDeviceToHost ));
        view->_seg_end_beta.clear();
    }

    // Compute target cells for each source pixel
    int *c_in = new int[ N ];
    for ( int oy=0; oy<H; oy++ ) {
        for ( int ox=0; ox<W; ox++ ) {
            int o = ox + oy*W;
            if ( tmp_warp_x[o] < 0 || tmp_warp_y[o] < 0 || tmp_warp_x[o] > W || tmp_warp_y[o] > H  ) {
                c_in[ o ] = N;
                continue;
            }

            // get location in u
            cuflt uxv = tmp_warp_x[o] - 0.5;
            cuflt uyv = tmp_warp_y[o] - 0.5;
            int px = (int)floor(uxv);
            int py = (int)floor(uyv);
            if ( px < 0 || py < 0 || px > W-1 || py > H-1 ) {
                c_in[ o ] = N;
                continue;
            }
            int po = px + py*W;
            c_in[ o ] = po;
        }
    }

    // Group into non-overlapping segments
    // Needs array c_in
    TRACE6( "grouping cells ..." );

    int *grouped = new int[ N ];
    int ngrouped = 0;
    memset( grouped, 0, sizeof(int) * N );
    int *count = new int[ N ];
    vector<int> cells;

    // BUG FIX: the variable margin was introduced because :
    //          in some cases the forward warp computed on CPU gives a slightly different result
    //          as in GPU. The extreme case being  floor(CPU value) != floor(GPU value)
    //          This makes that the non-overlapping segments, in fact, overlap,
    //          creating a non-determined behaviour when threads collide.
    //          Expanding the "confort zone" to a 3x3 neighborhood solves for this problem.
    //          The "drawback" is that there are slightly more segments.
    int margin = 1;

    while ( ngrouped < N ) {

        memset( count, 0, sizeof(int) * N );
        for ( int i=0 ; i < N ; i++ ) {
            if ( grouped[i] ) {
                continue;
            }

            int target = c_in[i];

            // check targets is unused
            if ( target == N ) {
                grouped[i] = 1;
                ngrouped++;
                continue;
            }

            bool ok = true;
            int px = target % W;
            int py = target / W;

            for (int x = -margin; x<=margin; ++x) {
                if (0<= px+x && px+x <W ) {
                    for (int y = -margin; y<=margin; ++y) {
                        if (0<= py+y && py+y <H ) {
                            if ( count[px+x + (py+y) *W] != 0 ) {
                                ok = false;
                            }
                        }
                    }
                }
            }
            if ( !ok ) {
                continue;
            }

            // add cell to group, mark all targets as used
            cells.push_back( i );
            ngrouped++;
            grouped[i] = 1;

            for (int x = -margin; x<=margin; ++x) {
                if (0<= px+x && px+x <W ) {
                    for (int y = -margin; y<=margin; ++y) {
                        if (0<= py+y && py+y <H ) {
                            assert ( count[px+x + (py+y) *W] == 0 );
                            count[px+x + (py+y) *W] = 1;
                        }
                    }
                }
            }
        }

        if ( direction == 0 ) { // forward warp is tau

            view->_seg_end_tau.push_back( cells.size() );

        } else { // forward warp is beta

            view->_seg_end_beta.push_back( cells.size() );
        }

        TRACE6( "  ... " << ngrouped << " grouped, " << cells.size() << " cells." << endl );
    }
    TRACE6( "done." << endl );
    assert( ngrouped == N );

    // Copy new cell grouping to GPU
    if ( direction == 0 ) { // forward warp is tau

        if ( view->_cells_tau != NULL ) {
            CUDA_SAFE_CALL( cudaFree( view->_cells_tau ));
        }
        CUDA_SAFE_CALL( cudaMalloc( &view->_cells_tau, sizeof(int) * cells.size() ));
        CUDA_SAFE_CALL( cudaMemcpy( view->_cells_tau, &cells[0], sizeof(int) * cells.size(), cudaMemcpyHostToDevice ));
    } else { // forward warp is beta

        if ( view->_cells_beta != NULL ) {
            CUDA_SAFE_CALL( cudaFree( view->_cells_beta ));
        }
        CUDA_SAFE_CALL( cudaMalloc( &view->_cells_beta, sizeof(int) * cells.size() ));
        CUDA_SAFE_CALL( cudaMemcpy( view->_cells_beta, &cells[0], sizeof(int) * cells.size(), cudaMemcpyHostToDevice ));
    }

    // Cleanup
    delete[] tmp_warp_x;
    delete[] tmp_warp_y;
    delete[] grouped;
    delete[] count;
    delete[] c_in;
    return true;
}


// Currently completely on host, TODO: try to parallelize (hard)
// fills _dmap_u with the depth of the warped pixels (Greg)
bool coco::vtv_sr_init_u_dmap_unstructured( coco_vtv_data *data, size_t nview )
{
    // check for required data
    assert( data != NULL );
    coco_vtv_sr_data *sr = data->_sr_data;
    assert( sr != NULL );
    int W = data->_W;
    int H = data->_H;
    assert( W*H > 0 );
    int N = W*H;
    assert( nview < sr->_nviews );
    coco_vtv_sr_view_data *view = sr->_views[nview];

    // Initialize Mask and u depth map
    view->_dmap_u = new float[ N ];
    for ( int i=0; i<N; i++ ) {
        view->_dmap_u[i] = INVALID_DEPTH;
    }

    // write target depth to all cells (_dmap_u)
    for ( int oy=0; oy<H; oy++ ) {
        for ( int ox=0; ox<W; ox++ ) {
            int o = ox + oy*W;

            // get location in u
            float d = view->_dmap_v[o];
            float uxv = ox + d * view->_dx_vu;
            float uyv = oy + d * view->_dy_vu;
            int px = (int)floor(uxv);
            int py = (int)floor(uyv);
            if ( px<0 || py<0 || px>W-2 || py>H-2 ) {
                continue;
            }

            int po = px + py*W;
            if ( d > view->_dmap_u[po] ) {
                view->_dmap_u[po] = d;
            }
            if ( px<W-1 ) {
                if ( d > view->_dmap_u[po+1] ) {
                    view->_dmap_u[po+1] = d;
                }
            }
            if ( py<H-1 ) {
                if ( d > view->_dmap_u[po+W] ) {
                    view->_dmap_u[po+W] = d;
                }
            }
            if (  px<W-1  && py<H-1 ) {
                if ( d > view->_dmap_u[po+W+1] ) {
                    view->_dmap_u[po+W+1] = d;
                }
            }
        }
    }

    return true;
}

// Compute the beta warped image u for a single view vi to test only (Greg)
bool coco::coco_vtv_sr_compute_warped_image( coco_vtv_data *data, size_t nview )
{
    // check for required data
    assert( data != NULL );
    coco_vtv_sr_data *sr = data->_sr_data;
    assert( sr != NULL );
    size_t W = data->_W;
    size_t H = data->_H;
    assert( W*H > 0 );
    coco_vtv_workspace *w = data->_workspace;
    assert( nview < sr->_nviews );
    coco_vtv_sr_view_data *view = sr->_views[nview];

    // Temp buffers
    float *tmp_viewvi = w->_temp[0];

    // Add view contributions via backwards warp
    for ( size_t i=0; i<data->_nchannels; i++ ) {
        // Clear target image and masks
        CUDA_SAFE_CALL( cudaMemset( w->_U[i], 0.0f, w->_nfbytes ));

        // Upsample view in V
        vtv_sr_bilinear_upsample_device<<< w->_dimGrid, w->_dimBlock >>>( W, H, // Hi-res size
                                                                          sr->_w, sr->_h,// Lo-res size
                                                                          sr->_dsf,      // Scale factor
                                                                          view->_image_f + i*sr->_w*sr->_h, // lo-res matrix
                                                                          tmp_viewvi,    // hi-res result
                                                                          view->_visibility_mask);
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        coco_vtv_sr_warp_unstructured( data, nview, Gamma_from_Omegai, tmp_viewvi, w->_U[i] );
    }

    return true;
}

// Compute tau and beta warps and dpart of a single view  (Greg)
bool coco::coco_vtv_sr_warps_from_disp( coco_vtv_data *data )
{
    // check for required data
    assert( data != NULL );
    coco_vtv_sr_data *sr = data->_sr_data;
    assert( sr != NULL );
    size_t W = data->_W;
    size_t H = data->_H;
    size_t N = W*H;
    assert( N > 0 );
    coco_vtv_workspace *ws = data->_workspace;
    assert( ws != NULL );

    // Init target disparity map
    vtv_sr_init_target_disparity_map_unstructured( data );

    for (size_t nview = 0 ; nview < sr->_nviews ; nview++) {

        // compute cell structure (non-overlapping segments
        coco_vtv_sr_view_data *view = sr->_views[nview];

        float *tmp_disp = ws->_temp[0]; // tmp disparity
        CUDA_SAFE_CALL( cudaMemcpy( tmp_disp, view->_dmap_v, sr->_nfbytes_hi, cudaMemcpyHostToDevice ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        // Compute binary visibility mask comparing _dmap_v to _dmap_u using _disp_threshold
        vtv_sr_compute_visibility_mask_device<<< ws->_dimGrid, ws->_dimBlock >>>
                                                                               ( W,H, tmp_disp, view->_dx_vu, view->_dy_vu, sr->_dmap_u,
                                                                                 view->_visibility_mask, data->_disp_threshold);
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        // compute tau warp and deformation weights |det D tau|^(-1)
        coco_vtv_sr_compute_tau_warp( data, nview );

        // compute non-overlapping segments (cell structure)
        // just tau cells (to create beta warps afterwards
        vtv_sr_init_forward_warp_structure_unstructured( data, nview, 0 );

        // compute beta warp and deformation weights |det D beta|^(-1)
        coco_vtv_sr_compute_beta_warp( data, nview );

        // compute dpart
        coco_vtv_sr_compute_dpart( data, nview );
    }

    return true;
}

// Compute the tau warp of a view (Greg)
// Also compute the deformation weights
bool coco::coco_vtv_sr_compute_tau_warp( coco_vtv_data *data, size_t nview )
{
    // check for required data
    assert( data != NULL );
    coco_vtv_sr_data *sr = data->_sr_data;
    assert( sr != NULL );
    size_t W = data->_W;
    size_t H = data->_H;
    assert( W*H > 0 );
    coco_vtv_workspace *w = data->_workspace;
    assert( nview < sr->_nviews );
    coco_vtv_sr_view_data *view = sr->_views[nview];

    // Clear warps and tmp weights
    CUDA_SAFE_CALL( cudaMemset( view->warp_tau_x, 0, sr->_nfbytes_hi ));
    CUDA_SAFE_CALL( cudaMemset( view->warp_tau_y, 0, sr->_nfbytes_hi ));

    // Temp buffers for diparities
    float *tmp_disp = w->_temp[0];
    CUDA_SAFE_CALL( cudaMemcpy( tmp_disp, view->_dmap_v, sr->_nfbytes_hi, cudaMemcpyHostToDevice ));
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    vtv_sr_compute_tau_device<<< w->_dimGrid, w->_dimBlock >>>
                                                             ( data->_W, data->_H,
                                                               tmp_disp, view->_dx_vu, view->_dy_vu,
                                                               view->warp_tau_x, // output warp
                                                               view->warp_tau_y);
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    // compute the deformation weights |det D tau_i|^(-1) // same for all channels
    // when computing the deformation weights, we take the non visible pixels into account
    vtv_sr_init_tau_mask_weight_device<<< w->_dimGrid, w->_dimBlock >>> // overwrite deform_weight_tau
                                                                      ( W, H, view->_visibility_mask, view->warp_tau_x, view->warp_tau_y, view->deform_weight_tau );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    // compute the deformation weights with disparities to compare
    // --------------------------------------------------------------------------------------
    /*
    float *tmp_weight = w->_temp[1];

    vtv_sr_init_mask_gradient_weight_device<<< w->_dimGrid, w->_dimBlock >>> // overwrite tmp_weight
                                                                             ( W,H, view->_dx_vu, view->_dy_vu, tmp_disp, view->_visibility_mask, tmp_weight );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    TRACE("Write deformation weights of view " << nview << std::endl);
    write_pfm_image_signed(W,H, view->deform_weight_tau, data->_basedir + "deform_weight_tau_greg_%03lu.pfm", nview);
    // deformation weights computed with disparities
    write_pfm_image_signed(W,H, tmp_weight, data->_basedir + "deform_weight_tau_gold_%03lu.pfm", nview);
    */
    // --------------------------------------------------------------------------------------

    vtv_sr_set_tau_visibility_device<<< w->_dimGrid, w->_dimBlock >>>
                                                                    ( data->_W, data->_H,
                                                                      view->_visibility_mask,
                                                                      view->warp_tau_x, // output warp
                                                                      view->warp_tau_y);
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    /*
    write_pfm_image_signed(W,H, view->warp_tau_x, data->_basedir + "tau_warp_x_%03lu.pfm", nview);
    write_pfm_image_signed(W,H, view->warp_tau_y, data->_basedir + "tau_warp_y_%03lu.pfm", nview);
    printf( "tau warp x y %03lu", nview );
    TRACE(std::endl);
    */

    printf( "tau warp x y and deformation weights %02lu", nview );
    TRACE(std::endl);
    write_pfm_image_signed(W, H, view->warp_tau_x, view->warp_tau_y, view->deform_weight_tau, data->_basedir + "tau_%02lu.pfm", nview);

    return true;
}

// Compute the beta warp of a view (Greg)
bool coco::coco_vtv_sr_compute_beta_warp( coco_vtv_data *data, size_t nview )
{
    // check for required data
    assert( data != NULL );
    coco_vtv_sr_data *sr = data->_sr_data;
    assert( sr != NULL );
    size_t W = data->_W;
    size_t H = data->_H;
    assert( W*H > 0 );
    coco_vtv_workspace *w = data->_workspace;
    assert( nview < sr->_nviews );
    coco_vtv_sr_view_data *view = sr->_views[nview];

    // Temp buffers
    float *tmp_weights = w->_temp[0];
    float *tmp_disp = w->_temp[1];

    // fill tpm disp
    CUDA_SAFE_CALL( cudaMemcpy( tmp_disp, view->_dmap_v, sr->_nfbytes_hi, cudaMemcpyHostToDevice ));
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    // Clear warps and tmp weights
    CUDA_SAFE_CALL( cudaMemset( tmp_weights, 0, sr->_nfbytes_hi ));
    CUDA_SAFE_CALL( cudaMemset( view->warp_beta_x, 0, sr->_nfbytes_hi ));
    CUDA_SAFE_CALL( cudaMemset( view->warp_beta_y, 0, sr->_nfbytes_hi ));
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    // Forward warp, non-overlap regions sequentially
    // First pass to set the visibility mask, pixels of u visible from vi
    int seg_start = 0;
    for ( size_t j=0; j<view->_seg_end_tau.size(); j++ ) {
        int seg_end = view->_seg_end_tau[j];
        int seg_size = seg_end - seg_start;

        // forward warp call for this segment, cannot overlap
        int seg_width = cuda_default_block_size_x() * cuda_default_block_size_y();
        dim3 dimBlock = dim3( seg_width, 1 );
        dim3 dimGrid = dim3( seg_size / seg_width + 1, 1 );

        vtv_sr_compute_beta_visibility_accumulate_device<<< dimGrid, dimBlock >>>
                                                                                ( data->_W, data->_H, seg_width,
                                                                                  view->_cells_tau, seg_start, seg_end,
                                                                                  tmp_disp, view->_dx_vu, view->_dy_vu,
                                                                                  view->_visibility_mask,
                                                                                  view->warp_beta_x, // output beta visibility
                                                                                  view->warp_beta_y);
        CUDA_SAFE_CALL( cudaThreadSynchronize() );
        seg_start = seg_end;
    }

    // set the non visible pixels to (-W, -H) and the visible ones to 0.0
    vtv_sr_set_beta_visibility_device<<< w->_dimGrid, w->_dimBlock >>>
                                                                     ( data->_W, data->_H,
                                                                       view->warp_beta_x, // output warp
                                                                       view->warp_beta_y);
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    /*
    write_pfm_image_signed( W, H, view->warp_beta_x, view->warp_beta_y, view->warp_beta_y, data->_basedir + "beta_visi_x_%03lu.pfm", nview );
    TRACE( "beta visibility x y " << nview << std::endl );
    */

    // Forward warp, non-overlap regions sequentially
    // Set the value for visible pixels
    seg_start = 0;
    for ( size_t j=0; j<view->_seg_end_tau.size(); j++ ) {
        int seg_end = view->_seg_end_tau[j];
        int seg_size = seg_end - seg_start;

        // forward warp call for this segment, cannot overlap
        int seg_width = cuda_default_block_size_x() * cuda_default_block_size_y();
        dim3 dimBlock = dim3( seg_width, 1 );
        dim3 dimGrid = dim3( seg_size / seg_width + 1, 1 );

        vtv_sr_compute_beta_accumulate_device<<< dimGrid, dimBlock >>>
                                                                     ( data->_W, data->_H, seg_width,
                                                                       view->_cells_tau, seg_start, seg_end,
                                                                       tmp_disp, view->_dx_vu, view->_dy_vu,
                                                                       view->_visibility_mask,
                                                                       view->warp_beta_x, // output warp
                                                                       view->warp_beta_y,
                                                                       tmp_weights ); // output weights
        CUDA_SAFE_CALL( cudaThreadSynchronize() );
        seg_start = seg_end;
    }

    /*
    write_pfm_image_signed(W, H, view->warp_beta_x, view->warp_beta_y, view->deform_weight_beta, data->_basedir + "beta_no_norm%02lu.pfm", nview);
    TRACE( "beta warp x y before normalization " << nview << std::endl);
    */

    // Normalize
    cuda_normalize_device<<< w->_dimGrid, w->_dimBlock >>>
                                                         ( data->_W, data->_H, view->warp_beta_x, tmp_weights );
    cuda_normalize_device<<< w->_dimGrid, w->_dimBlock >>>
                                                         ( data->_W, data->_H, view->warp_beta_y, tmp_weights );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    // compute the deformation weights |det D beta_i|^(-1) // same for all channels
    vtv_sr_init_beta_mask_weight_device<<< w->_dimGrid, w->_dimBlock >>> // overwrite deform_weight_beta
                                                                       ( W, H,
                                                                         view->warp_beta_x, view->warp_beta_y,
                                                                         view->deform_weight_beta );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    write_pfm_image_signed(W, H, view->warp_beta_x, view->warp_beta_y, view->deform_weight_beta, data->_basedir + "beta_%02lu.pfm", nview);
    TRACE( "beta warp x y " << nview << std::endl);

    return true;
}

// Compute the dpart of a view (sigma_dmap * tau partial derivative) (Greg)
bool coco::coco_vtv_sr_compute_dpart( coco_vtv_data *data, size_t nview )
{
    // check for required data
    assert( data != NULL );
    coco_vtv_sr_data *sr = data->_sr_data;
    assert( sr != NULL );
    size_t W = data->_W;
    size_t H = data->_H;
    assert( W*H > 0 );
    coco_vtv_workspace *w = data->_workspace;
    assert( nview < sr->_nviews );
    coco_vtv_sr_view_data *view = sr->_views[nview];

    // Clear dparts
    CUDA_SAFE_CALL( cudaMemset( view->dpart_x, 0, sr->_nfbytes_hi ));
    CUDA_SAFE_CALL( cudaMemset( view->dpart_y, 0, sr->_nfbytes_hi ));

    vtv_sr_compute_dpart_device<<< w->_dimGrid, w->_dimBlock >>>
                                                               ( W, H,
                                                                 view->_dx_vu, view->_dy_vu,
                                                                 view->_visibility_mask,
                                                                 view->dpart_x,
                                                                 view->dpart_y);
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    // create uniform sigma_disp map
    /*
    float *sigma_disp = (float*) malloc(W * H * sizeof(float));
    for (int m=0; m< (W * H); ++m) {
        sigma_disp[m] = sr->_aux_dmap_sigma;
    }
    */
    float *tmp_sigma = w->_temp[0]; // weight in vi tmp buffer (total mask)
    CUDA_SAFE_CALL( cudaMemcpy( tmp_sigma, view->_dmap_sigma, sr->_nfbytes_hi, cudaMemcpyHostToDevice ));

    // hack: we store (c-ci) instead of dtaui/dz and sigma_dmapi instead of sigma_zi so that in the end
    // the product dpart*sigma_dmap=sigma_z*dtau/dz anyway

    write_pfm_image_signed(W, H, tmp_sigma, view->dpart_x, view->dpart_y, data->_basedir + "partial_tau_%02lu.pfm", nview);
    //write_pfm_image_signed(W,H, view->dpart_x, data->_basedir + "dpart_x_%03lu.pfm", nview);
    //write_pfm_image_signed(W,H, view->dpart_y, data->_basedir + "dpart_y_%03lu.pfm", nview);
    printf( "partial_tau x y %02lu", nview );
    TRACE(std::endl);

    return true;
}

// compute visibility mask from the tau warps
bool coco::coco_vtv_visibility_from_tau( coco_vtv_data *data ) {

    // check for required data
    assert( data != NULL );
    coco_vtv_sr_data *sr = data->_sr_data;
    assert( sr != NULL );
    size_t W = data->_W;
    size_t H = data->_H;
    assert( W*H > 0 );
    coco_vtv_workspace *w = data->_workspace;

    for ( size_t nview=0; nview < sr->_nviews; nview++ ) {

        coco_vtv_sr_view_data *view = sr->_views[nview];

        // Clear visibility mask
        CUDA_SAFE_CALL( cudaMemset( view->_visibility_mask, false, W*H * sizeof(bool) ));

        vtv_sr_visibility_from_tau_device<<< w->_dimGrid, w->_dimBlock >>>
                                                                         ( W, H,
                                                                           view->warp_tau_x, // input tau warp
                                                                           view->warp_tau_y,
                                                                           view->_visibility_mask); // output visibility mask

        CUDA_SAFE_CALL( cudaThreadSynchronize() );


        //write_test_image_bool(W,H, view->_visibility_mask, data->_basedir + "visibility_mask%03lu.png", nview, 0);
        //printf( "visibility_mask of view%03lu", nview );
        //TRACE(std::endl)
    }

    return true;
}

static int meta_iter = -1;

// Update the weights (Greg)
bool coco::coco_vtv_sr_compute_weights_unstructured( coco_vtv_data *data ) {

    // check for required data
    assert( data != NULL );
    coco_vtv_sr_data *sr = data->_sr_data;
    assert( sr != NULL );
    size_t W = data->_W;
    size_t H = data->_H;
    assert( W*H > 0 );
    coco_vtv_workspace *ws = data->_workspace;
    assert(ws->_nfbytes == sr->_nfbytes_hi);

    //float *luminance = ws->_temp[0];
    //CUDA_SAFE_CALL( cudaMemset( luminance, 0, sr->_nfbytes_hi ));

    meta_iter++;

    if (!( sr->_lumigraph )) {
        if (data->_nchannels == 1) {
            cuda_compute_gradient_device <<< ws->_dimGrid, ws->_dimBlock >>>
                                                                           ( W,H, ws->_U[0], ws->_X1[0], ws->_X2[0]);
        } else { // rgb

            vtv_sr_compute_gradient_device <<< ws->_dimGrid, ws->_dimBlock >>>
                            ( W,H, ws->_U[0], ws->_U[1], ws->_U[2], sr->_target_mask, ws->_X1[0], ws->_X2[0]);
        }
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );


        //compute_luminance_device <<< ws->_dimGrid, ws->_dimBlock >>>
        //    ( W,H, ws->_U[0], ws->_U[1], ws->_U[2], luminance);
        //CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    }

    //write_pfm_image_signed(W, H,ws->_U[0], ws->_U[1], ws->_U[2], data->_basedir + "u_%02d.pfm", meta_iter);
    //write_pfm_image_signed(W, H,ws->_Uq[0], ws->_Uq[1], ws->_Uq[2], data->_basedir + "uq_%02d.pfm", meta_iter);

    // init _target_mask
    // accumulation of _vmask contributions will be stored into it
    CUDA_SAFE_CALL( cudaMemset( sr->_target_mask, 0, sr->_nfbytes_hi ));

    // Temp buffers
    float *tmp_weights = ws->_temp[0]; // weight in vi tmp buffer (target mask)

    // update each view
    for ( size_t nview=0; nview < sr->_nviews; nview++ ) {
        coco_vtv_sr_view_data *view = sr->_views[nview];

        CUDA_SAFE_CALL( cudaMemset( view->_vmask, 0, sr->_nfbytes_hi ));

        //write_test_image_bool( W,H, view->_visibility_mask, data->_basedir + "visibility_mask_prev_%05i.png", nview );

        if (!( sr->_lumigraph )) {
            // compute weights with u gradient
            // dot product of grad u with partial tau partial z

            cuda_dot_product_device <<< ws->_dimGrid, ws->_dimBlock >>>
               (W, H, ws->_X1[0], ws->_X2[0], view->dpart_x, view->dpart_y, view->_vmask);
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            filter_invalid_device <<< ws->_dimGrid, ws->_dimBlock >>>  (W,H, view->_vmask, view->warp_beta_x);
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            //write_pfm_image_signed(W, H, view->_vmask, data->_basedir + "sigma_z_grad_%03d.pfm", meta_iter*100+nview);

            // compute variance
            //cuda_variance_epipolar_device <<< ws->_dimGrid, ws->_dimBlock >>>
            //            (W, H, luminance, view->dpart_x, view->dpart_y, view->warp_beta_x, view->_vmask);
            //        CUDA_SAFE_CALL( cudaDeviceSynchronize() );
            //write_pfm_image_signed(W, H, view->_vmask, data->_basedir + "sigma_z_var_%03d.pfm", meta_iter*100+nview);

            // use _X1 as temp variable
            /*
            cuda_probability_epipolar_device <<< ws->_dimGrid, ws->_dimBlock >>>
                        (W, H, view->dpart_x, view->dpart_y, view->warp_beta_x, ws->_X1[0]);
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            write_pfm_image_signed(W, H, ws->_X1[0], data->_basedir + "proba_%03d.pfm", meta_iter*100+nview);
            */

            // WE STAY IN GAMMA
            // warp to Omegai
            //CUDA_SAFE_CALL( cudaMemset( tmp_weights, 0, sr->_nfbytes_hi ));
            //coco_vtv_sr_warp_unstructured( data, nview, Omegai_from_Gamma, view->_vmask, tmp_weights );
            //CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            // DOWNSAMPLING IS 1
            // downsample to obtain sigma_g
            // set invalid value to -1, so that cuda_sigma_g_to_omegai knows it
            //vtv_sr_downsample_view_device<<< sr->_dimGrid, sr->_dimBlock >>>
            //          ( sr->_w, sr->_h, data->_W, tmp_weights, view->_visibility_mask, sr->_dsf, view->_vmask_lo, -1.);
            //CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            //write_pfm_image_signed(sr->_w, sr->_h, view->_vmask_lo, data->_basedir + "grad_u_prod_warped_%03d.pfm", meta_iter*100+nview);

            // now we have sigma_g
            // square it , add it to sigma sensor square and invert
            cuda_sigma_g_to_omegai_def <<< ws->_dimGrid, ws->_dimBlock >>> ( W, H, view->_vmask, view->deform_weight_beta,
                                                                             sr->_sigma_sensor, sr->_ugrad_threshold);
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            if (sr->_ugrad_threshold > 0.5) {
              //cuda_multiply_with_device <<< ws->_dimGrid, ws->_dimBlock >>> ( W, H, view->_vmask, ws->_X1[0]);
            }

            //write_pfm_image_signed(W, H, view->_vmask_lo, data->_basedir + "omega_i_%03d.pfm", meta_iter*100+nview);

            // UPSAMPLE is 1
            // angular weights are now upsampled (without weights) to be multiplied with deformation weights
            /*vtv_sr_bilinear_upsample_device<<< ws->_dimGrid, ws->_dimBlock >>>( W, H, // Hi-res size
                                                                                sr->_w, sr->_h,// Lo-res size
                                                                                sr->_dsf,      // Scale factor
                                                                                view->_vmask_lo,     // lo-res matrix
                                                                                view->_vmask,
                                                                                view->_visibility_mask);    // hi-res result
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );*/


            // multiply with deformation weights in GAMMA
            //multiply_with_deformation_weights<<< ws->_dimGrid, ws->_dimBlock >>> ( W, H, view->_vmask, view->deform_weight_beta);
            //CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        } else { // if lumigraph rendering, set all weights to one ( lumigraph weights are in Gamma )

            // vmask is set to 1/ sigma2 * lumi to be coherent with no-lumigraph
            // to have equivalent lambda parameter with different methods
            cuda_add_scaled_to_device <<< ws->_dimGrid, ws->_dimBlock >>> ( W, H, view->lumi_weights,
                                                                                  1./(sr->_sigma_sensor * sr->_sigma_sensor),
                                                                                  view->_vmask);
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            //write_pfm_image_signed(W, H, view->_vmask, data->_basedir + "vmask_%03d.pfm", meta_iter*100+nview);
        }

        // smooth the weights
        if (false) {
          cuda_convolution_row ( 0.25, 0.5, 0.25, W, H, view->_vmask, view->_vmask_lo);
          CUDA_SAFE_CALL( cudaDeviceSynchronize() );

          cuda_convolution_column ( 0.25, 0.5, 0.25, W, H, view->_vmask_lo, view->_vmask);
          CUDA_SAFE_CALL( cudaDeviceSynchronize() );

          filter_invalid_device <<< ws->_dimGrid, ws->_dimBlock >>>  (W,H, view->_vmask, view->warp_beta_x);
          CUDA_SAFE_CALL( cudaDeviceSynchronize() );
        }

        // accumulate the masks to fill target_mask
        cuda_add_to_device<<< ws->_dimGrid, ws->_dimBlock >>> (W, H, view->_vmask, sr->_target_mask);
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

       // write_pfm_image_signed(W, H, view->_vmask, data->_basedir + "vmask_%03d.pfm", meta_iter*100+nview);

        /*
        TRACE("Test: write vmask of view " << nview << std::endl);
        write_pfm_image_signed(W, H, view->_vmask, data->_basedir + "vmask_greg_%02lu.pfm", nview);
        exit(0);
*/
    }

    //write_pfm_image_signed(W, H, sr->_target_mask, data->_basedir + "target_mask_%03d.pfm", meta_iter);

    return true;
}


bool coco::coco_warp_input_images(coco_vtv_data *data) {
  assert( data != NULL );
  coco_vtv_sr_data *sr = data->_sr_data;
  assert( sr != NULL );
  size_t W = data->_W;
  size_t H = data->_H;
  assert( W*H > 0 );
  coco_vtv_workspace *w = data->_workspace;

  // Temp buffers
  float *tmp_view = w->_temp[0]; //input view vi

  // Warp all views
  for (size_t nview = 0 ; nview < sr->_views.size() ; nview++) {
    coco_vtv_sr_view_data *view = sr->_views[nview];

    for (size_t n = 0 ; n < data->_nchannels ; n++) {

      CUDA_SAFE_CALL( cudaMemset(tmp_view, 0, sr->_nfbytes_hi ));
      coco_vtv_sr_warp_unstructured( data, nview, Gamma_from_Omegai, view->_image_f + n*sr->_w*sr->_h, tmp_view );
      CUDA_SAFE_CALL( cudaDeviceSynchronize() );

      // cudaMemCpy with device device is very slow. Set to 0 and add with a device
      CUDA_SAFE_CALL( cudaMemset(view->_image_f + n*sr->_w*sr->_h, 0, sr->_nfbytes_hi ));

      cuda_add_to_device <<< w->_dimGrid, w->_dimBlock >>> (W, H, tmp_view, view->_image_f + n*sr->_w*sr->_h);
      CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    }

    write_pfm_image_signed(W, H, view->_image_f + 0*sr->_w*sr->_h,
                                 view->_image_f + 1*sr->_w*sr->_h,
                                 view->_image_f + 2*sr->_w*sr->_h,
                           data->_basedir + "warped_input_%03lu.pfm", nview);
  }
  return true;
}

// Compute the starting image by using beta warps (Greg)
bool coco::coco_vtv_sr_compute_averaged_beta_warp(coco_vtv_data *data) {

    // check for required data
    assert( data != NULL );
    coco_vtv_sr_data *sr = data->_sr_data;
    assert( sr != NULL );
    size_t W = data->_W;
    size_t H = data->_H;
    assert( W*H > 0 );
    coco_vtv_workspace *w = data->_workspace;

    // Temp buffers
    float *tmp_viewvi = w->_temp[0]; //input view vi
    float *tmp_view = w->_temp[1]; // warped image in gamma domain

    // Clear target image
    for (size_t n = 0 ; n < data->_nchannels ; n++) {
        CUDA_SAFE_CALL(cudaMemset(w->_U[n], 0, w->_nfbytes));
    }

    // Add view contributions
    for (size_t nview = 0 ; nview < sr->_views.size() ; nview++) {

        coco_vtv_sr_view_data *view = sr->_views[nview];

        for (size_t n = 0 ; n < data->_nchannels ; n++) {

            // Clear tmp buffer
            CUDA_SAFE_CALL(cudaMemset(tmp_view, 0, w->_nfbytes));

            // View already in GAMMA
            // Upsample view in V
            //TRACE("Test bilinear upsampling function, view " << nview << " channel " << n << std::endl);
            //write_pfm_image_signed(sr->_w, sr->_h, view->_image_f + n*sr->_w*sr->_h, data->_basedir + "lores_img_%03lu.pfm", nview*100+n);
            /*vtv_sr_bilinear_upsample_device<<< w->_dimGrid, w->_dimBlock >>>( W, H, // Hi-res size
                                                                              sr->_w, sr->_h, // Lo-res size
                                                                              sr->_dsf,      // Scale factor
                                                                              view->_image_f + n*sr->_w*sr->_h, // lo-res matrix
                                                                              tmp_viewvi,    // hi-res result
                                                                              view->_visibility_mask);
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );*/

            //write_pfm_image_signed(W, H, tmp_viewvi, data->_basedir + "av_inital_upsampled_%03lu.pfm", nview);


            cuda_add_to_device<<< w->_dimGrid, w->_dimBlock >>> (W, H, view->_image_f + n*sr->_w*sr->_h, tmp_view);
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            // apply the weights
            cuda_multiply_with_device<<< w->_dimGrid, w->_dimBlock >>> ( W, H, tmp_view, view->_vmask );
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            //write_pfm_image_signed(W, H, tmp_viewvi, data->_basedir + "av_inital_upsampled_vmaks_%03lu.pfm", nview);

            // Warp image
            // init to 0
            //CUDA_SAFE_CALL( cudaMemset(tmp_view, 0, sr->_nfbytes_hi ));
            //coco_vtv_sr_warp_unstructured( data, nview, Gamma_from_Omegai, tmp_viewvi, tmp_view );

            //write_pfm_image_signed(W, H, tmp_view, data->_basedir + "av_inital_gamma_%03lu.pfm", nview);


            // accumulate the images to create the blend
            cuda_add_to_device<<< w->_dimGrid, w->_dimBlock >>>
                                                              (W, H, tmp_view, w->_U[n]);
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        } // for channels
    } // for nviews

    // Normalize
    for (size_t n = 0 ; n < data->_nchannels ; n++) {
        cuda_normalize_device<<< w->_dimGrid, w->_dimBlock >>>
                                                             (data->_W, data->_H, w->_U[n], sr->_target_mask);
    }
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    ////// TEST WARPS COHERENCE
    //// EXPERIMENTAL CODE COMMENTED OUT
    if ( false ){

      int nview = 0;
      coco_vtv_sr_view_data *view = sr->_views[nview];

      for (int i=0; i<3; ++i) {
        CUDA_SAFE_CALL(cudaMemset(w->_U[i], 0, w->_nfbytes));
        //cuda_add_to_device<<< w->_dimGrid, w->_dimBlock >>> (W, H, view->_image_f + i*sr->_w*sr->_h, w->_U[i]);
      }
      cuda_add_to_device<<< w->_dimGrid, w->_dimBlock >>> (W, H, view->_image_f + 0*sr->_w*sr->_h, w->_U[0]);
      cuda_add_to_device<<< w->_dimGrid, w->_dimBlock >>> (W, H, view->warp_tau_x, w->_U[1]);
      cuda_add_to_device<<< w->_dimGrid, w->_dimBlock >>> (W, H, view->warp_tau_y, w->_U[2]);

      cuda_add_to_device<<< w->_dimGrid, w->_dimBlock >>> (W, H, view->warp_beta_x, w->_Uq[1]);
      cuda_add_to_device<<< w->_dimGrid, w->_dimBlock >>> (W, H, view->warp_beta_y, w->_Uq[2]);

      for (int molt = 0 ; molt < 20 ; molt++) {

        write_pfm_image_signed(W, H, w->_U[0], w->_U[1], w->_U[2], data->_basedir + "aaa_warp_back_and_forth_gamma%02lu.pfm", molt);
        write_pfm_image_signed(W, H, w->_Uq[0], w->_Uq[1], w->_Uq[2], data->_basedir + "aaa_warp_back_and_forth_omegai%02lu.pfm", molt);

        for (size_t n = 0 ; n < data->_nchannels ; n++) {

          CUDA_SAFE_CALL(cudaMemset(tmp_view, 0, w->_nfbytes));

          //TRACE("Test bilinear upsampling function, view " << nview << " channel " << n << std::endl);
          //write_pfm_image_signed(sr->_w, sr->_h, view->_image_f + n*sr->_w*sr->_h, data->_basedir + "lores_img_%03lu.pfm", nview*100+n);
          /*
          vtv_sr_bilinear_upsample_device<<< w->_dimGrid, w->_dimBlock >>>( W, H, // Hi-res size
              sr->_w, sr->_h, // Lo-res size
              sr->_dsf,      // Scale factor
              w->_U[n], // lo-res matrix
              tmp_view,    // hi-res result
              view->_visibility_mask);
          CUDA_SAFE_CALL( cudaDeviceSynchronize() );

          if (n==-1) {
            write_pfm_image_signed(W, H, tmp_view, data->_basedir + "aaa_upsampled_%03lu.pfm", molt);
          }*/

          // Warp image to GAMMA
          // init to 0
          CUDA_SAFE_CALL( cudaMemset(w->_Uq[n], 0, sr->_nfbytes_hi ));
          coco_vtv_sr_warp_unstructured( data, nview, Gamma_from_Omegai, w->_U[n], w->_Uq[n] );

          if (n==-1) {
            write_pfm_image_signed(W, H, tmp_viewvi, data->_basedir + "aaa_upsampled_warped_%03lu.pfm", molt);
          }

          // UNDO WARP: back to Omegai
          CUDA_SAFE_CALL(cudaMemset(w->_U[n], 0, w->_nfbytes));
          coco_vtv_sr_warp_unstructured( data, nview, Omegai_from_Gamma, w->_Uq[n], w->_U[n]);

          if (n==-1) {
            write_pfm_image_signed(W, H, tmp_view, data->_basedir + "aaa_upsampled_back_%03lu.pfm", molt);
          }
/*
          vtv_sr_downsample_view_device<<< sr->_dimGrid, sr->_dimBlock >>>
              ( sr->_w, sr->_h, data->_W, tmp_view, view->_visibility_mask, sr->_dsf, w->_U[n]);
          CUDA_SAFE_CALL( cudaDeviceSynchronize() );*/
        } // for channels
      } // for molt

      exit(0);
    }

    TRACE("Write starting image for the algorithm" << endl);
    //write_pfm_image_signed(W, H, w->_U[0], w->_U[1], w->_U[2], data->_basedir + "u_init.pfm", 0);

    return true;
}

static int iterations =0;
// Compute primal energy (Greg)
double coco::coco_vtv_sr_primal_energy_unstructured( coco_vtv_data *data )
{
    assert( data != NULL );
    coco_vtv_workspace *w = data->_workspace;
    coco_vtv_sr_data *sr = data->_sr_data;
    assert( sr != NULL );
    size_t W = data->_W;
    size_t H = data->_H;

    // Compute gradient of current solution
    for ( size_t i=0; i<data->_nchannels; i++ ) {
        cuda_compute_gradient_device<<< w->_dimGrid, w->_dimBlock >>>
                                                                    ( data->_W, data->_H, w->_U[i], w->_X1q[i], w->_X2q[i] );
    }
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );


    // TV component
    switch ( data->_regularizer ) {
    case 0:
        assert(false && "Not implemented for now\n");
        break;
    case 1:
        if ( data->_nchannels == 1 ) {
            cuda_compute_norm_device<<< w->_dimGrid, w->_dimBlock >>>
                                                                    ( W,H,
                                                                      w->_X1q[0], w->_X2q[0], w->_X1q[0], w->_X2q[0], w->_X1q[0], w->_X2q[0],
                    w->_G[0] );
        } else if ( data->_nchannels == 3 ) {
            cuda_compute_norm_device<<< w->_dimGrid, w->_dimBlock >>>
                                                                    ( W,H,
                                                                      w->_X1q[0], w->_X2q[0], w->_X1q[1], w->_X2q[1], w->_X1q[2], w->_X2q[2],
                    w->_G[0] );
        }
        break;
    case 2:
        // Compute largest singular value of gradient matrix
        if ( data->_nchannels == 1 ) {
            cuda_compute_largest_singular_value_device<<< w->_dimGrid, w->_dimBlock >>>
                                                                                      ( W,H,
                                                                                        w->_X1q[0], w->_X2q[0], w->_X1q[0], w->_X2q[0], w->_X1q[0], w->_X2q[0],
                    w->_G[0] );
        } else if ( data->_nchannels == 3 ) {
            cuda_compute_largest_singular_value_device<<< w->_dimGrid, w->_dimBlock >>>
                                                                                      ( W,H,
                                                                                        w->_X1q[0], w->_X2q[0], w->_X1q[1], w->_X2q[1], w->_X1q[2], w->_X2q[2],
                    w->_G[0] );
        }
        break;
    default:
        break;
    }

    float *E_TV = new float[ W*H ];
    CUDA_SAFE_CALL( cudaMemcpy( E_TV, w->_G[0], sr->_nfbytes_hi, cudaMemcpyDeviceToHost ));
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    // Compute tv energy
    double e_tv = 0.0;
    for ( size_t i=0; i<W*H; i++ ) {
        e_tv += E_TV[i];
    }

    delete[] E_TV;


    // Data Term
    float *hires_tmp = w->_temp[0]; // image for hi-res warp
    float *tmp = w->_temp[1]; // image for residu computation
    float *hires_energy =  w->_temp[2];

    CUDA_SAFE_CALL( cudaMemset( hires_energy, 0, sr->_nfbytes_hi ));

    // Sum contributions for all views
    for ( size_t nview=0; nview < sr->_views.size(); nview++ ) {
        coco_vtv_sr_view_data *view = sr->_views[nview];

        for ( size_t i=0; i<data->_nchannels; i++ ) {
            CUDA_SAFE_CALL( cudaMemset( hires_tmp, 0, sizeof(float) * data->_W * data->_H ));

            // warp current solution in omega_i domain
            //coco_vtv_sr_warp_unstructured( data, nview, Omegai_from_Gamma, w->_U[i], hires_tmp );
            //write_pfm_image_signed(sr->_w, sr->_h, w->_U[i], data->_basedir + "current_u_%d.pfm", iterations*1000+nview*10+i);

            // downsample
            //vtv_sr_downsample_view_device<<< sr->_dimGrid, sr->_dimBlock >>>
            //             ( sr->_w, sr->_h, data->_W, hires_tmp, view->_visibility_mask, sr->_dsf, tmp );
            //CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            //write_pfm_image_signed(sr->_w, sr->_h, lores_tmp, data->_basedir + "tau_u_%d.pfm", iterations*1000+nview*10+i);

            // init to U
            cuda_add_to_device <<< sr->_dimGrid, sr->_dimBlock >>> (W,H, w->_U[i], hires_tmp);
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            // subtract image from input float image
            cuda_subtract_from_device<<< sr->_dimGrid, sr->_dimBlock >>>
                                                                       ( sr->_w, sr->_h, view->_image_f + i*sr->_w*sr->_h, hires_tmp );
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            // square the result
            cuda_multiply_with_device<<< sr->_dimGrid, sr->_dimBlock >>>
                                                                       ( sr->_w, sr->_h, hires_tmp, hires_tmp );
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            // upsample (without weights)
            /*vtv_sr_bilinear_upsample_device<<< w->_dimGrid, w->_dimBlock >>>( W, H, // Hi-res size
                                                                              sr->_w, sr->_h,// Lo-res size
                                                                              sr->_dsf,      // Scale factor
                                                                              tmp,     // lo-res matrix
                                                                              hires_tmp,     // hi-res result
                                                                              view->_visibility_mask);
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );*/

            // apply the weights
            cuda_multiply_with_device<<< w->_dimGrid, w->_dimBlock >>> ( W, H, hires_tmp, view->_vmask );
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            //coco_vtv_sr_warp_unstructured( data, nview, Gamma_from_Omegai, hires_tmp, tmp );

            if ( sr->_lumigraph ) { // if lumigraph rendering multiply with the lumi_weights
                cuda_multiply_with_device<<< w->_dimGrid, w->_dimBlock >>>
                                                                         ( W, H, hires_tmp, view->lumi_weights );
                CUDA_SAFE_CALL( cudaDeviceSynchronize() );
            }

            // cumulate energy
            cuda_add_to_device<<< sr->_dimGrid, sr->_dimBlock >>>
                                                                ( sr->_w, sr->_h, hires_tmp, hires_energy);
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );
        }
    }

    float *E_DATA = new float[ sr->_w * sr->_h ];
    CUDA_SAFE_CALL( cudaMemcpy( E_DATA, hires_energy, sr->_nfbytes_lo, cudaMemcpyDeviceToHost ));
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    // Compute tv energy
    double e_data = 0.0;
    for ( size_t i=0; i<sr->_w * sr->_h; i++ ) {
        e_data += E_DATA[i];
    }

    delete[] E_DATA;

    double energy = data->_lambda * e_tv + e_data;

    if ( traceLevel() > 1 ) {
        TRACE("Energy : " << energy << " : E_TV = " << e_tv << " * " << data->_lambda << " = " << e_tv *  data->_lambda
              << " | E_DATA =  "<< e_data << endl);
    }

    return energy;
}

// Perform one iteration of Algorithm 1, Chambolle-Pock (Goldluecke)
bool coco::coco_vtv_sr_iteration_fista_unstructured( coco_vtv_data *data )
{
    // Todo: verify correct maximum step sizes.
    data->_tau = 0.3 / sqrt( 8.0 );
    data->_sigma = 0.3 / sqrt( 8.0 );
    data->_L = 1.0 / data->_lambda; //float(data->_sr_data->_nviews) / data->_lambda ;
    //data->_L = float(data->_sr_data->_nviews) / data->_lambda ;
    vtv_sr_ista_step_unstructured( data );
    cuflt alpha_new = 0.5 * ( 1.0 + sqrt( 1.0 + 4.0 * pow( data->_alpha, 2.0 ) ));
    coco_vtv_rof_overrelaxation( data, ( data->_alpha - 1.0 ) / alpha_new );
    data->_alpha = alpha_new;
    data->_workspace->_iteration ++;
    return true;
}


// Perform one single shrinkage step (ISTA) (Greg)
bool coco::vtv_sr_ista_step_unstructured( coco_vtv_data *data )
{
    assert( data != NULL );
    coco_vtv_workspace *w = data->_workspace;
    coco_vtv_sr_data *sr = data->_sr_data;
    assert( sr != NULL );

    // Start descent from current solution
    for ( size_t i=0; i<data->_nchannels; i++ ) {
        CUDA_SAFE_CALL( cudaMemcpy( w->_Uq[i], w->_U[i], w->_nfbytes, cudaMemcpyDeviceToDevice ));
    }
    CUDA_SAFE_CALL( cudaThreadSynchronize() );


    // Compute gradient of data term
    vtv_sr_dataterm_gradient_unstructured( data );

    // Compute F for ROF steps
    for ( size_t i=0; i<data->_nchannels; i++ ) {
        cuda_scale_device<<< w->_dimGrid, w->_dimBlock >>>
                                                         ( data->_W, data->_H, w->_G[i], -1.0 / ( data->_lambda * data->_L ));

        // Add current solution
        cuda_add_to_device<<< w->_dimGrid, w->_dimBlock >>>
                                                          ( data->_W, data->_H, w->_Uq[i], w->_G[i] );

        // Clamp to 0-1
        cuda_clamp_device<<< w->_dimGrid, w->_dimBlock >>>
                                                         ( data->_W, data->_H, w->_G[i], 0.0f, 1.0f );
    }


    // Perform a number of primal/dual ROF iterations
    profiler()->beginTask( "prox" );
    data->_tau = 0.3 / sqrt( 8.0 );
    data->_sigma = 0.3 / sqrt( 8.0 );
    for ( size_t k=0; k<data->_inner_iterations; k++ ) {
        coco_vtv_rof_dual_step( data );

        // Primal step kernel call for each channel
        for ( size_t i=0; i<data->_nchannels; i++ ) {
            cuda_rof_primal_prox_step_device<<< w->_dimGrid, w->_dimBlock >>>
                                                                            ( data->_W, data->_H, data->_tau, 1.0 / data->_L,
                                                                              w->_Uq[i], w->_Uq[i], w->_G[i], w->_X1[i], w->_X2[i] );
            CUDA_SAFE_CALL( cudaThreadSynchronize() );
        }
    }
    profiler()->endTask( "prox" );

    //write_test_image_signed( data->_W, data->_H, w->_Uq[0], "out/Uq_total.png", 0 );
    return true;
}

// Compute gradient of data term (Greg)
bool coco::vtv_sr_dataterm_gradient_unstructured( coco_vtv_data *data )
{
    assert( data != NULL );
    coco_vtv_workspace *w = data->_workspace;
    coco_vtv_sr_data *sr = data->_sr_data;
    assert( sr != NULL );
    size_t W = data->_W;
    size_t H = data->_H;

    // Start descent from current solution
    for ( size_t i=0; i<data->_nchannels; i++ ) {
        CUDA_SAFE_CALL( cudaMemcpy( w->_Uq[i], w->_U[i], w->_nfbytes, cudaMemcpyDeviceToDevice ));
    }
    CUDA_SAFE_CALL( cudaThreadSynchronize() );


    int print_channel = -1;
    //write_pfm_image_signed(W, H,  w->_U[0],w->_U[1], w->_U[2], data->_basedir + "current_solution_%03lu.pfm", meta_iter*100+iterations);

    // Clear derivative
    for ( size_t i=0; i<data->_nchannels; i++ ) {
        CUDA_SAFE_CALL( cudaMemset( w->_G[i], 0, w->_nfbytes )); //hi res tmp
    }

    float *tmp0 = w->_temp[0];
    //float *tmp1 = w->_temp[1]; //lo res tmp (in omega_i)
    //float *tmp2 = w->_temp[2]; //hi res tmp in gamma

    // Sum contributions for all views
    for ( size_t nview=0; nview < sr->_views.size(); nview++ ) {
        coco_vtv_sr_view_data *view = sr->_views[nview];

        for ( size_t i=0; i<data->_nchannels; i++ ) {

            CUDA_SAFE_CALL( cudaMemset(tmp0, 0, sr->_nfbytes_hi ));
            cuda_add_to_device<<< sr->_dimGrid, sr->_dimBlock >>> ( sr->_w, sr->_h, w->_U[i], tmp0 );

            /*coco_vtv_sr_warp_unstructured( data, nview, Omegai_from_Gamma, w->_U[i], hires_tmp_omega_i );

            if (i==print_channel){
              write_pfm_image_signed(W, H, hires_tmp_omega_i, data->_basedir + "warped_current_solution_%03lu.pfm",
                                     meta_iter*10000+100*iterations+nview);
            }

            // downsample the warped view Uq (in vi domain)
            vtv_sr_downsample_view_device<<< sr->_dimGrid, sr->_dimBlock >>>
                                                                           ( sr->_w, sr->_h, data->_W,
                                                                             hires_tmp_omega_i, view->_visibility_mask,
                                                                             sr->_dsf, lores_tmp );
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            if (i==print_channel){
              write_pfm_image_signed(sr->_w, sr->_h, lores_tmp, data->_basedir + "warped_current_solution_down_%03lu.pfm",
                  meta_iter*10000+100*iterations+nview);
            }*/

            cuda_subtract_from_device<<< sr->_dimGrid, sr->_dimBlock >>> ( sr->_w, sr->_h, view->_image_f + i*sr->_w*sr->_h, tmp0 );
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            if (i==print_channel){
              write_pfm_image_signed(sr->_w, sr->_h, view->_image_f + i*sr->_w*sr->_h, data->_basedir + "input_image_for_diff_%03lu.pfm",
                               meta_iter*10000+100*iterations+nview);

              write_pfm_image_signed(sr->_w, sr->_h, tmp0, data->_basedir + "warped_current_diff_%03lu.pfm",
                  meta_iter*10000+100*iterations+nview);
            }

            /*// upsample (without weights)
            vtv_sr_bilinear_upsample_device<<< w->_dimGrid, w->_dimBlock >>>( W, H, // Hi-res size
                                                                              sr->_w, sr->_h,// Lo-res size
                                                                              sr->_dsf,      // Scale factor
                                                                              lores_tmp,     // lo-res matrix
                                                                              hires_tmp_omega_i,    // hi-res result
                                                                              view->_visibility_mask);
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            if (i==print_channel){
              write_pfm_image_signed(sr->_w, sr->_h, hires_tmp_omega_i, data->_basedir + "warped_current_diff_up_%03lu.pfm",
                  meta_iter*10000+100*iterations+nview);
            }*/

            // masked by vmask (hi res) : apply the weights
            // weights here are 1. if lumigraph rendering
            cuda_multiply_with_device<<< w->_dimGrid, w->_dimBlock >>> ( W, H, tmp0, view->_vmask );
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            if (i==print_channel){
              write_pfm_image_signed(sr->_w, sr->_h, tmp0, data->_basedir + "warped_current_diff_up_scaled_%03lu.pfm",
                  meta_iter*10000+100*iterations+nview);
            }

            // Warp difference into Gamma
            // initialize result to 0
            /*CUDA_SAFE_CALL( cudaMemset(hires_tmp_gamma, 0, sr->_nfbytes_hi ));
            coco_vtv_sr_warp_unstructured( data, nview, Gamma_from_Omegai, hires_tmp_omega_i, hires_tmp_gamma );

            if (i==print_channel){
              write_pfm_image_signed(sr->_w, sr->_h, tmp0, data->_basedir + "warped_gamma_final_diff_%03lu.pfm",
                  meta_iter*10000+100*iterations+nview);
            }*/

            // accumulate the masks to fill target_mask
            cuda_add_to_device<<< w->_dimGrid, w->_dimBlock >>> (W, H, tmp0, w->_G[i]);
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );
        }
    }

    //write_pfm_image_signed(data->_W, data->_H, w->_G[0], w->_G[1], w->_G[2], data->_basedir + "dataterm_gradient_%03lu.pfm", meta_iter*100 + iterations);
    // Normalize
    // Note: this is a detail skipped in the paper.
    // For optimization, it is more stable to divide both
    // the data term as well as the regularizer by the number
    // of contributing views, so that the gradient values
    // are more uniform.

    for ( size_t i=0; i<data->_nchannels; i++ ) {
        cuda_normalize_device<<< w->_dimGrid, w->_dimBlock >>>
                                                             ( W, H, w->_G[i], sr->_target_mask );

        cuda_clamp_device<<< w->_dimGrid, w->_dimBlock >>>
                                                         ( W, H, w->_G[i], -1.0f, 1.0f );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    }

    //write_pfm_image_signed(data->_W, data->_H, w->_G[0], w->_G[1], w->_G[2], data->_basedir + "dataterm_gradient_normalized_%03lu.pfm", meta_iter*100+iterations);
    ++iterations;

    if (iterations == -1) {
      exit(0);
    }

    return true;
}

// Read the beta warps: from gsl_image to device float* (Greg)
bool coco::coco_vtv_sr_read_beta( coco_vtv_data *data, gsl_image** beta_warps )
{
    // check for required data
    assert( data != NULL );
    coco_vtv_sr_data *sr = data->_sr_data;
    assert( sr != NULL );
    size_t W = data->_W;
    size_t H = data->_H;
    size_t N = W*H;
    assert( N > 0 );

    float *buffer_f = new float[N];
    gsl_matrix *channel;

    for ( size_t nview=0; nview < sr->_views.size(); nview++ ) {
        coco_vtv_sr_view_data *view = sr->_views[nview];
        assert(beta_warps[nview]);

        channel = gsl_image_get_channel( beta_warps[nview], GSL_IMAGE_RED ); // load beta x
        gsl_matrix_copy_to_buffer( channel,  buffer_f );
        CUDA_SAFE_CALL( cudaMemcpy( view->warp_beta_x, buffer_f, sr->_nfbytes_hi, cudaMemcpyHostToDevice ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        channel = gsl_image_get_channel( beta_warps[nview], GSL_IMAGE_GREEN ); // load beta y
        gsl_matrix_copy_to_buffer( channel,  buffer_f );
        CUDA_SAFE_CALL( cudaMemcpy( view->warp_beta_y, buffer_f, sr->_nfbytes_hi, cudaMemcpyHostToDevice ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        channel = gsl_image_get_channel( beta_warps[nview], GSL_IMAGE_BLUE ); // deformation weights |det D beta|^(-1)
        gsl_matrix_copy_to_buffer( channel,  buffer_f );
        CUDA_SAFE_CALL( cudaMemcpy( view->deform_weight_beta, buffer_f, sr->_nfbytes_hi, cudaMemcpyHostToDevice ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        //cuda_set_all_device<<< data->_workspace->_dimGrid, data->_workspace->_dimBlock >>> (W,H, view->deform_weight_beta, 1. );
        //CUDA_SAFE_CALL( cudaThreadSynchronize() );
        //TRACE("Test: write pfm beta warp, view " << nview << std::endl);
        //write_pfm_image_signed(W, H, view->warp_beta_x, view->warp_beta_y, view->deform_weight_beta, data->_basedir + "beta_%02lu.pfm", nview);
    }
    delete [] buffer_f;

    return true;
}

// Read the tau warps and deformation weights: from gsl_image to device float* (Greg)
bool coco::coco_vtv_sr_read_tau( coco_vtv_data *data, gsl_image** tau_warps )
{
    // check for required data
    assert( data != NULL );
    coco_vtv_sr_data *sr = data->_sr_data;
    assert( sr != NULL );
    size_t W = data->_W;
    size_t H = data->_H;
    size_t N = W*H;
    assert( N > 0 );

    float *buffer_f = new float[N];
    gsl_matrix *channel;

    for ( size_t nview=0; nview < sr->_views.size(); nview++ ) {

        coco_vtv_sr_view_data *view = sr->_views[nview];

        channel = gsl_image_get_channel( tau_warps[nview], GSL_IMAGE_RED); // load tau x
        gsl_matrix_copy_to_buffer( channel,  buffer_f );
        CUDA_SAFE_CALL( cudaMemcpy( view->warp_tau_x, buffer_f, sr->_nfbytes_hi, cudaMemcpyHostToDevice ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        channel = gsl_image_get_channel( tau_warps[nview], GSL_IMAGE_GREEN ); // load tau y
        gsl_matrix_copy_to_buffer( channel,  buffer_f );
        CUDA_SAFE_CALL( cudaMemcpy( view->warp_tau_y, buffer_f, sr->_nfbytes_hi, cudaMemcpyHostToDevice ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        channel = gsl_image_get_channel( tau_warps[nview], GSL_IMAGE_BLUE ); // deformation weights |det D tau|^(-1)
        gsl_matrix_copy_to_buffer( channel,  buffer_f );
        CUDA_SAFE_CALL( cudaMemcpy( view->deform_weight_tau, buffer_f, sr->_nfbytes_hi, cudaMemcpyHostToDevice ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        //cuda_set_all_device<<< data->_workspace->_dimGrid, data->_workspace->_dimBlock >>> (W,H, view->deform_weight_tau, 1. );
        //CUDA_SAFE_CALL( cudaThreadSynchronize() );
        //TRACE("Test: write pfm tau warp, view " << nview << std::endl);
        //write_pfm_image_signed(W, H, view->warp_tau_x, view->warp_tau_y, view->deform_weight_tau, data->_basedir + "tau_%02lu.pfm", nview);
    }
    delete [] buffer_f;

    return true;
}

// Read the partial tau: from gsl_image to device float* (Greg)
bool coco::coco_vtv_sr_read_partial_tau( coco_vtv_data *data, gsl_image** partial_tau, float sigma_const )
{
    // check for required data
    assert( data != NULL );
    coco_vtv_workspace *w = data->_workspace;
    assert( w != NULL );
    coco_vtv_sr_data *sr = data->_sr_data;
    assert( sr != NULL );
    size_t W = data->_W;
    size_t H = data->_H;
    size_t N = W*H;
    assert( N > 0 );

    float *buffer_f = new float[N];
    gsl_matrix *channel;

    for ( size_t nview=0; nview < sr->_views.size(); nview++ ) {

        coco_vtv_sr_view_data *view = sr->_views[nview];
        cuflt *sigma_z = w->_temp[0]; //hi res tmp

        channel = gsl_image_get_channel( partial_tau[nview], GSL_IMAGE_RED ); // load sigma_z
        gsl_matrix_copy_to_buffer( channel,  buffer_f );
        CUDA_SAFE_CALL( cudaMemcpy( sigma_z, buffer_f, sr->_nfbytes_hi, cudaMemcpyHostToDevice ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        if ( sigma_const >= 0 ) {
            cuda_set_all_device<<< w->_dimGrid, w->_dimBlock >>> ( W, H, sigma_z, sigma_const);
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );
        }

        channel = gsl_image_get_channel( partial_tau[nview], GSL_IMAGE_GREEN ); // load dtau/dy x
        gsl_matrix_copy_to_buffer( channel,  buffer_f );
        CUDA_SAFE_CALL( cudaMemcpy( view->dpart_x, buffer_f, sr->_nfbytes_hi, cudaMemcpyHostToDevice ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        // do the product sigma_z*dtau/dz
        cuda_multiply_with_device<<< w->_dimGrid, w->_dimBlock >>> ( W, H, view->dpart_x, sigma_z );

        channel = gsl_image_get_channel( partial_tau[nview], GSL_IMAGE_BLUE ); // load dtau/dy y
        gsl_matrix_copy_to_buffer( channel,  buffer_f );
        CUDA_SAFE_CALL( cudaMemcpy( view->dpart_y, buffer_f, sr->_nfbytes_hi, cudaMemcpyHostToDevice ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        // do the product sigma_z*dtau/dz
        cuda_multiply_with_device<<< w->_dimGrid, w->_dimBlock >>> ( W, H, view->dpart_y, sigma_z );

        //TRACE("Test: write pfm partial tau, view " << nview << std::endl);
        //write_pfm_image_signed(W, H, sigma_z, view->dpart_x, view->dpart_y, data->_basedir + "partial_tau_%02lu.pfm", nview);
    }
    delete [] buffer_f;

    return true;
}

// Read the lumigraph weights: from gsl_image to device float* (Greg)
bool coco::coco_vtv_sr_read_lumi_weights( coco_vtv_data *data, gsl_image** lumi_weights )
{
    // check for required data
    assert( data != NULL );
    coco_vtv_workspace *w = data->_workspace;
    assert( w != NULL );
    coco_vtv_sr_data *sr = data->_sr_data;
    assert( sr != NULL );
    size_t W = data->_W;
    size_t H = data->_H;
    size_t N = W*H;
    assert( N > 0 );

    float *buffer_f = new float[N];
    gsl_matrix *channel;

    for ( size_t nview=0; nview < sr->_views.size(); nview++ ) {

        coco_vtv_sr_view_data *view = sr->_views[nview];

        channel = gsl_image_get_channel( lumi_weights[nview], GSL_IMAGE_RED ); // load sigma_z
        gsl_matrix_copy_to_buffer( channel,  buffer_f );
        CUDA_SAFE_CALL( cudaMemcpy( view->lumi_weights, buffer_f, sr->_nfbytes_hi, cudaMemcpyHostToDevice ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );
        /*
        // reverse buffer:
        std::vector<cuflt> cpu(N);
        CUDA_SAFE_CALL( cudaMemcpy( cpu.data(), view->lumi_weights, sizeof(cuflt) * N, cudaMemcpyDeviceToHost ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );
        for ( int i = 0 ; i < H/2 ; ++i ) {
            std::swap_ranges( cpu.begin() + i*W, cpu.begin()+ i*W + W, cpu.end() - W - i*W );
        }
        CUDA_SAFE_CALL( cudaMemcpy( view->lumi_weights, cpu.data(), sizeof(cuflt) * N, cudaMemcpyHostToDevice ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        TRACE("Test: write pfm lumigraph weights, view " << nview << std::endl);
        write_pfm_image_signed(W, H, view->lumi_weights, data->_basedir + "lumi_weights_%02lu.pfm", nview);
        */
    }
    delete [] buffer_f;

    return true;
}

// Write current solution in pfm format
bool coco::coco_vtv_sr_write_pfm_solution( coco_vtv_data *data )
{
    coco_vtv_workspace *w = data->_workspace;
    size_t W = data->_W;
    size_t H = data->_H;
    assert( W*H > 0 );

    if (data->_nchannels == 3) {
        write_pfm_image_signed(W, H, w->_U[0], w->_U[1], w->_U[2], data->_basedir + "output.pfm", 0);
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    }

    return true;
}

bool coco::coco_vtv_sr_init_regularizer_weight_unstructured( coco_vtv_data *data ) {

    // check for required data
    assert( data != NULL );
    coco_vtv_sr_data *sr = data->_sr_data;
    assert( sr != NULL );
    size_t W = data->_W;
    size_t H = data->_H;
    assert( W*H > 0 );
    coco_vtv_workspace *w = data->_workspace;

    // Use target mask as a regularizer weight
    for ( size_t i=0; i<data->_nchannels; i++ ) {
        // Use target mask as a regularizer weight
        if ( w->_g[i] == NULL ) {
            CUDA_SAFE_CALL( cudaMalloc( &w->_g[i], w->_nfbytes ));
            CUDA_SAFE_CALL( cudaMemset( w->_g[i], 0, w->_nfbytes ));
        }
        vtv_sr_init_regularizer_weight_device<<< w->_dimGrid, w->_dimBlock >>>
                                                                             ( data->_W, data->_H,
                                                                               data->_lambda_max_factor * data->_lambda,
                                                                               data->_lambda,
                                                                               sr->_nviews,
                                                                               sr->_target_mask, w->_g[i] );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        // Convolve weight
        cuda_convolution( sr->_vmask_filter, W,H, w->_g[i], w->_temp[0] );
        CUDA_SAFE_CALL( cudaMemcpy( w->_g[i], w->_temp[0], w->_nfbytes, cudaMemcpyDeviceToDevice ));

        // Write final mask and result as a reference
        if ( traceLevel() > 2 ) {
            write_test_image_unsigned( W, H, w->_temp[0], data->_basedir + "target_mask.png", 0 );
            write_test_image_unsigned( W, H, w->_g[i], data->_basedir + "regweight.png", 0 );
        }
    }

    return true;
}


// compute non-overlapping segments (cell structure) for all views and both for tau and beta splatting
bool coco::coco_vtv_sr_init_cells_unstructured( coco_vtv_data *data ) {

    coco_vtv_sr_data *sr = data->_sr_data;
    assert( sr != NULL );

    for ( size_t nview=0; nview < sr->_views.size(); nview++ ) { // for all views

        vtv_sr_init_forward_warp_structure_unstructured( data, nview, 0 ); // tau splatting
        vtv_sr_init_forward_warp_structure_unstructured( data, nview, 1 ); // beta splatting
    }

    return true;
}

// warp (in high res) the input image given the warping direction (Greg)
// omega_i to gamma: 0, gamma to omega_i: 1
// input and output can be in omega_i or gamma
bool coco::coco_vtv_sr_warp_unstructured( coco_vtv_data *data, size_t nview, t_warp_direction direction, cuflt* input, cuflt* output ) {

    coco_vtv_sr_data *sr = data->_sr_data;
    assert( sr != NULL );
    size_t W = data->_W;
    size_t H = data->_H;
    assert( W*H > 0 );
    coco_vtv_workspace *w = data->_workspace;
    coco_vtv_sr_view_data *view = sr->_views[nview];

    // warp i->o
    cuflt *forward_warp_x(0);
    cuflt *forward_warp_y(0);
    // warp o->i
    cuflt *backward_warp_x(0);
    cuflt *backward_warp_y(0);

    cuflt *deformation_weights(0);
    int *cells(0);
    std::vector<int> seg_end;

    // Temp output for hybrid method
    cuflt *tmp_output = w->_temp[4];
    CUDA_SAFE_CALL( cudaMemset( tmp_output, 0, w->_nfbytes ));

    if ( direction == Gamma_from_Omegai ) {
        // from omega_i to gamma, o is in gamma, forward warp is tau, backward warp is beta

        backward_warp_x = view->warp_beta_x;
        backward_warp_y = view->warp_beta_y;
        forward_warp_x = view->warp_tau_x;
        forward_warp_y = view->warp_tau_y;
        deformation_weights = view->deform_weight_beta; // |det D beta_i|^(-1)
        cells = view->_cells_tau;
        seg_end = view->_seg_end_tau;

    } else {
        assert(direction == Omegai_from_Gamma);
        // from gamma to omega_i, o is in omega_i, forward warp is beta, backward warp is tau

        backward_warp_x = view->warp_tau_x;
        backward_warp_y = view->warp_tau_y;
        forward_warp_x = view->warp_beta_x;
        forward_warp_y = view->warp_beta_y;
        deformation_weights = view->deform_weight_tau; // |det D tau_i|^(-1)
        cells = view->_cells_beta;
        seg_end = view->_seg_end_beta;
    }

    if ( warping == backwards_warp || warping == hybrid_warp ) { // backward warping (also needed for hybrid method)

        vtv_sr_bw_warp_device_unstructured<<< w->_dimGrid, w->_dimBlock >>>
                                                                          ( W, H,
                                                                            input, // input image high res
                                                                            backward_warp_x,
                                                                            backward_warp_y,
                                                                            forward_warp_x,
                                                                            output, // output image high res
                                                                            sampling );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    } else { // forward splatting

        coco_vtv_sr_fw_splatting_unstructured( data,
                                               input,
                                               forward_warp_x,
                                               forward_warp_y,
                                               backward_warp_x, // to test visibility
                                               backward_warp_y,
                                               cells,
                                               seg_end,
                                               output );

    }

    if ( warping == hybrid_warp ) {

        coco_vtv_sr_fw_splatting_unstructured( data,
                                               input,
                                               forward_warp_x,
                                               forward_warp_y,
                                               backward_warp_x, // to test visibility
                                               backward_warp_y,
                                               cells,
                                               seg_end,
                                               tmp_output );

        vtv_sr_blend_warped_images_device<<< w->_dimGrid, w->_dimBlock >>>( W, H,
                                                                            deformation_weights,
                                                                            tmp_output, // forward warped
                                                                            output); // backward warped
        CUDA_SAFE_CALL( cudaThreadSynchronize() );
    }

    return true;
}

int splat_helper = 0;
// Forward splatting with warp (either beta or tau) of a single input view (high res)
bool coco::coco_vtv_sr_fw_splatting_unstructured( coco_vtv_data *data,
                                                  cuflt *input,
                                                  cuflt *fw_warp_x,
                                                  cuflt *fw_warp_y,
                                                  cuflt *bw_warp_x, // to test visibility
                                                  cuflt *bw_warp_y,
                                                  int *cells,
                                                  std::vector<int> &seg_ends,
                                                  cuflt *output ) {

    // check for required data
    assert( data != NULL );
    coco_vtv_sr_data *sr = data->_sr_data;
    assert( sr != NULL );
    size_t W = data->_W;
    size_t H = data->_H;
    assert( W*H > 0 );
    coco_vtv_workspace *w = data->_workspace;

    // Temp buffer
    float *tmp_weights = w->_temp[3];

    // Clear output and weights
    CUDA_SAFE_CALL( cudaMemset( output, 0, w->_nfbytes ));
    CUDA_SAFE_CALL( cudaMemset( tmp_weights, 0, w->_nfbytes ));

    // Forward warp, non-overlap regions sequentially
    int seg_start = 0;
    for ( size_t j = 0 ; j < seg_ends.size() ; j++ ) {

        int seg_end = seg_ends[j];
        int seg_size = seg_end - seg_start;

        // forward warp call for this segment, cannot overlap
        int seg_width = cuda_default_block_size_x() * cuda_default_block_size_y();
        dim3 dimBlock = dim3( seg_width, 1 );
        dim3 dimGrid = dim3( seg_size / seg_width + 1, 1 );

        vtv_sr_fw_warp_device_unstructured<<< dimGrid, dimBlock >>>( data->_W, data->_H, seg_width,
                                                                     input,
                                                                     cells,
                                                                     seg_start, seg_end,
                                                                     fw_warp_x, fw_warp_y,
                                                                     bw_warp_x, bw_warp_y,
                                                                     output,
                                                                     tmp_weights );

        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        seg_start = seg_end;
    }

    if (splat_helper < 10) {
      //write_pfm_image_signed(W, H, tmp_weights, data->_basedir + "splatting_weights%02d.pfm", splat_helper);
      splat_helper++;
    }

    // Normalize
    cuda_normalize_device<<< w->_dimGrid, w->_dimBlock >>>
                                                         ( W, H, output, tmp_weights );

    return true;
}


// Read the partial tau: from gsl_image to device float* (Greg)
bool coco::check_warp_coherence( coco_vtv_data *data)
{
    // check for required data
    assert( data != NULL );
    coco_vtv_workspace *w = data->_workspace;
    assert( w != NULL );
    coco_vtv_sr_data *sr = data->_sr_data;
    assert( sr != NULL );
    size_t W = data->_W;
    size_t H = data->_H;

    // Temp buffer
    float *tmp_warp_x = w->_temp[0];
    float *tmp_warp_y = w->_temp[1];

    for ( size_t nview=0; nview < sr->_views.size(); nview++ ) {
        coco_vtv_sr_view_data *view = sr->_views[nview];

        // BETA
        // gradient computation completely overkill, use 3 channels just to add a mask
        vtv_sr_compute_gradient_device <<< w->_dimGrid, w->_dimBlock >>>
                                    ( W,H, view->deform_weight_beta, view->deform_weight_beta, view->deform_weight_beta,
                                      view->deform_weight_beta, w->_X1[0], w->_X2[0]);
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        CUDA_SAFE_CALL( cudaMemcpy(tmp_warp_x, view->warp_beta_x,  w->_nfbytes, cudaMemcpyDeviceToDevice ));
        CUDA_SAFE_CALL( cudaMemcpy(tmp_warp_y, view->warp_beta_y,  w->_nfbytes, cudaMemcpyDeviceToDevice ));

        check_warp_coherence_device <<< w->_dimGrid, w->_dimBlock >>>
                                                                    ( W, H, view->warp_beta_x, view->warp_beta_y,
                                                                      w->_X1[0], w->_X2[0], //view->deform_weight_beta,
                                                                      W, H, view->warp_tau_x, view->warp_tau_y);
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        filter_invalid_device <<< w->_dimGrid, w->_dimBlock >>> (W, H, tmp_warp_x, view->warp_beta_x);
        filter_invalid_device <<< w->_dimGrid, w->_dimBlock >>> (W, H, tmp_warp_y, view->warp_beta_y);

        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        // TAU COHERENCE
        // gradient computation  completely overkill, use 3 channels just to add a mask
        vtv_sr_compute_gradient_device <<< w->_dimGrid, w->_dimBlock >>>
                                   ( W,H, view->deform_weight_tau, view->deform_weight_tau, view->deform_weight_tau,
                                     view->deform_weight_tau, w->_X1[0], w->_X2[0]);
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        check_warp_coherence_device <<< w->_dimGrid, w->_dimBlock >>>
                                                                    ( W, H, view->warp_tau_x, view->warp_tau_y,
                                                                      w->_X1[0], w->_X2[0], //view->deform_weight_tau,
                                                                      W, H, tmp_warp_x, tmp_warp_y);
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        check_warp_perfect_coherence_device <<< w->_dimGrid, w->_dimBlock >>>( W, H, view->warp_beta_x, view->warp_beta_y,
                                             W, H, view->warp_tau_x, view->warp_tau_y);
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        //check_warp_perfect_coherence_device <<< w->_dimGrid, w->_dimBlock >>>( W, H, view->warp_tau_x, view->warp_tau_y,
        //                                     W, H, view->warp_beta_x, view->warp_beta_y);
        //CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        write_pfm_image_signed(W, H, view->warp_beta_x, view->warp_beta_y, view->deform_weight_beta, data->_basedir + "beta_warp_fixed_%02d.pfm", nview);
        write_pfm_image_signed(W, H, view->warp_tau_x, view->warp_tau_y, view->deform_weight_tau, data->_basedir + "tau_warp_fixed_%02d.pfm", nview);
    }
    exit(0);

    return true;
}

// Get current target mask
bool coco::coco_vtv_get_target_mask( coco_vtv_data *data,
                  gsl_matrix* U )
{
    coco_vtv_sr_data *sr = data->_sr_data;

    assert( U->size2 == data->_W );
    assert( U->size1 == data->_H );
    cuda_memcpy( U, sr->_target_mask );

    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    return true;
}

