/* -*-c++-*- */
#include <iostream>
#include <algorithm>

#include "vtv.h"
#include "vtv.cuh"
#include "vtv_sr_unstructured.cuh"

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
#include "vtv_sr_kernels_unstructured.cu"

using namespace std;

/*****************************************************************************
       TV_x Superresolution
*****************************************************************************/

// Setup unstructured SR algorithm: init view and resolution data
bool coco::coco_vtv_sr_init_unstructured( coco_vtv_data *data, size_t nviews, size_t ds_factor ) {

    // can only be initialized once.
    assert( data->_sr_data_unstructured == NULL );
    coco_vtv_sr_data_unstructured *sr = new coco_vtv_sr_data_unstructured;
    size_t W = data->_W;
    size_t H = data->_H;
    sr->_nviews = nviews;
    sr->_dsf = ds_factor;
    sr->_w = W / sr->_dsf;
    sr->_h = H / sr->_dsf;
    // validate downscale factor (exact multiple of size)
    assert( sr->_w * sr->_dsf == W );
    assert( sr->_h * sr->_dsf == H );

    // each kernel is the size of the downscale factor
    sr->_ks = ds_factor;

    // default for 8-bit normalized
    sr->_sigma_sensor = 1./255.0f;
    sr->_ugrad_threshold = 0.01;

    // compute mem layout
    sr->_nfbytes_lo = sr->_w * sr->_h * sizeof(cuflt);
    sr->_nfbytes_hi = W*H*sizeof(cuflt);
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

    size_t MB = 1048576;
    size_t bytes_per_view = data->_nchannels *  sr->_nfbytes_lo // image_f
            + sr->_nfbytes_lo  // _weights_omega_i
            + 4 * sr->_nfbytes_lo  // warps tau x/y and dparts x/y
            + sr->_ks*sr->_ks * sr->_nfbytes_lo; // sparse matrix A_i (N kernels of size sr->_ks^2)
    // TODO: covariance weights
    size_t bytes_view_total = nviews * bytes_per_view;

    TRACE( "Allocating mem:" << std::endl );
    TRACE( "  " << bytes_per_view / MB << " Mb per view, " << bytes_view_total/MB << " total." << std::endl );

    for ( size_t i = 0 ; i < nviews ; i++ ) {

        coco_vtv_sr_view_data_unstructured *view = new coco_vtv_sr_view_data_unstructured;

        CUDA_SAFE_CALL( cudaMalloc( &view->_image_f, data->_nchannels * sr->_nfbytes_lo ));
        CUDA_SAFE_CALL( cudaMemset( view->_image_f, 0, data->_nchannels * sr->_nfbytes_lo ));

        CUDA_SAFE_CALL( cudaMalloc( &view->warp_tau_x, sr->_nfbytes_lo ));
        CUDA_SAFE_CALL( cudaMemset( view->warp_tau_x, 0.0, sr->_nfbytes_lo ));
        CUDA_SAFE_CALL( cudaMalloc( &view->warp_tau_y, sr->_nfbytes_lo ));
        CUDA_SAFE_CALL( cudaMemset( view->warp_tau_y, 0.0, sr->_nfbytes_lo ));

        CUDA_SAFE_CALL( cudaMalloc( &view->dpart_x, sr->_nfbytes_lo ));
        CUDA_SAFE_CALL( cudaMemset( view->dpart_x, 0.0, sr->_nfbytes_lo ));
        CUDA_SAFE_CALL( cudaMalloc( &view->dpart_y, sr->_nfbytes_lo ));
        CUDA_SAFE_CALL( cudaMemset( view->dpart_y, 0.0, sr->_nfbytes_lo ));

        CUDA_SAFE_CALL( cudaMalloc( &view->_weights_omega_i, sr->_nfbytes_lo ));
        CUDA_SAFE_CALL( cudaMemset( view->_weights_omega_i, 0.0, sr->_nfbytes_lo ));

        CUDA_SAFE_CALL( cudaMalloc( &view->_A, sr->_ks*sr->_ks * sr->_nfbytes_lo ));
        CUDA_SAFE_CALL( cudaMemset( view->_A, 0.0, sr->_ks*sr->_ks * sr->_nfbytes_lo ));

        sr->_views.push_back( view );
    }

    // Additional work mem (TODO: reduce, use temp buffers w->F[...])
    size_t srbytes = sr->_nfbytes_hi // _norm_mask
            + sr->_nfbytes_hi * (data->_nchannels+2); // temp buffers

    TRACE( "  " << srbytes/MB << " Mb for additional work structures." << std::endl );

    // Target coverage
    CUDA_SAFE_CALL( cudaMalloc( &(sr->_norm_mask), sr->_nfbytes_hi ));
    CUDA_SAFE_CALL( cudaMemset( sr->_norm_mask, 0, sr->_nfbytes_hi ));

    // Check for grayscale and add temp buffers if necessary
    coco_vtv_workspace *w = data->_workspace;
    while ( w->_temp.size() < data->_nchannels+2 ) {
        cuflt *tmp = NULL;
        CUDA_SAFE_CALL( cudaMalloc( &tmp, sr->_nfbytes_hi ));
        CUDA_SAFE_CALL( cudaMemset( tmp, 0, sr->_nfbytes_hi ));
        w->_temp.push_back( tmp );
    }

    // Filter for visibility masks
    gsl_vector *gaussian = gsl_kernel_gauss_1xn( 11, 2.0f );
    sr->_vmask_filter = cuda_kernel_alloc_separable( gaussian, gaussian );
    gsl_vector_free( gaussian );

    // Finalize
    data->_sr_data_unstructured = sr;
    return true;
}

// Free up data for unstructured SR algorithm
bool coco::coco_vtv_sr_free_unstructured( coco_vtv_data *data ) {

    assert( data != NULL );
    coco_vtv_sr_data_unstructured *sr = data->_sr_data_unstructured;
    assert( sr != NULL );

    for ( size_t i=0; i<sr->_nviews; i++ ) {

        coco_vtv_sr_view_data_unstructured *view = sr->_views[i];

        CUDA_SAFE_CALL( cudaFree( view->_image_f ));

        CUDA_SAFE_CALL( cudaFree( view->warp_tau_x ));
        CUDA_SAFE_CALL( cudaFree( view->warp_tau_y ));

        CUDA_SAFE_CALL( cudaFree( view->dpart_x ));
        CUDA_SAFE_CALL( cudaFree( view->dpart_y ));

        CUDA_SAFE_CALL( cudaFree( view->_weights_omega_i ));

        CUDA_SAFE_CALL( cudaFree( view->_A ));

        CUDA_SAFE_CALL( cudaFree( view->_cells ));

        delete view;
    }

    CUDA_SAFE_CALL( cudaFree( sr->_norm_mask ));

    cuda_kernel_free( sr->_vmask_filter );

    // finalize
    delete data->_sr_data_unstructured;
    data->_sr_data_unstructured = NULL;
    return true;
}

// Compute the sparse matrix A
bool coco::coco_vtv_sr_compute_sparse_matrix( coco_vtv_data *data ) {

    // check for required data
    assert( data != NULL );
    coco_vtv_sr_data_unstructured *sr = data->_sr_data_unstructured;
    assert( sr != NULL );
    size_t w = sr->_w;
    size_t h = sr->_h;

    dim3 dimBlock = sr->_dimBlock; // low res
    dim3 dimGrid = sr->_dimGrid;

    // Box filtering
    for ( size_t i=0; i<sr->_nviews; i++ ) {

        coco_vtv_sr_view_data_unstructured *view = sr->_views[i];

        cuda_set_A_for_box_filtering<<< dimGrid, dimBlock >>>( w, h,
                                                               sr->_ks,
                                                               view->_A );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    }

    return true;
}

// Init forward warp for a view : uses warps (make sure they are computed)
// warp=0: tau, warp=1:beta
// Currently completely on host, TODO: try to parallelize (hard)
bool coco::vtv_sr_init_forward_warp_structure_unstructured( coco_vtv_data *data, size_t nview ) {

    // check for required data
    assert( data != NULL );
    coco_vtv_sr_data_unstructured *sr = data->_sr_data_unstructured;
    assert( sr != NULL );
    size_t w = sr->_w;
    size_t h = sr->_h;
    assert( w*h > 0 );
    size_t W = data->_W;
    size_t H = data->_H;
    assert( W*H > 0 );
    coco_vtv_workspace *ws = data->_workspace;
    assert( ws != NULL );
    assert( nview < sr->_nviews );
    coco_vtv_sr_view_data_unstructured *view = sr->_views[nview];

    // Need warps from GPU
    cuflt *tmp_warp_x = new cuflt[w*h];
    cuflt *tmp_warp_y = new cuflt[w*h];

    CUDA_SAFE_CALL( cudaMemcpy( tmp_warp_x, view->warp_tau_x, sizeof(cuflt) * w*h, cudaMemcpyDeviceToHost ));
    CUDA_SAFE_CALL( cudaMemcpy( tmp_warp_y, view->warp_tau_y, sizeof(cuflt) * w*h, cudaMemcpyDeviceToHost ));
    view->_seg_end.clear();

    // Compute target cells for each source pixel
    int *c_in = new int[ w*h ];
    for ( size_t oy=0; oy<h; oy++ ) {
        for ( size_t ox=0; ox<w; ox++ ) {
            size_t o = ox + oy*w;
            if ( tmp_warp_x[o] < 0 || tmp_warp_y[o] < 0 || tmp_warp_x[o] > W || tmp_warp_y[o] > H  ) {
                c_in[ o ] = W*H;
                continue;
            }

            // get location in u
            cuflt uxv = tmp_warp_x[o] - 0.5;
            cuflt uyv = tmp_warp_y[o] - 0.5;
            int px = (int)floor(uxv);
            int py = (int)floor(uyv);
            if ( px < 0 || py < 0 || px > (int)W-1 || py > (int)H-1 ) {
                c_in[ o ] = W*H;
                continue;
            }
            int po = px + py*W;
            c_in[ o ] = po;
        }
    }

    // Group into non-overlapping segments
    // Needs array c_in
    TRACE( "grouping cells ..." );

    int *grouped = new int[ w*h ];
    size_t ngrouped = 0;
    memset( grouped, 0, sizeof(int) * w*h );
    int *count = new int[ W*H ];
    vector<int> cells;

    // BUG FIX: the variable margin was introduced because :
    //          in some cases the forward warp computed on CPU gives a slightly different result
    //          as in GPU. The extreme case being  floor(CPU value) != floor(GPU value)
    //          This makes that the non-overlapping segments, in fact, overlap,
    //          creating a non-determined behaviour when threads collide.
    //          Expanding the "confort zone" to a 3x3 neighborhood solves for this problem.
    //          The "drawback" is that there are slightly more segments.
    int margin = 1;

    while ( ngrouped < w*h ) {

        memset( count, 0, sizeof(int) * W*H );
        for ( size_t i=0 ; i < w*h ; i++ ) {
            if ( grouped[i] ) {
                continue;
            }

            size_t target = c_in[i];

            // check targets is unused
            if ( target == W*H ) {
                grouped[i] = 1;
                ngrouped++;
                continue;
            }

            bool ok = true;
            int px = target % W;
            int py = target / W;

            for (int x = -margin; x<=margin; ++x) {
                if (0<= px+x && px+x < (int)W ) {
                    for (int y = -margin; y<=margin; ++y) {
                        if (0<= py+y && py+y <(int)H ) {
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
                if (0<= px+x && px+x <(int)W ) {
                    for (int y = -margin; y<=margin; ++y) {
                        if (0<= py+y && py+y <(int)H ) {
                            assert ( count[px+x + (py+y) *W] == 0 );
                            count[px+x + (py+y) *W] = 1;
                        }
                    }
                }
            }
        }

        view->_seg_end.push_back( cells.size() );

        //TRACE( "  ... " << ngrouped << " grouped, " << cells.size() << " cells." << endl );
    }
    TRACE( "done." << endl );
    assert( ngrouped == w*h );

    // Copy new cell grouping to GPU

    if ( view->_cells != NULL ) {
        CUDA_SAFE_CALL( cudaFree( view->_cells ));
    }
    CUDA_SAFE_CALL( cudaMalloc( &view->_cells, sizeof(int) * cells.size() ));
    CUDA_SAFE_CALL( cudaMemcpy( view->_cells, &cells[0], sizeof(int) * cells.size(), cudaMemcpyHostToDevice ));

    // Cleanup
    delete[] tmp_warp_x;
    delete[] tmp_warp_y;
    delete[] grouped;
    delete[] count;
    delete[] c_in;
    return true;
}

// Setup a single view
bool coco::coco_vtv_sr_create_view_unstructured( coco_vtv_data *data, size_t nview, gsl_image *I) {

    // check for required data
    assert( data != NULL );
    coco_vtv_sr_data_unstructured *sr = data->_sr_data_unstructured;
    assert( sr != NULL );
    size_t W = data->_W;
    size_t H = data->_H;

    assert( W*H > 0 );
    assert( nview < sr->_nviews );
    coco_vtv_sr_view_data_unstructured *view = sr->_views[nview];

    // view image should be equal to downsampled size
    assert( I->_w == W / sr->_dsf );
    assert( I->_h == H / sr->_dsf );

    // Image
    size_t N = sr->_w * sr->_h;
    float *buffer_f = new cuflt[ N*data->_nchannels ];

    for ( size_t n = 0 ; n < data->_nchannels ; n++ ) {
        // load view to device
        gsl_matrix *channel = gsl_image_get_channel( I, (coco::gsl_image_channel)n );

        for ( size_t i=0; i<N; i++ ) {
            buffer_f[N*n+i] = (cuflt)channel->data[i];
        }
    }

    CUDA_SAFE_CALL( cudaMemcpy( view->_image_f, buffer_f, data->_nchannels*N*sizeof(cuflt), cudaMemcpyHostToDevice ));
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    delete[] buffer_f;

    return true;
}

static int meta_iter = -1;
static int iterations = 0;

// Update weight_omega_i
bool coco::coco_vtv_sr_compute_weights_unstructured( coco_vtv_data *data ) {

    // check for required data
    assert( data != NULL );
    coco_vtv_sr_data_unstructured *sr = data->_sr_data_unstructured;
    assert( sr != NULL );
    size_t W = data->_W; // high res
    size_t H = data->_H;
    assert( W*H > 0 );
    size_t w = sr->_w; // low res
    size_t h = sr->_h;
    assert( w*h > 0 );
    coco_vtv_workspace *ws = data->_workspace;
    assert(ws->_nfbytes == sr->_nfbytes_hi);
    dim3 DimBlock = ws->_dimBlock; // high res
    dim3 DimGrid = ws->_dimGrid;
    dim3 dimBlock = sr->_dimBlock; // low res
    dim3 dimGrid = sr->_dimGrid;

    ++meta_iter;

    if (data->_nchannels == 1) { // grayscale
        cuda_compute_gradient_device <<< DimGrid, DimBlock >>>
                                                             ( W, H, ws->_U[0], ws->_X1[0], ws->_X2[0] );
    } else { // rgb
        vtv_sr_compute_gradient_device <<< DimGrid, DimBlock >>>
                                                               ( W, H, ws->_U[0], ws->_U[1], ws->_U[2], ws->_X1[0], ws->_X2[0] );
    }
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    //write_pfm_image_signed(W, H,ws->_U[0], ws->_U[1], ws->_U[2], data->_basedir + "u_%02d.pfm", meta_iter);
    //write_pfm_image_signed(W, H,ws->_Uq[0], ws->_Uq[1], ws->_Uq[2], data->_basedir + "uq_%02d.pfm", meta_iter);

    // update each view
    for ( size_t nview=0; nview < sr->_nviews; nview++ ) {

        coco_vtv_sr_view_data_unstructured *view = sr->_views[nview];

        CUDA_SAFE_CALL( cudaMemset( view->_weights_omega_i, 0, sr->_nfbytes_lo ));

        //write_test_image_bool( W,H, view->_visibility_mask, data->_basedir + "visibility_mask_prev_%05i.png", nview );

        // compute angular weights with u gradient
        // dot product of grad u with partial tau partial z

        vtv_sr_u_gradient_weight_unstructured_device <<< dimGrid, dimBlock >>>
                                                                             ( W, H,
                                                                               w, h,
                                                                               ws->_X1[0], // u domain, high res
                ws->_X2[0],
                view->warp_tau_x, // vi domain, low res, values high res
                view->warp_tau_y,
                sr->_sigma_sensor,
                view->dpart_x, // vi domain, low res
                view->dpart_y, // dpart replaces aux_dmap_sigma*dtau/dz
                sr->_ugrad_threshold,
                view->_weights_omega_i ); // in low res
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        //write_pfm_image_signed( w, h, view->_weights_omega_i, data->_basedir + "view->_weights_omega_%03d.pfm", meta_iter*100+nview );
    }

    return true;
}

void coco::coco_vtv_sr_set_ugrad_threshold_unstructured( coco_vtv_data *data, cuflt ugrad_threshold) {
    data->_sr_data_unstructured->_ugrad_threshold = ugrad_threshold;
}

void coco::coco_vtv_sr_set_sigma_sensor_unstructured( coco_vtv_data *data, cuflt sigma_sensor) {
    data->_sr_data_unstructured->_sigma_sensor = sigma_sensor;
}

// Read the tau warps and deformation weights: from gsl_image to device cuflt*
bool coco::coco_vtv_sr_read_tau( coco_vtv_data *data, gsl_image** tau_warps ) {

    // check for required data
    assert( data != NULL );
    coco_vtv_sr_data_unstructured *sr = data->_sr_data_unstructured;
    assert( sr != NULL );
    size_t w = sr->_w;
    size_t h = sr->_h;
    size_t N = w*h;
    assert( N > 0 );

    cuflt *buffer_f = new cuflt[N];
    gsl_matrix *channel;

    for ( size_t nview = 0 ; nview < sr->_views.size() ; nview++ ) {

        coco_vtv_sr_view_data_unstructured *view = sr->_views[nview];

        channel = gsl_image_get_channel( tau_warps[nview], GSL_IMAGE_RED ); // load tau x
        gsl_matrix_copy_to_buffer( channel,  buffer_f );
        CUDA_SAFE_CALL( cudaMemcpy( view->warp_tau_x, buffer_f, sr->_nfbytes_lo, cudaMemcpyHostToDevice ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        channel = gsl_image_get_channel( tau_warps[nview], GSL_IMAGE_GREEN ); // load tau y
        gsl_matrix_copy_to_buffer( channel,  buffer_f );
        CUDA_SAFE_CALL( cudaMemcpy( view->warp_tau_y, buffer_f, sr->_nfbytes_lo, cudaMemcpyHostToDevice ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        //TRACE("Test: write pfm tau warp, view " << nview << std::endl);
        //write_pfm_image_signed(w, h, view->warp_tau_x, view->warp_tau_y, view->warp_tau_y, data->_basedir + "tau_%02lu.pfm", nview);
    }

    delete [] buffer_f;

    return true;
}

// Read the partial tau: from gsl_image to device cuflt*
bool coco::coco_vtv_sr_read_partial_tau( coco_vtv_data *data, gsl_image** partial_tau ) {

    // check for required data
    assert( data != NULL );
    coco_vtv_workspace *ws = data->_workspace;
    assert( ws != NULL );
    coco_vtv_sr_data_unstructured *sr = data->_sr_data_unstructured;
    assert( sr != NULL );
    size_t w = sr->_w;
    size_t h = sr->_h;
    size_t N = w*h;
    assert( N > 0 );

    cuflt *buffer_f = new cuflt[N];
    gsl_matrix *channel;

    for ( size_t nview = 0 ; nview < sr->_views.size() ; nview++ ) {

        coco_vtv_sr_view_data_unstructured *view = sr->_views[nview];
        cuflt *sigma_z = ws->_temp[0];

        channel = gsl_image_get_channel( partial_tau[nview], GSL_IMAGE_RED ); // load sigma_z
        gsl_matrix_copy_to_buffer( channel,  buffer_f );
        CUDA_SAFE_CALL( cudaMemcpy( sigma_z, buffer_f, sr->_nfbytes_lo, cudaMemcpyHostToDevice ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        channel = gsl_image_get_channel( partial_tau[nview], GSL_IMAGE_GREEN ); // load dtau/dy x
        gsl_matrix_copy_to_buffer( channel,  buffer_f );
        CUDA_SAFE_CALL( cudaMemcpy( view->dpart_x, buffer_f, sr->_nfbytes_lo, cudaMemcpyHostToDevice ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        // do the product sigma_z*dtau/dz
        cuda_multiply_with_device<<< sr->_dimGrid, sr->_dimBlock >>> ( w, h, view->dpart_x, sigma_z );

        channel = gsl_image_get_channel( partial_tau[nview], GSL_IMAGE_BLUE ); // load dtau/dy y
        gsl_matrix_copy_to_buffer( channel,  buffer_f );
        CUDA_SAFE_CALL( cudaMemcpy( view->dpart_y, buffer_f, sr->_nfbytes_lo, cudaMemcpyHostToDevice ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        // do the product sigma_z*dtau/dz
        cuda_multiply_with_device<<< sr->_dimGrid, sr->_dimBlock >>> ( w, h, view->dpart_y, sigma_z );

        //TRACE("Test: write pfm partial tau, view " << nview << std::endl);
        //write_pfm_image_signed(w, h, sigma_z, view->dpart_x, view->dpart_y, data->_basedir + "partial_tau_%02lu.pfm", nview);
    }
    delete [] buffer_f;

    return true;
}

// Compute the initial image, starting point of the algorithm
bool coco::coco_vtv_sr_compute_initial_image( coco_vtv_data *data ) {

    // check for required data
    assert( data != NULL );
    coco_vtv_sr_data_unstructured *sr = data->_sr_data_unstructured;
    assert( sr != NULL );
    coco_vtv_workspace *ws = data->_workspace;
    size_t W = data->_W;
    size_t H = data->_H;
    assert( W*H );

    // Clear target image
    for ( size_t i = 0 ; i < data->_nchannels ; i++ ) {
        CUDA_SAFE_CALL( cudaMemset( ws->_U[i], 0, ws->_nfbytes ));
    }

    // read starting image for the algorithm (to test only)
    TRACE("Read starting image for the algorithm" << std::endl);
    gsl_image *initialization = gsl_image_load( data->_basedir + "u.png" );

    if ( initialization != NULL ) {

        std::vector<gsl_matrix*> init_vector;
        init_vector.push_back( initialization->_r );
        if ( data->_nchannels == 3 ) {
            init_vector.push_back( initialization->_g );
            init_vector.push_back( initialization->_b );
        }

        for ( size_t i = 0 ; i < data->_nchannels ; i++ ) {

            gsl_matrix *u = init_vector[i]; // source
            assert( u->size2 == data->_W );
            assert( u->size1 == data->_H );
            cuda_memcpy( ws->_U[i], u );
            CUDA_SAFE_CALL( cudaThreadSynchronize() );
        }

        gsl_image_free( initialization );

    } else {

        for ( size_t i = 0 ; i < data->_nchannels; i++ ) {

            // Clear the normalization mask
            CUDA_SAFE_CALL( cudaMemset( sr->_norm_mask, 0, ws->_nfbytes ));

            // Perform splatting for every input view
            for ( size_t nview = 0 ; nview < sr->_views.size() ; nview++ ) {

                coco_vtv_sr_view_data_unstructured *view = sr->_views[nview];



                // Forward warp, non-overlap regions sequentially
                int seg_start = 0;
                for ( size_t j = 0 ; j < view->_seg_end.size() ; j++ ) {

                    int seg_end = view->_seg_end[j];
                    int seg_size = seg_end - seg_start;

                    // forward warp call for this segment, cannot overlap
                    int seg_width = cuda_default_block_size_x() * cuda_default_block_size_y();
                    dim3 dimBlock = dim3( seg_width, 1 );
                    dim3 dimGrid = dim3( seg_size / seg_width + 1, 1 );

                    cuda_deconvolution_nonsep_device_param<<< dimGrid, dimBlock >>>( W, H, seg_width,
                                                                                     view->_image_f + i*sr->_w*sr->_h,
                                                                                     view->_cells,
                                                                                     seg_start, seg_end,
                                                                                     sr->_ks,
                                                                                     0, // no weights for splatting
                                                                                     view->_weights_omega_i,
                                                                                     view->warp_tau_x, view->warp_tau_y,
                                                                                     ws->_U[i],
                                                                                     sr->_norm_mask );

                    CUDA_SAFE_CALL( cudaThreadSynchronize() );

                    seg_start = seg_end;
                }
            }

            // Normalize
            cuda_normalize_device<<< ws->_dimGrid, ws->_dimBlock >>>
                                                                   ( W, H, ws->_U[i], sr->_norm_mask );
        }
    }

    TRACE("Write starting image for the algorithm" << std::endl);
    write_pfm_image_signed( W, H, ws->_U[0], ws->_U[1], ws->_U[2], data->_basedir + "u_init.pfm", 0 );

    return true;
}

// Blur high res image to test the kernels
bool coco::coco_vtv_sr_downsample( coco_vtv_data *data ) {

    // check for required data
    assert( data != NULL );
    coco_vtv_sr_data_unstructured *sr = data->_sr_data_unstructured;
    assert( sr != NULL );
    size_t W = data->_W; // high res
    size_t H = data->_H;
    assert( W*H > 0 );
    size_t w = sr->_w; // low res
    size_t h = sr->_h;
    assert( w*h > 0 );
    coco_vtv_workspace *ws = data->_workspace;
    assert(ws->_nfbytes == sr->_nfbytes_hi);

    for ( size_t nview = 0 ; nview < sr->_views.size() ; nview++ ) {

        coco_vtv_sr_view_data_unstructured *view = sr->_views[nview];

        for ( size_t i = 0 ; i < data->_nchannels; i++ ) {

            CUDA_SAFE_CALL( cudaMemset(  ws->_temp[i], 0, ws->_nfbytes ));

            cuda_convolution_nonsep_param( data,
                                           view->_A,
                                           view->warp_tau_x, view->warp_tau_y,
                                           ws->_U[i],
                                           ws->_temp[i] );
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );
        }

        TRACE("Write temp image v" << nview << std::endl);
        write_pfm_image_signed( w, h, ws->_temp[0], ws->_temp[1], ws->_temp[2], data->_basedir + "image_lo_%02lu.pfm", nview );
    }

    return true;
}

// Compute primal energy
double coco::coco_vtv_sr_primal_energy_unstructured( coco_vtv_data *data ) {

    // check for required data
    assert( data != NULL );
    coco_vtv_sr_data_unstructured *sr = data->_sr_data_unstructured;
    assert( sr != NULL );
    size_t W = data->_W; // high res
    size_t H = data->_H;
    assert( W*H > 0 );
    size_t w = sr->_w; // low res
    size_t h = sr->_h;
    assert( w*h > 0 );
    coco_vtv_workspace *ws = data->_workspace;
    assert(ws->_nfbytes == sr->_nfbytes_hi);
    dim3 DimBlock = ws->_dimBlock; // high res
    dim3 DimGrid = ws->_dimGrid;
    dim3 dimBlock = sr->_dimBlock; // low res
    dim3 dimGrid = sr->_dimGrid;

    if (data->_nchannels == 1) { // grayscale
        cuda_compute_gradient_device <<< DimGrid, DimBlock >>>
                                                             ( W, H, ws->_U[0], ws->_X1q[0], ws->_X2q[0] );
    } else { // rgb
        vtv_sr_compute_gradient_device <<< DimGrid, DimBlock >>>
                                                               ( W, H, ws->_U[0], ws->_U[1], ws->_U[2], ws->_X1q[0], ws->_X2q[0] );
    }
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    // TV component
    switch ( data->_regularizer ) {
    case 0:
        printf("Not implemented for now\n");
        exit(0);
        break;
    case 1:
        if ( data->_nchannels == 1 ) {
            cuda_compute_norm_device<<< DimGrid, DimBlock >>>
                                                            ( W, H,
                                                              ws->_X1q[0], ws->_X2q[0], ws->_X1q[0], ws->_X2q[0], ws->_X1q[0], ws->_X2q[0],
                    ws->_G[0] );
        } else if ( data->_nchannels == 3 ) {
            cuda_compute_norm_device<<< DimGrid, DimBlock >>>
                                                            ( W, H,
                                                              ws->_X1q[0], ws->_X2q[0], ws->_X1q[1], ws->_X2q[1], ws->_X1q[2], ws->_X2q[2],
                    ws->_G[0] );
        }
        break;
    case 2:
        // Compute largest singular value of gradient matrix
        if ( data->_nchannels == 1 ) {
            cuda_compute_largest_singular_value_device<<< DimGrid, DimBlock >>>
                                                                              ( W, H,
                                                                                ws->_X1q[0], ws->_X2q[0], ws->_X1q[0], ws->_X2q[0], ws->_X1q[0], ws->_X2q[0],
                    ws->_G[0] );
        } else if ( data->_nchannels == 3 ) {
            cuda_compute_largest_singular_value_device<<< DimGrid, DimBlock >>>
                                                                              ( W, H,
                                                                                ws->_X1q[0], ws->_X2q[0], ws->_X1q[1], ws->_X2q[1], ws->_X1q[2], ws->_X2q[2],
                    ws->_G[0] );
        }
        break;
    default:
        break;
    }

    float *E_TV = new cuflt[ W*H ];
    CUDA_SAFE_CALL( cudaMemcpy( E_TV, ws->_G[0], sr->_nfbytes_hi, cudaMemcpyDeviceToHost ));
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    // Compute tv energy (integral over gamma)
    cuflt e_tv = 0.0;
    for ( size_t i=0; i<W*H; i++ ) {
        e_tv += E_TV[i];
    }

    delete[] E_TV;

    // Data Term
    cuflt *v_i = ws->_temp[0]; // tmp image from U downsampling
    cuflt *tmp_energy = ws->_temp[1]; // low res tmp for data energy

    CUDA_SAFE_CALL( cudaMemset( tmp_energy, 0, sr->_nfbytes_hi ));

    // Sum contributions for all views
    for ( size_t nview = 0 ; nview < sr->_views.size() ; nview++ ) {

        coco_vtv_sr_view_data_unstructured *view = sr->_views[nview];

        for ( size_t i = 0 ; i < data->_nchannels; i++ ) {

            CUDA_SAFE_CALL( cudaMemset( v_i, 0,  sr->_nfbytes_hi ));

            // filter the high res image to get v_i in low res
            cuda_convolution_nonsep_param( data,
                                           view->_A,
                                           view->warp_tau_x, view->warp_tau_y,
                                           ws->_U[i], v_i );
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            // subtract image v_i from input cuflt image v_i*
            cuda_subtract_from_device<<< dimGrid, dimBlock >>>
                                                             ( w, h, view->_image_f + i*w*h, v_i );
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            // square the result
            cuda_multiply_with_device<<< dimGrid, dimBlock >>>
                                                             ( w, h, v_i, v_i );
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            // apply the weights
            cuda_multiply_with_device<<< dimGrid, dimBlock >>> ( w, h, v_i, view->_weights_omega_i );
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            // cumulate energy
            cuda_add_to_device<<< dimGrid, dimBlock >>>
                                                      ( w, h, v_i, tmp_energy);
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );
        }
    }

    cuflt *E_DATA = new cuflt[ w * h ];
    CUDA_SAFE_CALL( cudaMemcpy( E_DATA, tmp_energy, sr->_nfbytes_lo, cudaMemcpyDeviceToHost ));
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    // Compute tv energy
    cuflt e_data = 0.0;
    for ( size_t i = 0 ; i < w * h ; i++ ) {
        e_data += E_DATA[i];
    }

    delete[] E_DATA;

    cuflt energy = data->_lambda * e_tv + e_data;

//    TRACE("Energy : " << energy << " : E_TV = " << e_tv << " * " << data->_lambda << " = " << e_tv *  data->_lambda
//          << " | E_DATA =  "<< e_data << std::endl);

    return energy;
}

// Slow nonseparable convolution with parametrized kernel
bool coco::cuda_convolution_nonsep_param( coco_vtv_data *data,
                                          cuflt *A,
                                          cuflt *warp_tau_x, // vi domain, low res, values high res
                                          cuflt *warp_tau_y,
                                          const cuflt* in, cuflt *out ) {

    assert( data != NULL );
    coco_vtv_sr_data_unstructured *sr = data->_sr_data_unstructured;
    assert( sr != NULL );
    size_t W = data->_W; // high res
    size_t H = data->_H;
    assert( W*H > 0 );
    size_t w = sr->_w; // low res
    size_t h = sr->_h;
    assert( w*h > 0 );

    dim3 dimBlock = sr->_dimBlock; // low res
    dim3 dimGrid = sr->_dimGrid;

    // Compute divergence step
    cuda_convolution_nonsep_device_param<<< dimGrid, dimBlock >>>
                                                                ( W, H,
                                                                  w, h,
                                                                  sr->_ks,
                                                                  A,
                                                                  warp_tau_x,
                                                                  warp_tau_y,
                                                                  in,
                                                                  out );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    return true;
}

// Forward splatting with warp tau of a single input view (low res)
bool coco::cuda_deconvolution_nonsep_param( coco_vtv_data *data,
                                            cuflt *input, // low res input
                                            size_t nview,
                                            cuflt *output ) { // high res output

    // check for required data
    assert( data != NULL );
    coco_vtv_sr_data_unstructured *sr = data->_sr_data_unstructured;
    assert( sr != NULL );
    size_t W = data->_W; // high res
    size_t H = data->_H;
    assert( W*H > 0 );
    coco_vtv_sr_view_data_unstructured *view = sr->_views[nview];

    // Forward warp, non-overlap regions sequentially
    int seg_start = 0;
    for ( size_t j = 0 ; j < view->_seg_end.size() ; j++ ) {

        int seg_end = view->_seg_end[j];
        int seg_size = seg_end - seg_start;

        // forward warp call for this segment, cannot overlap
        int seg_width = cuda_default_block_size_x() * cuda_default_block_size_y();
        dim3 dimBlock = dim3( seg_width, 1 );
        dim3 dimGrid = dim3( seg_size / seg_width + 1, 1 );

        cuda_deconvolution_nonsep_device_param<<< dimGrid, dimBlock >>>( W, H, seg_width,
                                                                         input,
                                                                         view->_cells,
                                                                         seg_start, seg_end,
                                                                         sr->_ks,
                                                                         view->_A,
                                                                         view->_weights_omega_i,
                                                                         view->warp_tau_x, view->warp_tau_y,
                                                                         output,
                                                                         sr->_norm_mask );

        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        seg_start = seg_end;
    }

    return true;
}

// Write current solution in pfm format
bool coco::coco_vtv_sr_write_pfm_solution( coco_vtv_data *data ) {

    coco_vtv_workspace *ws = data->_workspace;
    size_t W = data->_W;
    size_t H = data->_H;
    assert( W*H > 0 );

    if ( data->_nchannels == 3 ) {
        write_pfm_image_signed( W, H, ws->_U[0], ws->_U[1], ws->_U[2], data->_basedir + "output.pfm", 0 );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    }

    return true;
}


// Perform one iteration of Algorithm 1, Chambolle-Pock
bool coco::coco_vtv_sr_iteration_fista_unstructured( coco_vtv_data *data ) {

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

// Perform one single shrinkage step (ISTA)
bool coco::vtv_sr_ista_step_unstructured( coco_vtv_data *data ) {

    assert( data != NULL );
    coco_vtv_workspace *w = data->_workspace;
    coco_vtv_sr_data_unstructured *sr = data->_sr_data_unstructured;
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

// Perform one single shrinkage step (ISTA)
bool coco::vtv_sr_dataterm_gradient_unstructured( coco_vtv_data *data ) {

    assert( data != NULL );
    coco_vtv_workspace *ws = data->_workspace;
    coco_vtv_sr_data_unstructured *sr = data->_sr_data_unstructured;
    assert( sr != NULL );
    size_t W = data->_W; // high res
    size_t H = data->_H;
    assert( W*H > 0 );
    size_t w = sr->_w; // low res
    size_t h = sr->_h;
    assert( w*h > 0 );
    dim3 DimBlock = ws->_dimBlock; // high res
    dim3 DimGrid = ws->_dimGrid;
    dim3 dimBlock = sr->_dimBlock; // low res
    dim3 dimGrid = sr->_dimGrid;

    // Compute gradient of data term

    for ( size_t i=0; i<data->_nchannels; i++ ) {

        // Start descent from current solution
        CUDA_SAFE_CALL( cudaMemcpy( ws->_Uq[i], ws->_U[i], ws->_nfbytes, cudaMemcpyDeviceToDevice ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        // Clear derivative and normalization weights
        CUDA_SAFE_CALL( cudaMemset( ws->_G[i], 0, ws->_nfbytes ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );
        CUDA_SAFE_CALL( cudaMemset( sr->_norm_mask, 0, ws->_nfbytes ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        cuflt *lores_tmp = ws->_temp[0];

        // Sum contributions for all views
        for ( size_t nview=0; nview < sr->_views.size(); nview++ ) {

            coco_vtv_sr_view_data_unstructured *view = sr->_views[nview];

            // clear the tmp buffer
            CUDA_SAFE_CALL( cudaMemset( lores_tmp, 0, ws->_nfbytes ));

            // reconstruct the low res image v_k given the current high res solution u
            cuda_convolution_nonsep_param( data,
                                           view->_A,
                                           view->warp_tau_x, view->warp_tau_y,
                                           ws->_U[i],
                                           lores_tmp );
            CUDA_SAFE_CALL( cudaThreadSynchronize() );

            //write_pfm_image_signed( w, h, lores_tmp, data->_basedir + "lores_tmp_%03lu.pfm", iterations * 8 + nview );

            // compute dv that is the difference between reconstructed low res image v_k and data v_k*
            cuda_subtract_from_device<<< dimGrid, dimBlock >>>
                                                             ( w, h, view->_image_f + i*w*h, lores_tmp );
            CUDA_SAFE_CALL( cudaThreadSynchronize() );

            //write_pfm_image_signed( w, h, lores_tmp, data->_basedir + "subs_%03lu.pfm", iterations * 8 + nview );

            // deconvolution step (applying the weights)
            cuda_deconvolution_nonsep_param( data,
                                             lores_tmp, // low res input
                                             nview,
                                             ws->_G[i] );
            CUDA_SAFE_CALL( cudaThreadSynchronize() );
        }
    }

    //write_pfm_image_signed( W, H, ws->_G[0], ws->_G[1], ws->_G[2], data->_basedir + "G_%03lu.pfm", iterations );

    for ( size_t i=0; i<data->_nchannels; i++ ) {

        cuda_normalize_device<<< DimGrid, DimBlock >>>
                                                     ( W, H, ws->_G[i], sr->_norm_mask );
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        cuda_clamp_device<<< DimGrid, DimBlock >>>
                                                 ( W, H, ws->_G[i], -1.0f, 1.0f );
        CUDA_SAFE_CALL( cudaThreadSynchronize() );
    }

    //write_pfm_image_signed( W, H, ws->_G[0], ws->_G[1], ws->_G[2], data->_basedir + "dataterm_gradient_normalized_%03lu.pfm", iterations );
    iterations++;

    return true;
}

bool coco::coco_vtv_sr_init_regularizer_weight_unstructured( coco_vtv_data *data ) {

    // check for required data
    assert( data != NULL );
    coco_vtv_sr_data_unstructured *sr = data->_sr_data_unstructured;
    assert( sr != NULL );
    size_t W = data->_W;
    size_t H = data->_H;
    assert( W*H > 0 );
    coco_vtv_workspace *ws = data->_workspace;

    // Use target mask as a regularizer weight
    for ( size_t i=0; i<data->_nchannels; i++ ) {
        // Use target mask as a regularizer weight
        if ( ws->_g[i] == NULL ) {
            CUDA_SAFE_CALL( cudaMalloc( &ws->_g[i], ws->_nfbytes ));
            CUDA_SAFE_CALL( cudaMemset( ws->_g[i], 0, ws->_nfbytes ));
        }
        vtv_sr_init_regularizer_weight_device<<< ws->_dimGrid, ws->_dimBlock >>>
                                                                               ( W, H,
                                                                                 data->_lambda_max_factor * data->_lambda,
                                                                                 data->_lambda,
                                                                                 sr->_nviews,
                                                                                 sr->_norm_mask, ws->_g[i] );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        // Convolve weight
        cuda_convolution( sr->_vmask_filter, W, H, ws->_g[i], ws->_temp[0] );
        CUDA_SAFE_CALL( cudaMemcpy( ws->_g[i], ws->_temp[0], ws->_nfbytes, cudaMemcpyDeviceToDevice ));

        // Write final mask and result as a reference
        if ( traceLevel() > 2 ) {
            write_test_image_unsigned( W, H, ws->_temp[0], data->_basedir + "target_mask.png", 0 );
            write_test_image_unsigned( W, H, ws->_g[i], data->_basedir + "regweight.png", 0 );
        }
    }

    return true;
}
