/* -*-c++-*- */

#include <cuda/cuda_kernels.cuh>
#include <cuda/cuda_helper.h>

#include "gradientIBR_kernels.cuh"
#include "gradientIBR.cuh"
#include "config.h"

static const cuflt epsilon = 1e-3;
static const int nbChannels = 3;

using namespace std;

/*****************************************************************************
       TV_x Superresolution
*****************************************************************************/

// Get current solution
void get_solution( Data* data, std::vector<coco::gsl_matrix*> &U ) {

    for ( size_t i = 0 ; i < nbChannels ; ++i ) {

        coco::gsl_matrix *u = U[i];
        assert( u->size2 == data->_W );
        assert( u->size1 == data->_H );
        cuda_memcpy( u, data->_U[i] );
    }
    for ( size_t i = nbChannels ; i < U.size() ; ++i ) {

        coco::gsl_matrix *u = U[i];
        assert( u->size2 == data->_W );
        assert( u->size1 == data->_H );
        cuda_memcpy( u, data->_U[0] );
    }
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}

// Compute backward visibility (gamma domain)
void setup_target_visibility( Data* data ) {

    // check for required data
    assert( data != NULL );
    size_t W = data->_W; // high res
    size_t H = data->_H;
    assert( W*H > 0 );

    // Clear the normalization mask
    CUDA_SAFE_CALL( cudaMemset( data->_visibility_mask, 0, W*H*sizeof(bool) ));
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    // Create the mask
    for ( size_t nview = 0 ; nview < data->_views.size() ; nview++ ) {

        ViewData *view = data->_views[nview];

        // Forward warp, non-overlap regions sequentially
        int seg_start = 0;
        for ( size_t j = 0 ; j < view->_seg_end.size() ; j++ ) {

            int seg_end = view->_seg_end[j];
            int seg_size = seg_end - seg_start;

            // forward warp call for this segment, cannot overlap
            int seg_width = coco::cuda_default_block_size_x() * coco::cuda_default_block_size_y();
            dim3 dimBlock_splatting = dim3( seg_width, 1 );
            dim3 dimGrid_splatting = dim3( seg_size / seg_width + 1, 1 );

            setup_visibility_mask<<< dimGrid_splatting, dimBlock_splatting >>>( W, H, seg_width,
                                                                                view->_cells,
                                                                                seg_start, seg_end,
                                                                                data->_ks,
                                                                                data->_dsf,
                                                                                view->_warp_tau_x, view->_warp_tau_y,
                                                                                data->_visibility_mask );
            CUDA_SAFE_CALL( cudaThreadSynchronize() );

            seg_start = seg_end;
        }
    }

    coco::write_test_image_bool( W, H, data->_visibility_mask, data->_outdir + "/_visibility_mask.png", 0 );
}


// Perform TV on init image to fill holes
void hole_filling( Data* data ) {

    // check for required data
    assert( data != NULL );
    size_t W = data->_W; // high res
    size_t H = data->_H;
    assert( W*H > 0 );
    dim3 DimGrid = data->_DimGrid;
    dim3 DimBlock = data->_DimBlock;

    cuflt *laplacian = data->_temp[0];
    cuflt energy = 0.0;
    cuflt previous_energy = 0.0;

    TRACE("Perform Poisson diffusion on init image to fill holes   [");
    int iterations = 500;
    for ( int k = 0 ; k < iterations ; ++k ) {

        if ( (k%(iterations/10)) == 0 ) {
            TRACE( "." );
        }

        previous_energy = energy;

        // Poisson diffusion
        for ( size_t i = 0 ; i < nbChannels; i++ ) {

            // compute the laplacian nabla(u) in the holes, 0 elsewhere
            compute_laplacian<<< DimGrid, DimBlock >>>( W, H,
                                                        data->_U[i],
                                                        laplacian,
                                                        data->_visibility_mask );
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );
            // u(t+i) = u(t) + nabla(u)
            cuda_add_to_device<<< DimGrid, DimBlock >>>( W, H, laplacian, data->_U[i] );
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            //compute energy
            cuda_multiply_with_device<<< DimGrid, DimBlock >>>( W, H, laplacian, data->_U[i] );

            cuflt *E = new cuflt[ W * H ];
            CUDA_SAFE_CALL( cudaMemcpy( E, laplacian, data->_nfbytes_hi, cudaMemcpyDeviceToHost ));
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            energy = 0.0;
            for ( size_t p = 0 ; p < W*H ; ++p ) {
                energy += E[p];
            }

            delete[] E;
        }
        //TRACE("Energy: " << energy << endl);
        if ( abs(previous_energy - energy) < epsilon ) {

            TRACE("] Energy minimum reached at iteration " << k << endl);
            break;
        }
    }
    if ( abs(previous_energy - energy) >= epsilon ) {

        TRACE( "] maximum number of iterations reached" << endl );
    }

    TRACE("Write filled starting image" << std::endl);
    coco::write_pfm_image_signed( W, H, data->_U[0], data->_U[1], data->_U[2], data->_outdir + "/u_init_filled.pfm", 0 );
}

// Setup unstructured SR algorithm: init view and resolution data
Data* init_data( Config_data *config_data ) {

    Data *data = new Data;
    data->_nviews = config_data->_nviews;

    data->_dsf = config_data->_dsf;
    // low res
    data->_w = config_data->_w;
    data->_h = config_data->_h;
    // high res
    data->_W = data->_dsf * data->_w;
    data->_H = data->_dsf * data->_h;

    // default for 8-bit normalized
    data->_sigma_sensor = config_data->_sigma_sensor;
    data->_ugrad_threshold = config_data->_ugrad_threshold;
    data->_ks = data->_dsf + 1;
    data->_dw_type = config_data->_dw_type;
    assert( data->_dw_type == 0 ||
            data->_dw_type == 1 ||
            data->_dw_type == 2 );
    data->_gw_type = config_data->_gw_type;
    assert( data->_gw_type == 0 ||
            data->_gw_type == 1 ||
            data->_gw_type == 2 );

    data->_dt_alpha = config_data->_dt_alpha;
    data->_dt_beta = config_data->_dt_beta;

    data->_gradient_step = config_data->_gradient_step;

    data->_outdir = config_data->_outdir;
    data->_init_name = config_data->_init_name;

    // CUDA Block dimensions
    // low res
    data->_nfbytes_lo = data->_w * data->_h * sizeof(cuflt);
    data->_nfbytes_hi = data->_W * data->_H * sizeof(cuflt);
    data->_dimBlock = dim3( coco::cuda_default_block_size_x(),
                          coco::cuda_default_block_size_y() );
    size_t blocks_w = data->_w / data->_dimBlock.x;
    if ( data->_w % data->_dimBlock.x != 0 ) {
        blocks_w += 1;
    }
    size_t blocks_h = data->_h / data->_dimBlock.y;
    if ( data->_h % data->_dimBlock.y != 0 ) {
        blocks_h += 1;
    }
    data->_dimGrid = dim3(blocks_w, blocks_h);

    // high res
    data->_DimBlock = dim3( coco::cuda_default_block_size_x(),
                          coco::cuda_default_block_size_y() );
    size_t blocks_W = data->_W / data->_DimBlock.x;
    if ( data->_W % data->_DimBlock.x != 0 ) {
        blocks_W += 1;
    }
    size_t blocks_H = data->_H / data->_DimBlock.y;
    if ( data->_H % data->_DimBlock.y != 0 ) {
        blocks_H += 1;
    }
    data->_DimGrid = dim3(blocks_W, blocks_H);

    size_t MB = 1048576;
    size_t bytes_per_view = nbChannels *  data->_nfbytes_lo // image_f
            + data->_nfbytes_lo  // _weights_omega_i
            + 4 * data->_nfbytes_lo  // warps tau x/y and dparts x/y
            + data->_ks*data->_ks * data->_nfbytes_lo; // sparse matrix A_i (N kernels of size data->_ks^2)
    // TODO: covariance weights
    size_t bytes_view_total = data->_nviews * bytes_per_view;

    TRACE( "Allocating mem:" << std::endl );
    TRACE( "  " << bytes_per_view / MB << " Mb per view, " << bytes_view_total/MB << " total." << std::endl );

    for ( size_t nview = 0 ; nview < data->_nviews ; ++nview ) {

        ViewData *view = new ViewData;

        CUDA_SAFE_CALL( cudaMalloc( &view->_image_f, nbChannels * data->_nfbytes_lo ));
        CUDA_SAFE_CALL( cudaMemset( view->_image_f, 0, nbChannels * data->_nfbytes_lo ));

        CUDA_SAFE_CALL( cudaMalloc( &view->_warp_tau_x, data->_nfbytes_lo ));
        CUDA_SAFE_CALL( cudaMemset( view->_warp_tau_x, 0.0, data->_nfbytes_lo ));
        CUDA_SAFE_CALL( cudaMalloc( &view->_warp_tau_y, data->_nfbytes_lo ));
        CUDA_SAFE_CALL( cudaMemset( view->_warp_tau_y, 0.0, data->_nfbytes_lo ));

        CUDA_SAFE_CALL( cudaMalloc( &view->dpart_x, data->_nfbytes_lo ));
        CUDA_SAFE_CALL( cudaMemset( view->dpart_x, 0.0, data->_nfbytes_lo ));
        CUDA_SAFE_CALL( cudaMalloc( &view->dpart_y, data->_nfbytes_lo ));
        CUDA_SAFE_CALL( cudaMemset( view->dpart_y, 0.0, data->_nfbytes_lo ));

        CUDA_SAFE_CALL( cudaMalloc( &view->_weights_omega_i, data->_nfbytes_lo ));
        CUDA_SAFE_CALL( cudaMemset( view->_weights_omega_i, 0.0, data->_nfbytes_lo ));

        CUDA_SAFE_CALL( cudaMalloc( &view->_B, data->_ks*data->_ks * data->_nfbytes_lo ));
        CUDA_SAFE_CALL( cudaMemset( view->_B, 0.0, data->_ks*data->_ks * data->_nfbytes_lo ));

        view->_cells = NULL;

        data->_views.push_back( view );
    }

    // Additional work mem (TODO: reduce, use temp buffers ws->F[...])
    size_t srbytes = data->_nfbytes_hi // _norm_mask
            + data->_W * data->_H * sizeof(bool) // _visibility_mask
            + data->_nfbytes_hi * (nbChannels+2); // temp buffers

    TRACE( "  " << srbytes/MB << " Mb for additional work structures." << std::endl );

    CUDA_SAFE_CALL( cudaMalloc( &(data->_norm_mask), data->_nfbytes_hi ));
    CUDA_SAFE_CALL( cudaMemset( data->_norm_mask, 0, data->_nfbytes_hi ));

    CUDA_SAFE_CALL( cudaMalloc( &(data->_norm_mask), data->_nfbytes_hi ));
    CUDA_SAFE_CALL( cudaMemset( data->_norm_mask, 0, data->_nfbytes_hi ));

    CUDA_SAFE_CALL( cudaMalloc( &data->_u_grad_x, data->_nfbytes_hi ));
    CUDA_SAFE_CALL( cudaMemset( data->_u_grad_x, 0, data->_nfbytes_hi ));
    CUDA_SAFE_CALL( cudaMalloc( &data->_u_grad_y, data->_nfbytes_hi ));
    CUDA_SAFE_CALL( cudaMemset( data->_u_grad_y, 0, data->_nfbytes_hi ));

    CUDA_SAFE_CALL( cudaMalloc( &data->_target_mask, data->_nfbytes_hi ));
    CUDA_SAFE_CALL( cudaMemset( data->_target_mask, 0, data->_nfbytes_hi ));

    CUDA_SAFE_CALL( cudaMalloc( &(data->_visibility_mask), data->_W * data->_H * sizeof(bool) ));
    CUDA_SAFE_CALL( cudaMemset( data->_visibility_mask, 0, data->_W * data->_H * sizeof(bool) ));

    for ( int i = 0 ; i < nbChannels ; ++i ) {

        cuflt *tmp = NULL;
        CUDA_SAFE_CALL( cudaMalloc( &tmp, data->_nfbytes_hi ));
        CUDA_SAFE_CALL( cudaMemset( tmp, 0, data->_nfbytes_hi ));
        data->_U.push_back( tmp );
    }

    // temp vectors
    while ( data->_temp.size() < 2*nbChannels ) {

        cuflt *tmp = NULL;
        CUDA_SAFE_CALL( cudaMalloc( &tmp, data->_nfbytes_hi ));
        CUDA_SAFE_CALL( cudaMemset( tmp, 0, data->_nfbytes_hi ));
        data->_temp.push_back( tmp );
    }

    return data;
}

// Free up data for unstructured SR algorithm
void free_data( Data *data ) {

    assert( data != NULL );

    for ( size_t nview = 0 ; nview < data->_nviews; ++nview ) {

        ViewData *view = data->_views[nview];

        CUDA_SAFE_CALL( cudaFree( view->_image_f ));

        CUDA_SAFE_CALL( cudaFree( view->_warp_tau_x ));
        CUDA_SAFE_CALL( cudaFree( view->_warp_tau_y ));

        CUDA_SAFE_CALL( cudaFree( view->dpart_x ));
        CUDA_SAFE_CALL( cudaFree( view->dpart_y ));

        CUDA_SAFE_CALL( cudaFree( view->_weights_omega_i ));

        CUDA_SAFE_CALL( cudaFree( view->_B ));

        CUDA_SAFE_CALL( cudaFree( view->_cells ));

        delete view;
    }

    CUDA_SAFE_CALL( cudaFree( data->_norm_mask ));

    CUDA_SAFE_CALL( cudaFree( data->_u_grad_x ));
    CUDA_SAFE_CALL( cudaFree( data->_u_grad_y ));

    CUDA_SAFE_CALL( cudaFree( data->_target_mask ));

    CUDA_SAFE_CALL( cudaFree( data->_visibility_mask ));

    for ( int i = 0 ; i < nbChannels ; ++i ) {

        CUDA_SAFE_CALL( cudaFree( data->_U[i] ));
    }

    for ( int i = 0 ; i < 2*nbChannels ; ++i ) {

        CUDA_SAFE_CALL( cudaFree( data->_temp[i] ));
    }

    // finalize
    delete data;
}

// Compute the sparse matrix B
void compute_sparse_matrix( Data* data ) {

    // check for required data
    assert( data != NULL );
    size_t w = data->_w;
    size_t h = data->_h;
    assert( w*h > 0 );
    size_t W = data->_W;
    size_t H = data->_H;
    assert( W*H > 0 );

    dim3 dimBlock = data->_dimBlock; // low res
    dim3 dimGrid = data->_dimGrid;

    // Box filtering
    for ( size_t i=0; i<data->_nviews; i++ ) {

        ViewData *view = data->_views[i];

        set_B_bilinear<<< dimGrid, dimBlock >>>( W, H,
                                                 w, h,
                                                 data->_ks,
                                                 view->_warp_tau_x,
                                                 view->_warp_tau_y,
                                                 view->_B );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    }
}

// Init forward warp for a view : uses warps (make sure they are computed)
// warp=0: tau, warp=1:beta
// Currently completely on host, TODO: try to parallelize (hard)
void init_forward_warp_structure( Data* data, size_t nview ) {

    // check for required data
    assert( data != NULL );
    size_t w = data->_w;
    size_t h = data->_h;
    assert( w*h > 0 );
    size_t W = data->_W;
    size_t H = data->_H;
    assert( W*H > 0 );
    assert( nview < data->_nviews );
    ViewData *view = data->_views[nview];

    // Need warps from GPU
    cuflt *tmp_warp_x = new cuflt[w*h];
    cuflt *tmp_warp_y = new cuflt[w*h];

    CUDA_SAFE_CALL( cudaMemcpy( tmp_warp_x, view->_warp_tau_x, sizeof(cuflt) * w*h, cudaMemcpyDeviceToHost ));
    CUDA_SAFE_CALL( cudaMemcpy( tmp_warp_y, view->_warp_tau_y, sizeof(cuflt) * w*h, cudaMemcpyDeviceToHost ));
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
    //    TRACE( "grouping cells ..." );

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
    int margin = 1 + data->_ks/2;

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
    //    TRACE( "done." << endl );
    assert( ngrouped == w*h );

    // Copy new cell grouping to GPU

    if ( view->_cells != NULL ) {
        CUDA_SAFE_CALL( cudaFree( view->_cells ));
    }
    CUDA_SAFE_CALL( cudaMalloc( &view->_cells, sizeof(int) * cells.size() ));
    CUDA_SAFE_CALL( cudaMemcpy( view->_cells, &cells[0], sizeof(int) * cells.size(), cudaMemcpyHostToDevice ));

    // Clean up
    delete[] tmp_warp_x;
    delete[] tmp_warp_y;
    delete[] grouped;
    delete[] count;
    delete[] c_in;
}

// Setup a single view
void create_view( Data* data, size_t nview, coco::gsl_image *I) {

    // check for required data
    assert( data != NULL );
    size_t W = data->_W;
    size_t H = data->_H;

    assert( W*H > 0 );
    assert( nview < data->_nviews );
    ViewData *view = data->_views[nview];

    // view image should be equal to downsampled size
    assert( I->_w == W / data->_dsf );
    assert( I->_h == H / data->_dsf );

    // Image
    size_t N = data->_w * data->_h;
    float *buffer_f = new cuflt[ N*nbChannels ];

    for ( size_t n = 0 ; n < nbChannels ; n++ ) {
        // load view to device
        coco::gsl_matrix *channel = coco::gsl_image_get_channel( I, (coco::gsl_image_channel)n );

        for ( size_t i=0; i<N; i++ ) {
            buffer_f[N*n+i] = (cuflt)channel->data[i];
        }
    }

    CUDA_SAFE_CALL( cudaMemcpy( view->_image_f, buffer_f, nbChannels*N*sizeof(cuflt), cudaMemcpyHostToDevice ));
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    delete[] buffer_f;
}

// Update weight_omega_k
void compute_weights( Data* data ) {

    // check for required data
    assert( data != NULL );
    size_t W = data->_W; // high res
    size_t H = data->_H;
    assert( W*H > 0 );
    size_t w = data->_w; // low res
    size_t h = data->_h;
    assert( w*h > 0 );
    dim3 DimBlock = data->_dimBlock; // high res
    dim3 DimGrid = data->_dimGrid;
    dim3 dimBlock = data->_dimBlock; // low res
    dim3 dimGrid = data->_dimGrid;

    cuflt *tmp_deform = data->_temp[0];
    cuflt *tmp_gradient_x = data->_temp[1];
    cuflt *tmp_gradient_y = data->_temp[2];

    assert( nbChannels == 3 ); // rgb is required
    compute_gradient<<< DimGrid, DimBlock >>>
                                            ( W, H, data->_U[0], data->_U[1], data->_U[2], data->_u_grad_x, data->_u_grad_y );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    // update each view
    for ( size_t nview = 0 ; nview < data->_nviews ; ++nview ) {

        ViewData *view = data->_views[nview];

        CUDA_SAFE_CALL( cudaMemset( view->_weights_omega_i, 0, data->_nfbytes_lo ));

        // compute angular weights with u gradient
        // dot product of grad u with partial tau partial z
        convolution_nonsep_param<<< dimGrid, dimBlock >>>
                                                        ( W, H,
                                                          w, h,
                                                          data->_ks,
                                                          view->_B,
                                                          view->_warp_tau_x,
                                                          view->_warp_tau_y,
                                                          data->_u_grad_x,
                tmp_gradient_x );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );
        convolution_nonsep_param<<< dimGrid, dimBlock >>>
                                                        ( W, H,
                                                          w, h,
                                                          data->_ks,
                                                          view->_B,
                                                          view->_warp_tau_x,
                                                          view->_warp_tau_y,
                                                          data->_u_grad_y,
                tmp_gradient_y );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        angular_weights<<< dimGrid, dimBlock >>>
                                               ( w, h,
                                                 tmp_gradient_x, // u domain, high res
                                                 tmp_gradient_y,
                                                 data->_sigma_sensor,
                                                 view->dpart_x, // vi domain, low res
                                                 view->dpart_y, // dpart replaces aux_dmap_sigma*dtau/dz
                                                 data->_ugrad_threshold,
                                                 view->_weights_omega_i ); // in low res
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        switch( data->_dw_type ){

        case 0:
            // compute Wanner's deformation weights, same for all channels
            gold_deform_weights <<< dimGrid, dimBlock >>> ( W, H,
                                                            w, h,
                                                            view->_warp_tau_x, // vi domain, low res, values high res
                                                            view->_warp_tau_y,
                                                            tmp_deform );
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            break;

        case 1:
            assert(false);
            break;
        case 2:
            assert(false);
            break;
        }

        // multiply with deformation weights
        cuda_multiply_with_device<<< data->_dimGrid, data->_dimBlock >>> ( w, h, view->_weights_omega_i, tmp_deform );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    }
}

// Read the tau warps and deformation weights: from gsl_image to device cuflt*
void read_tau( Data* data, coco::gsl_image** tau_warps ) {

    // check for required data
    assert( data != NULL );
    size_t w = data->_w;
    size_t h = data->_h;
    size_t N = w*h;
    assert( N > 0 );

    cuflt *buffer_f = new cuflt[N];
    coco::gsl_matrix *channel;

    for ( size_t nview = 0 ; nview < data->_views.size() ; nview++ ) {

        ViewData *view = data->_views[nview];

        channel = coco::gsl_image_get_channel( tau_warps[nview], coco::GSL_IMAGE_RED ); // load tau x
        coco::gsl_matrix_copy_to_buffer( channel,  buffer_f );
        CUDA_SAFE_CALL( cudaMemcpy( view->_warp_tau_x, buffer_f, data->_nfbytes_lo, cudaMemcpyHostToDevice ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        channel = coco::gsl_image_get_channel( tau_warps[nview], coco::GSL_IMAGE_GREEN ); // load tau y
        coco::gsl_matrix_copy_to_buffer( channel,  buffer_f );
        CUDA_SAFE_CALL( cudaMemcpy( view->_warp_tau_y, buffer_f, data->_nfbytes_lo, cudaMemcpyHostToDevice ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );
    }

    delete [] buffer_f;
}

// Read the partial tau: from gsl_image to device cuflt*
void read_partial_tau( Data* data, coco::gsl_image** partial_tau ) {

    // check for required data
    assert( data != NULL );
    size_t w = data->_w;
    size_t h = data->_h;
    size_t N = w*h;
    assert( N > 0 );

    cuflt *buffer_f = new cuflt[N];
    coco::gsl_matrix *channel;

    for ( size_t nview = 0 ; nview < data->_views.size() ; nview++ ) {

        ViewData *view = data->_views[nview];
        cuflt *sigma_z = data->_temp[0];

        channel = coco::gsl_image_get_channel( partial_tau[nview], coco::GSL_IMAGE_RED ); // load sigma_z
        coco::gsl_matrix_copy_to_buffer( channel,  buffer_f );
        CUDA_SAFE_CALL( cudaMemcpy( sigma_z, buffer_f, data->_nfbytes_lo, cudaMemcpyHostToDevice ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        channel = coco::gsl_image_get_channel( partial_tau[nview], coco::GSL_IMAGE_GREEN ); // load dtau/dy x
        coco::gsl_matrix_copy_to_buffer( channel,  buffer_f );
        CUDA_SAFE_CALL( cudaMemcpy( view->dpart_x, buffer_f, data->_nfbytes_lo, cudaMemcpyHostToDevice ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        // do the product sigma_z*dtau/dz
        cuda_multiply_with_device<<< data->_dimGrid, data->_dimBlock >>> ( w, h, view->dpart_x, sigma_z );

        channel = coco::gsl_image_get_channel( partial_tau[nview], coco::GSL_IMAGE_BLUE ); // load dtau/dy y
        coco::gsl_matrix_copy_to_buffer( channel,  buffer_f );
        CUDA_SAFE_CALL( cudaMemcpy( view->dpart_y, buffer_f, data->_nfbytes_lo, cudaMemcpyHostToDevice ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        // do the product sigma_z*dtau/dz
        cuda_multiply_with_device<<< data->_dimGrid, data->_dimBlock >>> ( w, h, view->dpart_y, sigma_z );
    }
    delete [] buffer_f;
}

// Compute the initial image, starting point of the algorithm
void compute_initial_image( Data* data ) {

    // check for required data
    assert( data != NULL );
    size_t W = data->_W;
    size_t H = data->_H;
    assert( W*H );

    // Clear target image
    for ( size_t i = 0 ; i < nbChannels ; i++ ) {
        CUDA_SAFE_CALL( cudaMemset( data->_U[i], 0, data->_nfbytes_hi ));
    }

    // read starting image for the algorithm (to test only)
    TRACE("Read starting image for the algorithm" << std::endl);
    coco::gsl_image *initialization = coco::gsl_image_load_pfm( data->_init_name );

    if ( initialization != NULL ) {

        TRACE("Found image " << data->_init_name << endl);

        std::vector<coco::gsl_matrix*> init_vector;
        init_vector.push_back( initialization->_r );
        if ( nbChannels == 3 ) {
            init_vector.push_back( initialization->_g );
            init_vector.push_back( initialization->_b );
        }

        for ( size_t i = 0 ; i < nbChannels ; i++ ) {

            coco::gsl_matrix *u = init_vector[i]; // source
            assert( u->size2 == data->_W );
            assert( u->size1 == data->_H );
            cuda_memcpy( data->_U[i], u );
            CUDA_SAFE_CALL( cudaThreadSynchronize() );
        }

        coco::gsl_image_free( initialization );

    } else {

        // Clear the normalization mask
        CUDA_SAFE_CALL( cudaMemset( data->_norm_mask, 0, data->_nfbytes_hi ));

        TRACE("Starting image doesn't exist yet, computing it..." << endl);

        // Perform splatting for every input view
        for ( size_t nview = 0 ; nview < data->_views.size() ; nview++ ) {

            ViewData *view = data->_views[nview];

            // Forward warp, non-overlap regions sequentially
            int seg_start = 0;
            for ( size_t j = 0 ; j < view->_seg_end.size() ; j++ ) {

                int seg_end = view->_seg_end[j];
                int seg_size = seg_end - seg_start;

                // forward warp call for this segment, cannot overlap
                int seg_width = coco::cuda_default_block_size_x() * coco::cuda_default_block_size_y();
                dim3 dimBlock_splatting = dim3( seg_width, 1 );
                dim3 dimGrid_splatting = dim3( seg_size / seg_width + 1, 1 );

                weighted_deconvolution_nonsep_param<<< dimGrid_splatting, dimBlock_splatting >>>( W, H, seg_width,
                                                                                                  view->_image_f + 0*data->_w*data->_h,
                                                                                                  view->_image_f + 1*data->_w*data->_h,
                                                                                                  view->_image_f + 2*data->_w*data->_h,
                                                                                                  view->_cells,
                                                                                                  seg_start, seg_end,
                                                                                                  data->_ks,
                                                                                                  data->_dsf,
                                                                                                  view->_B,
                                                                                                  view->_weights_omega_i,
                                                                                                  view->_warp_tau_x, view->_warp_tau_y,
                                                                                                  data->_U[0], data->_U[1], data->_U[2],
                        data->_norm_mask );

                CUDA_SAFE_CALL( cudaThreadSynchronize() );

                seg_start = seg_end;
            }
        }

        // Normalize
        for ( size_t i = 0 ; i < nbChannels ; i++ ) {
            cuda_normalize_device<<< data->_DimGrid, data->_DimBlock >>>
                                                                   ( W, H, data->_U[i], data->_norm_mask );
        }

        TRACE("Write starting image for the algorithm" << std::endl);
        coco::write_pfm_image_signed( W, H, data->_U[0], data->_U[1], data->_U[2], data->_outdir + "/u_init.pfm", 0 );

        // Perform TV on init image to fill holes
        hole_filling( data );
    }
}

// Write current solution in pfm format
void write_pfm_solution( Data* data ) {

    size_t W = data->_W;
    size_t H = data->_H;
    assert( W*H > 0 );

    coco::write_pfm_image_signed( W, H, data->_U[0], data->_U[1], data->_U[2], data->_outdir + "/output.pfm", 0 );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}

// initialize the gradient of the target view by splatting
void init_u_gradient( Data* data ) {

    // check for required data
    assert( data != NULL );
    size_t W = data->_W;
    size_t H = data->_H;
    assert( W*H > 0 );
    size_t w = data->_w; // low res
    size_t h = data->_h;
    assert( w*h > 0 );
    dim3 DimBlock = data->_DimBlock; // high res
    dim3 DimGrid = data->_DimGrid;
    dim3 dimBlock = data->_dimBlock; // low res
    dim3 dimGrid = data->_dimGrid;

    // clear the target images
    CUDA_SAFE_CALL( cudaMemset( data->_u_grad_x, 0, data->_nfbytes_hi ));
    CUDA_SAFE_CALL( cudaMemset( data->_u_grad_y, 0, data->_nfbytes_hi ));
    CUDA_SAFE_CALL( cudaMemset( data->_target_mask, 0, data->_nfbytes_hi ));

    // Clear the normalization mask
    CUDA_SAFE_CALL( cudaMemset( data->_norm_mask, 0, data->_nfbytes_hi ));

    TRACE("Gradient splatting..." << endl);

    // Perform splatting for every input view
    for ( size_t nview = 0 ; nview < data->_views.size() ; nview++ ) {

        ViewData *view = data->_views[nview];

        // Compute the gradient of the input view
        compute_gradient <<< dimGrid, dimBlock >>>
                                                 ( w, h,
                                                   view->_image_f + 0*w*h,
                                                   view->_image_f + 1*w*h,
                                                   view->_image_f + 2*w*h,
                                                   data->_temp[0], data->_temp[1] );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        // Forward warp, non-overlap regions sequentially
        int seg_start = 0;
        for ( size_t j = 0 ; j < view->_seg_end.size() ; j++ ) {

            int seg_end = view->_seg_end[j];
            int seg_size = seg_end - seg_start;

            // forward warp call for this segment, cannot overlap
            int seg_width = coco::cuda_default_block_size_x() * coco::cuda_default_block_size_y();
            dim3 dimBlock_splatting = dim3( seg_width, 1 );
            dim3 dimGrid_splatting = dim3( seg_size / seg_width + 1, 1 );

            deconvolution_nonsep_param<<< dimGrid_splatting, dimBlock_splatting >>>( W, H, seg_width,
                                                                                     data->_temp[0], data->_temp[1], view->_weights_omega_i,
                    view->_cells,
                    seg_start, seg_end,
                    data->_ks,
                    data->_dsf,
                    view->_B,
                    view->_warp_tau_x, view->_warp_tau_y,
                    data->_u_grad_x, data->_u_grad_y, data->_target_mask,
                    data->_norm_mask );

            CUDA_SAFE_CALL( cudaThreadSynchronize() );

            seg_start = seg_end;
        }
    }

    // Normalization
    cuda_normalize_device<<< DimGrid, DimBlock >>>
                                                 ( W, H, data->_u_grad_x, data->_norm_mask );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    cuda_normalize_device<<< DimGrid, DimBlock >>>
                                                 ( W, H, data->_u_grad_y, data->_norm_mask );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    cuda_normalize_device<<< DimGrid, DimBlock >>>
                                                 ( W, H, data->_target_mask, data->_norm_mask );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    // uniformly normalize the weights
    cuda_scale_device<<< DimGrid, DimBlock >>>( W, H, data->_target_mask, data->_sigma_sensor*data->_sigma_sensor );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    TRACE("...done!" << std::endl);

    TRACE("Write gradient of target view" << std::endl);
    coco::write_pfm_image_signed( W, H, data->_u_grad_x, data->_u_grad_y, data->_target_mask, data->_outdir + "/u_grad.pfm", 0 );
}

// Perform Poisson integration with Jacobi method
void poisson_jacobi( Data* data ) {

    // check for required data
    assert( data != NULL );
    size_t W = data->_W;
    size_t H = data->_H;
    assert( W*H > 0 );
    dim3 DimBlock = data->_DimBlock; // high res
    dim3 DimGrid = data->_DimGrid;

    cuflt energy = 0.0;
    cuflt previous_energy = 1.0;

    cuflt jac_lambda = 1.0;

    vector< cuflt *> z;
    for ( size_t i = 0 ; i < nbChannels; i++ ) {

        z.push_back( data->_temp[i] );
        CUDA_SAFE_CALL( cudaMemset( z[i], 0, data->_nfbytes_hi ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        // compte divergence of estimated gradient
        compute_divergence<<< DimGrid, DimBlock >>>( W, H,
                                                     data->_u_grad_x, data->_u_grad_y,
                                                     z[i] );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        // compute z from Jacobi method x(i+1) = B*x(i) + z
        jacobi_z<<< DimGrid, DimBlock >>>( W, H,
                                           data->_U[i], // roughly estimated solution
                                           jac_lambda,
                                           data->_target_mask, // weights for the data term
                                           z[i] );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    }

    TRACE("Perform Poisson integration of gradient  [");
    int iterations = 200;
    for ( int k = 0 ; k < iterations ; ++k ) {

        if ( (k%(iterations/10)) == 0 ) {
            TRACE( "." );
        }

        //previous_energy = energy;

        // Poisson integration
        for ( size_t i = 0 ; i < nbChannels; i++ ) {

            CUDA_SAFE_CALL( cudaMemcpy( data->_temp[3], data->_U[i], data->_nfbytes_hi, cudaMemcpyDeviceToDevice ));
            CUDA_SAFE_CALL( cudaThreadSynchronize() );

            // compute B from Jacobi method x(i+1) = B*x(i) + z
            jacobi_B<<< DimGrid, DimBlock >>>( W, H,
                                               data->_temp[3],
                                               z[i],
                                               jac_lambda,
                                               data->_target_mask, // weights for the data term
                                               data->_U[i] );
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            //            cuflt *E = new cuflt[ W * H ];
            //            CUDA_SAFE_CALL( cudaMemcpy( E, laplacian, data->_nfbytes_hi, cudaMemcpyDeviceToHost ));
            //            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            //            energy = 0.0;
            //            for ( size_t p = 0 ; p < W*H ; ++p ) {
            //                energy += E[p];
            //            }

            //            delete[] E;
        }

        //        write_pfm_image_signed( W, H, data->_U[0], data->_U[1], data->_U[2], data->_outdir + "/u_poisson_%03lu.pfm", k );

        //        TRACE("Energy: " << energy << endl);
        //        if ( abs(previous_energy - energy) < epsilon ) {

        //            TRACE("] Energy minimum reached at iteration " << k << endl);
        //            break;
        //        }
    }
    if ( abs(previous_energy - energy) >= epsilon ) {

        TRACE( "] maximum number of iterations reached" << endl );
    }

    TRACE("Write final image" << std::endl);
    coco::write_pfm_image_signed( W, H, data->_U[0], data->_U[1], data->_U[2], data->_outdir + "/u_integ.pfm", 0 );
}

