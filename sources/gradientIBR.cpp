#include "gradientIBR.h"
#include "config.h"
#include "import.h"
#include "gradientIBR.cuh"
#include "solver.cuh"

#include <cocolib/cocolib/common/debug.h>

using namespace std;

// Main entry function: compute compute superresolved novel view from unstructured lf
void IBR_gradient( Config_data *config_data ) {

    TRACE( "Computing novel view from unstructured lightfield" << endl );

    vector<coco::gsl_image*> lightfield = import_lightfield( config_data );

    // Structure for multilabel integration
    size_t w = config_data->_w; // low res
    size_t h = config_data->_h;
    size_t nviews = config_data->_nviews; // inline lf
    size_t dsf = config_data->_dsf;
    size_t W = w * dsf; // high res
    size_t H = h * dsf;

    TRACE( "  lightfield resolution " << w << " x " << h << endl );
    TRACE( "  target     resolution " << W << " x " << H << endl );

    // Zero initialization for testing
    coco::gsl_image *R = coco::gsl_image_alloc( W, H );
    coco::gsl_matrix_set_all( R->_r, 0.0 );
    coco::gsl_matrix_set_all( R->_g, 0.0 );
    coco::gsl_matrix_set_all( R->_b, 0.0 );
    std::vector<coco::gsl_matrix*> ZERO;
    ZERO.push_back( R->_r );
    if ( config_data->_nchannels == 3 ) {
        ZERO.push_back( R->_g );
        ZERO.push_back( R->_b );
    }

    TRACE("First image: " << config_data->_s_min << endl);
    TRACE("Last image: " << config_data->_s_max << endl);

    TRACE( "configuring views ... " << nviews << endl );
    TRACE( "  downscaling factor is " << dsf << endl );

    Data* data = init_data( config_data );

    for ( size_t nview = 0 ; nview < nviews ; ++nview ) {

        create_view( data, nview, lightfield[nview] );
    }

    TRACE( "Read input warps and weights." << endl );
    coco::gsl_image** input_data = new coco::gsl_image* [nviews];

    if ( !config_data->_depth_name.empty() &&
         !config_data->_ply_name.empty() &&
         !config_data->_cam_name.empty() ) {

        import_depth_from_ply( config_data, lightfield );

    }
    if ( !config_data->_depth_name.empty() && !config_data->_cam_name.empty() ) {

        import_warps_from_depth( config_data, input_data );

    } else {

        TRACE("Import warps from files" << endl);
    }

    import_warps( config_data, input_data, config_data->_tau_name );
    read_tau( data, input_data );

    import_warps( config_data, input_data, config_data->_dpart_name );
    read_partial_tau( data, input_data );

    for (size_t nview = 0 ; nview < nviews ; ++nview) {
        coco::gsl_image_free(input_data[nview]);
        coco::gsl_image_free( lightfield[nview] );
    }

    delete[] input_data;

    TRACE("Create sparse matrix structure" << std::endl);
    compute_sparse_matrix( data );

    TRACE("Create cells for splatting..." << std::endl);
    // create cells for splatting
    for (size_t nview = 0 ; nview < nviews ; ++nview) {
        init_forward_warp_structure( data, nview );
    }

    TRACE( "done." << endl );

    TRACE("Compute backward visibility..." << endl);
    setup_target_visibility( data );
    TRACE("...done!" << endl);

    // compute weights with current image (null image)
    compute_weights( data );

    // compute averaged view as start point for algorithm
    TRACE( "  computing averaged view init ..." << endl );
    compute_initial_image( data );
    TRACE( "...done!" << endl );

    // blur the initial image to test the kernels
    //coco_vtv_sr_downsample( data );

    TRACE( "FINISHED INIT" << endl );

    TRACE( "Computing VTV Superresolution for light field novel view " << endl );

    // compare with prev_meta each meta iteration
    coco::gsl_image *R_prev_meta = coco::gsl_image_alloc( W, H );
    get_solution( data, ZERO );
    coco::gsl_image_copy_to(R, R_prev_meta, 0,0);

    // compare with prev_meta each iteration
    coco::gsl_image *R_prev = coco::gsl_image_alloc( W, H );
    get_solution( data, ZERO );
    coco::gsl_image_copy_to(R, R_prev, 0,0);
    coco::gsl_image_save( (config_data->_outdir + "/u_init_filled.png").c_str(), R );

    // initialize the target gradient by splatting
    init_u_gradient( data );

    TRACE("Integrate the gradient to reconstruct the target view" << std::endl);
    // poisson_jacobi( data );
    possion_conj_grad( data );

    // Write solution to output image
    get_solution( data, ZERO );
    if ( config_data->_nchannels == 1 ) {
        assert(false);
        coco::gsl_matrix_copy_to( R->_r, R->_g );
        coco::gsl_matrix_copy_to( R->_r, R->_b );
    }

    coco::gsl_image_save( (config_data->_outdir + "/" + config_data->_outfile).c_str(), R );
    write_pfm_solution( data );

    // Cleanup
    free_data( data );

    coco::gsl_image_free( R_prev );
    coco::gsl_image_free( R_prev_meta );
}


