#include <time.h>

#include "super-resolution.h"
#include "config.h"
#include "import.h"
#include "optimization.cuh"

#include <cocolib/cocolib/vtv/vtv.h>
#include <cocolib/cocolib/common/debug.h>
#include <cocolib/cocolib/common/gsl_image.h>

using namespace std;
using namespace coco; // goldluecke's files

// Compare current and previous solutions to evaluate the number of moving pixels
bool sr_compare_images( const gsl_image *A, const gsl_image *B, double MAX_DIFF_THRESHOLD, int *nS, int *nM ) {

    assert(A->_h == B->_h && A->_w == B->_w);

    *nS = 0;
    *nM = 0;
    float maxdiff = 0.;

    bool equal = true;
    for (size_t j=0; j<A->_h; ++j) {
        for (size_t i=0; i<A->_w; ++i) {
            double ra,ga,ba,rb,gb,bb;
            gsl_image_get_color( A, i, j, ra, ga, ba );
            gsl_image_get_color( B, i, j, rb, gb, bb );

            double diffr = fabs(ra-rb);
            double diffg = fabs(ga-gb);
            double diffb = fabs(ba-bb);

            double maxval = max(diffr, max(diffg, diffb));

            if (maxval < MAX_DIFF_THRESHOLD) {
                ++(*nS);
            } else {
                equal = false;
                ++(*nM);
            }

            if (maxval > maxdiff) {
                maxdiff = maxval;
            }
        }
    }

    return equal;
}

// Main entry function: compute compute superresolved novel view from unstructured lf
void sr_synthesize_view( Config_data *config_data, int frame ) {

    TRACE( "Computing novel view from unstructured lightfield" << endl );

    vector<gsl_image*> lightfield = import_lightfield( config_data );

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
    gsl_image *R = gsl_image_alloc( W, H );
    gsl_matrix_set_all( R->_r, 0.0 );
    gsl_matrix_set_all( R->_g, 0.0 );
    gsl_matrix_set_all( R->_b, 0.0 );
    vector<gsl_matrix*> ZERO;
    ZERO.push_back( R->_r );
    if ( config_data->_nchannels == 3 ) {
        ZERO.push_back( R->_g );
        ZERO.push_back( R->_b );
    }

    // Create solver workspaces
    coco_vtv_data* mtv = coco_vtv_alloc( ZERO );
    mtv->_basedir = config_data->_outdir;
    mtv->_lambda = config_data->_lambda;
    mtv->_lambda_max_factor = config_data->_lambda_max_factor;
    assert( config_data->_niter > 0 );
    mtv->_inner_iterations = config_data->_niter;
    TRACE( mtv->_inner_iterations << " inner iterations." << endl );
    TRACE( config_data->_nmeta_iter << " meta-iterations." << endl );
    mtv->_regularizer = 1;
    mtv->_data_term_p = 2;

    // Previous nasty bug:
    // Functional needs to be initialized here, since the following
    // already uses U, Uq
    coco_vtv_initialize( mtv, ZERO );

    TRACE("First image: " << config_data->_s_min << endl);
    TRACE("Last image: " << config_data->_s_max << endl);

    TRACE( "configuring views ... " << nviews << endl );
    TRACE( "  downscaling factor is " << dsf << endl );

    coco_vtv_sr_init_unstructured( mtv, config_data );

    for ( size_t nview = 0 ; nview < nviews ; ++nview ) {

        coco_vtv_sr_create_view_unstructured( mtv, nview, lightfield[nview] );
    }

    TRACE( "Read input warps and weights." << endl );
    gsl_image** input_data = new gsl_image* [nviews];

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
    coco_vtv_sr_read_tau( mtv, input_data );

    import_warps( config_data, input_data, config_data->_dpart_name );
    coco_vtv_sr_read_partial_tau( mtv, input_data );

    for (size_t nview = 0 ; nview < nviews ; ++nview) {
        gsl_image_free(input_data[nview]);
        gsl_image_free( lightfield[nview] );
    }

    delete[] input_data;

    TRACE("Create sparse matrix structure" << std::endl);
    coco_vtv_sr_compute_sparse_matrix( mtv );

    TRACE("Create cells for splatting..." << std::endl);
    // create cells for splatting
    for (size_t nview = 0 ; nview < nviews ; ++nview) {
        vtv_sr_init_forward_warp_structure_unstructured( mtv, nview );
    }

    TRACE( "done." << endl );

    // compute weights with current image (null image)
    coco_vtv_sr_compute_weights_unstructured( mtv );

    TRACE("Compute backward visibility..." << endl);
    coco_vtv_setup_visibility_mask( mtv );
    TRACE("...done!" << endl);

    // compute averaged view as start point for algorithm
     TRACE( "  computing averaged view init ..." << endl );
     coco_vtv_sr_compute_initial_image( mtv, config_data ); // TODO boolean to test if initialization
     TRACE( "...done!" << endl );

    // coco_vtv_sr_test_warps( mtv );

    // blur the initial image to test the kernels
    //coco_vtv_sr_downsample( mtv );

    TRACE( "FINISHED INIT" << endl );

    TRACE( "Computing VTV Superresolution for light field novel view " << endl );

    const cuflt EPSILON_TOL = 1e-4;
    const cuflt MAX_DIFF_THRESHOLD = 1e-3;
    const cuflt PERCENT_MOVING = 0.01;

    cuflt energy = 0;
    cuflt previous_energy = energy;
    cuflt previous_energy_reweight = energy;

    // compare with prev_meta each meta iteration
    gsl_image *R_prev_meta = gsl_image_alloc( W, H );
    coco_vtv_get_solution( mtv, ZERO );
    gsl_image_copy_to(R, R_prev_meta, 0,0);

    // compare with prev_meta each iteration
    gsl_image *R_prev = gsl_image_alloc( W, H );
    coco_vtv_get_solution( mtv, ZERO );
    gsl_image_copy_to(R, R_prev, 0,0);
    // gsl_image_save( (config_data->_outdir + "/u_init_filled.png").c_str(), R ); // TODO boolean to test if initialization

    double total_t = 0;

    // Start iteration of the reweighted least square
    int r;
    for ( r=0; r<config_data->_nmeta_iter; r++) {

        // recompute weights with current solution
        coco_vtv_sr_compute_weights_unstructured( mtv );

        // update regularizer with new weights
        coco_vtv_sr_init_regularizer_weight_unstructured( mtv );

        // weights may be different, energy may change
        energy = coco_vtv_sr_primal_energy_unstructured( mtv );

        previous_energy = energy;

        TRACE( "pass " << r  << ", no displacement" << endl );
        {
            clock_t t0 = clock();

            TRACE( "  [" );
            int iterations = config_data->_niter;

            int k;
            for ( k=0; k<iterations; k++ ) {

                if ( iterations >= 10 ) {
                    if ( (k%(iterations/10)) == 0 ) {
                        TRACE( "." );
                    }
                }

                coco_vtv_sr_iteration_fista_unstructured( mtv );

                // Check convergence using the energy
                energy = coco_vtv_sr_primal_energy_unstructured( mtv, previous_energy );

                if (energy < EPSILON_TOL) {
                    TRACE( "]" << endl << " -> Fista convergence: very small energy (close to zero) at iteration " << k << endl );
                    break;
                } else {
                    double rel_error_fista = fabs( (energy - previous_energy) / energy);
                    if ( rel_error_fista < EPSILON_TOL) {
                        TRACE( "]" << endl << " -> Fista convergence: relative error small at iteration " << k << endl );
                        break;
                    } else {
                        previous_energy = energy;
                    }
                }

                int nS = 0, nM = 0;

                // compare solution to previous iteration one
                coco_vtv_get_solution( mtv, ZERO );
                sr_compare_images( R, R_prev, MAX_DIFF_THRESHOLD, &nS, &nM );

                TRACE1( "It " << r << " : " << nM << " Moving pixels : " << nS << " Static pixels" << endl );

                if ((float)nM / (float)(nM+nS) < PERCENT_MOVING) {
                    TRACE( "] done. " << endl << "  -> Convergence: Less than " << PERCENT_MOVING*100 << "% of pixels moving : Convergence Reached at iteration " << k << endl);
                    break; // finish reweighted for
                } else {
                    gsl_image_copy_to(R, R_prev, 0,0);
                }
            }
            if ( k == iterations ) {
                TRACE( "] done. " << endl << "  -> Max iterations reached" << endl );
            }

            clock_t t1 = clock();
            double secs = double(t1 - t0) / double(CLOCKS_PER_SEC);
            TRACE( "Fista " << r << " runtime : " << secs << "s." << endl );
            TRACE( "per iteration : " << secs / double(k+1) << "s." << endl );
            TRACE( "iter / s      : " << double(k+1) / secs  << endl );

            total_t += secs;
        } // end of block FISTA Iterations

        if (energy < EPSILON_TOL) {
            if ( previous_energy_reweight < EPSILON_TOL ) {
                TRACE( " -> Reweighted Convergence : Very small energy (close to zero) at meta-iteration " << r << endl );
                break; // finish reweighted for
            } else {
                previous_energy_reweight = energy;
                continue;
            }
        }

        int nS = 0, nM = 0;

        // compare solution to previous iteration one
        coco_vtv_get_solution( mtv, ZERO );
        sr_compare_images( R, R_prev_meta, MAX_DIFF_THRESHOLD, &nS, &nM );

        TRACE( "It " << r << " : " << nM << " Moving pixels : " << nS << " Static pixels" << endl << endl );

        if ((float)nM / (float)(nM+nS) < PERCENT_MOVING) {
            TRACE( " -> Reweighted Convergence: Less than " << PERCENT_MOVING*100 << "% of pixels moving : Convegence Reached" << endl << endl );
            break; // finish reweighted for
        } else {
            gsl_image_copy_to(R, R_prev_meta, 0,0);
        }

    } // end for reweighted least squares

    if ( r == config_data->_nmeta_iter ) {
        TRACE( "Finished reweighted iterations. " << endl << "  -> Max iterations reached" << endl << endl );
    }

    TRACE("Total duration: " << total_t << " seconds." << endl);

//    TRACE("Perform Push/Pull hole filling..." << endl);
//    coco_vtv_push_pull( mtv );
//    TRACE("... done!" << endl);

    // Write solution to output image
    coco_vtv_get_solution( mtv, ZERO );
    if ( config_data->_nchannels == 1 ) {
        assert(false);
        gsl_matrix_copy_to( R->_r, R->_g );
        gsl_matrix_copy_to( R->_r, R->_b );
    }

    std::string outName = config_data->_outdir + "/" + config_data->_outfile;
    if( frame != -1) {
        char outNameChar[500];
        sprintf( outNameChar, outName.c_str(), frame );
        gsl_image_save( outNameChar, R );
    } else {
        gsl_image_save( outName.c_str(), R );
    }

    coco_vtv_sr_write_pfm_solution( mtv );

    // Cleanup
    coco_vtv_sr_free_unstructured( mtv );

    coco_vtv_free( mtv );

    gsl_image_free( R_prev );
    gsl_image_free( R_prev_meta );
}


