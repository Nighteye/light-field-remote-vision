/* -*-c++-*- */
// colorIBR functions -- IBR with gradient constraints, as opposed to gradient IBR

#include <vtv/vtv.h>
#include <vtv/vtv.cuh>

#include <common/gsl_matrix_helper.h>
#include <common/gsl_matrix_convolutions.h>

#include <cuda/cuda_helper.h>
#include <cuda/cuda_reduce.h>
#include <cuda/cuda_kernels.cuh>
#include <cuda/cuda_convolutions.h>
#include <defs.h>
#include <common/linalg3d.h>

#include <cuda/cuda_inline_device_functions.cu>
#include <vtv/vtv_sr_kernels.cu>
#include <vtv/vtv_sr_kernels_unstructured.cu>

#include "optimization.cuh"
#include "config.h"

//static const cuflt epsilon = 1e-3;

using namespace std;

/*****************************************************************************
       TV_x Superresolution
*****************************************************************************/

// Compute backward visibility (gamma domain)
void coco::coco_vtv_setup_visibility_mask( coco_vtv_data *data ) {

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
    dim3 DimBlock = ws->_dimBlock; // high res
    dim3 DimGrid = ws->_dimGrid;
//    dim3 dimBlock = sr->_dimBlock; // low res
//    dim3 dimGrid = sr->_dimGrid;

    // Clear the normalization mask
    CUDA_SAFE_CALL( cudaMemset( sr->_visibility_mask, 0, W*H*sizeof(bool) ));
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    // Create the mask
//    for ( size_t nview = 0 ; nview < sr->_views.size() ; nview++ ) {

//        coco_vtv_sr_view_data_unstructured *view = sr->_views[nview];

//        // Forward warp, non-overlap regions sequentially
//        int seg_start = 0;
//        for ( size_t j = 0 ; j < view->_seg_end.size() ; j++ ) {

//            int seg_end = view->_seg_end[j];
//            int seg_size = seg_end - seg_start;

//            // forward warp call for this segment, cannot overlap
//            int seg_width = cuda_default_block_size_x() * cuda_default_block_size_y();
//            dim3 dimBlock_splatting = dim3( seg_width, 1 );
//            dim3 dimGrid_splatting = dim3( seg_size / seg_width + 1, 1 );

//            cuda_setup_visibility_mask_device<<< dimGrid_splatting, dimBlock_splatting >>>( W, H, seg_width,
//                                                                                            view->_cells,
//                                                                                            seg_start, seg_end,
//                                                                                            sr->_ks,
//                                                                                            sr->_dsf,
//                                                                                            view->_warp_tau_x, view->_warp_tau_y,
//                                                                                            sr->_visibility_mask );
//            CUDA_SAFE_CALL( cudaThreadSynchronize() );

//            seg_start = seg_end;
//        }
//    }

    cuda_set_all_device<<< DimGrid, DimBlock >>>( W, H, ws->_temp[0], 1.0 );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    CUDA_SAFE_CALL( cudaMemset( ws->_temp[1], 0, ws->_nfbytes ));
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    // Sum contributions for all views
    for ( size_t nview=0; nview < sr->_views.size(); nview++ ) {

        coco_vtv_sr_view_data_unstructured *view = sr->_views[nview];

        // deconvolution step (applying the weights)
        // Forward warp, non-overlap regions sequentially
        int seg_start = 0;
        for ( size_t j = 0 ; j < view->_seg_end.size() ; j++ ) {

            int seg_end = view->_seg_end[j];
            int seg_size = seg_end - seg_start;

            // forward warp call for this segment, cannot overlap
            int seg_width = cuda_default_block_size_x() * cuda_default_block_size_y();
            dim3 dimBlock_splatting = dim3( seg_width, 1 );
            dim3 dimGrid_splatting = dim3( seg_size / seg_width + 1, 1 );

            cuda_deconvolution_nonsep_device_param<<< dimGrid_splatting, dimBlock_splatting >>>( W, H, seg_width,
                                                                                                 ws->_temp[0],
                    view->_cells,
                    seg_start, seg_end,
                    sr->_ks,
                    sr->_dsf,
                    view->_A,
                    view->_warp_tau_x, view->_warp_tau_y,
                    ws->_temp[1],
                    sr->_norm_mask );

            CUDA_SAFE_CALL( cudaThreadSynchronize() );

            seg_start = seg_end;
        }
        CUDA_SAFE_CALL( cudaThreadSynchronize() );
    }

//    write_pfm_image_signed( W, H, ws->_temp[1], data->_basedir + "/_target.pfm", 0 );

    cuda_setup_visibility_mask_device<<< DimGrid, DimBlock >>>
                                                 ( W, H, ws->_temp[1], sr->_visibility_mask );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    //write_pfm_image_signed( W, H, ws->_temp[2], data->_basedir + "/_norm_target.pfm", 0 );

//    write_test_image_bool( W, H, sr->_visibility_mask, data->_basedir + "/_visibility_mask.png", 0 );
}


// Perform TV on init image to fill holes
//void coco::coco_vtv_hole_filling( coco_vtv_data *data ) {

//    // check for required data
//    assert( data != NULL );
//    coco_vtv_sr_data_unstructured *sr = data->_sr_data_unstructured;
//    assert( sr != NULL );
//    size_t W = data->_W; // high res
//    size_t H = data->_H;
//    assert( W*H > 0 );
//    coco_vtv_workspace *ws = data->_workspace;
//    assert(ws->_nfbytes == sr->_nfbytes_hi);
//    dim3 DimGrid = ws->_dimGrid;
//    dim3 DimBlock = ws->_dimBlock;

//    cuflt *laplacian = ws->_temp[0];
//    cuflt energy = 0.0;
//    cuflt previous_energy = 0.0;

//    TRACE("Perform Poisson diffusion on init image to fill holes   [");
//    int iterations = 500;
//    for ( int k = 0 ; k < iterations ; ++k ) {

//        if ( (k%(iterations/10)) == 0 ) {
//            TRACE( "." );
//        }

//        previous_energy = energy;

//        // Poisson diffusion
//        for ( size_t i = 0 ; i < data->_nchannels; i++ ) {

//            // compute the laplacian nabla(u) in the holes, 0 elsewhere
//            cuda_laplacian_device<<< DimGrid, DimBlock >>>( W, H,
//                                                            ws->_U[i],
//                                                            laplacian,
//                                                            sr->_visibility_mask );
//            CUDA_SAFE_CALL( cudaDeviceSynchronize() );
//            // u(t+i) = u(t) + nabla(u)
//            cuda_add_to_device<<< DimGrid, DimBlock >>>( W, H, laplacian, ws->_U[i] );
//            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

//            //compute energy
//            cuda_multiply_with_device<<< DimGrid, DimBlock >>>( W, H, laplacian, ws->_U[i] );

//            cuflt *E = new cuflt[ W * H ];
//            CUDA_SAFE_CALL( cudaMemcpy( E, laplacian, sr->_nfbytes_hi, cudaMemcpyDeviceToHost ));
//            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

//            energy = 0.0;
//            for ( size_t p = 0 ; p < W*H ; ++p ) {
//                energy += E[p];
//            }

//            delete[] E;
//        }
//        //TRACE("Energy: " << energy << endl);
//        if ( abs(previous_energy - energy) < epsilon ) {

//            TRACE("] Energy minimum reached at iteration " << k << endl);
//            break;
//        }
//    }
//    if ( abs(previous_energy - energy) >= epsilon ) {

//        TRACE( "] maximum number of iterations reached" << endl );
//    }

//    TRACE("Write filled starting image" << std::endl);
//    write_pfm_image_signed( W, H, ws->_U[0], ws->_U[1], ws->_U[2], data->_basedir + "/u_init_filled.pfm", 0 );
//}

void reduce(int W, int H, int w, int h, const float* const input, float* const output) {

    const float a = 0.4f;
    const uint kernelSize = 5;

    float kernel[kernelSize] = {0.25f - 0.5f*a, 0.25f, a, 0.25f, 0.25f - 0.5f*a};

    float *tempArray = new float[W*h];

    for(int i = 0 ; i < h ; ++i) {
        for(int j = 0 ; j < W ; ++j) {

            float weight = 0.0;

            tempArray[i*W+j] = 0;

            if(2*i-2 < 0) {
                if(input[j] != 0.0) {
                    tempArray[i*W+j] += kernel[0]*input[j];
                    weight += kernel[0];
                }
            } else {
                if(input[(2*i-2)*W+j] != 0.0) {
                    tempArray[i*W+j] += kernel[0]*input[(2*i-2)*W+j];
                    weight += kernel[0];
                }
            }

            if(2*i-1 < 0) {
                if(input[j] != 0.0) {
                    tempArray[i*W+j] += kernel[1]*input[j];
                    weight += kernel[1];
                }
            } else {
                if(input[(2*i-1)*W+j] != 0.0) {
                    tempArray[i*W+j] += kernel[1]*input[(2*i-1)*W+j];
                    weight += kernel[1];
                }
            }

            if(input[(2*i)*W+j] != 0.0) {
                tempArray[i*W+j] += kernel[2]*input[(2*i)*W+j];
                weight += kernel[2];
            }

            if(H-1 < 2*i+1) {
                if(input[(H-1)*W+j] != 0.0) {
                    tempArray[i*W+j] += kernel[3]*input[(H-1)*W+j];
                    weight += kernel[3];
                }
            } else {
                if(input[(2*i+1)*W+j] != 0.0) {
                    tempArray[i*W+j] += kernel[3]*input[(2*i+1)*W+j];
                    weight += kernel[3];
                }
            }

            if(H-1 < 2*i+2) {
                if(input[(H-1)*W+j] != 0.0) {
                    tempArray[i*W+j] += kernel[4]*input[(H-1)*W+j];
                    weight += kernel[4];
                }
            } else {
                if(input[(2*i+2)*W+j] != 0.0) {
                    tempArray[i*W+j] += kernel[4]*input[(2*i+2)*W+j];
                    weight += kernel[4];
                }
            }

            if(weight == 0) {
                tempArray[i*W+j] = 0.0;
            } else {
                tempArray[i*W+j] /= weight;
            }
        }
    }

    for(int i = 0 ; i < h ; ++i) {
        for(int j = 0 ; j < w ; ++j) {

            float weight = 0.0;

            output[i*w+j] = 0;

            if(2*j-2 < 0) {
                if(tempArray[i*W] != 0.0) {
                    output[i*w+j] += kernel[0]*tempArray[i*W];
                    weight += kernel[0];
                }
            } else {
                if(tempArray[i*W+(2*j-2)] != 0.0) {
                    output[i*w+j] += kernel[0]*tempArray[i*W+(2*j-2)];
                    weight += kernel[0];
                }
            }

            if(2*j-1 < 0) {
                if(tempArray[i*W] != 0.0) {
                    output[i*w+j] += kernel[1]*tempArray[i*W];
                    weight += kernel[1];
                }
            } else {
                if(tempArray[i*W+(2*j-1)] != 0.0) {
                    output[i*w+j] += kernel[1]*tempArray[i*W+(2*j-1)];
                    weight += kernel[1];
                }
            }

            if(tempArray[i*W+(2*j)] != 0.0) {
                output[i*w+j] += kernel[2]*tempArray[i*W+(2*j)];
                weight += kernel[2];
            }

            if(W-1 < 2*j+1) {
                if(tempArray[i*W+(W-1)] != 0.0) {
                    output[i*w+j] += kernel[3]*tempArray[i*W+(W-1)];
                    weight += kernel[3];
                }
            } else {
                if(tempArray[i*W+(2*j+1)] != 0.0) {
                    output[i*w+j] += kernel[3]*tempArray[i*W+(2*j+1)];
                    weight += kernel[3];
                }
            }

            if(W-1 < 2*j+2) {
                if(tempArray[i*W+(W-1)] != 0.0) {
                    output[i*w+j] += kernel[4]*tempArray[i*W+(W-1)];
                    weight += kernel[4];
                }
            } else {
                if(tempArray[i*W+(2*j+2)] != 0.0) {
                    output[i*w+j] += kernel[4]*tempArray[i*W+(2*j+2)];
                    weight += kernel[4];
                }
            }

            if(weight == 0) {
                output[i*w+j] = 0.0;
            } else {
                output[i*w+j] /= weight;
            }
        }
    }

    delete[] tempArray;
}
void reduce(int W, int H, int w, int h, const bool* const input, bool* const output) {

    bool *tempArray = new bool[W*h];

    for(int i = 0 ; i < h ; ++i) {
        for(int j = 0 ; j < W ; ++j) {

            tempArray[i*W+j] = false;

            if(2*i-2 < 0) {
                tempArray[i*W+j] = tempArray[i*W+j] || input[j];
            } else {
                tempArray[i*W+j] = tempArray[i*W+j] || input[(2*i-2)*W+j];
            }

            if(2*i-1 < 0) {
                tempArray[i*W+j] = tempArray[i*W+j] || input[j];
            } else {
                tempArray[i*W+j] = tempArray[i*W+j] || input[(2*i-1)*W+j];
            }

            if(input[(2*i)*W+j] != 0.0) {
                tempArray[i*W+j] = tempArray[i*W+j] || input[(2*i)*W+j];
            }

            if(H-1 < 2*i+1) {
                tempArray[i*W+j] = tempArray[i*W+j] || input[(H-1)*W+j];
            } else {
                tempArray[i*W+j] = tempArray[i*W+j] || input[(2*i+1)*W+j];
            }

            if(H-1 < 2*i+2) {
                tempArray[i*W+j] = tempArray[i*W+j] || input[(H-1)*W+j];
            } else {
                tempArray[i*W+j] = tempArray[i*W+j] || input[(2*i+2)*W+j];
            }
        }
    }

    for(int i = 0 ; i < h ; ++i) {
        for(int j = 0 ; j < w ; ++j) {

            output[i*w+j] = false;

            if(2*j-2 < 0) {
                output[i*w+j] = output[i*w+j] || tempArray[i*W];
            } else {
                output[i*w+j] = output[i*w+j] || tempArray[i*W+(2*j-2)];
            }

            if(2*j-1 < 0) {
                output[i*w+j] = output[i*w+j] || tempArray[i*W];
            } else {
                output[i*w+j] = output[i*w+j] || tempArray[i*W+(2*j-1)];
            }

            if(tempArray[i*W+(2*j)] != 0.0) {
                output[i*w+j] = output[i*w+j] || tempArray[i*W+(2*j)];
            }

            if(W-1 < 2*j+1) {
                output[i*w+j] = output[i*w+j] || tempArray[i*W+(W-1)];
            } else {
                output[i*w+j] = output[i*w+j] || tempArray[i*W+(2*j+1)];
            }

            if(W-1 < 2*j+2) {
                output[i*w+j] = output[i*w+j] || tempArray[i*W+(W-1)];
            } else {
                output[i*w+j] = output[i*w+j] || tempArray[i*W+(2*j+2)];
            }
        }
    }

    delete[] tempArray;
}

void expand(int W, int H, int w, int h, const float* const input, float* const output, const bool* const visibility) {

    const float a = 0.4;
    const uint kernelSize = 5;

    float kernel[kernelSize] = {0.25f - 0.5f*a, 0.25f, a, 0.25f, 0.25f - 0.5f*a};

    float *tempArray = new float[w*H];

    for(int i = 0 ; i < H ; ++i) {
        for(int j = 0 ; j < w ; ++j) {

            tempArray[i*w+j] = 0;

            if(i%2 == 0) {

                if((i-2)/2 < 0) {
                    tempArray[i*w+j] += 2*kernel[0]*input[j];
                } else {
                    tempArray[i*w+j] += 2*kernel[0]*input[(i-2)/2*w+j];
                }
                tempArray[i*w+j] += 2*kernel[2]*input[i/2*w+j];
                if(h-1 < (i+2)/2) {
                    tempArray[i*w+j] += 2*kernel[4]*input[(h-1)*w+j];
                } else {
                    tempArray[i*w+j] += 2*kernel[4]*input[(i+2)/2*w+j];
                }

            } else {

                if((i-1)/2 < 0) {
                    tempArray[i*w+j] += 2*kernel[1]*input[j];
                } else {
                    tempArray[i*w+j] += 2*kernel[1]*input[(i-1)/2*w+j];
                }
                if(h-1 < (i+1)/2) {
                    tempArray[i*w+j] += 2*kernel[3]*input[(h-1)*w+j];
                } else {
                    tempArray[i*w+j] += 2*kernel[3]*input[(i+1)/2*w+j];
                }
            }
        }
    }

    for(int i = 0 ; i < H ; ++i) {
        for(int j = 0 ; j < W ; ++j) {

            if(!visibility[i*W+j]) {

                output[i*W+j] = 0;

                if(j%2 == 0) {

                    if((j-2)/2 < 0) {
                        output[i*W+j] += 2*kernel[0]*tempArray[i*w];
                    } else {
                        output[i*W+j] += 2*kernel[0]*tempArray[i*w+(j-2)/2];
                    }
                    output[i*W+j] += 2*kernel[2]*tempArray[i*w+j/2];
                    if(w-1 < (j+2)/2) {
                        output[i*W+j] += 2*kernel[4]*tempArray[i*w+(w-1)];
                    } else {
                        output[i*W+j] += 2*kernel[4]*tempArray[i*w+(j+2)/2];
                    }

                } else {

                    if((j-1)/2 < 0) {
                        output[i*W+j] += 2*kernel[1]*tempArray[i*w];
                    } else {
                        output[i*W+j] += 2*kernel[1]*tempArray[i*w+(j-1)/2];
                    }
                    if(w-1 < (j+1)/2) {
                        output[i*W+j] += 2*kernel[3]*tempArray[i*w+(w-1)];
                    } else {
                        output[i*W+j] += 2*kernel[3]*tempArray[i*w+(j+1)/2];
                    }
                }
            }
        }
    }

    delete[] tempArray;
}

void erode(int W, int H, bool* const image) {

    for(int i = 0 ; i < H ; ++i) {
        for(int j = 0 ; j < W ; ++j) {

            if(!image[i*W+j]) {

                if(0 < i) {
                    image[(i-1)*W+j] = false;
                }
                if(0 < j) {
                    image[i*W+(j-1)] = false;
                }
                if(i < H-1) {
                    image[(i+1)*W+j] = false;
                }
                if(j < W-1) {
                    image[i*W+(j+1)] = false;
                }
            }
        }
    }
}

// Perform push pull Laplacian hole-filling
void coco::coco_vtv_push_pull( coco_vtv_data *data ) {

    // check for required data
    assert( data != NULL );
    coco_vtv_sr_data_unstructured *sr = data->_sr_data_unstructured;
    assert( sr != NULL );
    size_t W = data->_W; // high res
    size_t H = data->_H;
    assert( W*H > 0 );
    coco_vtv_workspace *ws = data->_workspace;
    assert(ws->_nfbytes == sr->_nfbytes_hi);
    dim3 DimGrid = ws->_dimGrid;
    dim3 DimBlock = ws->_dimBlock;

    // Compute pyramid size
    unsigned int a(W), b(H), r(W%H);
    while(r != 0) {
        a = b;
        b = r;
        r = a%b;
    }
    unsigned int pyramidHeight = 0;
    unsigned int res = b;
    while(res%2 == 0) {
        res = res/2;
        ++pyramidHeight;
    }
    // HACK
    // TODO: fix the height computation

    pyramidHeight = 10;

    TRACE("Pyramid height: " << pyramidHeight << std::endl);

    for ( size_t i = 0 ; i < data->_nchannels; i++ ) {

        std::vector< float* > imagePyramid(pyramidHeight+1);
        std::vector< bool* > visibilityPyramid(pyramidHeight+1);
        for(unsigned int s = 0 ; s < imagePyramid.size() ; ++s) {
            imagePyramid[s] = 0;
        }
        for(unsigned int s = 0 ; s < visibilityPyramid.size() ; ++s) {
            visibilityPyramid[s] = 0;
        }

        unsigned int scale = 0;
        imagePyramid[scale] = new float[ W * H ];
        visibilityPyramid[scale] = new bool[ W * H ];
        CUDA_SAFE_CALL( cudaMemcpy( imagePyramid[scale], ws->_U[i], sr->_nfbytes_hi, cudaMemcpyDeviceToHost ));
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );
        CUDA_SAFE_CALL( cudaMemcpy( visibilityPyramid[scale], sr->_visibility_mask, W*H*sizeof(bool), cudaMemcpyDeviceToHost ));
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        for(unsigned int s = 1 ; s <= (unsigned int)pyramidHeight ; ++s) {

            unsigned int Wscale = (unsigned int)W / (uint)pow(2.0, (double)(s-1));
            unsigned int Hscale = (unsigned int)H / (uint)pow(2.0, (double)(s-1));
            unsigned int wscale = (unsigned int)W / (uint)pow(2.0, (double)s);
            unsigned int hscale = (unsigned int)H / (uint)pow(2.0, (double)s);

            imagePyramid[s] = new float[ wscale * hscale ];
            memset(imagePyramid[s], 0, wscale*hscale*sizeof(float));
            visibilityPyramid[s] = new bool[ wscale * hscale ];
            memset(visibilityPyramid[s], false, wscale*hscale*sizeof(bool));

            reduce(Wscale, Hscale, wscale, hscale, imagePyramid[s-1], imagePyramid[s]);
            reduce(Wscale, Hscale, wscale, hscale, visibilityPyramid[s-1], visibilityPyramid[s]);

//            CUDA_SAFE_CALL( cudaMemcpy( ws->_temp[0], imagePyramid[s], wscale*hscale*sizeof(float), cudaMemcpyHostToDevice ));
//            write_pfm_image_signed( wscale, hscale, ws->_temp[0], data->_basedir + "/u_scale_%02i.pfm", s );
        }

        for(unsigned int s = (unsigned int)pyramidHeight ; 0 < s ; --s) {

            unsigned int Wscale = (unsigned int)W / (uint)pow(2.0, (double)(s-1));
            unsigned int Hscale = (unsigned int)H / (uint)pow(2.0, (double)(s-1));
            unsigned int wscale = (unsigned int)W / (uint)pow(2.0, (double)s);
            unsigned int hscale = (unsigned int)H / (uint)pow(2.0, (double)s);

            expand(Wscale, Hscale, wscale, hscale, imagePyramid[s], imagePyramid[s-1], visibilityPyramid[s-1]);

//            CUDA_SAFE_CALL( cudaMemcpy( ws->_temp[0], imagePyramid[s-1], Wscale*Hscale*sizeof(float), cudaMemcpyHostToDevice ));
//            write_pfm_image_signed( Wscale, Hscale, ws->_temp[0], data->_basedir + "/u_expand_%02i.pfm", s-1 );
        }

        CUDA_SAFE_CALL( cudaMemcpy( ws->_temp[i], ws->_U[i], sr->_nfbytes_hi, cudaMemcpyDeviceToDevice ));
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        CUDA_SAFE_CALL( cudaMemcpy( ws->_U[i], imagePyramid[0], sr->_nfbytes_hi, cudaMemcpyHostToDevice ));
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        for(unsigned int s = 0 ; s < imagePyramid.size() ; ++s) {
            delete[] imagePyramid[s];
            imagePyramid[s] = 0;
        }
        for(unsigned int s = 0 ; s < visibilityPyramid.size() ; ++s) {
            delete[] visibilityPyramid[s];
            visibilityPyramid[s] = 0;
        }

        cuda_subtract_from_device<<< DimGrid, DimBlock >>>
                                                         ( W, H, ws->_temp[i], ws->_U[i] );
        CUDA_SAFE_CALL( cudaThreadSynchronize() );
    }

    TRACE("Write filled starting image" << std::endl);
    write_pfm_image_signed( W, H, ws->_U[0], ws->_U[1], ws->_U[2], data->_basedir + "/u_init_filled.pfm", 0 );
}

// Setup unstructured SR algorithm: init view and resolution data
bool coco::coco_vtv_sr_init_unstructured( coco_vtv_data *data, Config_data *config_data ) {

    // can only be initialized once.
    assert( data->_sr_data_unstructured == NULL );
    coco_vtv_sr_data_unstructured *sr = new coco_vtv_sr_data_unstructured;
    size_t W = data->_W; // high res
    size_t H = data->_H;
    sr->_nviews = config_data->_nviews;
    sr->_dsf = config_data->_dsf;
    sr->_w = config_data->_w;
    sr->_h = config_data->_h;
    // validate downscale factor (exact multiple of size)
    assert( sr->_w * sr->_dsf == W );
    assert( sr->_h * sr->_dsf == H );

    // default for 8-bit normalized
    sr->_sigma_sensor = config_data->_sigma_sensor;
    sr->_ugrad_threshold = config_data->_ugrad_threshold;
    sr->_ks = sr->_dsf + 1;
    sr->_dw_type = config_data->_dw_type;
    assert( sr->_dw_type == 0 ||
            sr->_dw_type == 1 ||
            sr->_dw_type == 2 );
    sr->_gw_type = config_data->_gw_type;
    assert( sr->_gw_type == 0 ||
            sr->_gw_type == 1 ||
            sr->_gw_type == 2 );

    sr->_dt_alpha = config_data->_dt_alpha;
    sr->_dt_beta = config_data->_dt_beta;

    sr->_gradient_step = config_data->_gradient_step;

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
    size_t bytes_view_total = sr->_nviews * bytes_per_view;

    TRACE( "Allocating mem:" << std::endl );
    TRACE( "  " << bytes_per_view / MB << " Mb per view, " << bytes_view_total/MB << " total." << std::endl );

    for ( size_t nview = 0 ; nview < sr->_nviews ; ++nview ) {

        coco_vtv_sr_view_data_unstructured *view = new coco_vtv_sr_view_data_unstructured;

        CUDA_SAFE_CALL( cudaMalloc( &view->_image_f, data->_nchannels * sr->_nfbytes_lo ));
        CUDA_SAFE_CALL( cudaMemset( view->_image_f, 0, data->_nchannels * sr->_nfbytes_lo ));

        CUDA_SAFE_CALL( cudaMalloc( &view->_warp_tau_x, sr->_nfbytes_lo ));
        CUDA_SAFE_CALL( cudaMemset( view->_warp_tau_x, 0.0, sr->_nfbytes_lo ));
        CUDA_SAFE_CALL( cudaMalloc( &view->_warp_tau_y, sr->_nfbytes_lo ));
        CUDA_SAFE_CALL( cudaMemset( view->_warp_tau_y, 0.0, sr->_nfbytes_lo ));

        CUDA_SAFE_CALL( cudaMalloc( &view->dpart_x, sr->_nfbytes_lo ));
        CUDA_SAFE_CALL( cudaMemset( view->dpart_x, 0.0, sr->_nfbytes_lo ));
        CUDA_SAFE_CALL( cudaMalloc( &view->dpart_y, sr->_nfbytes_lo ));
        CUDA_SAFE_CALL( cudaMemset( view->dpart_y, 0.0, sr->_nfbytes_lo ));

        CUDA_SAFE_CALL( cudaMalloc( &view->_weights_omega_i, sr->_nfbytes_lo ));
        CUDA_SAFE_CALL( cudaMemset( view->_weights_omega_i, 0.0, sr->_nfbytes_lo ));

        CUDA_SAFE_CALL( cudaMalloc( &view->_A, sr->_ks*sr->_ks * sr->_nfbytes_lo ));
        CUDA_SAFE_CALL( cudaMemset( view->_A, 0.0, sr->_ks*sr->_ks * sr->_nfbytes_lo ));

        view->_cells = NULL;

        sr->_views.push_back( view );
    }

    // Additional work mem (TODO: reduce, use temp buffers ws->F[...])
    size_t srbytes = sr->_nfbytes_hi // _norm_mask
            + W*H*sizeof(bool) // _visibility_mask
            + sr->_nfbytes_hi * (data->_nchannels+2); // temp buffers

    TRACE( "  " << srbytes/MB << " Mb for additional work structures." << std::endl );

    for ( size_t i = 0 ; i < data->_nchannels ; ++i ) {

        cuflt *G_intensities = NULL;
        cuflt *G_gradients = NULL;
        CUDA_SAFE_CALL( cudaMalloc( &G_intensities, sr->_nfbytes_hi ));
        CUDA_SAFE_CALL( cudaMemset( G_intensities, 0, sr->_nfbytes_hi ));
        CUDA_SAFE_CALL( cudaMalloc( &G_gradients, sr->_nfbytes_hi ));
        CUDA_SAFE_CALL( cudaMemset( G_gradients, 0, sr->_nfbytes_hi ));
        sr->_G_intensities.push_back( G_intensities );
        sr->_G_gradients.push_back( G_gradients );
    }

    CUDA_SAFE_CALL( cudaMalloc( &(sr->_norm_mask), sr->_nfbytes_hi ));
    CUDA_SAFE_CALL( cudaMemset( sr->_norm_mask, 0, sr->_nfbytes_hi ));

    // Target coverage
    CUDA_SAFE_CALL( cudaMalloc( &(sr->_norm_mask), sr->_nfbytes_hi ));
    CUDA_SAFE_CALL( cudaMemset( sr->_norm_mask, 0, sr->_nfbytes_hi ));

    CUDA_SAFE_CALL( cudaMalloc( &(sr->_visibility_mask), W*H*sizeof(bool) ));
    CUDA_SAFE_CALL( cudaMemset( sr->_visibility_mask, 0, W*H*sizeof(bool) ));

    // Check for grayscale and add temp buffers if necessary
    coco_vtv_workspace *ws = data->_workspace;
    assert( ws != NULL );
    while ( ws->_temp.size() < 2*data->_nchannels ) {
        cuflt *tmp = NULL;
        CUDA_SAFE_CALL( cudaMalloc( &tmp, sr->_nfbytes_hi ));
        CUDA_SAFE_CALL( cudaMemset( tmp, 0, sr->_nfbytes_hi ));
        ws->_temp.push_back( tmp );
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

    for ( size_t nview = 0 ; nview < sr->_nviews; ++nview ) {

        coco_vtv_sr_view_data_unstructured *view = sr->_views[nview];

        CUDA_SAFE_CALL( cudaFree( view->_image_f ));

        CUDA_SAFE_CALL( cudaFree( view->_warp_tau_x ));
        CUDA_SAFE_CALL( cudaFree( view->_warp_tau_y ));

        CUDA_SAFE_CALL( cudaFree( view->dpart_x ));
        CUDA_SAFE_CALL( cudaFree( view->dpart_y ));

        CUDA_SAFE_CALL( cudaFree( view->_weights_omega_i ));

        CUDA_SAFE_CALL( cudaFree( view->_A ));

        CUDA_SAFE_CALL( cudaFree( view->_cells ));

        delete view;
    }

    for ( size_t i = 0 ; i < sr->_G_intensities.size() ; ++i ) {
        CUDA_SAFE_CALL( cudaFree( sr->_G_intensities[i] ));
    }
    for ( size_t i = 0 ; i < sr->_G_gradients.size() ; ++i ) {
        CUDA_SAFE_CALL( cudaFree( sr->_G_gradients[i] ));
    }

    CUDA_SAFE_CALL( cudaFree( sr->_norm_mask ));

    CUDA_SAFE_CALL( cudaFree( sr->_visibility_mask ));

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
    assert( w*h > 0 );
    size_t W = data->_W;
    size_t H = data->_H;
    assert( W*H > 0 );

    dim3 dimBlock = sr->_dimBlock; // low res
    dim3 dimGrid = sr->_dimGrid;

    // Box filtering
    for ( size_t i=0; i<sr->_nviews; i++ ) {

        coco_vtv_sr_view_data_unstructured *view = sr->_views[i];

        //        cuda_set_A_box_filtering<<< dimGrid, dimBlock >>>( W, H,
        //                                                               w, h,
        //                                                               sr->_ks,
        //                                                               view->_warp_tau_x,
        //                                                               view->_warp_tau_y,
        //                                                               view->_A );
        cuda_set_A_bilinear<<< dimGrid, dimBlock >>>( W, H,
                                                      w, h,
                                                      sr->_ks,
                                                      view->_warp_tau_x,
                                                      view->_warp_tau_y,
                                                      view->_A );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        //        write_pfm_image_signed( w*sr->_ks, h*sr->_ks, view->_A, data->_basedir + "/A_%03lu.pfm", i );
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
    int margin = 1 + sr->_ks/2;

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

    cuflt *tmp_deform = ws->_temp[0];
    cuflt *tmp_gradient_x = ws->_temp[1];
    cuflt *tmp_gradient_y = ws->_temp[2];

    assert( data->_nchannels == 3 ); // rgb is required
    vtv_sr_compute_gradient_device <<< DimGrid, DimBlock >>>
                                                           ( W, H, ws->_U[0], ws->_U[1], ws->_U[2], ws->_X1[0], ws->_X2[0], sr->_visibility_mask );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    // update each view
    for ( size_t nview = 0 ; nview < sr->_nviews ; ++nview ) {

        coco_vtv_sr_view_data_unstructured *view = sr->_views[nview];

        CUDA_SAFE_CALL( cudaMemset( view->_weights_omega_i, 0, sr->_nfbytes_lo ));

        // compute angular weights with u gradient
        // dot product of grad u with partial tau partial z
        cuda_convolution_nonsep_device_param<<< dimGrid, dimBlock >>>
                                                                    ( W, H,
                                                                      w, h,
                                                                      sr->_ks,
                                                                      view->_A,
                                                                      view->_warp_tau_x,
                                                                      view->_warp_tau_y,
                                                                      ws->_X1[0],
                tmp_gradient_x );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );
        cuda_convolution_nonsep_device_param<<< dimGrid, dimBlock >>>
                                                                    ( W, H,
                                                                      w, h,
                                                                      sr->_ks,
                                                                      view->_A,
                                                                      view->_warp_tau_x,
                                                                      view->_warp_tau_y,
                                                                      ws->_X2[0],
                tmp_gradient_y );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        vtv_sr_angular_weights_device <<< dimGrid, dimBlock >>>
                                                              ( w, h,
                                                                tmp_gradient_x, // u domain, high res
                                                                tmp_gradient_y,
                                                                sr->_sigma_sensor,
                                                                view->dpart_x, // vi domain, low res
                                                                view->dpart_y, // dpart replaces aux_dmap_sigma*dtau/dz
                                                                sr->_ugrad_threshold,
                                                                view->_weights_omega_i ); // in low res
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        switch( sr->_dw_type ){

        case 0:
            // compute Wanner's deformation weights, same for all channels
            vtv_sr_gold_deform_weights_device <<< dimGrid, dimBlock >>> ( W, H,
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

        //                write_pfm_image_signed( w, h, view->_weights_omega_i, data->_basedir + "/weights_omega_%05i.pfm", nview );
        //                write_pfm_image_signed( w, h, tmp_deform, data->_basedir + "/tmp_deform%05i.pfm", nview );

        // multiply with deformation weights
        cuda_multiply_with_device<<< sr->_dimGrid, sr->_dimBlock >>> ( w, h, view->_weights_omega_i, tmp_deform );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        //        write_pfm_image_signed( w, h, view->_weights_omega_i, data->_basedir + "/final_weights_%05i.pfm", nview );
    }

    return true;
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
        CUDA_SAFE_CALL( cudaMemcpy( view->_warp_tau_x, buffer_f, sr->_nfbytes_lo, cudaMemcpyHostToDevice ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        channel = gsl_image_get_channel( tau_warps[nview], GSL_IMAGE_GREEN ); // load tau y
        gsl_matrix_copy_to_buffer( channel,  buffer_f );
        CUDA_SAFE_CALL( cudaMemcpy( view->_warp_tau_y, buffer_f, sr->_nfbytes_lo, cudaMemcpyHostToDevice ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        //TRACE("Test: write pfm tau warp, view " << nview << std::endl);
        //write_pfm_image_signed(w, h, view->_warp_tau_x, view->_warp_tau_y, view->_warp_tau_y, data->_basedir + "/tau_%02lu.pfm", nview);
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
        //write_pfm_image_signed(w, h, sigma_z, view->dpart_x, view->dpart_y, data->_basedir + "/partial_tau_%02lu.pfm", nview);
    }
    delete [] buffer_f;

    return true;
}

// Compute the initial image, starting point of the algorithm
void coco::coco_vtv_sr_compute_initial_image( coco_vtv_data *data, Config_data *config_data ) {

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
    gsl_image *initialization = gsl_image_load_pfm( config_data->_init_name );

    if ( initialization != NULL ) {

        TRACE("Found image " << config_data->_init_name << endl);

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

        // Clear the normalization mask
        CUDA_SAFE_CALL( cudaMemset( sr->_norm_mask, 0, ws->_nfbytes ));

        TRACE("Starting image doesn't exist yet, computing it..." << endl);

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
                dim3 dimBlock_splatting = dim3( seg_width, 1 );
                dim3 dimGrid_splatting = dim3( seg_size / seg_width + 1, 1 );

                cuda_weighted_deconvolution_nonsep_device_param<<< dimGrid_splatting, dimBlock_splatting >>>( W, H, seg_width,
                                                                                                              view->_image_f + 0*sr->_w*sr->_h,
                                                                                                              view->_image_f + 1*sr->_w*sr->_h,
                                                                                                              view->_image_f + 2*sr->_w*sr->_h,
                                                                                                              view->_cells,
                                                                                                              seg_start, seg_end,
                                                                                                              sr->_ks,
                                                                                                              sr->_dsf,
                                                                                                              view->_A,
                                                                                                              view->_weights_omega_i,
                                                                                                              view->_warp_tau_x, view->_warp_tau_y,
                                                                                                              ws->_U[0], ws->_U[1], ws->_U[2],
                        sr->_norm_mask );

                CUDA_SAFE_CALL( cudaThreadSynchronize() );

                seg_start = seg_end;
            }
        }

        // Normalize
        for ( size_t i = 0 ; i < data->_nchannels ; i++ ) {
            cuda_normalize_device<<< ws->_dimGrid, ws->_dimBlock >>>
                                                                   ( W, H, ws->_U[i], sr->_norm_mask );
        }

        TRACE("Write starting image for the algorithm" << std::endl);
        write_pfm_image_signed( W, H, ws->_U[0], ws->_U[1], ws->_U[2], data->_basedir + "/u_init.pfm", 0 );
        //        write_pfm_image_signed( W, H, sr->_norm_mask, data->_basedir + "/norm_mask.pfm", 0 );

        // Perform TV on init image to fill holes
        //coco_vtv_hole_filling( data );

        coco_vtv_push_pull( data );
    }
}

// warp each input view separately by splatting
bool coco::coco_vtv_sr_test_warps( coco_vtv_data *data ) {

    // check for required data
    assert( data != NULL );
    coco_vtv_sr_data_unstructured *sr = data->_sr_data_unstructured;
    assert( sr != NULL );
    coco_vtv_workspace *ws = data->_workspace;
    size_t W = data->_W;
    size_t H = data->_H;
    assert( W*H );

    // Perform splatting for every input view
    for ( size_t nview = 0 ; nview < sr->_views.size() ; nview++ ) {

        coco_vtv_sr_view_data_unstructured *view = sr->_views[nview];

        // Clear target image
        for ( size_t i = 0 ; i < data->_nchannels ; i++ ) {
            CUDA_SAFE_CALL( cudaMemset(  ws->_temp[i], 0, ws->_nfbytes ));
        }

        // Clear the normalization mask
        CUDA_SAFE_CALL( cudaMemset( sr->_norm_mask, 0, ws->_nfbytes ));

        // Forward warp, non-overlap regions sequentially
        int seg_start = 0;
        for ( size_t j = 0 ; j < view->_seg_end.size() ; j++ ) {

            int seg_end = view->_seg_end[j];
            int seg_size = seg_end - seg_start;

            // forward warp call for this segment, cannot overlap
            int seg_width = cuda_default_block_size_x() * cuda_default_block_size_y();
            dim3 dimBlock_splatting = dim3( seg_width, 1 );
            dim3 dimGrid_splatting = dim3( seg_size / seg_width + 1, 1 );

            cuda_deconvolution_nonsep_device_param<<< dimGrid_splatting, dimBlock_splatting >>>( W, H, seg_width,
                                                                                                 view->_image_f + 0*sr->_w*sr->_h,
                                                                                                 view->_image_f + 1*sr->_w*sr->_h,
                                                                                                 view->_image_f + 2*sr->_w*sr->_h,
                                                                                                 view->_cells,
                                                                                                 seg_start, seg_end,
                                                                                                 sr->_ks,
                                                                                                 sr->_dsf,
                                                                                                 0, // no weights for splatting
                                                                                                 view->_warp_tau_x, view->_warp_tau_y,
                                                                                                 ws->_temp[0], ws->_temp[1], ws->_temp[2],
                    sr->_norm_mask );

            CUDA_SAFE_CALL( cudaThreadSynchronize() );

            seg_start = seg_end;
        }

        // Normalize
        for ( size_t i = 0 ; i < data->_nchannels; i++ ) {

            cuda_normalize_device<<< ws->_dimGrid, ws->_dimBlock >>>
                                                                   ( W, H, ws->_temp[i], sr->_norm_mask );
        }

        TRACE("Write temp image " << nview << std::endl);
        write_pfm_image_signed( W, H, ws->_temp[0], ws->_temp[1], ws->_temp[2], data->_basedir + "/warped_v_%02lu.pfm", nview );
    }

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
    dim3 dimBlock = sr->_dimBlock; // low res
    dim3 dimGrid = sr->_dimGrid;

    for ( size_t nview = 0 ; nview < sr->_views.size() ; nview++ ) {

        coco_vtv_sr_view_data_unstructured *view = sr->_views[nview];

        for ( size_t i = 0 ; i < data->_nchannels; i++ ) {
            CUDA_SAFE_CALL( cudaMemset(  ws->_temp[i], 0, ws->_nfbytes ));
        }

        cuda_convolution_nonsep_device_param<<< dimGrid, dimBlock >>>
                                                                    ( W, H,
                                                                      w, h,
                                                                      sr->_ks,
                                                                      view->_A,
                                                                      view->_warp_tau_x,
                                                                      view->_warp_tau_y,
                                                                      ws->_U[0], ws->_U[1], ws->_U[2],
                ws->_temp[0], ws->_temp[1], ws->_temp[2] );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        TRACE("Write temp image v" << nview << std::endl);
        write_pfm_image_signed( w, h, ws->_temp[0], ws->_temp[1], ws->_temp[2], data->_basedir + "/image_lo_%02lu.pfm", nview );
    }

    return true;
}

// Compute primal energy
double coco::coco_vtv_sr_primal_energy_unstructured( coco_vtv_data *data, float previous_energy ) {

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

    //    if (data->_nchannels == 1) { // grayscale
    //        cuda_compute_gradient_device <<< DimGrid, DimBlock >>>
    //                                                             ( W, H, ws->_U[0], ws->_X1q[0], ws->_X2q[0] );
    //    } else { // rgb
    //        vtv_sr_compute_gradient_device <<< DimGrid, DimBlock >>>
    //                                                               ( W, H, ws->_U[0], ws->_U[1], ws->_U[2], ws->_X1q[0], ws->_X2q[0] );
    //    }
    //    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    assert( data->_nchannels == 3 );
    vtv_sr_compute_gradient_device<<< DimGrid, DimBlock >>>( W, H,
                                                             ws->_U[0],
            ws->_U[1],
            ws->_U[2],
            ws->_X1q[0],
            ws->_X2q[0], sr->_visibility_mask );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );



    // TV component
    cuflt *E_TV = new cuflt[ W*H ];
    cuflt e_tv = 0.0;
    // Compute tv energy (integral over gamma)

    CUDA_SAFE_CALL( cudaMemcpy( E_TV, ws->_X1q[0], sr->_nfbytes_hi, cudaMemcpyDeviceToHost ));
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    for ( size_t i=0; i<W*H; i++ ) {
        e_tv += abs(E_TV[i]);
    }

    CUDA_SAFE_CALL( cudaMemcpy( E_TV, ws->_X2q[0], sr->_nfbytes_hi, cudaMemcpyDeviceToHost ));
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    // Compute tv energy (integral over gamma)
    for ( size_t i=0; i<W*H; i++ ) {
        e_tv += abs(E_TV[i]);
    }

    delete[] E_TV;

    // Data term

    cuflt *E_DATA = new cuflt[ w * h ];

    cuflt* tmp_energy;
    CUDA_SAFE_CALL( cudaMalloc( &tmp_energy, sr->_nfbytes_lo ));
    CUDA_SAFE_CALL( cudaMemset( tmp_energy, 0, sr->_nfbytes_lo ));

    // --------------------------------------DATA INTENSITIES----------------------------------

    // Sum contributions for all views
    for ( size_t nview = 0 ; nview < sr->_views.size() ; nview++ ) {

        coco_vtv_sr_view_data_unstructured *view = sr->_views[nview];

        for ( size_t i = 0 ; i < data->_nchannels; ++i ) {
            CUDA_SAFE_CALL( cudaMemset( ws->_G[i], 0, sr->_nfbytes_hi ));
        }

        // filter the high res image to get v_i in low res
        cuda_convolution_nonsep_device_param<<< dimGrid, dimBlock >>>
                                                                    ( W, H,
                                                                      w, h,
                                                                      sr->_ks,
                                                                      view->_A,
                                                                      view->_warp_tau_x,
                                                                      view->_warp_tau_y,
                                                                      ws->_U[0], ws->_U[1], ws->_U[2],
                ws->_G[0], ws->_G[1], ws->_G[2] );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        for ( size_t i = 0 ; i < data->_nchannels; ++i ) {

            // subtract image v_i from input cuflt image v_i*
            cuda_subtract_from_device<<< dimGrid, dimBlock >>>
                                                             ( w, h, view->_image_f + i*w*h, ws->_G[i] );
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            // square the result
            cuda_multiply_with_device<<< dimGrid, dimBlock >>>
                                                             ( w, h, ws->_G[i], ws->_G[i] );
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            // apply the weights
            cuda_multiply_with_device<<< dimGrid, dimBlock >>> ( w, h, ws->_G[i], view->_weights_omega_i );
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            // normalize the weights
            cuda_scale_device<<< dimGrid, dimBlock >>>( w, h, ws->_G[i], sr->_sigma_sensor*sr->_sigma_sensor );
            CUDA_SAFE_CALL( cudaThreadSynchronize() );

            // cumulate energy
            cuda_add_to_device<<< dimGrid, dimBlock >>>
                                                      ( w, h, ws->_G[i], tmp_energy);
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );
        }
    }

    CUDA_SAFE_CALL( cudaMemcpy( E_DATA, tmp_energy, sr->_nfbytes_lo, cudaMemcpyDeviceToHost ));
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    // clear temp energy buffer
    CUDA_SAFE_CALL( cudaMemset( tmp_energy, 0, sr->_nfbytes_lo ));

    // Compute data energy, intensity term
    cuflt e_data_intensities = 0.0;
    for ( size_t i = 0 ; i < w * h ; i++ ) {
        e_data_intensities += E_DATA[i];
    }

    // --------------------------------------DATA GRADIENTS----------------------------------

    // Sum contributions for all views
    for ( size_t nview = 0 ; nview < sr->_views.size() ; nview++ ) {

        CUDA_SAFE_CALL( cudaMemset( ws->_temp[0], 0,  sr->_nfbytes_hi ));
        CUDA_SAFE_CALL( cudaMemset( ws->_temp[1], 0,  sr->_nfbytes_hi ));
        CUDA_SAFE_CALL( cudaMemset( ws->_temp[2], 0,  sr->_nfbytes_hi ));
        CUDA_SAFE_CALL( cudaMemset( ws->_temp[3], 0,  sr->_nfbytes_hi ));
        CUDA_SAFE_CALL( cudaMemset( ws->_temp[4], 0,  sr->_nfbytes_hi ));
        CUDA_SAFE_CALL( cudaMemset( ws->_temp[5], 0,  sr->_nfbytes_hi ));

        coco_vtv_sr_view_data_unstructured *view = sr->_views[nview];

        vtv_sr_compute_gradient_device <<< DimGrid, DimBlock >>>
                                                               ( W, H,
                                                                 ws->_U[0], ws->_U[1], ws->_U[2],
                ws->_temp[0], ws->_temp[1], 0 );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        vtv_sr_compute_gradient_device <<< dimGrid, dimBlock >>>
                                                               ( w, h,
                                                                 view->_image_f + 0*w*h,
                                                                 view->_image_f + 1*w*h,
                                                                 view->_image_f + 2*w*h,
                                                                 ws->_temp[4], ws->_temp[5], 0 );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        cuda_convolution_nonsep_device_param<<< dimGrid, dimBlock >>>
                                                                    ( W, H,
                                                                      w, h,
                                                                      sr->_ks,
                                                                      view->_A,
                                                                      view->_warp_tau_x,
                                                                      view->_warp_tau_y,
                                                                      ws->_temp[0],
                ws->_temp[2] );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        cuda_convolution_nonsep_device_param<<< dimGrid, dimBlock >>>
                                                                    ( W, H,
                                                                      w, h,
                                                                      sr->_ks,
                                                                      view->_A,
                                                                      view->_warp_tau_x,
                                                                      view->_warp_tau_y,
                                                                      ws->_temp[1],
                ws->_temp[3] );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        cuda_subtract_from_device<<< dimGrid, dimBlock >>>
                                                         ( w, h, ws->_temp[4], ws->_temp[2] );
        CUDA_SAFE_CALL( cudaThreadSynchronize() );
        cuda_subtract_from_device<<< dimGrid, dimBlock >>>
                                                         ( w, h, ws->_temp[5], ws->_temp[3] );
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        cuda_square_device<<< dimGrid, dimBlock >>> ( w, h, ws->_temp[2] );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );
        cuda_square_device<<< dimGrid, dimBlock >>> ( w, h, ws->_temp[3] );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        if ( sr->_gw_type != 0 ) {

            // apply the weights
            cuda_multiply_with_device<<< dimGrid, dimBlock >>> ( w, h, ws->_temp[2], view->_weights_omega_i );
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );
            cuda_multiply_with_device<<< dimGrid, dimBlock >>> ( w, h, ws->_temp[3], view->_weights_omega_i );
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            // normalize the weights
            cuda_scale_device<<< dimGrid, dimBlock >>>( w, h, ws->_temp[2], sr->_sigma_sensor*sr->_sigma_sensor );
            CUDA_SAFE_CALL( cudaThreadSynchronize() );
            cuda_scale_device<<< dimGrid, dimBlock >>>( w, h, ws->_temp[3], sr->_sigma_sensor*sr->_sigma_sensor );
            CUDA_SAFE_CALL( cudaThreadSynchronize() );
        }

        // cumulate energy
        cuda_add_to_device<<< dimGrid, dimBlock >>>
                                                  ( w, h, ws->_temp[2], tmp_energy );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );
        cuda_add_to_device<<< dimGrid, dimBlock >>>
                                                  ( w, h, ws->_temp[3], tmp_energy );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    }

    CUDA_SAFE_CALL( cudaMemcpy( E_DATA, tmp_energy, sr->_nfbytes_lo, cudaMemcpyDeviceToHost ));
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    // can free the temp energy buffer
    CUDA_SAFE_CALL( cudaFree( tmp_energy ));

    // Compute data energy, gradient term
    cuflt e_data_gradients = 0.0;
    for ( size_t i = 0 ; i < w * h ; i++ ) {
        e_data_gradients += E_DATA[i];
    }

    //write_pfm_image_signed( w, h, tmp_energy_1, data->_basedir + "/energy%03lu.pfm", ws->_iteration );

    delete[] E_DATA;

    cuflt energy = data->_lambda * e_tv + sr->_dt_alpha * e_data_intensities + sr->_dt_beta * e_data_gradients;

    TRACE("Energy : " << energy << " : E_TV = " << e_tv << " * " << data->_lambda << " = " << e_tv *  data->_lambda
          << " | E_DATA_INTENSITIES =  " << e_data_intensities << " * " << sr->_dt_alpha << " = " << sr->_dt_alpha * e_data_intensities
          << " | E_DATA_GRADIENTS =  " << e_data_gradients << " * " << sr->_dt_beta << " = " << sr->_dt_beta * e_data_gradients << endl );

    return energy;
}

// Write current solution in pfm format
bool coco::coco_vtv_sr_write_pfm_solution( coco_vtv_data *data ) {

    coco_vtv_workspace *ws = data->_workspace;
    size_t W = data->_W;
    size_t H = data->_H;
    assert( W*H > 0 );

    if ( data->_nchannels == 3 ) {
        write_pfm_image_signed( W, H, ws->_U[0], ws->_U[1], ws->_U[2], data->_basedir + "/output.pfm", 0 );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    }

    return true;
}

// Perform one iteration of Algorithm 1, Chambolle-Pock
bool coco::coco_vtv_sr_iteration_fista_unstructured( coco_vtv_data *data, bool data_term ) {

    // TODO: verify correct maximum step sizes.
    data->_tau = 0.3 / sqrt( 8.0 );
    data->_sigma = 0.3 / sqrt( 8.0 );
    data->_L = 1.0 / data->_lambda; //float(data->_sr_data->_nviews) / data->_lambda ;
    //data->_L = float(data->_sr_data->_nviews) / data->_lambda ;
    vtv_sr_ista_step_unstructured( data, data_term );
    cuflt alpha_new = 0.5 * ( 1.0 + sqrt( 1.0 + 4.0 * pow( data->_alpha, 2.0 ) ));
    coco_vtv_rof_overrelaxation( data, ( data->_alpha - 1.0 ) / alpha_new );
    data->_alpha = alpha_new;
    return true;
}

// Perform one single shrinkage step (ISTA)
bool coco::vtv_sr_ista_step_unstructured( coco_vtv_data *data, bool data_term ) {

    assert( data != NULL );
    size_t W = data->_W; // high res
    size_t H = data->_H;
    assert( W*H > 0 );
    coco_vtv_workspace *ws = data->_workspace;
    coco_vtv_sr_data_unstructured *sr = data->_sr_data_unstructured;
    assert( sr != NULL );

    dim3 DimBlock = ws->_dimBlock; // high res
    dim3 DimGrid = ws->_dimGrid;

    // Start descent from current solution
    for ( size_t i=0; i<data->_nchannels; i++ ) {
        CUDA_SAFE_CALL( cudaMemcpy( ws->_Uq[i], ws->_U[i], ws->_nfbytes, cudaMemcpyDeviceToDevice ));
    }
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    for ( size_t i=0; i<data->_nchannels; i++ ) {
        CUDA_SAFE_CALL( cudaMemset( ws->_G[i], 0, ws->_nfbytes ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );
    }

    if ( data_term ) { // Compute gradient of data term

        if ( sr->_dt_alpha > 0.0 ) { // speed up algo when only gradient term is needed
            vtv_sr_dataterm_intensities( data );
        }

        if ( sr->_dt_beta > 0.0 ) { // speed up algo when only intensity term is needed
            vtv_sr_dataterm_gradients( data );
        }

        for ( size_t i=0; i<data->_nchannels; i++ ) {
            cuda_clamp_device<<< DimGrid, DimBlock >>>
                                                     ( W, H, ws->_G[i], -1.0f, 1.0f );
            CUDA_SAFE_CALL( cudaThreadSynchronize() );
        }

//        write_pfm_image_signed( W, H, ws->_U[0], ws->_U[1], ws->_U[2], data->_basedir + "/u_iter_%03lu.pfm", ws->_iteration );
//        write_pfm_image_signed( W, H, ws->_G[0], ws->_G[1], ws->_G[2], data->_basedir + "/G_iter_%03lu.pfm", ws->_iteration );
//        write_pfm_image_signed( W, H, sr->_G_intensities[0], sr->_G_intensities[1], sr->_G_intensities[2], data->_basedir + "/sr->_G_intensities_iter_%03lu.pfm", ws->_iteration );
//        write_pfm_image_signed( W, H, sr->_G_gradients[0], sr->_G_gradients[1], sr->_G_gradients[2], data->_basedir + "/sr->_G_gradients_iter_%03lu.pfm", ws->_iteration );

        ++data->_workspace->_iteration;
        //if (data->_workspace->_iteration == 10) exit(0);
    }

    // Compute F for ROF steps
    for ( size_t i=0; i<data->_nchannels; i++ ) {
        cuda_scale_device<<< ws->_dimGrid, ws->_dimBlock >>>
                                                           ( W, H, ws->_G[i], -1.0 / ( data->_lambda * data->_L ));

        // Add current solution
        cuda_add_to_device<<< ws->_dimGrid, ws->_dimBlock >>>
                                                            ( W, H, ws->_Uq[i], ws->_G[i] );

        // Clamp to 0-1
        cuda_clamp_device<<< ws->_dimGrid, ws->_dimBlock >>>
                                                           ( W, H, ws->_G[i], 0.0f, 1.0f );
    }

    // Perform a number of primal/dual ROF iterations
    data->_tau = 0.3 / sqrt( 8.0 );
    data->_sigma = 0.3 / sqrt( 8.0 );
    for ( size_t k=0; k<data->_inner_iterations; k++ ) {

        coco_vtv_rof_dual_step( data );

        // Primal step kernel call for each channel
        for ( size_t i=0; i<data->_nchannels; i++ ) {
            cuda_rof_primal_prox_step_device<<< ws->_dimGrid, ws->_dimBlock >>>
                                                                              ( W, H, data->_tau, 1.0 / data->_L,
                                                                                ws->_Uq[i], ws->_Uq[i], ws->_G[i], ws->_X1[i], ws->_X2[i] );
            CUDA_SAFE_CALL( cudaThreadSynchronize() );
        }
    }

    //write_test_image_signed( data->_W, data->_H, ws->_Uq[0], "out/Uq_total.png", 0 );
    return true;
}

// Compute gradient of dataterm (difference of intensities)
bool coco::vtv_sr_dataterm_intensities( coco_vtv_data *data ) {

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
    // Start descent from current solution

    for ( size_t i=0 ; i<data->_nchannels ; ++i ) {

        CUDA_SAFE_CALL( cudaMemcpy( ws->_Uq[i], ws->_U[i], ws->_nfbytes, cudaMemcpyDeviceToDevice ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        // Clear derivative
        CUDA_SAFE_CALL( cudaMemset( sr->_G_intensities[i], 0, ws->_nfbytes ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );
    }

    // Clear normalization weights
    CUDA_SAFE_CALL( cudaMemset( sr->_norm_mask, 0, ws->_nfbytes ));
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    // Sum contributions for all views
    for ( size_t nview=0; nview < sr->_views.size(); nview++ ) {

        coco_vtv_sr_view_data_unstructured *view = sr->_views[nview];

        // clear tmp buffer
        for ( size_t i=0 ; i<data->_nchannels ; ++i ) {

            CUDA_SAFE_CALL( cudaMemset( ws->_temp[i], 0, ws->_nfbytes ));
        }

        // reconstruct the low res image v_k given the current high res solution u
        cuda_convolution_nonsep_device_param<<< dimGrid, dimBlock >>>
                                                                    ( W, H,
                                                                      w, h,
                                                                      sr->_ks,
                                                                      view->_A,
                                                                      view->_warp_tau_x,
                                                                      view->_warp_tau_y,
                                                                      ws->_U[0], ws->_U[1], ws->_U[2],
                ws->_temp[0], ws->_temp[1], ws->_temp[2] );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        // compute dv that is the difference between reconstructed low res image v_k and data v_k*
        for ( size_t i=0 ; i<data->_nchannels ; ++i ) {

            cuda_subtract_from_device<<< dimGrid, dimBlock >>>
                                                             ( w, h, view->_image_f + i*w*h, ws->_temp[i] );
            CUDA_SAFE_CALL( cudaThreadSynchronize() );
        }

        // write_pfm_image_signed( w, h,  ws->_temp[0],  ws->_temp[1],  ws->_temp[2], data->_basedir + "/ws->_temp_%03lu.pfm", ws->_iteration );

        // write_pfm_image_signed( w, h, view->_weights_omega_i, data->_basedir + "/view->_weights_omega_%03lu.pfm", nview );

        // multiply by weights
        cuda_multiply_with_device<<< dimGrid, dimBlock >>> ( w, h, ws->_temp[0], view->_weights_omega_i );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );
        cuda_multiply_with_device<<< dimGrid, dimBlock >>> ( w, h, ws->_temp[1], view->_weights_omega_i );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );
        cuda_multiply_with_device<<< dimGrid, dimBlock >>> ( w, h, ws->_temp[2], view->_weights_omega_i );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        // deconvolution step (applying the weights)
        // Forward warp, non-overlap regions sequentially
        int seg_start = 0;
        for ( size_t j = 0 ; j < view->_seg_end.size() ; j++ ) {

            int seg_end = view->_seg_end[j];
            int seg_size = seg_end - seg_start;

            // forward warp call for this segment, cannot overlap
            int seg_width = cuda_default_block_size_x() * cuda_default_block_size_y();
            dim3 dimBlock_splatting = dim3( seg_width, 1 );
            dim3 dimGrid_splatting = dim3( seg_size / seg_width + 1, 1 );

            cuda_deconvolution_nonsep_device_param<<< dimGrid_splatting, dimBlock_splatting >>>( W, H, seg_width,
                                                                                                 ws->_temp[0], ws->_temp[1], ws->_temp[2],
                    view->_cells,
                    seg_start, seg_end,
                    sr->_ks,
                    sr->_dsf,
                    view->_A,
                    view->_warp_tau_x, view->_warp_tau_y,
                    sr->_G_intensities[0], sr->_G_intensities[1], sr->_G_intensities[2],
                    sr->_norm_mask );

            CUDA_SAFE_CALL( cudaThreadSynchronize() );

            seg_start = seg_end;
        }
        CUDA_SAFE_CALL( cudaThreadSynchronize() );
    }

    //write_pfm_image_signed( W, H,  sr->_G_intensities[0],  sr->_G_intensities[1],  sr->_G_intensities[2], data->_basedir + "/sr->_G_intensities_%03lu.pfm", ws->_iteration );

    for ( size_t i=0 ; i<data->_nchannels ; ++i ) {

        cuda_normalize_device<<< DimGrid, DimBlock >>>
                                                     ( W, H, sr->_G_intensities[i], sr->_norm_mask );
        CUDA_SAFE_CALL( cudaThreadSynchronize() );
    }
    // write_pfm_image_signed( W, H,  sr->_G_intensities[0],  sr->_G_intensities[1],  sr->_G_intensities[2], data->_basedir + "/sr->_G_intensities_norm1_%03lu.pfm", ws->_iteration );
    for ( size_t i=0 ; i<data->_nchannels ; ++i ) {

        // uniformly normalize the weights
        cuda_scale_device<<< DimGrid, DimBlock >>>( W, H, sr->_G_intensities[i], sr->_sigma_sensor*sr->_sigma_sensor );
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        cuda_add_scaled_to_device<<< DimGrid, DimBlock >>>
                                                         ( W, H, sr->_G_intensities[i], sr->_dt_alpha, ws->_G[i] );
        CUDA_SAFE_CALL( cudaThreadSynchronize() );
    }

    //write_pfm_image_signed( W, H,  sr->_G_intensities[0],  sr->_G_intensities[1],  sr->_G_intensities[2], data->_basedir + "/sr->_G_intensities_norm_%03lu.pfm", ws->_iteration );

    return true;
}

// Compute gradient of dataterm (difference of gradients)
bool coco::vtv_sr_dataterm_gradients( coco_vtv_data *data ) {

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
    // Start descent from current solution

    for ( size_t i=0 ; i<data->_nchannels ; ++i ) {

        CUDA_SAFE_CALL( cudaMemcpy( ws->_Uq[i], ws->_U[i], ws->_nfbytes, cudaMemcpyDeviceToDevice ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        // Clear derivative
        CUDA_SAFE_CALL( cudaMemset( sr->_G_gradients[i], 0, ws->_nfbytes ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );
    }

    // Clear normalization weights
    CUDA_SAFE_CALL( cudaMemset( sr->_norm_mask, 0, ws->_nfbytes ));
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    // Sum contributions for all views
    for ( size_t nview=0; nview < sr->_views.size(); nview++ ) {

        coco_vtv_sr_view_data_unstructured *view = sr->_views[nview];

        // clear tmp buffer
        for ( size_t i=0 ; i<data->_nchannels ; ++i ) {

            CUDA_SAFE_CALL( cudaMemset( ws->_temp[i], 0, ws->_nfbytes ));
            CUDA_SAFE_CALL( cudaMemset( ws->_temp[3+i], 0, ws->_nfbytes ));

            // compute the Laplacian of the warped solution
            cuda_laplacian_device<<< DimGrid, DimBlock >>>( W, H,
                                                            ws->_U[i],
                                                            ws->_temp[i], /*sr->_visibility_mask*/ 0 );
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );
        }

        // reconstruct the low res image v_k given the current high res solution u
        cuda_convolution_nonsep_device_param<<< dimGrid, dimBlock >>>
                                                                    ( W, H,
                                                                      w, h,
                                                                      sr->_ks,
                                                                      view->_A,
                                                                      view->_warp_tau_x,
                                                                      view->_warp_tau_y,
                                                                      ws->_temp[0], ws->_temp[1], ws->_temp[2],
                ws->_temp[3], ws->_temp[4], ws->_temp[5] );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        for ( size_t i=0 ; i<data->_nchannels ; ++i ) {

            // we scale the Laplacian of the current solution to approximate the Laplacian of the warped solution
            cuda_scale_device<<< dimGrid, dimBlock >>>( w, h, ws->_temp[3+i], sr->_dsf*sr->_dsf );
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            cuda_laplacian_device<<< dimGrid, dimBlock >>>( w, h,
                                                            view->_image_f + i*w*h,
                                                            ws->_temp[i], 0 );
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );

            // compute dv that is the difference between reconstructed low res image v_k and data v_k*
            cuda_subtract_from_device<<< dimGrid, dimBlock >>>
                                                             ( w, h,  ws->_temp[i],  ws->_temp[3+i] );
            CUDA_SAFE_CALL( cudaThreadSynchronize() );
        }

        //write_pfm_image_signed( w, h, lores_tmp, data->_basedir + "/subs_%03lu.pfm", iterations * 8 + nview );

        if ( sr->_gw_type != 0 ) {

            // multiply by weights
            cuda_multiply_with_device<<< dimGrid, dimBlock >>> ( w, h, ws->_temp[3], view->_weights_omega_i );
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );
            cuda_multiply_with_device<<< dimGrid, dimBlock >>> ( w, h, ws->_temp[4], view->_weights_omega_i );
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );
            cuda_multiply_with_device<<< dimGrid, dimBlock >>> ( w, h, ws->_temp[5], view->_weights_omega_i );
            CUDA_SAFE_CALL( cudaDeviceSynchronize() );
        }

        // deconvolution step (applying the weights)
        // Forward warp, non-overlap regions sequentially
        int seg_start = 0;
        for ( size_t j = 0 ; j < view->_seg_end.size() ; j++ ) {

            int seg_end = view->_seg_end[j];
            int seg_size = seg_end - seg_start;

            // forward warp call for this segment, cannot overlap
            int seg_width = cuda_default_block_size_x() * cuda_default_block_size_y();
            dim3 dimBlock_splatting = dim3( seg_width, 1 );
            dim3 dimGrid_splatting = dim3( seg_size / seg_width + 1, 1 );

            cuda_deconvolution_nonsep_device_param<<< dimGrid_splatting, dimBlock_splatting >>>( W, H, seg_width,
                                                                                                 ws->_temp[3], ws->_temp[4], ws->_temp[5],
                    view->_cells,
                    seg_start, seg_end,
                    sr->_ks,
                    sr->_dsf,
                    view->_A,
                    view->_warp_tau_x, view->_warp_tau_y,
                    sr->_G_gradients[0], sr->_G_gradients[1], sr->_G_gradients[2],
                    sr->_norm_mask );

            CUDA_SAFE_CALL( cudaThreadSynchronize() );

            seg_start = seg_end;
        }
        CUDA_SAFE_CALL( cudaThreadSynchronize() );
    }

    for ( size_t i=0 ; i<data->_nchannels ; ++i ) {

        cuda_normalize_device<<< DimGrid, DimBlock >>>
                                                     ( W, H, sr->_G_gradients[i], sr->_norm_mask );
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        if ( sr->_gw_type != 0 ) {

            // uniformly normalize the weights
            cuda_scale_device<<< DimGrid, DimBlock >>>( W, H, sr->_G_gradients[i], sr->_sigma_sensor*sr->_sigma_sensor );
            CUDA_SAFE_CALL( cudaThreadSynchronize() );
        }

        cuda_scale_device<<< DimGrid, DimBlock >>>( W, H, sr->_G_gradients[i], -sr->_gradient_step );
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        cuda_add_scaled_to_device<<< DimGrid, DimBlock >>>
                                                         ( W, H, sr->_G_gradients[i], sr->_dt_beta, ws->_G[i] );
        CUDA_SAFE_CALL( cudaThreadSynchronize() );
    }

    return true;
}

// Compute gradient of dataterm (difference of gradients)
bool coco::vtv_sr_dataterm_gradients2( coco_vtv_data *data ) {

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
    // Start descent from current solution

    for ( size_t i=0 ; i<data->_nchannels ; ++i ) {

        CUDA_SAFE_CALL( cudaMemcpy( ws->_Uq[i], ws->_U[i], ws->_nfbytes, cudaMemcpyDeviceToDevice ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        // Clear derivative
        CUDA_SAFE_CALL( cudaMemset( sr->_G_gradients[i], 0, ws->_nfbytes ));
        CUDA_SAFE_CALL( cudaThreadSynchronize() );
    }

    // Clear normalization weights
    CUDA_SAFE_CALL( cudaMemset( sr->_norm_mask, 0, ws->_nfbytes ));
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    // clear tmp buffer
    for ( size_t i=0 ; i<data->_nchannels ; ++i ) {

        CUDA_SAFE_CALL( cudaMemset( ws->_temp[i], 0, ws->_nfbytes ));
        CUDA_SAFE_CALL( cudaMemset( ws->_temp[3+i], 0, ws->_nfbytes ));
    }

    // Perform splatting for every input view
    for ( size_t nview = 0 ; nview < sr->_views.size() ; nview++ ) {

        coco_vtv_sr_view_data_unstructured *view = sr->_views[nview];

        // Compute the gradient of the input view
        cuda_gradient_device <<< dimGrid, dimBlock >>>
                                                     ( w, h,
                                                       view->_image_f + 0*w*h,
                                                       view->_image_f + 1*w*h,
                                                       view->_image_f + 2*w*h,
                                                       ws->_temp[0], ws->_temp[1] );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        // multiply by weights
        cuda_multiply_with_device<<< dimGrid, dimBlock >>> ( w, h, ws->_temp[0], view->_weights_omega_i );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );
        cuda_multiply_with_device<<< dimGrid, dimBlock >>> ( w, h, ws->_temp[1], view->_weights_omega_i );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        // Forward warp, non-overlap regions sequentially
        int seg_start = 0;
        for ( size_t j = 0 ; j < view->_seg_end.size() ; j++ ) {

            int seg_end = view->_seg_end[j];
            int seg_size = seg_end - seg_start;

            // forward warp call for this segment, cannot overlap
            int seg_width = cuda_default_block_size_x() * cuda_default_block_size_y();
            dim3 dimBlock_splatting = dim3( seg_width, 1 );
            dim3 dimGrid_splatting = dim3( seg_size / seg_width + 1, 1 );

            cuda_deconvolution_nonsep_device_param<<< dimGrid_splatting, dimBlock_splatting >>>( W, H, seg_width,
                                                                                                 ws->_temp[0],ws->_temp[1], ws->_temp[1],
                    view->_cells,
                    seg_start, seg_end,
                    sr->_ks,
                    sr->_dsf,
                    view->_A,
                    view->_warp_tau_x, view->_warp_tau_y,
                    ws->_temp[2], ws->_temp[3], ws->_temp[3],
                    sr->_norm_mask );

            CUDA_SAFE_CALL( cudaThreadSynchronize() );

            seg_start = seg_end;
        }
    }

    // Normalize
    cuda_normalize_device<<< DimGrid, DimBlock >>>
                                                 ( W, H, ws->_temp[2], sr->_norm_mask );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    cuda_normalize_device<<< DimGrid, DimBlock >>>
                                                 ( W, H, ws->_temp[3], sr->_norm_mask );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    // Compute the divergencce of the target gradient
    cuda_divergence_device<<< DimGrid, DimBlock >>>( W, H,
                                                     ws->_temp[2],
            ws->_temp[3],
            ws->_temp[0] );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    // we scale the Laplacian of the current solution to approximate the Laplacian of the warped solution
    cuda_scale_device<<< DimGrid, DimBlock >>>( W, H, ws->_temp[0], 1.0/(sr->_dsf*sr->_dsf) );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    for ( size_t i=0 ; i<data->_nchannels ; ++i ) {

        // compute the Laplacian of the warped solution
        cuda_laplacian_device<<< DimGrid, DimBlock >>>( W, H,
                                                        ws->_U[i],
                                                        sr->_G_gradients[i], 0 );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        cuda_subtract_from_device<<< DimGrid, DimBlock >>>
                                                         ( W, H, ws->_temp[0],  sr->_G_gradients[i] );
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        cuda_scale_device<<< DimGrid, DimBlock >>>( W, H, sr->_G_gradients[i], -sr->_gradient_step );
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        cuda_add_scaled_to_device<<< DimGrid, DimBlock >>>
                                                         ( W, H, sr->_G_gradients[i], sr->_dt_beta, ws->_G[i] );
        CUDA_SAFE_CALL( cudaThreadSynchronize() );
    }

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
    dim3 DimBlock = ws->_dimBlock; // high res
    dim3 DimGrid = ws->_dimGrid;

    CUDA_SAFE_CALL( cudaMemset( ws->_temp[1], 0, ws->_nfbytes ));

    // Sum contributions for all views
    for ( size_t nview=0; nview < sr->_views.size(); nview++ ) {

        coco_vtv_sr_view_data_unstructured *view = sr->_views[nview];

        CUDA_SAFE_CALL( cudaMemcpy( ws->_temp[0], view->_weights_omega_i, sr->_nfbytes_lo, cudaMemcpyDeviceToDevice ));

        // deconvolution step (applying the weights)
        // Forward warp, non-overlap regions sequentially
        int seg_start = 0;
        for ( size_t j = 0 ; j < view->_seg_end.size() ; j++ ) {

            int seg_end = view->_seg_end[j];
            int seg_size = seg_end - seg_start;

            // forward warp call for this segment, cannot overlap
            int seg_width = cuda_default_block_size_x() * cuda_default_block_size_y();
            dim3 dimBlock_splatting = dim3( seg_width, 1 );
            dim3 dimGrid_splatting = dim3( seg_size / seg_width + 1, 1 );

            cuda_deconvolution_nonsep_device_param<<< dimGrid_splatting, dimBlock_splatting >>>( W, H, seg_width,
                                                                                                 ws->_temp[0],
                    view->_cells,
                    seg_start, seg_end,
                    sr->_ks,
                    sr->_dsf,
                    view->_A,
                    view->_warp_tau_x, view->_warp_tau_y,
                    ws->_temp[1], //target mask
                    sr->_norm_mask );

            CUDA_SAFE_CALL( cudaThreadSynchronize() );

            seg_start = seg_end;
        }
        CUDA_SAFE_CALL( cudaThreadSynchronize() );
    }

    cuda_normalize_device<<< DimGrid, DimBlock >>>
                                                 ( W, H, ws->_temp[1], sr->_norm_mask );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    // uniformly normalize the weights
    cuda_scale_device<<< DimGrid, DimBlock >>>( W, H, ws->_temp[1], sr->_sigma_sensor*sr->_sigma_sensor );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    // Use target mask as a regularizer weight
    for ( size_t i=0; i<data->_nchannels; i++ ) {
        // Use target mask as a regularizer weight
        if ( ws->_g[i] == NULL ) {
            CUDA_SAFE_CALL( cudaMalloc( &ws->_g[i], ws->_nfbytes ));
            CUDA_SAFE_CALL( cudaMemset( ws->_g[i], 0, ws->_nfbytes ));
        }
        vtv_sr_init_regularizer_weight_device<<< DimGrid, DimBlock >>>
                                                                     ( W, H,
                                                                       data->_lambda_max_factor * data->_lambda,
                                                                       data->_lambda,
                                                                       sr->_nviews,
                                                                       ws->_temp[1], ws->_g[i] );
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );

        // Convolve weight
        // NASTY BUG: border conditions?
        cuda_convolution( sr->_vmask_filter, W, H, ws->_g[i], ws->_temp[0] );
        CUDA_SAFE_CALL( cudaMemcpy( ws->_g[i], ws->_temp[0], ws->_nfbytes, cudaMemcpyDeviceToDevice ));
    }

//    TRACE("data->_lambda_max_factor * data->_lambda: " << data->_lambda_max_factor * data->_lambda << endl);
//    write_pfm_image_signed( W, H, ws->_g[0], ws->_g[1], ws->_g[2], data->_basedir + "/g_%03lu.pfm", ws->_iteration );
//    write_pfm_image_signed( W, H, ws->_temp[1], data->_basedir + "/norm_mask_%03lu.pfm", ws->_iteration );

    return true;
}



