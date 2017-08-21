/* -*-c++-*- */
#include <iostream>
#include <algorithm>

#include "vtv.h"
#include "vtv.cuh"

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

using namespace std;

const float INVALID_DEPTH = -1e30;

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
    vector<float*> _vmask;
    
    // depth map on v (with u resolution) 
    float *_dmap_v;

    // variance of the disparity measure
    float *_dmap_sigma;

    // depth vote for u (CPU)
    float *_dmap_u;

    // visibility weight on lores-v
    // its the sum of the weights of each pixel
    // corresponding high resolution pixel
    vector<float*> _vmask_lo;

    // target cell array in u
    // this is an array of index of non overlapping pixel groups
    // used to parallellize the forward mapping tasks
    int *_cells; 
    vector<int> _seg_end;

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
    vector<coco_vtv_sr_view_data*> _views;

    // disparity map created using vtv_sr_init_target_disparity_map
    float *_dmap_u;
    float _disp_max;

    // Target mask (in hires)
    vector<float*> _target_mask;

    // Kernel for visibility map smoothing
    cuda_kernel *_vmask_filter;

    // grid data
    dim3 _dimGrid;
    dim3 _dimBlock;
  };

  // init u_dmap of the view with the v_dmap. Nearest disparity wins
  bool vtv_sr_init_u_dmap( coco_vtv_data *data, size_t nview );

  // Init forward warp for a view
  // Currently completely on host, TODO: try to parallelize (hard)
  bool vtv_sr_init_forward_warp_structure( coco_vtv_data *data, size_t nview );

  // Compute the disparity map in u, using the disparities in v_i
  // sr->_dmap_u is filled using the votes of each view->_dmap_u (warped disparities into u)
  // _dmap_u for each view must be already computed
  // The max or median value of all votes is used (test what works best)
  bool vtv_sr_init_target_disparity_map( coco_vtv_data *data );

  // Filter the visibility mask 6 times (hard-coded) NOT USED
  // Add the nine neighbors values into 'sum': 
  // if sum<4 -> 0
  // if sum==4 -> keep current value
  // if sum>4 -> 1
  bool vtv_sr_filter_mask( coco_vtv_data *data, int *mask );
  
  // Filter the disparity map : NOT USED
  bool vtv_sr_filter_disparity_map( coco_vtv_data *data, float *mask );

  // Compute gradient of data term
  bool vtv_sr_dataterm_gradient( coco_vtv_data *data );
  // Compute dataterm gradient with pre-initialized optimized depth map
  bool vtv_sr_dataterm_gradient_dopt( coco_vtv_data *data );

  // Perform one single shrinkage step (ISTA)
  bool vtv_sr_ista_step( coco_vtv_data *data );
};




/*****************************************************************************
       TV_x Superresolution
*****************************************************************************/

// Setup SR algorithm: init view and resolution data
bool coco::coco_vtv_sr_init( coco_vtv_data *data, size_t nviews, size_t ds_factor )
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
                          + data->_nchannels * sr->_nfbytes_lo // _vmask_lo
                          + data->_nchannels * sr->_vmask_bytes  // _vmask
                          + 2 * sr->_nfbytes_hi; // _dmap & _dmap_sigma
  size_t bytes_view_total = nviews * bytes_per_view;
  TRACE( "Allocating mem:" << endl );
  TRACE( "  " << bytes_per_view / MB << " Mb per view, " << bytes_view_total/MB << " total." << endl );
  
  for ( size_t i=0; i<nviews; i++ ) {
    coco_vtv_sr_view_data *view = new coco_vtv_sr_view_data;

    CUDA_SAFE_CALL( cudaMalloc( &view->_image_f, data->_nchannels * sr->_nfbytes_lo ));
    CUDA_SAFE_CALL( cudaMemset( view->_image_f, 0, data->_nchannels * sr->_nfbytes_lo ));

    CUDA_SAFE_CALL( cudaMalloc( &view->_visibility_mask, W*H * sizeof(bool) ));
    CUDA_SAFE_CALL( cudaMemset( view->_visibility_mask, 0, W*H * sizeof(bool) ));

    // TODO: optimize : dmap_v and dmap_sigma could be in lo-res
    CUDA_SAFE_CALL( cudaMalloc( &view->_dmap_v, sr->_nfbytes_hi ));
    CUDA_SAFE_CALL( cudaMemset( view->_dmap_v, 0, sr->_nfbytes_hi ));

    CUDA_SAFE_CALL( cudaMalloc( &view->_dmap_sigma, sr->_nfbytes_hi ));
    CUDA_SAFE_CALL( cudaMemset( view->_dmap_sigma, 0, sr->_nfbytes_hi ));

    view->_vmask_lo.resize(data->_nchannels, 0);
    view->_vmask.resize(data->_nchannels, 0);
    for( size_t n=0; n< data->_nchannels; ++n) {
      CUDA_SAFE_CALL( cudaMalloc( &(view->_vmask_lo[n]), sr->_nfbytes_lo ));
      CUDA_SAFE_CALL( cudaMemset( view->_vmask_lo[n], 0, sr->_nfbytes_lo ));

      CUDA_SAFE_CALL( cudaMalloc( &(view->_vmask[n]), sr->_vmask_bytes ));
      CUDA_SAFE_CALL( cudaMemset( view->_vmask[n], 0, sr->_vmask_bytes ));
    }

    view->_cells = NULL;
    view->_dmap_u = NULL;

    // done
    sr->_views.push_back( view );
  }

  // Additional work mem (TODO: reduce, use temp buffers w->F[...])
  size_t srbytes = data->_nchannels * sr->_nfbytes_hi;
  TRACE( "  " << srbytes/MB << " Mb for additional work structures." << endl );

  // Target coverage
  CUDA_SAFE_CALL( cudaMalloc( &sr->_dmap_u, sr->_nfbytes_hi ));
  CUDA_SAFE_CALL( cudaMemset( sr->_dmap_u, 0, sr->_nfbytes_hi ));
  sr->_target_mask.resize(data->_nchannels, NULL );
  for ( size_t n=0; n<data->_nchannels; n++ ) {
    CUDA_SAFE_CALL( cudaMalloc( &(sr->_target_mask[n]), sr->_nfbytes_hi ));
    CUDA_SAFE_CALL( cudaMemset( sr->_target_mask[n], 0, sr->_nfbytes_hi ));
  }

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

// Free up data for SR algorithm
bool coco::coco_vtv_sr_free( coco_vtv_data *data )
{
  assert( data != NULL );
  coco_vtv_sr_data *sr = data->_sr_data;
  assert( sr != NULL );

  for ( size_t i=0; i<sr->_nviews; i++ ) {
    coco_vtv_sr_view_data *view = sr->_views[i];
    CUDA_SAFE_CALL( cudaFree( view->_image_f ));
    CUDA_SAFE_CALL( cudaFree( view->_visibility_mask ));
    CUDA_SAFE_CALL( cudaFree( view->_cells ));
    CUDA_SAFE_CALL( cudaFree( view->_dmap_v ));
    CUDA_SAFE_CALL( cudaFree( view->_dmap_sigma ));

    for ( size_t n=0; n<data->_nchannels; n++ ) {
      CUDA_SAFE_CALL( cudaFree( view->_vmask[n] ));
      CUDA_SAFE_CALL( cudaFree( view->_vmask_lo[n] ));
    }

    delete[] view->_dmap_u;
    delete view;
  }

  for ( size_t n=0; n<data->_nchannels; n++ ) {
    CUDA_SAFE_CALL( cudaFree( sr->_target_mask[n] ));
  }

  CUDA_SAFE_CALL( cudaFree( sr->_dmap_u ));
  cuda_kernel_free( sr->_vmask_filter );

  // finalize
  delete data->_sr_data;
  data->_sr_data = NULL;
  return true;
}

void coco::coco_vtv_sr_set_ugrad_threshold( coco_vtv_data *data, float ugrad_threshold) {
  data->_sr_data->_ugrad_threshold = ugrad_threshold;
}

void coco::coco_vtv_sr_set_sigma_sensor( coco_vtv_data *data, float sigma_sensor){
  data->_sr_data->_sigma_sensor = sigma_sensor;
}

void coco::coco_vtv_sr_set_sigma_disp( coco_vtv_data *data, float aux_dmap_sigma){
  data->_sr_data->_aux_dmap_sigma = aux_dmap_sigma;
}


void coco::coco_vtv_sr_remove_last_view( coco_vtv_data *data )
{
  assert( data != NULL );
  coco_vtv_sr_data *sr = data->_sr_data;
  assert( sr != NULL );
  coco_vtv_sr_view_data *view = sr->_views.back();
  CUDA_SAFE_CALL( cudaFree( view->_image_f ));
  for ( size_t n=0; n<data->_nchannels; n++ ) {
    CUDA_SAFE_CALL( cudaFree( view->_vmask[n] ));
    CUDA_SAFE_CALL( cudaFree( view->_vmask_lo[n] ));
  }
  CUDA_SAFE_CALL( cudaFree( view->_cells ));
  CUDA_SAFE_CALL( cudaFree( view->_dmap_v ));
  CUDA_SAFE_CALL( cudaFree( view->_dmap_sigma ));
  delete[] view->_dmap_u;
  delete view;
  sr->_views.pop_back();
  sr->_nviews = data->_sr_data->_views.size();
}


// Setup a single test view with globally constant displacement for testing.
// displacement is measured in percent of an original pixel.
bool coco::coco_vtv_sr_create_test_view( coco_vtv_data *data, size_t nview, double dx, double dy )
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
  size_t dsf = sr->_dsf;
  size_t w = W / dsf;
  size_t h = H / dsf;
  assert( w==sr->_w );
  assert( h==sr->_h );
  coco_vtv_workspace *ws = data->_workspace;

  // Warp geometry
  view->_dx_vu = dx*dsf;
  view->_dy_vu = -dy*dsf;

  // no image : call coco_vtv_sr_render_test_view

  // Disparity
  float *dmap_lo = NULL;
  CUDA_SAFE_CALL( cudaMalloc( &dmap_lo, w*h*sizeof(float) ));
  cuda_set_all_device<<< sr->_dimBlock, sr->_dimGrid >>> ( w,h, dmap_lo, 1.0f );

  // upsample to hires DMAP
  vtv_sr_upsample_dmap_device<<< ws->_dimGrid, ws->_dimBlock >>>
    ( W,H, w,h, sr->_dsf, dmap_lo, view->_dmap_v );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );

  // set dmap sigma to 1. also
  vtv_sr_upsample_dmap_device<<< ws->_dimGrid, ws->_dimBlock >>>
      ( W,H, w,h, sr->_dsf, dmap_lo, view->_dmap_sigma );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );

  CUDA_SAFE_CALL( cudaFree( dmap_lo ));

  // Vote for u_dmap
  vtv_sr_init_u_dmap( data, nview );

  // max disparity
  sr->_disp_max = 1.0f;

  return true;
}

// Render a single test view using current warp structure
bool coco::coco_vtv_sr_render_test_view( coco_vtv_data *data, size_t nview )
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
  coco_vtv_workspace *ws = data->_workspace;

  // Compute backward warp of original image
  gsl_image *I = gsl_image_alloc( sr->_w, sr->_h );
  vector<gsl_matrix*> IC = gsl_image_get_channels(I);

  for ( size_t i=0; i<data->_nchannels; i++ ) {
    
    CUDA_SAFE_CALL( cudaMemset( ws->_G[i], 0, sizeof(float) * data->_W * data->_H ));

    vtv_sr_warp_view_device<<< ws->_dimGrid, ws->_dimBlock >>>
      ( data->_W, data->_H, ws->_F[i],
        view->_dmap_v, view->_dx_vu, view->_dy_vu, view->_visibility_mask, ws->_G[i] );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
 
    vtv_sr_downsample_view_device<<< sr->_dimGrid, sr->_dimBlock >>>
      ( sr->_w, sr->_h, data->_W, ws->_G[i], view->_visibility_mask,
        sr->_dsf, view->_image_f + i*sr->_w* sr->_h );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    cuda_memcpy( IC[i], view->_image_f + i*sr->_w* sr->_h );
  }
 
  char filename[200];
  sprintf( filename, "%sinput_view_%04i.png", data->_basedir.c_str(), (int)nview );
  gsl_image_save( filename, I );

  gsl_image_free( I );
  return true;
}



// Setup a single view given its displacement, the source image and its disparity map
// (dx,dy) : optical center displacement measured in disparity map units
// I input image (low resolution size)
// disparity : its size is low_resolution, but its values are in hi-resolution (what's the rule??)
bool coco::coco_vtv_sr_create_view( coco_vtv_data *data, size_t nview,
				    double dx, double dy,
				    gsl_image *I, float *disparity, float *disp_sigma)
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
  size_t dsf = sr->_dsf;
  size_t w = W / dsf;
  size_t h = H / dsf;
  assert( I->_w == w );
  assert( I->_h == h );
  coco_vtv_workspace *ws = data->_workspace;

  // Warp geometry
  view->_dx_vu = dx*dsf;
  view->_dy_vu = -dy*dsf;

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

  // Disparity
  assert ( disparity );
  float *dmap_lo = NULL;
  CUDA_SAFE_CALL( cudaMalloc( &dmap_lo, w*h*sizeof(float) ));
  CUDA_SAFE_CALL( cudaMemcpy( dmap_lo, disparity, w*h*sizeof(float), cudaMemcpyHostToDevice ));
  // upsample to hires DMAP; values are not scaled
  vtv_sr_upsample_dmap_device<<< ws->_dimGrid, ws->_dimBlock >>>
      ( W,H, w,h, sr->_dsf, dmap_lo, view->_dmap_v );
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // Disparity confidence -- reuse dmap_lo temp var
  // if it's NULL set it to 0
  bool free_it = false;
  if ( disp_sigma == 0 ) {
    disp_sigma = (float*) malloc( w*h*sizeof(float));
    memset(disp_sigma, 0, w*h*sizeof(float));
    free_it = true;
  }

  CUDA_SAFE_CALL( cudaMemcpy( dmap_lo, disp_sigma, w*h*sizeof(float), cudaMemcpyHostToDevice ));

  if ( free_it ) {
    free (disp_sigma);
  }
  // upsample to hires; values are not scaled
  vtv_sr_upsample_dmap_device<<< ws->_dimGrid, ws->_dimBlock >>>
      ( W,H, w,h, sr->_dsf, dmap_lo, view->_dmap_sigma );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() );

  CUDA_SAFE_CALL( cudaFree( dmap_lo ));

  // Vote for u_dmap
  vtv_sr_init_u_dmap( data, nview );

  // max disparity
  for ( size_t i=0; i<w*h; i++ ) {
    float d = fabs( disparity[i] );
    if ( d > sr->_disp_max ) {
      sr->_disp_max = d;
    }
  }
  
  // Test forward warp
  if ( traceLevel() > 2 ) {
    gsl_image *I = gsl_image_alloc( W,H );
    vector<gsl_matrix*> IC = gsl_image_get_channels( I );
    coco_vtv_sr_compute_forward_warp( data, nview ); // but the cell structure hasn't been initialized
    coco_vtv_get_solution( data, IC );
    char str[200];
    sprintf( str, "forward_warp_%03lu.png", nview );
    gsl_image_save( data->_basedir + str, I );
    gsl_image_free( I );
  }

  return true;
}

//static int meta_iter = -1;

// Update weights
// this can be done after the disparity maps have been computed
bool coco::coco_vtv_sr_compute_weights( coco_vtv_data *data )
{
  // check for required data 
  assert( data != NULL );
  coco_vtv_sr_data *sr = data->_sr_data;
  assert( sr != NULL );
  size_t W = data->_W;
  size_t H = data->_H;
  assert( W*H > 0 );
  coco_vtv_workspace *ws = data->_workspace;

  assert(ws->_nfbytes == sr->_nfbytes_hi);

  //++meta_iter;

  // temps images used to compute the _vmaks contributions
  float *warp_img = ws->_temp[0];
  CUDA_SAFE_CALL( cudaMemset( warp_img, 0, ws->_nfbytes ));

  // init _target_mask
  // accumulation of _vmask contributions will be stored into it
  for ( size_t n=0; n<data->_nchannels; n++ ) {
    CUDA_SAFE_CALL( cudaMemset( sr->_target_mask[n], 0, sr->_nfbytes_hi ));
  }

  // Reuse _X1, _X2  for gradient values
  for ( size_t n=0; n<data->_nchannels; n++ ) {
    cuda_compute_gradient_device <<< ws->_dimGrid, ws->_dimBlock >>>
        ( W,H, ws->_U[n], ws->_X1[n], ws->_X2[n]);
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
  }

  // update each view
  for ( size_t nview=0; nview < sr->_nviews; nview++ ) {
    coco_vtv_sr_view_data *view = sr->_views[nview];

    float *tmpgrad = ws->_G[1];

    //write_test_image_bool( W,H, view->_visibility_mask, data->_basedir + "visibility_mask_prev_%05i.png", nview );

    // compute gradient mask weight // the same for all channels
    vtv_sr_init_mask_gradient_weight_device<<< ws->_dimGrid, ws->_dimBlock >>>
        ( W,H, view->_dx_vu, view->_dy_vu, view->_dmap_v, view->_visibility_mask, tmpgrad );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    // compute weights with u gradient
    for ( size_t n=0; n<data->_nchannels; n++ ) {

      vtv_sr_compute_u_gradient_weight_device<<< ws->_dimGrid, ws->_dimBlock >>>
        ( W, H, view->_visibility_mask, view->_dx_vu, view->_dy_vu, ws->_X1[n], ws->_X2[n],
          view->_dmap_v, view->_dmap_sigma, sr->_sigma_sensor, sr->_aux_dmap_sigma, sr->_ugrad_threshold, view->_vmask[n]);   
      CUDA_SAFE_CALL( cudaDeviceSynchronize() );
      //write_pfm_image_signed(W,H, view->_vmask[n], data->_basedir + "vmask_ang_%05i.pfm", meta_iter*100+nview*10+n);

      cuda_multiply_with_device<<< ws->_dimGrid, ws->_dimBlock >>>
        ( W, H, view->_vmask[n], tmpgrad);
      CUDA_SAFE_CALL( cudaDeviceSynchronize() );

      // downsample mask
      vtv_sr_downsample_mask_device<<< sr->_dimGrid, sr->_dimBlock >>>
        ( sr->_w, sr->_h, data->_W, view->_vmask[n], view->_vmask_lo[n], sr->_dsf );
      CUDA_SAFE_CALL( cudaDeviceSynchronize() );

      if ( traceLevel() > 3 ) {
        write_test_image_signed( W,H, view->_vmask[n], data->_basedir + "vmask_%05i.png", nview );
        if ( traceLevel() > 4 ) {
          write_test_image_signed( sr->_w, sr->_h, view->_vmask_lo[n], data->_basedir + "vmask_lo_%05i.png", nview );
        }
      }

      //write_pfm_image_signed(W,H, view->_vmask[n], data->_basedir + "vmaksDown_%05i.pfm", k*100+nview*10+n);
    }

//    write_pfm_image_signed( sr->_w, sr->_h,
//                            view->_vmask_lo[0],
//            view->_vmask_lo[1],
//            view->_vmask_lo[2], data->_basedir + "/_vmask_lo%05i.pfm", nview );

    /*
    TRACE("Test: write vmask of view " << nview << std::endl);
    write_pfm_image_signed(W, H, view->_vmask[0], view->_vmask[1], view->_vmask[2], data->_basedir + "vmask_gold_%02lu.pfm", nview);
    */

    // Use forward warp structures to cummulate the _vmaks of each view into the _target_mask
    float *warped_img = ws->_temp[1];

    // accumulation of _vmask contributions will be stored into _target_mask
    for ( size_t n=0; n<data->_nchannels; n++ ) {
      CUDA_SAFE_CALL( cudaMemset( warped_img, 0, ws->_nfbytes ));

      int seg_start = 0;

      for ( size_t j=0; j<view->_seg_end.size(); j++ ) {
        int seg_end = view->_seg_end[j];
        int seg_size = seg_end - seg_start;

        // forward warp call for this segment, cannot overlap
        int seg_width = cuda_default_block_size_x() * cuda_default_block_size_y();
        dim3 dimBlock = dim3( seg_width, 1 );
        dim3 dimGrid = dim3( seg_size / seg_width + 1, 1 );

        // the important thing is to fill the _target_mask.
        // we warp an empty image
        vtv_sr_forward_warp_accumulate_device<<< dimGrid, dimBlock >>>
            ( data->_W, data->_H, seg_width,
                warp_img,
                view->_cells, seg_start, seg_end,
                view->_dmap_v, view->_dx_vu, view->_dy_vu,
                view->_vmask[n],
                warped_img, sr->_target_mask[n] );

        CUDA_SAFE_CALL( cudaDeviceSynchronize() );
        seg_start = seg_end;
      }
    }
    
    // Test forward warp
    if ( traceLevel() > 2 ) {
      gsl_image *I = gsl_image_alloc( W,H );
      vector<gsl_matrix*> IC = gsl_image_get_channels( I );
      coco_vtv_sr_compute_forward_warp( data, nview );
      coco_vtv_get_solution( data, IC );
      char str[200];
      sprintf( str, "forward_warp_updated_%03lu.png", nview );
      gsl_image_save( data->_basedir + str, I );
      gsl_image_free( I );
    }
  }

  return true;
}


bool coco::vtv_sr_init_target_disparity_map( coco_vtv_data *data )
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



// End view creation
bool coco::coco_vtv_sr_end_view_creation( coco_vtv_data *data )
{
  // check for required data 
  assert( data != NULL );
  coco_vtv_sr_data *sr = data->_sr_data;
  assert( sr != NULL );
  size_t W = data->_W;
  size_t H = data->_H;
  assert( W*H > 0 );

  TRACE( "Finalizing views ..." );

  // Init target disparity map
  vtv_sr_init_target_disparity_map( data );

  // Write test images for depth map on u
  if ( traceLevel() > 1 ) {
    write_test_image_signed( W,H, sr->_dmap_u, data->_basedir + "dmap_u.png", 0 );
  }

  // compute visibility in each view comparing to u depth map
  // and compute the forward warp structure
  coco_vtv_workspace *ws = data->_workspace;
  for ( size_t nview=0; nview < sr->_nviews; nview++ ) {
    coco_vtv_sr_view_data *view = sr->_views[nview];

    // Compute binary visibility mask comparing _dmap_v to _dmap_u using _disp_threshold
    vtv_sr_compute_visibility_mask_device<<< ws->_dimGrid, ws->_dimBlock >>>
        ( W,H, view->_dmap_v, view->_dx_vu, view->_dy_vu, sr->_dmap_u,
            view->_visibility_mask, data->_disp_threshold);
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    if ( traceLevel() > 4 ) {
      write_test_image_bool( W,H, view->_visibility_mask, data->_basedir + "visibility_mask_prev_%05i.png", nview );
    }

    // Initialize warp masks
    vtv_sr_init_forward_warp_structure( data, nview);
  }

  coco_vtv_sr_compute_weights( data );

  TRACE( " done " << endl );

  if ( traceLevel() > 2 ) {
      for ( size_t nview=0; nview < sr->_nviews; nview++ ) {

          // Test forward warp

          TRACE("compute test forward warped image " << nview << std::endl);
          gsl_image *I = gsl_image_alloc( W,H );
          vector<gsl_matrix*> IC = gsl_image_get_channels( I );
          coco_vtv_sr_compute_forward_warp( data, nview );
          coco_vtv_get_solution( data, IC );
          char str[200];
          sprintf( str, "forward_warp_%03lu.png", nview );
          gsl_image_save( data->_basedir + str, I );
          gsl_image_free( I );
      }
  }

  return true;
}
    
// Currently completely on host, TODO: try to parallelize (hard)
// fills _dmap_u with the depth of the warped pixels
bool coco::vtv_sr_init_u_dmap( coco_vtv_data *data, size_t nview )
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

  // Need depth map for v from GPU
  float *dmap_v = new float[ N ];
  CUDA_SAFE_CALL( cudaMemcpy( dmap_v, view->_dmap_v, sizeof(float) * N, cudaMemcpyDeviceToHost ));
  
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
      float d = dmap_v[o];
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

  // Cleanup
  delete[] dmap_v;
  return true;
}



// Init forward warp for a view : uses disparities and visibility (make sure they are computed)
// Currently completely on host, TODO: try to parallelize (hard)
bool coco::vtv_sr_init_forward_warp_structure( coco_vtv_data *data, size_t nview )
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

  // Need depth map for v from GPU
  float *dmap_v = new float[ N ];
  CUDA_SAFE_CALL( cudaMemcpy( dmap_v, view->_dmap_v, sizeof(float) * N, cudaMemcpyDeviceToHost ));
  bool *visibility = new bool[ N ];
  CUDA_SAFE_CALL( cudaMemcpy( visibility, view->_visibility_mask, sizeof(bool) * N, cudaMemcpyDeviceToHost ));

  // Compute target cells for each source pixel
  int *c_in = new int[ N ];
  for ( int oy=0; oy<H; oy++ ) {
    for ( int ox=0; ox<W; ox++ ) {
      int o = ox + oy*W;
      if ( visibility[o] == 0 ) {
        c_in[ o ] = N;
        continue;
      }

      // get location in u
      float d = dmap_v[o];
      float uxv = ox + d * view->_dx_vu;
      float uyv = oy + d * view->_dy_vu;
      int px = (int)floor(uxv);
      int py = (int)floor(uyv);
      if ( px<0 || py<0 || px>W-1 || py>H-1 ) {
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

  int *grouped = new int[N];
  int ngrouped = 0;
  memset( grouped, 0, sizeof(int) * N );
  int *count = new int[ N ];
  vector<int> cells;
  view->_seg_end.clear();

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

    for ( int i=0; i<N; i++ ) {
      if ( grouped[i] ) {
        continue;
      }

      int target = c_in[i];

      // check targets is unused
      if ( target == N ) {
        grouped[i] = 1;
        ngrouped ++;
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

    view->_seg_end.push_back( cells.size() );
    TRACE6( "  ... " << ngrouped << " grouped, " << cells.size() << " cells." << endl );
  }
  TRACE6( "done." << endl );
  assert( ngrouped == N );

  // Copy new cell grouping to GPU
  if ( view->_cells != NULL ) {
    CUDA_SAFE_CALL( cudaFree( view->_cells ));
  }
  CUDA_SAFE_CALL( cudaMalloc( &view->_cells, sizeof(int) * cells.size() ));
  CUDA_SAFE_CALL( cudaMemcpy( view->_cells, &cells[0], sizeof(int) * cells.size(), cudaMemcpyHostToDevice ));

  // Cleanup
  delete[] dmap_v;
  delete[] grouped;
  delete[] count;
  delete[] c_in;
  return true;
}


// get view image, lores version
coco::gsl_image *coco::coco_vtv_sr_get_view_lores( coco_vtv_data *data, size_t nview )
{
  // check for required data 
  assert( data != NULL );
  coco_vtv_sr_data *sr = data->_sr_data;
  assert( sr != NULL );
  size_t w = sr->_w;
  size_t h = sr->_h;
  assert( w*h > 0 );
  assert( nview < sr->_nviews );
  coco_vtv_sr_view_data *view = sr->_views[nview];

  // copy view image
  gsl_image *I = gsl_image_alloc( sr->_w, sr->_h );
  if ( data->_nchannels == 1 ) {
    cuda_memcpy( I->_r, view->_image_f );
    cuda_memcpy( I->_g, view->_image_f );
    cuda_memcpy( I->_b, view->_image_f );
  }
  else if ( data->_nchannels == 3 ) {
    cuda_memcpy( I->_r, view->_image_f + 0*w*h );
    cuda_memcpy( I->_g, view->_image_f + 1*w*h );
    cuda_memcpy( I->_b, view->_image_f + 2*w*h );
  }
  else {
    TRACE( "Unsupported number of image channels" << endl );
    assert( data->_nchannels > 0 );
    cuda_memcpy( I->_r, view->_image_f );
    cuda_memcpy( I->_g, view->_image_f );
    cuda_memcpy( I->_b, view->_image_f );
  }

  return I;
}

using namespace coco;

// write cuda array to image file, unsigned version
bool write_clamped_test_image_unsigned( size_t W, size_t H, float *data,
					const string &spattern, int hint,
					bool normalize, float vmin, float vmax )
{
  char str[500];
  sprintf( str, spattern.c_str(), hint );
  size_t N = W*H;
  assert( N>0 );
  float *cpu = new float[N];
  CUDA_SAFE_CALL( cudaMemcpy( cpu, data, sizeof(float) * N, cudaMemcpyDeviceToHost ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  gsl_matrix *M = gsl_matrix_alloc( H,W );
  for ( size_t i=0; i<N; i++ ) {
    M->data[i] = clamp( cpu[i], vmin, vmax );
  }
  delete[] cpu;
  gsl_image *I = gsl_image_alloc( W,H );
  gsl_image_from_matrix( I, M );
  if ( normalize ) {
    gsl_image_normalize( I );
  }
  gsl_image_save( str, I );
  gsl_image_free( I );
  return true;
}


// Compute the averaged image as a starting point for the algorithm
bool coco::coco_vtv_sr_compute_averaged_forward_warp( coco_vtv_data *data )
{
  // check for required data 
  assert( data != NULL );
  coco_vtv_sr_data *sr = data->_sr_data;
  assert( sr != NULL );
  size_t W = data->_W;
  size_t H = data->_H;
  assert( W*H > 0 );
  coco_vtv_workspace *w = data->_workspace;

  // Temp buffers
  float *hires_tmp = w->_G[0];

  // Clear target image
  for ( size_t i=0; i<data->_nchannels; i++ ) {
    CUDA_SAFE_CALL( cudaMemset( w->_U[i], 0, w->_nfbytes ));
  }

  // Add view contributions via backwards warp
  for ( size_t nview=0; nview < sr->_views.size(); nview++ ) {
    coco_vtv_sr_view_data *view = sr->_views[nview];

    // Add view contributions via backwards warp
    for ( size_t i=0; i<data->_nchannels; i++ ) {
      CUDA_SAFE_CALL( cudaMemset( hires_tmp, 0, w->_nfbytes ));

      // Upsample view in V
      cuda_upsample_matrix_device<<< w->_dimGrid, w->_dimBlock >>>
        ( W,H, sr->_w, sr->_h, sr->_dsf, view->_image_f + i*sr->_w*sr->_h, hires_tmp );
      
      int seg_start = 0;
      for ( size_t j=0; j<view->_seg_end.size(); j++ ) {
        int seg_end = view->_seg_end[j];
        int seg_size = seg_end - seg_start;
        
        // forward warp call for this segment, cannot overlap
        int seg_width = cuda_default_block_size_x() * cuda_default_block_size_y();
        dim3 dimBlock = dim3( seg_width, 1 );
        dim3 dimGrid = dim3( seg_size / seg_width + 1, 1 );
        
        vtv_sr_forward_warp_accumulate_device<<< dimGrid, dimBlock >>>
          ( data->_W, data->_H, seg_width,
          hires_tmp,
          view->_cells, seg_start, seg_end,
          view->_dmap_v, view->_dx_vu, view->_dy_vu,
          view->_vmask[i],
          w->_U[i], 0 );

        CUDA_SAFE_CALL( cudaThreadSynchronize() );
        seg_start = seg_end;
      }
    }
  }
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // Add view contributions via backwards warp
  for ( size_t i=0; i<data->_nchannels; i++ ) {
    // Normalize
    cuda_normalize_device<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H, w->_U[i], sr->_target_mask[i] );
  }

  /*
  TRACE("Write starting image for the algorithm" << endl);
  gsl_image *I = gsl_image_alloc( W,H );
  vector<gsl_matrix*> IC = gsl_image_get_channels( I );
  coco_vtv_get_solution( data, IC );
  gsl_image_save( data->_basedir + "u_gold_init.png", I );
  gsl_image_free( I );
  write_pfm_image_signed(W, H, w->_U[0], w->_U[1], w->_U[2], data->_basedir + "u_gold_init.pfm", 0);
  */

  return true;
}

bool coco::coco_vtv_sr_init_regularizer_weight( coco_vtv_data *data ){
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
            sr->_target_mask[i], w->_g[i] );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

//    write_pfm_image_signed( W, H, w->_g[i], data->_basedir + "g.pfm", 0 );

    // Convolve weight
    cuda_convolution( sr->_vmask_filter, W,H, w->_g[i], w->_temp[0] );
    CUDA_SAFE_CALL( cudaMemcpy( w->_g[i], w->_temp[0], w->_nfbytes, cudaMemcpyDeviceToDevice ));

    // Write final mask and result as a reference
    if ( traceLevel() > 2 ) {
      write_test_image_unsigned( W, H, w->_g[i], data->_basedir + "regweight.png", 0 );
    }
  }
//  TRACE("data->_lambda_max_factor * data->_lambda: " << data->_lambda_max_factor * data->_lambda << endl);

//  write_pfm_image_signed( W, H, sr->_target_mask[0], sr->_target_mask[1], sr->_target_mask[2], data->_basedir + "target_mask.pfm", 0 );

  return true;
}



// Compute the forward warp of a view to test coord conversion matrix
bool coco::coco_vtv_sr_compute_forward_warp( coco_vtv_data *data, size_t nview )
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
  float *hires_tmp = w->_temp[0];
  float *temp2 = w->_temp[1];
  
  // Add view contributions via backwards warp
  for ( size_t i=0; i<data->_nchannels; i++ ) {
    // Clear target image and mask
    CUDA_SAFE_CALL( cudaMemset( temp2, 0, w->_nfbytes ));
    CUDA_SAFE_CALL( cudaMemset( w->_U[i], 0, w->_nfbytes ));
    
    // Upsample view in V
    cuda_upsample_matrix_device<<< w->_dimGrid, w->_dimBlock >>>
    ( data->_W, data->_H, sr->_w, sr->_h, sr->_dsf,
      view->_image_f + i*sr->_w*sr->_h, hires_tmp );
    
    // Forward warp, non-overlap regions sequentially
    int seg_start = 0;
    for ( size_t j=0; j<view->_seg_end.size(); j++ ) {
      int seg_end = view->_seg_end[j];
      int seg_size = seg_end - seg_start;
      
      // forward warp call for this segment, cannot overlap
      int seg_width = cuda_default_block_size_x() * cuda_default_block_size_y();
      dim3 dimBlock = dim3( seg_width, 1 );
      dim3 dimGrid = dim3( seg_size / seg_width + 1, 1 );

      vtv_sr_forward_warp_accumulate_device<<< dimGrid, dimBlock >>>
          ( data->_W, data->_H, seg_width,
              hires_tmp,
              view->_cells, seg_start, seg_end,
              view->_dmap_v, view->_dx_vu, view->_dy_vu,
              view->_vmask[i],
              w->_U[i], temp2 );
      CUDA_SAFE_CALL( cudaThreadSynchronize() );
      seg_start = seg_end;
    }
  
    // Normalize
    cuda_normalize_device<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H, w->_U[i], temp2 );
  }

  return true;
}




// Perform one single shrinkage step (ISTA)
bool coco::vtv_sr_dataterm_gradient( coco_vtv_data *data )
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

  // Compute gradient of data term
  profiler()->beginTask( "gradient" );
  for ( size_t i=0; i<data->_nchannels; i++ ) {
    // Clear derivative
    CUDA_SAFE_CALL( cudaMemset( w->_G[i], 0, w->_nfbytes ));
  }
  float *hires_tmp = w->_temp[data->_nchannels+0];
  float *lores_tmp = w->_temp[data->_nchannels+1];

  // Sum contributions for all views
  gsl_image *itmp_hi = gsl_image_alloc( data->_W, data->_H );
  for ( size_t nview=0; nview < sr->_views.size(); nview++ ) {
    coco_vtv_sr_view_data *view = sr->_views[nview];

    profiler()->beginTask( "warp v->u" );
    for ( size_t i=0; i<data->_nchannels; i++ ) {

      vtv_sr_warp_view_device<<< w->_dimGrid, w->_dimBlock >>>
        ( data->_W, data->_H, w->_Uq[i],
        view->_dmap_v, view->_dx_vu, view->_dy_vu, view->_visibility_mask, hires_tmp );
      CUDA_SAFE_CALL( cudaThreadSynchronize() );

      /*
      write_test_image_unsigned( W,H, w->_U[i], "./out/U_in_%03i.png", i, true );
      write_test_image_unsigned( W,H, w->_Uq[i], "./out/Uq_in_%03i.png", i, true );
      write_test_image_unsigned( W,H, hires_tmp, "./out/warp_%03i.png", i, true );
      */
      
      if ( w->_iteration == 99 && nview == 0) {
        cuda_memcpy( gsl_image_get_channel( itmp_hi,(gsl_image_channel)i ), hires_tmp );
      }

      // downsample
      vtv_sr_downsample_view_device<<< sr->_dimGrid, sr->_dimBlock >>>
        ( sr->_w, sr->_h, data->_W, hires_tmp, view->_visibility_mask, sr->_dsf, lores_tmp );
      CUDA_SAFE_CALL( cudaThreadSynchronize() );
      
      /*
      write_test_image_unsigned( sr->_w,sr->_h, lores_tmp, "./out/lotmp_%03i.png", i, true );
      */

      // subtract image
      cuda_subtract_from_device<<< sr->_dimGrid, sr->_dimBlock >>>
        ( sr->_w, sr->_h, view->_image_f + i*sr->_w*sr->_h, lores_tmp );

      /*
      write_test_image_unsigned( sr->_w,sr->_h, view->_image_f + i*sr->_w+i*sr->_h, "./out/img_%03i.png", i, true );
      write_test_image_unsigned( sr->_w,sr->_h, lores_tmp, "./out/lotmp_sub_%03i.png", i, true );
      */

      // masked by vmask_lo : apply the weights
      cuda_multiply_with_device<<< sr->_dimGrid, sr->_dimBlock >>>
        ( sr->_w, sr->_h, lores_tmp, view->_vmask_lo[i] );

      //write_test_image_unsigned( sr->_w,sr->_h, lores_tmp, "./out/lotmp_sub_masked_%03i.png", i, true );

      // upsample
      vtv_sr_ds_transpose_view_device<<< sr->_dimGrid, sr->_dimBlock >>>
        ( sr->_w, sr->_h, data->_W, lores_tmp, view->_vmask[i], view->_vmask_lo[i], sr->_dsf, w->_temp[i]/*hires_buffer[i]*/ );
      CUDA_SAFE_CALL( cudaThreadSynchronize() );
    }
    profiler()->endTask( "warp v->u" );

    if ( w->_iteration == 99 && nview==0 && traceLevel() > 1 ) {
      gsl_image_save( data->_basedir + "99_backward_warp.png", itmp_hi );
      write_test_image_unsigned( W, H, view->_vmask[0], data->_basedir + "99_mask.png", 0 );
      write_clamped_test_image_unsigned( W, H, view->_dmap_v, data->_basedir + "99_dmap_v.png", 0, true, -10.0f, 10.0f );
    }

    /*
    write_test_image_rgb( sr->_w, sr->_h, view->_vmask_lo[0], view->_vmask_lo[1], view->_vmask_lo[2], "./out/vmask_%03i.png", nview, true );
    write_test_image_rgb( W, H, w->_temp[0], w->_temp[1], w->_temp[2], "./out/HB_%03i.png", nview, true );
    write_test_image_rgb( W, H, view->_vmask[0], view->_vmask[1], view->_vmask[2], "./out/HB_%03i.png", nview, true );
    write_clamped_test_image_unsigned( W, H, view->_dmap_v, "./out/dmap_%03i.png", 0, true, -10.0f, 10.0f );
    */

    // copy new warp matrix to GPU
    profiler()->beginTask( "warp u->v" );
    for ( size_t i=0; i<data->_nchannels; i++ ) {

      int seg_start = 0;
      for ( size_t j=0; j<view->_seg_end.size(); j++ ) {
        int seg_end = view->_seg_end[j];
        int seg_size = seg_end - seg_start;
	
        // forward warp call for this segment, cannot overlap
        int seg_width = cuda_default_block_size_x() * cuda_default_block_size_y();
        dim3 dimBlock = dim3( seg_width, 1 );
        dim3 dimGrid = dim3( seg_size / seg_width + 1, 1 );
	
        vtv_sr_forward_warp_accumulate_weighted_device<<< dimGrid, dimBlock >>>
          ( data->_W, data->_H, seg_width,
	    w->_temp[i]/*hires_buffer[i]*/,
	    view->_cells, seg_start, seg_end,
	    view->_dmap_v, view->_dx_vu, view->_dy_vu,
	    view->_visibility_mask,
	    w->_G[i], 0 );

        CUDA_SAFE_CALL( cudaThreadSynchronize() );
        seg_start = seg_end;
      }
    }
    profiler()->endTask( "warp u->v" );

  }
  profiler()->endTask( "gradient" );
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  // Normalize
  // Note: this is a detail skipped in the paper.
  // For optimization, it is more stable to divide both
  // the data term as well as the regularizer by the number
  // of contributing views, so that the gradient values
  // are more uniform.
  for ( size_t i=0; i<data->_nchannels; i++ ) {
    cuda_normalize_device<<< w->_dimGrid, w->_dimBlock >>>
      ( W, H, w->_G[i], sr->_target_mask[i] );
    cuda_clamp_device<<< w->_dimGrid, w->_dimBlock >>>
      ( W, H, w->_G[i], -1.0f, 1.0f );
  }

  /*
  write_test_image_rgb( W, H, w->_G[0], w->_G[1], w->_G[2], "./out/G_%03i.png", w->_iteration, true );
  write_test_image_rgb( W, H, w->_U[0], w->_U[1], w->_U[2], "./out/U_%03i.png", w->_iteration, true );
  write_test_image_rgb( W, H, w->_Uq[0], w->_Uq[1], w->_Uq[2], "./out/Uq_%03i.png", w->_iteration, true );
  */

  gsl_image_free( itmp_hi );
  //assert( false );
  return true;
}

/*

// Optimize the disparity maps using the current solution, then
// re-initialize algorithm with new ones
bool coco::vtv_sr_optimize_disparity_maps( coco_vtv_data *data )
{
  assert( data != NULL );
  coco_vtv_workspace *w = data->_workspace;
  coco_vtv_sr_data *sr = data->_sr_data;
  assert( sr != NULL );
  float *hires_tmp = w->_temp[0];

  profiler()->beginTask( "optimize dmaps" );
  for ( size_t nview=0; nview < sr->_views.size(); nview++ ) {
    coco_vtv_sr_view_data *view = sr->_views[nview];
    
    // For now, just optimize for one channel (assume BW)
    // Upsample view image
    cuda_upsample_matrix_device<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H, sr->_w, sr->_h, sr->_dsf,
        &(view->_image_f[0*sr->_w*sr->_h]), hires_tmp );

    // compute optimized disparity map
    float dx = view->_dx_vu;
    float dy = view->_dy_vu;
    float drange = sr->_disp_max / max( 1.0f, max( dx,dy ));
    vtv_sr_optimize_dmap_device<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H, drange,
	hires_tmp, w->_U[0],
	view->_dmap_v, dx, dy, view->_visibility_mask,
	view->_dmap_v );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    // vote for u_dmap with new disparity values
    vtv_sr_init_u_dmap( data, nview );
  }
  profiler()->endTask( "optimize dmaps" );

  // update target disparity map with new votes
  // updates visibility and forward warp structures
  coco_vtv_sr_end_view_creation( data );

  return true;
}


// Compute dataterm gradient with pre-initialized optimized depth map
bool coco::vtv_sr_dataterm_gradient_dopt( coco_vtv_data *data )
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
  profiler()->beginTask( "gradient" );
  vector<float*> hires_buffer;
  for ( size_t i=0; i<data->_nchannels; i++ ) {
    float *tmp = NULL;
    CUDA_SAFE_CALL( cudaMalloc( &tmp, w->_nfbytes ));
    hires_buffer.push_back( tmp );
    // Clear derivative
    CUDA_SAFE_CALL( cudaMemset( w->_G[i], 0, w->_nfbytes ));
  }
  float *hires_tmp = w->_temp[0];
  float *lores_tmp = w->_temp[1];
  float *dopt = w->_temp[2];

  // Sum contributions for all views
  for ( size_t nview=0; nview < sr->_views.size(); nview++ ) {
    coco_vtv_sr_view_data *view = sr->_views[nview];

    profiler()->beginTask( "warp v->u" );
    
    for ( size_t i=0; i<data->_nchannels; i++ ) {

      // Upsample view image
      cuda_upsample_matrix_device<<< w->_dimGrid, w->_dimBlock >>>
        ( data->_W, data->_H, sr->_w, sr->_h, sr->_dsf,
          &(view->_image_f[i*sr->_w*sr->_h]), hires_tmp );

      // compute optimized disparity map
      float dx = view->_dx_vu;
      float dy = view->_dy_vu;
      float drange = sr->_disp_max / max( 1.0f, max( dx,dy ));
      vtv_sr_optimize_dmap_device<<< w->_dimGrid, w->_dimBlock >>>
	( data->_W, data->_H, drange,
	  hires_tmp, w->_Uq[i],
	  view->_dmap_v, dx, dy, view->_visibility_mask,
	  dopt );
      CUDA_SAFE_CALL( cudaThreadSynchronize() );

      // warp with optimized map
      vtv_sr_warp_view_device<<< w->_dimGrid, w->_dimBlock >>>
	( data->_W, data->_H, w->_Uq[i],
	  dopt, view->_dx_vu, view->_dy_vu, view->_visibility_mask, hires_tmp );
      CUDA_SAFE_CALL( cudaThreadSynchronize() );
      
      // downsample
      vtv_sr_downsample_view_device<<< sr->_dimGrid, sr->_dimBlock >>>
	( sr->_w, sr->_h, data->_W, hires_tmp, view->_visibility_mask, sr->_dsf, lores_tmp );
      CUDA_SAFE_CALL( cudaThreadSynchronize() );
      
      // subtract image
      cuda_subtract_from_device<<< sr->_dimGrid, sr->_dimBlock >>>
  ( sr->_w, sr->_h, &(view->_image_f[i*sr->_w*sr->_h]), lores_tmp );

      // masked by vmask_lo
      cuda_multiply_with_device<<< sr->_dimGrid, sr->_dimBlock >>>
	( sr->_w, sr->_h, lores_tmp, view->_vmask_lo[i] );

      // upsample
      vtv_sr_ds_transpose_view_device<<< sr->_dimGrid, sr->_dimBlock >>>
	( sr->_w, sr->_h, data->_W, lores_tmp, view->_vmask[i], view->_vmask_lo[i], sr->_dsf, hires_buffer[i] );
      CUDA_SAFE_CALL( cudaThreadSynchronize() );
    }
    profiler()->endTask( "warp v->u" );
      
    // copy new warp matrix to GPU
    profiler()->beginTask( "warp u->v" );
    for ( size_t i=0; i<data->_nchannels; i++ ) {

      int seg_start = 0;
      for ( size_t j=0; j<view->_seg_end.size(); j++ ) {
	int seg_end = view->_seg_end[j];
	int seg_size = seg_end - seg_start;
	
	// forward warp call for this segment, cannot overlap
	int seg_width = cuda_default_block_size_x() * cuda_default_block_size_y();
	dim3 dimBlock = dim3( seg_width, 1 );
	dim3 dimGrid = dim3( seg_size / seg_width + 1, 1 );
	
	vtv_sr_forward_warp_accumulate_weighted_device<<< dimGrid, dimBlock >>>
	  ( data->_W, data->_H, seg_width,
	    hires_buffer[i],
	    view->_cells, seg_start, seg_end,
	    dopt, view->_dx_vu, view->_dy_vu,
	    view->_visibility_mask,
	    w->_G[i], 0 );

	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	seg_start = seg_end;
      }
    }
    profiler()->endTask( "warp u->v" );

  }
  profiler()->endTask( "gradient" );
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  for ( size_t i=0; i<data->_nchannels; i++ ) {
    CUDA_SAFE_CALL( cudaFree( hires_buffer[i] ));
  }

  // Normalize
  for ( size_t i=0; i<data->_nchannels; i++ ) {
    cuda_normalize_device<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H, w->_G[i], sr->_target_mask[i] );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
  }

  return true;
}


*/

// Perform one single shrinkage step (ISTA)
bool coco::vtv_sr_ista_step( coco_vtv_data *data )
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
  vtv_sr_dataterm_gradient( data );

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


// Compute primal energy
double coco::coco_vtv_sr_primal_energy( coco_vtv_data *data )
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
  float *lores_tmp = w->_temp[1]; // image for residu computation
  float *lores_energy =  w->_temp[2];

  CUDA_SAFE_CALL( cudaMemset( lores_energy, 0, sr->_nfbytes_lo ));

  // Sum contributions for all views
  for ( size_t nview=0; nview < sr->_views.size(); nview++ ) {
    coco_vtv_sr_view_data *view = sr->_views[nview];

    for ( size_t i=0; i<data->_nchannels; i++ ) {
      CUDA_SAFE_CALL( cudaMemset( hires_tmp, 0, sizeof(float) * data->_W * data->_H ));

      // warp current solution
      vtv_sr_warp_view_device<<< w->_dimGrid, w->_dimBlock >>>
        ( data->_W, data->_H, w->_U[i],
        view->_dmap_v, view->_dx_vu, view->_dy_vu, view->_visibility_mask, hires_tmp );
      CUDA_SAFE_CALL( cudaDeviceSynchronize() );

      // downsample
      vtv_sr_downsample_view_device<<< sr->_dimGrid, sr->_dimBlock >>>
        ( sr->_w, sr->_h, data->_W, hires_tmp, view->_visibility_mask, sr->_dsf, lores_tmp );
      CUDA_SAFE_CALL( cudaDeviceSynchronize() );

      // subtract image from input float image
      cuda_subtract_from_device<<< sr->_dimGrid, sr->_dimBlock >>>
        ( sr->_w, sr->_h, view->_image_f + i*sr->_w*sr->_h, lores_tmp );
      CUDA_SAFE_CALL( cudaDeviceSynchronize() );

      // square the result
      cuda_multiply_with_device<<< sr->_dimGrid, sr->_dimBlock >>>
        ( sr->_w, sr->_h, lores_tmp, lores_tmp );

      // apply the weights
      cuda_multiply_with_device<<< sr->_dimGrid, sr->_dimBlock >>>
        ( sr->_w, sr->_h, lores_tmp, view->_vmask_lo[i] );

      // cumulate energy
      cuda_add_to_device<<< sr->_dimGrid, sr->_dimBlock >>>
        ( sr->_w, sr->_h, lores_tmp, lores_energy);
      CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    }
  }

  float *E_DATA = new float[ sr->_w * sr->_h ];
  CUDA_SAFE_CALL( cudaMemcpy( E_DATA, lores_energy, sr->_nfbytes_lo, cudaMemcpyDeviceToHost ));
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


// Perform one iteration of Algorithm 1, Chambolle-Pock
bool coco::coco_vtv_sr_iteration_fista( coco_vtv_data *data )
{
  // Todo: verify correct maximum step sizes.
  data->_tau = 0.3 / sqrt( 8.0 );
  data->_sigma = 0.3 / sqrt( 8.0 );
  data->_L = 1.0 / data->_lambda; //float(data->_sr_data->_nviews) / data->_lambda ;
  //data->_L = float(data->_sr_data->_nviews) / data->_lambda ;
  vtv_sr_ista_step( data );
  cuflt alpha_new = 0.5 * ( 1.0 + sqrt( 1.0 + 4.0 * pow( data->_alpha, 2.0 ) ));
  coco_vtv_rof_overrelaxation( data, ( data->_alpha - 1.0 ) / alpha_new );
  data->_alpha = alpha_new;
  data->_workspace->_iteration ++;
  return true;
}

// Perform one iteration of Algorithm 1, Chambolle-Pock
bool coco::coco_vtv_sr_iteration_chambolle_pock_1( coco_vtv_data *data )
{
  // CURRENTLY BUGGED, WRONG REGULARIZATION WEIGHT
  assert( false );

  // Todo: verify correct maximum step sizes.
  data->_tau = 1.0 / 8.0;
  data->_sigma = 1.0 / 8.0;
  coco_vtv_rof_dual_step( data );

  // set lambda
  coco_vtv_workspace *w = data->_workspace;
  if ( w->_g[0] == NULL ) {
    CUDA_SAFE_CALL( cudaMalloc( &w->_g[0], w->_nfbytes ));
    cuda_set_all_device<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H, w->_g[0], data->_lambda );
  }
  w->_iteration++;

  // Primal descent step for data term
  vtv_sr_dataterm_gradient( data );

  // Primal descent step for TV
  for ( size_t i=0; i<data->_nchannels; i++ ) {
    tv_primal_descent_step_device<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H,
	data->_tau,
	w->_U[i], w->_Uq[i], w->_X1[i], w->_X2[i] );
    cuda_add_scaled_to_device<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H,
	w->_G[i], -data->_tau, w->_Uq[i] );

    // write_test_image_signed( data->_W,data->_H, w->_G[i], data->_basedir + "G_%05i.png", i );
    // write_test_image_signed( data->_W,data->_H, w->_Uq[i], data->_basedir + "Uq_%05i.png", i );

  }
    
  // gradient step for U, store in Uq
  // overrelaxation, store extragradient in Uq, new solution in U
  coco_vtv_rof_overrelaxation( data, 1.0 );

  //for ( size_t i=0; i<data->_nchannels; i++ ) {
  //  write_test_image_signed( data->_W,data->_H, w->_Uq[i], data->_basedir + "U_%05i.png", i );
  //}


  return true;
}


/*
// Perform one iteration of Algorithm 1, Chambolle-Pock
bool coco::coco_vtv_sr_dmap_iteration_chambolle_pock_1( coco_vtv_data *data )
{
  // Todo: verify correct maximum step sizes.
  data->_tau = 1.0 / 8.0;
  data->_sigma = 1.0 / 8.0;
  coco_vtv_rof_dual_step( data );

  // set lambda
  coco_vtv_workspace *w = data->_workspace;
  if ( w->_g[0] == NULL ) {
    CUDA_SAFE_CALL( cudaMalloc( &w->_g[0], w->_nfbytes ));
    cuda_set_all_device<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H, w->_g[0], data->_lambda );
  }
  w->_iteration++;

  // Primal descent step for data term
  vtv_sr_dataterm_gradient_dopt( data );

  // Primal descent step for TV
  for ( size_t i=0; i<data->_nchannels; i++ ) {
    tv_primal_descent_step_device<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H,
	data->_tau,
	w->_U[i], w->_Uq[i], w->_X1[i], w->_X2[i] );
    cuda_add_scaled_to_device<<< w->_dimGrid, w->_dimBlock >>>
      ( data->_W, data->_H,
	w->_G[i], -data->_tau, w->_Uq[i] );


    //write_test_image_signed( data->_W,data->_H, w->_G[i], data->_basedir + "G_%05i.png", i );
    //write_test_image_signed( data->_W,data->_H, w->_Uq[i], data->_basedir + "Uq_%05i.png", i );

  }
    
  // gradient step for U, store in Uq
  // overrelaxation, store extragradient in Uq, new solution in U
  coco_vtv_rof_overrelaxation( data, 1.0 );

  //for ( size_t i=0; i<data->_nchannels; i++ ) {
  //  write_test_image_signed( data->_W,data->_H, w->_Uq[i], data->_basedir + "U_%05i.png", i );
  //}


  return true;
}

// Filter the visibility mask 6 times (hard-coded)
// Add the nine neighbors values -> sum: 
// if sum<4 -> 0
// if sum==4 -> keep current value
// if sum>4 -> 1
bool coco::vtv_sr_filter_mask( coco_vtv_data *data, int *mask )
{
  // check for required data 
  assert( data != NULL );
  coco_vtv_sr_data *sr = data->_sr_data;
  assert( sr != NULL );
  size_t W = data->_W;
  size_t H = data->_H;
  assert( W*H > 0 );

  // buffer memory
  int *buffer = NULL;
  CUDA_SAFE_CALL( cudaMalloc( &buffer, W*H*sizeof(int) ));

  // perform filtering
  coco_vtv_workspace *w = data->_workspace;
  vtv_sr_filter_mask_device<<< w->_dimGrid, w->_dimBlock >>>
    ( data->_W, data->_H, mask, buffer );
  vtv_sr_filter_mask_device<<< w->_dimGrid, w->_dimBlock >>>
    ( data->_W, data->_H, buffer, mask );
  vtv_sr_filter_mask_device<<< w->_dimGrid, w->_dimBlock >>>
    ( data->_W, data->_H, mask, buffer );
  vtv_sr_filter_mask_device<<< w->_dimGrid, w->_dimBlock >>>
    ( data->_W, data->_H, buffer, mask );
  vtv_sr_filter_mask_device<<< w->_dimGrid, w->_dimBlock >>>
    ( data->_W, data->_H, mask, buffer );
  vtv_sr_filter_mask_device<<< w->_dimGrid, w->_dimBlock >>>
    ( data->_W, data->_H, buffer, mask );

  // cleanup
  //CUDA_SAFE_CALL( cudaMemcpy( mask, buffer, W*H*sizeof(int), cudaMemcpyDeviceToDevice ) );
  CUDA_SAFE_CALL( cudaFree( buffer ));
  return true;
}

// Filter the disparity map
bool coco::vtv_sr_filter_disparity_map( coco_vtv_data *data, float *mask )
{
  // check for required data 
  assert( data != NULL );
  coco_vtv_sr_data *sr = data->_sr_data;
  assert( sr != NULL );
  size_t W = data->_W;
  size_t H = data->_H;
  assert( W*H > 0 );

  // upsample to hires DMAP
  float *buffer = NULL;
  CUDA_SAFE_CALL( cudaMalloc( &buffer, W*H*sizeof(float) ));



  CUDA_SAFE_CALL( cudaFree( buffer ));
  return true;
}*/
