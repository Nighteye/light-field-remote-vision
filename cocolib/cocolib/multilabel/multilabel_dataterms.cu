/* -*-c++-*- */
#include "multilabel_dataterms.h"
#include "multilabel.cuh"
#include "vectorial_multilabel.cuh"
#include "../optic_flow/vtv_l1_optical_flow.h"

#include "cuda.h"

#include "../defs.h"
#include "../cuda/cuda_helper.h"
#include "../cuda/cuda_interface.h"
#include "../cuda/cuda_kernels.cuh"
#include "../cuda/cuda_inline_device_functions.cu"
#include "../common/gsl_image.h"

using namespace std;

// Compute data term for multilabel segmentation (equidistant)
float *coco::multilabel_dataterm_segmentation( multilabel_data* data,
					       const gsl_matrix *image )
{
  // Set up 3D structure
  multilabel_workspace *w = data->_w;
  float *rho = new float[ w->_nfbytes ];
  size_t index = 0;
  for ( size_t y=0; y<data->_H; y++ ) {
    for ( size_t x=0; x<data->_W; x++ ) {
      float v = image->data[index];
      for ( size_t g=0; g<data->_G; g++ ) {
	size_t p = index + g*data->_N;
	rho[p] = fabs( data->_labels[g] - v );
      }
      index++;
    }
  }

  return rho;
}



////////////////////////////////////////////////////////////////////////////////
// Stereo dataterm from two images, 1 layer
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_stereo_dataterm_rgb_layer( int W, int H,
						cuflt lambda,
						cuflt dx, cuflt dy,
						cuflt *r0, cuflt *g0, cuflt* b0,
						cuflt *r1, cuflt *g1, cuflt* b1,
						cuflt *rho,
						cuflt *count )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;

  // Bilinear interpolation at target location
  float vx = float(ox) + dx;
  float vy = float(oy) + dy;
  if ( vx<0.0f || vx>float(W-1) || vy<0.0f || vy>float(H-1)) {
    return;
  }

  float r1v = bilinear_interpolation( W,H, r1, vx,vy );
  float g1v = bilinear_interpolation( W,H, g1, vx,vy );
  float b1v = bilinear_interpolation( W,H, b1, vx,vy );
  // Compute error term for this label
  r1v = fabs( r0[o]-r1v );
  g1v = fabs( g0[o]-g1v );
  b1v = fabs( b0[o]-b1v );

  //float err = max( r_err, max( g_err, b_err ));
  rho[o] += lambda * (r1v*r1v + g1v*g1v + b1v*b1v);
  if ( count != NULL ) {
    count[o] += 1.0f;
  }
}



vector<float*> gsl_image_to_gpu( const coco::gsl_image *I )
{
  vector<float*> channels;
  size_t N = I->_w * I->_h;
  assert( N>0 );
  float *r = coco::cuda_alloc_floats( N );
  cuda_memcpy( r, I->_r );
  channels.push_back( r );
  float *g = coco::cuda_alloc_floats( N );
  cuda_memcpy( g, I->_g );
  channels.push_back( g );
  float *b = coco::cuda_alloc_floats( N );
  cuda_memcpy( b, I->_b );
  channels.push_back( b );  
  return channels;
}

bool gpu_image_free( vector<float*> &channels )
{
  for ( size_t i=0; i<channels.size(); i++ ) {
    CUDA_SAFE_CALL( cudaFree( channels[i] ));
  }
  channels.clear();
  return true;
}



// Compute data term for multilabel segmentation
// Labels equal disparity values used for matching
float *coco::multilabel_dataterm_stereo( multilabel_data* data,
					 float lambda,
					 gsl_image *im0,
					 gsl_image *im1 )
{
  size_t W = data->_W;
  size_t H = data->_H;
  size_t N = data->_G;

  vector<float*> im0_gpu = gsl_image_to_gpu( im0 );
  vector<float*> im1_gpu = gsl_image_to_gpu( im1 );

  float *rho = new float[ W*H*N ];
  float *rho_gpu = data->_w->_rho;

  dim3 dimBlock;
  dim3 dimGrid;
  cuda_default_grid( W,H, dimGrid, dimBlock );

  CUDA_SAFE_CALL( cudaMemset( rho_gpu, 0, sizeof(float) * W*H*N ));
  for ( size_t i=0; i<N; i++ ) {

    float dx = data->_labels[i];
    float dy = 0.0f;
    float *rho_label = rho_gpu + i*W*H;
    cuda_stereo_dataterm_rgb_layer<<< dimGrid, dimBlock >>>
      ( W,H, 
	lambda, dx, dy,
	im0_gpu[0], im0_gpu[1], im0_gpu[2],
	im1_gpu[0], im1_gpu[1], im1_gpu[2],
	rho_label, NULL );
  }

  CUDA_SAFE_CALL( cudaMemcpy( rho, rho_gpu, W*H*N*sizeof(float), cudaMemcpyDeviceToHost ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  gpu_image_free( im0_gpu );
  gpu_image_free( im1_gpu );
  return rho;
}


// Compute data term for multilabel segmentation
// Labels equal disparity values used for matching
float *coco::multilabel_dataterm_stereo( size_t W, size_t H,
					 const vector<float> &labels,
					 float lambda,
					 const gsl_image *im0,
					 const gsl_image *im1 )
{
  size_t N = labels.size();

  vector<float*> im0_gpu = gsl_image_to_gpu( im0 );
  vector<float*> im1_gpu = gsl_image_to_gpu( im1 );

  float *rho = new float[ W*H*N ];
  float *rho_gpu = NULL;
  CUDA_SAFE_CALL( cudaMalloc( &rho_gpu, W*H*N*sizeof(float) ));
  assert( rho_gpu != NULL );

  dim3 dimBlock;
  dim3 dimGrid;
  cuda_default_grid( W,H, dimGrid, dimBlock );

  CUDA_SAFE_CALL( cudaMemset( rho_gpu, 0, sizeof(float) * W*H*N ));
  for ( size_t i=0; i<N; i++ ) {

    float dx = labels[i];
    float dy = 0.0f;
    float *rho_label = rho_gpu + i*W*H;
    cuda_stereo_dataterm_rgb_layer<<< dimGrid, dimBlock >>>
      ( W,H, 
	lambda, dx, dy,
	im0_gpu[0], im0_gpu[1], im0_gpu[2],
	im1_gpu[0], im1_gpu[1], im1_gpu[2],
	rho_label, NULL );
  }

  CUDA_SAFE_CALL( cudaMemcpy( rho, rho_gpu, W*H*N*sizeof(float), cudaMemcpyDeviceToHost ));
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  CUDA_SAFE_CALL( cudaFree( rho_gpu ));

  gpu_image_free( im0_gpu );
  gpu_image_free( im1_gpu );
  return rho;
}




// Compute RGB color value list, one number per label
// Corresponds to exact dataterm numbering
bool coco::vml_dataterm_segmentation_labels( vml_data *data, vector<color> &list )
{
  size_t G = data->_G;
  list.clear();
  assert( data->_K == 3 );

  for ( size_t g=0; g<G; g++ ) {
    int *label = data->_labels[g];

    float rl = data->_dim[0]->_values[ label[0] ];
    float gl = data->_dim[1]->_values[ label[1] ];
    float bl = data->_dim[2]->_values[ label[2] ];

    list.push_back( color( RGB, rl,gl,bl ));
  }

  return true;
}


// Create data term
float *coco::vml_dataterm_segmentation( vml_data *data,
					gsl_image *I,
					int *rlabels,
					int *glabels,
					int *blabels )
{
  TRACE( "Computing segmentation dataterm [" );

  size_t W = data->_W;
  size_t H = data->_H;
  size_t N = W*H;
  size_t G = data->_G;
  float *a = new float[ G*N ];

  for ( size_t n=0; n<N; n++ ) {
    if ( n % (N/10) == 0 ) {
      TRACE( "." );
    }

    float min_dist = 1e10;
    int *best_label = NULL;
    for ( size_t g=0; g<G; g++ ) {
      float *alayer = a + g * N;
      int *label = data->_labels[g];

      float ri = I->_r->data[n];
      float rl = data->_dim[0]->_values[ label[0] ];

      float gi = I->_g->data[n];
      float gl = data->_dim[1]->_values[ label[1] ];

      float bi = I->_b->data[n];
      float bl = data->_dim[2]->_values[ label[2] ];

      //TRACE( "Label " << g << "  value " << rl << " " << gl << " " << bl << endl );
      //TRACE( "          iv " << ri << " " << gi << " " << bi << endl );

      float dist = pow( fabs(ri-rl), 2.0f ) + pow( fabs(gi-gl), 2.0f ) + pow( fabs(bi-bl), 2.0f );
      //float dist = fabs( ri-rl ) + fabs( gi-gl ) + fabs( bi-bl );
      if ( dist < min_dist ) {
	min_dist = dist;
	best_label = label;
      }

      alayer[n] = dist;
    }

    assert( best_label != NULL );
    rlabels[n] = best_label[0];
    glabels[n] = best_label[1];
    blabels[n] = best_label[2];
  }

  TRACE( "] done." << endl );
  return a;
}


// Convert vectorial labeling result to image
coco::gsl_image *coco::vml_segmentation_result( vml_data *data, int *rlabel, int *glabel, int *blabel )
{
  size_t W = data->_W;
  size_t H = data->_H;
  size_t N = W*H;
  gsl_image *I = gsl_image_alloc( W,H );

  for ( size_t n=0; n<N; n++ ) {
    float rl = data->_dim[0]->_values[ rlabel[n] ];
    float gl = data->_dim[1]->_values[ glabel[n] ];
    float bl = data->_dim[2]->_values[ blabel[n] ];
    I->_r->data[n] = rl;
    I->_g->data[n] = gl;
    I->_b->data[n] = bl;
  }

  return I;
}


// Convert linear labeling result to RGB image, given label color list
coco::gsl_image *coco::vml_segmentation_result( size_t W, size_t H, const vector<color> &colors, int *labels )
{
  size_t N = W*H;
  gsl_image *I = gsl_image_alloc( W,H );

  for ( size_t n=0; n<N; n++ ) {
    const color &c = colors[ labels[n] ];
    I->_r->data[n] = c._r;
    I->_g->data[n] = c._g;
    I->_b->data[n] = c._b;
  }

  return I;
}





// Compute 2D vector list, one per label
// Corresponds to exact dataterm numbering
bool coco::vml_dataterm_opticflow_labels( vml_data *data, vector<Vec2f> &list )
{
  size_t G = data->_G;
  list.clear();

  for ( size_t g=0; g<G; g++ ) {
    int *label = data->_labels[g];

    float u = data->_dim[0]->_values[ label[0] ];
    float v = data->_dim[1]->_values[ label[1] ];

    list.push_back( Vec2f( u,v ));
    TRACE6( "OF label " << g << " " << u << " " << v << endl );
  }

  return true;
}


// Create data term
float *coco::vml_dataterm_opticflow( vml_data *data,
				     gsl_image *im0,
				     gsl_image *im1,
				     double rho,
				     double fx_min, double fx_max,
				     double fy_min, double fy_max,
				     int *ulabels,
				     int *vlabels )
{
  if ( data->_K != 2 ) {
    ERROR( "optic flow data term requires 2-dimensional label space." << endl );
    assert( false );
    return NULL;
  }
  vml_dimension_data_set_label_range( data->_dim[0], fx_min, fx_max );
  vml_dimension_data_set_label_range( data->_dim[1], fy_min, fy_max );


  TRACE( "Computing optic flow dataterm [" );

  vector<Vec2f> vecs;
  vml_dataterm_opticflow_labels( data, vecs );

  size_t W = data->_W;
  size_t H = data->_H;
  size_t N = W*H;
  size_t G = data->_G;
  float *a = new float[ G*N ];

  for ( size_t y=0; y<H; y++ ) {
    if ( y % (H/10) == 0 ) {
      TRACE( "." );
    }

    for ( size_t x=0; x<W; x++ ) {
      int n = x + y*W;

      double r1,g1,b1;
      gsl_image_get_color_interpolate( im0, x,y, r1,g1,b1 );

      float min_dist = 1e10;
      int *best_label = NULL;
      Vec2f best_vec;
      for ( size_t g=0; g<G; g++ ) {
	float *alayer = a + g * N;
	int *label = data->_labels[g];
	Vec2f vec = vecs[g];

	double r2,g2,b2;
	gsl_image_get_color_interpolate( im1, x+vec.x, y+vec.y, r2,g2,b2 );

	// Compute error term for this label
	float r_err = fabs( r1-r2 );
	float g_err = fabs( g1-g2 );
	float b_err = fabs( b1-b2 );
	//float err = max( r_err, max( g_err, b_err ));
	float dist = r_err + g_err + b_err;
	dist *= rho;

	if ( dist < min_dist ) {
	  min_dist = dist;
	  best_label = label;
	  best_vec = vec;
	}

	alayer[ n ] = dist;
      }

      assert( best_label != NULL );
      ulabels[n] = best_label[0];
      vlabels[n] = best_label[1];


      // TEST: COMPLETELY SMOOTH FIELD
      /*
      for ( size_t g=0; g<G; g++ ) {
	float *alayer = a + g * N;
	Vec2f vec = vecs[g];
	float dist = (best_vec - vec).length();
	dist *= rho;
	//TRACE( "dist label " << g << ": " << best_vec << "  ...  " << vec << "    = " << dist << endl );
	alayer[ n ] = dist;
      }
      */
    }
  }

  TRACE( "] done." << endl );
  return a;
}


// Convert vectorial labeling result to image
coco::gsl_image *coco::vml_opticflow_result( vml_data *data, int *ulabel, int *vlabel )
{
  vector<Vec2f> vecs;
  vml_dataterm_opticflow_labels( data, vecs );

  size_t W = data->_W;
  size_t H = data->_H;
  size_t N = W*H;
  int *labels = new int[N];

  for ( size_t n=0; n<N; n++ ) {
    int lv[2];
    lv[0] = ulabel[n];
    lv[1] = vlabel[n];
    labels[n] = vml_label_index( data, lv );
  }

  gsl_image *I = vml_opticflow_result( W,H, vecs, labels );
  delete[] labels;
  return I;
}


// Convert linear labeling result to RGB image, given label color list
coco::gsl_image *coco::vml_opticflow_result( size_t W, size_t H, const vector<Vec2f> &vecs, int *labels )
{
  size_t N = W*H;
  gsl_matrix *u = gsl_matrix_alloc( H,W );
  gsl_matrix *v = gsl_matrix_alloc( H,W );
  for ( size_t i=0; i<N; i++ ) {
    Vec2f vec = vecs[ labels[i] ];    
    u->data[i] = vec.x;
    v->data[i] = vec.y;
  }

  gsl_image *I = flow_field_to_image( u,v );
  gsl_matrix_free( u );
  gsl_matrix_free( v );
  return I;
}





/*****************************************************************************

    Nieuwenhuis and Cremers TPAMI 2012:
    Spatially varying color distributions

    Input:   I      - image on which the scribbles are defined
             masks  - vector of binary masks (one for each label)
             S      - image to be segmented

             sigma - standard deviation of color difference kernel
             alpha - standard deviation scale of distance difference kernel

    Output:  GPU array rho of label costs (layered by label index)
             will be filled (must be allocated on call)

*******************************************************************************/
#define MAX_SCRIBBLES 2000
__constant__ float __scribbles_r[MAX_SCRIBBLES];
__constant__ float __scribbles_g[MAX_SCRIBBLES];
__constant__ float __scribbles_b[MAX_SCRIBBLES];
__constant__ short __scribbles_x[MAX_SCRIBBLES];
__constant__ short __scribbles_y[MAX_SCRIBBLES];


__global__ void compute_dataterm_nieuwenhuis_layer_device
  ( int W, int H,
    float sigma_sq, float alpha_sq,
    int x, int y, float r, float g, float b,
    float *rdata, float *gdata, float *bdata,
    float *rho )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;

  // color contribution
  float rdiff_sq = rdata[o] - r;
  rdiff_sq = rdiff_sq * rdiff_sq;
  float gdiff_sq = gdata[o] - g;
  gdiff_sq = gdiff_sq * gdiff_sq;
  float bdiff_sq = bdata[o] - b;
  bdiff_sq = bdiff_sq * bdiff_sq;

  float cost = 1.0f; //(rdiff_sq + gdiff_sq + bdiff_sq) / sigma_sq; //rsqrtf( 3.14159265f * 2.0f * sigma_sq );
  cost *= expf( -0.5f / sigma_sq * ( rdiff_sq + gdiff_sq + bdiff_sq ));

  // spatial contribution: later
  /*
  float cost = 0.0f;
  if ( x==ox && y==oy ) {
    cost = 1.0f;
  }
  */

  rho[o] += cost;
}


__global__ void compute_dataterm_nieuwenhuis_full_layer_device
  ( int W, int H,
    int nscribbles,
    float sigma_sq, float alpha_sq,
    float *rdata, float *gdata, float *bdata,
    float *rho )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = oy*W + ox;

  float r = rdata[o];
  float g = gdata[o];
  float b = bdata[o];

  // color contribution
  float cost = 0.0f;
  for ( int i=0; i<nscribbles; i++ ) {
    float rdiff_sq = r - __scribbles_r[i];
    rdiff_sq = rdiff_sq * rdiff_sq;
    float gdiff_sq = g - __scribbles_g[i];
    gdiff_sq = gdiff_sq * gdiff_sq;
    float bdiff_sq = b - __scribbles_b[i];
    bdiff_sq = bdiff_sq * bdiff_sq;

    //float cost = 1.0f; //(rdiff_sq + gdiff_sq + bdiff_sq) / sigma_sq; //rsqrtf( 3.14159265f * 2.0f * sigma_sq );
    cost += expf( -0.5f / sigma_sq * ( rdiff_sq + gdiff_sq + bdiff_sq ));
  }

  // spatial contribution: later
  /*
  float cost = 0.0f;
  if ( x==ox && y==oy ) {
    cost = 1.0f;
  }
  */

  rho[o] += cost;
}







bool coco::vml_dataterm_segmentation_nieuwenhuis( gsl_image *I,
						  vector<gsl_matrix*> &masks,
						  gsl_image *S,
						  double sigma,
						  double alpha,
						  float *rho )
{
  // Block sizes
  int W = I->_w;
  int H = I->_h;
  assert( (int)S->_w == W );
  assert( (int)S->_h == H );
  dim3 dimBlock( cuda_default_block_size_x(),
		 cuda_default_block_size_y() );
  size_t blocks_w = W / dimBlock.x;
  if ( W % dimBlock.x != 0 ) {
    blocks_w += 1;
  }
  size_t blocks_h = H / dimBlock.y;
  if ( H % dimBlock.y != 0 ) {
    blocks_h += 1;
  }
  dim3 dimGrid( blocks_w, blocks_h );

  // Copy image to be segmented to GPU
  float *rdata = NULL;
  float *gdata = NULL;
  float *bdata = NULL;
  size_t nbytes = W*H*sizeof(float);
  CUDA_SAFE_CALL( cudaMalloc( &rdata, nbytes ));
  cuda_memcpy( rdata, S->_r );
  CUDA_SAFE_CALL( cudaMalloc( &gdata, nbytes ));
  cuda_memcpy( gdata, S->_g );
  CUDA_SAFE_CALL( cudaMalloc( &bdata, nbytes ));
  cuda_memcpy( bdata, S->_b );

  // Clear data term
  CUDA_SAFE_CALL( cudaMemset( rho, 0, nbytes * masks.size() ));

  //#define METHOD1
#ifdef METHOD1
  // Completely separate for each label
  for ( size_t i=0; i<masks.size(); i++ ) {

    // Extract scribbles for each mask
    gsl_matrix *m = masks[i];
    assert( m->size1 == H );
    assert( m->size2 == W );
    int count = 0;

    for ( size_t y=0; y<H; y++ ) {
      for ( size_t x=0; x<W; x++ ) {

	int n = x + y*W;
	if ( m->data[n] != 0.0 ) {

	  // Add data term contribution for this scribble
	  float r = I->_r->data[n];
	  float g = I->_g->data[n];
	  float b = I->_b->data[n];
	  
	  compute_dataterm_nieuwenhuis_layer_device<<< dimGrid, dimBlock >>>
	    ( W,H,
	      sigma*sigma, alpha*alpha,
	      x,y, r,g,b,
	      rdata, gdata, bdata,
	      rho + W*H*i
	    );

	  ++ count;
	}
      }
    }

    // Finally, scale cost by number of scribbles
    cuda_scale_device<<< dimGrid, dimBlock >>>
      ( W,H, rho + W*H*i, 1.0f / float(count) );
  }

#else

  float *scribbles_r = new float[MAX_SCRIBBLES];
  float *scribbles_g = new float[MAX_SCRIBBLES];
  float *scribbles_b = new float[MAX_SCRIBBLES];
  short *scribbles_x = new short[MAX_SCRIBBLES];
  short *scribbles_y = new short[MAX_SCRIBBLES];

  // Completely separate for each label
  for ( size_t i=0; i<masks.size(); i++ ) {

    // Extract scribbles for each mask
    gsl_matrix *m = masks[i];
    assert( (int)m->size1 == H );
    assert( (int)m->size2 == W );
    int count = 0;
    int scribble = 0;

    for ( int y=0; y<H; y++ ) {
      for ( int x=0; x<W; x++ ) {

	int n = x + y*W;
	if ( m->data[n] != 0.0 ) {

	  // Add data term contribution for this scribble
	  scribbles_r[scribble] = I->_r->data[n];
	  scribbles_g[scribble] = I->_g->data[n];
	  scribbles_b[scribble] = I->_b->data[n];
	  scribbles_x[scribble] = x;
	  scribbles_y[scribble] = y;
	  count++;
	  scribble++;
	}

	if ( scribble == MAX_SCRIBBLES ) {

	  // Copy scribbles
	  CUDA_SAFE_CALL( cudaMemcpyToSymbol( __scribbles_r, scribbles_r, scribble * sizeof(float) ));
	  CUDA_SAFE_CALL( cudaMemcpyToSymbol( __scribbles_g, scribbles_g, scribble * sizeof(float) ));
	  CUDA_SAFE_CALL( cudaMemcpyToSymbol( __scribbles_b, scribbles_b, scribble * sizeof(float) ));
	  CUDA_SAFE_CALL( cudaMemcpyToSymbol( __scribbles_x, scribbles_x, scribble * sizeof(short) ));
	  CUDA_SAFE_CALL( cudaMemcpyToSymbol( __scribbles_y, scribbles_y, scribble * sizeof(short) ));
	  
	  compute_dataterm_nieuwenhuis_full_layer_device<<< dimGrid, dimBlock >>>
	    ( W,H,
	      scribble,
	      sigma*sigma, alpha*alpha,
	      rdata, gdata, bdata,
	      rho + W*H*i
	      );

	  scribble = 0;
	}
      }
    }

    if ( scribble > 0 ) {

      // Copy scribbles
      CUDA_SAFE_CALL( cudaMemcpyToSymbol( __scribbles_r, scribbles_r, scribble * sizeof(float) ));
      CUDA_SAFE_CALL( cudaMemcpyToSymbol( __scribbles_g, scribbles_g, scribble * sizeof(float) ));
      CUDA_SAFE_CALL( cudaMemcpyToSymbol( __scribbles_b, scribbles_b, scribble * sizeof(float) ));
      CUDA_SAFE_CALL( cudaMemcpyToSymbol( __scribbles_x, scribbles_x, scribble * sizeof(short) ));
      CUDA_SAFE_CALL( cudaMemcpyToSymbol( __scribbles_y, scribbles_y, scribble * sizeof(short) ));
	  
      compute_dataterm_nieuwenhuis_full_layer_device<<< dimGrid, dimBlock >>>
	( W,H,
	  scribble,
	  sigma*sigma, alpha*alpha,
	  rdata, gdata, bdata,
	  rho + W*H*i
	  );

      scribble = 0;
    }

    // Finally, scale cost by number of scribbles
    cuda_scale_device<<< dimGrid, dimBlock >>>
      ( W,H, rho + W*H*i, 1.0f / float(count) );
  }

  delete[] scribbles_r;
  delete[] scribbles_g;
  delete[] scribbles_b;
  delete[] scribbles_x;
  delete[] scribbles_y;


#endif

  // Cleanup
  CUDA_SAFE_CALL( cudaFree( rdata ));
  CUDA_SAFE_CALL( cudaFree( gdata ));
  CUDA_SAFE_CALL( cudaFree( bdata ));
  return true;
}
