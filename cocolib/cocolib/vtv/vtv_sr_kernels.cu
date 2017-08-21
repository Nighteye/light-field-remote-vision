/* -*-c++-*- */
static __global__ void cuda_sparse_matrix_multiply_device( int W, int H, int N, int m,
							   float *A,
							   int *A_coord,
							   float *u,
							   float *v )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = ox + oy*W;

  float z = 0.0f;
  for ( int i=0; i<m; i++ ) {
    int oa = i*N + o;
    int cpos = A_coord[ oa ];
    if ( cpos == N ) {
      // done with this row.
      break;
    }
    z += A[ oa ] * u[cpos];
  }

  v[o] = z;
}


static __global__ void cuda_sparse_matrix_average_device( int W, int H, int N, int m,
							  int *A_coord,
							  float *u,
							  float *v,
							  float *cov )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = ox + oy*W;

  float count = 0.0f;
  float z = 0.0f;
  for ( int i=0; i<m; i++ ) {
    int oa = i*N + o;
    int cpos = A_coord[ oa ];
    if ( cpos == N ) {
      // done with this row.
      break;
    }

    z += u[cpos];
    count += 1.0f;
  }
  if ( count != 0.0f ) {
    cov[o] += 1.0f;
    z /= count;
  }

  v[o] = z;
}



// This is a backward map rendering method
// The dmap is in output coordinates and gives the displacement to the u image
// v is in the same resolution as u
// vmaks is used as a binary mask (0 or !0) designing occluded pixels
static __global__ void vtv_sr_warp_view_device( int W, int H,
            const float *u, // input image
            const float *dmap, // input dispariy map corresponding to output image
            float dx, float dy, // optical center displacement
            const bool *visibility_mask, // binary mask corresponding to output image
            float *v ) // output image
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = ox + oy*W;

  // set occluded pixels to black (they have 0 weight anyway)
  if ( visibility_mask[o] == 0 ) {
    v[o] = 0.0f;
    return;
  }

  // get location in u
  float d = dmap[o];
  float uxv = ox + d * dx;
  float uyv = oy + d * dy;
  int px = (int)floor(uxv);
  int py = (int)floor(uyv);
  // if pixels goes outside the image, set it to black
  // although its weight should be 0 (vmask[o])
  if ( px<0 || py<0 || px>W-1 || py>H-1 ) {
    v[o] = 0.0f;
    return;
  }
  int po = px + py*W;
  float ax = uxv - float(px);
  float ay = uyv - float(py);

  // transpose bilinear sampling
  float mxmym = (1.0f - ax) * (1.0f - ay);
  float mxpym = ax * (1.0f - ay);
  float mxmyp = (1.0f - ax) * ay;
  float mxpyp = ax * ay;

  float r = u[ po + 0 ] * mxmym;
  float weight = 1;
  if ( px<W-1 ) {
    r += u[ po + 1 ] * mxpym;
  } else {
    weight -= mxpym;
  }
  if ( py < H-1 ) {
    r += u[ po + W ] * mxmyp;
  } else {
    weight -= mxmyp;
  }
  if ( px<W-1  && py < H-1) {
    r += u[ po + W + 1 ] * mxpyp;
  } else {
    weight -= mxpyp;
  }

  // bilinear sampling (todo: texture lookup)
  v[o] = r/weight;
}

// refine disparity map, comparing along an optimization line
// TODO: optimization interval (-0.1, 0.1) should be a variable
static __global__ void vtv_sr_optimize_dmap_device( int W, int H,
						    float disp_max,
						    float *v,
						    float *u,
						    float *dmap,
						    float dx, float dy,
						    const bool *visibility_mask,
						    float *dmap_opt )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = ox + oy*W;

  // only warp inside mask pixels
  if ( visibility_mask[o] == 0 ) {
    v[o] = 0.0f;
    return;
  }

  // get depth and color in v
  float cv = v[o];
  float dv = dmap[o];
  float dopt = dv;
  float eopt = 1e10;

  // sample along optimization line
  for ( float disp = -0.1; disp <= 0.1; disp += 0.01 ) {
    float dtest = dv + disp*disp_max;
    float uxv = ox + dtest * dx;
    float uyv = oy + dtest * dy;
    float cu = bilinear_interpolation( W,H, u, uxv, uyv );
    float e = fabs(cu-cv);
    if ( e < eopt ) {
      eopt = e;
      dopt = dtest;
    }
  }

  // bilinear sampling (todo: texture lookup)
  dmap_opt[o] = dopt;
}

// warp u into out using the dmap. Each u element contributes to 4 neighboring pixels
// with its corresponding weight vmask.
static __global__ void vtv_sr_forward_warp_accumulate_device( int W, int H, int R,
							      float *u, // input image
							      int *index_array,
							      int start, int end, 
							      float *dmap,
							      float dx, float dy,
							      float *vmask, // input weight image
							      float *out, // u image forward warped and 'auto-blended'
							      float *out_weight ) // has the total weight contributed by u 
{
  // Global thread index
  int idx = R * blockIdx.x + threadIdx.x + start;
  if ( idx>=end ) {
    return;
  }
  int o = index_array[idx];
  int ox = o % W;
  int oy = o / W;

  // only warp inside mask pixels
  float w = vmask[o];
  if ( w == 0.0f ) {
    return;
  }

  // get location in u
  float uv = u[o];
  float d = dmap[o];
  float uxv = ox + d * dx;
  float uyv = oy + d * dy;
  int px = (int)floor(uxv);
  int py = (int)floor(uyv);
  if ( px<0 || py<0 || px>W-1 || py>H-1 ) {
    return;
  }
  int po = px + py*W;
  float ax = uxv - float(px);
  float ay = uyv - float(py);

  float mxmym = (1.0f - ax) * (1.0f - ay);
  float mxpym = ax * (1.0f - ay);
  float mxmyp = (1.0f - ax) * ay;
  float mxpyp = ax * ay;

  out[ po + 0 ] += w * uv * mxmym;
  if ( px < W-1 ) {
    out[ po + 1 ] += w * uv * mxpym;
  }
  if ( py < H-1 ) {
    out[ po + W ] += w * uv * mxmyp;
  }
  if ( px < W-1 && py < H-1 ) {
    out[ po + W + 1 ] += w * uv * mxpyp;
  }
  
  if ( out_weight ) {
    out_weight[ po + 0 ] += w * mxmym;
    if ( px < W-1 ) {
      out_weight[ po + 1 ] += w * mxpym;
    }
    if ( py < H-1 ) {
      out_weight[ po + W ] += w * mxmyp;
    }
    if ( px < W-1 && py < H-1 ) {
      out_weight[ po + W + 1 ] += w * mxpyp;
    }
  }
}

// This method is called "weighted" because the input values u will be considered as being already weigthed
// visibility_mask is only used to check visibility : if not visible the value won't be used)
static __global__ void vtv_sr_forward_warp_accumulate_weighted_device( int W, int H, int R,
                       float *in, // input values already weigthed
                       int *index_array,
                       int start, int end,
                       float *dmap, // disparity map to warp values
                       float dx, float dy,
                       const bool *visibility_mask, // weight mask: value only used if vmask != 0.
                       float *out, //
                       float *out_weight ) // cummulated weights (pointer may be 0 if values not needed)
{
  // Global thread index
  int idx = R * blockIdx.x + threadIdx.x + start;
  if ( idx>=end ) {
    return;
  }
  int o = index_array[idx];
  int ox = o % W;
  int oy = o / W;

  // only warp inside mask pixels
  if ( visibility_mask[o] == 0 ) {
    return;
  }

  // get location in u
  float d = dmap[o];
  float uxv = ox + d * dx;
  float uyv = oy + d * dy;
  int px = (int)floor(uxv);
  int py = (int)floor(uyv);
  if ( px<0 || py<0 || px>W-1 || py>H-1 ) {
    return;
  }
  int po = px + py*W;
  float ax = uxv - float(px);
  float ay = uyv - float(py);

  // transpose bilinear sampling
  float mxmym = (1.0f - ax) * (1.0f - ay);
  float mxpym = ax * (1.0f - ay);
  float mxmyp = (1.0f - ax) * ay;
  float mxpyp = ax * ay;
  //float m = mxmym + mxpym + mxmyp + mxpyp;

  float inv = in[o];

  out[ po + 0 ] += inv * mxmym;
  if ( px < W-1 ) {
    out[ po + 1 ] += inv * mxpym;
  }
  if ( py < H-1 ) {
    out[ po + W ] += inv * mxmyp;
  }
  if ( px < W-1 && py < H-1 ) {
    out[ po + W + 1 ] += inv * mxpyp;
  }
  
  if ( out_weight ) {
    out_weight[ po + 0 ] += mxmym;
    if ( px < W-1 ) {
      out_weight[ po + 1 ] += mxpym;
    }
    if ( py < H-1 ) {
      out_weight[ po + W ] += mxmyp;
    }
    if ( px < W-1 && py < H-1 ) {
      out_weight[ po + W + 1 ] += mxpyp;
    }
  }
}

/*
static __global__ void vtv_sr_warp_view_accumulate_device( int W, int H,
							   float *u, float *umask,
							   float *dmap, float dx, float dy,
							   int *vmask,
							   float *v, float *vweight )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = ox + oy*W;

  // only warp inside mask pixels
  if ( vmask[o] == 0 ) {
    return;
  }

  // get location in u
  float d = dmap[o];
  float uxv = ox + d * dx;
  float uyv = oy + d * dy;
  int px = (int)floor(uxv);
  int py = (int)floor(uyv);
  if ( px<0 || py<0 || px>W-2 || py>H-2 ) {
    return;
  }
  int po = px + py*W;
  float ax = uxv - float(px);
  float ay = uyv - float(py);

  // bilinear sampling (todo: texture lookup)

  //float mxmym = umask[po]; // * (1.0f - ax) * (1.0f - ay);
  //float mxpym = umask[po+1]; // * ax * (1.0f - ay);
  //float mxmyp = umask[po+W]; // * (1.0f - ax) * ay;
  //float mxpyp = umask[po+W+1]; // * ax * ay;

  float mxmym = umask[po] * (1.0f - ax) * (1.0f - ay);
  float mxpym = umask[po+1] * ax * (1.0f - ay);
  float mxmyp = umask[po+W] * (1.0f - ax) * ay;
  float mxpyp = umask[po+W+1] * ax * ay;
  float m = mxmym + mxpym + mxmyp + mxpyp;
  float vadd = u[po] * mxmym;
  vadd += u[po+1] * mxpym;
  vadd += u[po+W] * mxmyp;
  vadd += u[po+W+1] * mxpyp;
  v[o] += vadd;
  vweight[o] += m; //4.0f;
}


static __global__ void vtv_sr_compute_warp_matrix_device( int W, int H, int N,
							  float *dmap, float dx, float dy,
							  bool *visibility_mask,
							  float *mat_warp, int *mat_warp_coord )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = ox + oy*W;

  // only warp inside mask pixels
  if ( visibility_mask[o] == 0 ) {
    mat_warp[ o + 0 ] = 0.0f;
    mat_warp_coord[ o + 0 ] = N;
    return;
  }

  // get location in u
  float d = dmap[o];
  float uxv = ox + d * dx;
  float uyv = oy + d * dy;
  int px = (int)floor(uxv);
  int py = (int)floor(uyv);
  if ( px<0 || py<0 || px>W-2 || py>H-2 ) {
    mat_warp[ o + 0 ] = 0.0f;
    mat_warp_coord[ o + 0 ] = N;
    return;
  }
  int po = px + py*W;
  float ax = uxv - float(px);
  float ay = uyv - float(py);

  // bilinear sampling (todo: texture lookup)
  //float mxmym = umask[po]; // * (1.0f - ax) * (1.0f - ay);
  //float mxpym = umask[po+1]; // * ax * (1.0f - ay);
  //float mxmyp = umask[po+W]; // * (1.0f - ax) * ay;
  //float mxpyp = umask[po+W+1]; // * ax * ay;

  float mxmym = (1.0f - ax) * (1.0f - ay);
  float mxpym = ax * (1.0f - ay);
  float mxmyp = (1.0f - ax) * ay;
  float mxpyp = ax * ay;
  float m = mxmym + mxpym + mxmyp + mxpyp;
  if ( m==0.0f ) {
    mat_warp[ o + 0 ] = 0.0f;
    mat_warp_coord[ o + 0 ] = N;
    return;
  }

  mat_warp[ o + 0 ] = mxmym / m;
  mat_warp_coord[ o + 0 ] = po;

  mat_warp[ o + N ] = mxpym / m;
  mat_warp_coord[ o + N ] = po+1;

  mat_warp[ o + 2*N ] = mxmyp / m;
  mat_warp_coord[ o + 2*N ] = po+W;

  mat_warp[ o + 3*N ] = mxpyp / m;
  mat_warp_coord[ o + 3*N ] = po+W+1;
}

*/


// DMAP upsampling : same value is repeated in the result cells
static __global__ void vtv_sr_upsample_dmap_device( int W, int H, // Hi-res size
						    int w, int h, // Lo-res size
						    float F,        // Scale factor
						    float *m,     // lo-res matrix
						    float *M )    // hi-res result
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }

  int px = int(float(ox) / F); // coordinate truncation
  int py = int(float(oy) / F);
  if ( px>=w || py>=h ) {
    return;
  }

  M[ox+oy*W] = m[px+py*w];
}

// Given a Hi-resolution weight maks (vmask),
// vmask_lo will have the small resolution, and each pixel
// will contain the addition of visibility weights in the corresponding hi-res pixels 
static __global__ void vtv_sr_downsample_mask_device( int W, int H, // Low-res size
						      int W_hi, // Hi-res width
						      float *vmask, // Hi-res visbility weights matrix
						      float *vmask_lo, // Low-res visibility weights matrix (sum of all hi-res)
						      int dsf ) // Down scale factor
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }

  int o = ox + oy*W;
  int o_hi = (ox*dsf) + (oy*dsf)*W_hi;

  float weight = 0.0f;
  for ( int y=0; y<dsf; y++ ) {
    for ( int x=0; x<dsf; x++ ) {
      weight += vmask[o_hi];
      o_hi++;
    }
    o_hi += W_hi - dsf;
  }

  vmask_lo[o] = weight;
}


static __global__ void vtv_sr_init_mask_gradient_weight_device( int W, int H,
                float dx, float dy,
                const float *dmap,
                const bool *visibility_mask, // binary visibility mask
                float *vmask_weighted ) // output gradient weights
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }

  int o = ox + oy*W;
  if ( visibility_mask[o] == 0 ) {
    vmask_weighted[o] = 0.0f;
    return;
  }
  
  // Neumann condition : gradient = 0 -> weight = 1.0
  if ( ox == 1 || oy == 1 || ox == W-1 || oy == H-1 ) {
    vmask_weighted[o] = 1.0f;
    return;
  }


  // THEORETICALLY CORRECT VERSION
  float d = dmap[o];
  float dxp = (dmap[o+1] - d);
  float dxm = (d - dmap[o-1]);
  float dyp = (dmap[o+W] - d);
  float dym = (d - dmap[o-W]);

  float Dx = 0.5f * (dxp + dxm);
  float Dy = 0.5f * (dyp + dym);

  // original formula is with a minus sign, but dx and dy are
  // already reversed. We are using vu instead of uv
  float N = fabs( 1.0f + Dx*dx + Dy*dy );

#define GRAD_W_MAX 20.0f
  if ( N > 1.0f / GRAD_W_MAX ) {
    vmask_weighted[o] = 1.0f / N;
  }
  else {
    vmask_weighted[o] = GRAD_W_MAX;
  }
}


static __global__ void vtv_sr_compute_u_gradient_weight_device( int W, int H,
                                                                bool *visibility_mask,
                                                                float dx, float dy,
                                                                float *u_gradient_x,
                                                                float *u_gradient_y,
                                                                float *dmap,
                                                                float *dmap_sigma,
                                                                float sigma_sensor,
                                                                float aux_dmap_sigma,
                                                                float ugrad_threshold,
                                                                float *output )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }

  int o = ox + oy*W;

  // if not visible no need to compute anything
  if ( visibility_mask[o] == 0 ) {
    output[o] = 0.;
    return;
  }

  // get location in u
  float d = dmap[o];
  float uxv = ox + d * dx;
  float uyv = oy + d * dy;
  int px = (int)floor(uxv);
  int py = (int)floor(uyv);

  // if outside : weight = 0
  // the difference of intesities cannot be computed (and will not be used)
  if ( px<0 || py<0 || px>W-1 || py>H-1 ) {
    output[o] = 0.;
    return;
  }

  // bilinear interpolation of the gradient
  int po = px + py*W;
  float ax = uxv - float(px);
  float ay = uyv - float(py);

  float grad_x = (1.0f - ax) * (1.0f - ay) * u_gradient_x[po] +
                         ax  * (1.0f - ay) * u_gradient_x[po+1] +
                 (1.0f - ax) *         ay  * u_gradient_x[po+W] +
                         ax  *         ay  * u_gradient_x[po+W+1];
  float grad_y = (1.0f - ax) * (1.0f - ay) * u_gradient_y[po] +
                         ax  * (1.0f - ay) * u_gradient_y[po+1] +
                 (1.0f - ax) *         ay  * u_gradient_y[po+W] +
                         ax  *         ay  * u_gradient_y[po+W+1];

  // Threshold the gradient value. Do not threshold the dot product, direction is important
  float norm = sqrt(grad_x * grad_x + grad_y * grad_y);
  if( norm > ugrad_threshold ) {
    grad_x  = grad_x / norm * ugrad_threshold;
    grad_y  = grad_y / norm * ugrad_threshold;
  }

  float N = /*dmap_sigma[o]*/ aux_dmap_sigma * (grad_x * dx + grad_y * dy);

  // The maximum weight is achieved when the u gradient is 0
  // Weight = 1./(sigma_sensor * sigma_sensor)
  output[o] = 1.0 / (N*N + sigma_sensor * sigma_sensor);
}

// experimental: compute the deformation weights
static __global__ void vtv_sr_compute_weights_device( int W, int H,
                                                                bool *visibility_mask,
                                                                float dx, float dy,
                                                                float *u_gradient_x,
                                                                float *u_gradient_y,
                                                                float *dmap,
                                                                cuflt sigma_p_i, // standard deviation of the point-wise function
                                                                float *output ) {

    // Global thread index
    int ox = blockDim.x * blockIdx.x + threadIdx.x;
    int oy = blockDim.y * blockIdx.y + threadIdx.y;
    if ( ox>=W || oy>=H ) {
      return;
    }

    int o = ox + oy*W;

    // if not visible no need to compute anything
    if ( visibility_mask[o] == 0 ) {
      output[o] = 0.;
      return;
    }

    // get location in u
    float d = dmap[o];
    float uxv = ox + d * dx;
    float uyv = oy + d * dy;
    int px = (int)floor(uxv);
    int py = (int)floor(uyv);

    // if outside : weight = 0
    // the difference of intesities cannot be computed (and will not be used)
    if ( px<0 || py<0 || px>W-1 || py>H-1 ) {
      output[o] = 0.;
      return;
    }

    // bilinear interpolation of the gradient
    int po = px + py*W;
    float ax = uxv - float(px);
    float ay = uyv - float(py);

    float grad_x = (1.0f - ax) * (1.0f - ay) * u_gradient_x[po] +
                           ax  * (1.0f - ay) * u_gradient_x[po+1] +
                   (1.0f - ax) *         ay  * u_gradient_x[po+W] +
                           ax  *         ay  * u_gradient_x[po+W+1];
    float grad_y = (1.0f - ax) * (1.0f - ay) * u_gradient_y[po] +
                           ax  * (1.0f - ay) * u_gradient_y[po+1] +
                   (1.0f - ax) *         ay  * u_gradient_y[po+W] +
                           ax  *         ay  * u_gradient_y[po+W+1];

    // gradient of the disparity
    float dxp = (dmap[o+1] - d);
    float dxm = (d - dmap[o-1]);
    float dyp = (dmap[o+W] - d);
    float dym = (d - dmap[o-W]);

    float Dx = 0.5f * (dxp + dxm);
    float Dy = 0.5f * (dyp + dym);

    // square deformation sigma
    cuflt sigma_r_i = sigma_p_i*sigma_p_i * ( ( grad_x*( 1+dx*Dx ) + grad_y*( dy*Dx ) )*( grad_x*( 1+dx*Dx ) + grad_y*( dy*Dx ) ) +

                                              ( grad_x*( dx*Dy ) + grad_y*( 1+dy*Dy ) )*( grad_x*( dx*Dy ) + grad_y*( 1+dy*Dy ) ) );

    if (output[o] == 0) {
        return;
    }
    cuflt inv = 1.0 / output[o];
    inv += sigma_r_i;
    output[o] = 1.0 / inv;
}

static __global__ void vtv_sr_add_dmap_vote_device( int W, int H,
						    float *dmap_vote, int *mask,
						    float *dmap, float *dmap_weight )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }

  int o = ox + oy*W;
  if ( mask[o] != 0 ) {
    // TEST CONSERVATIVE OCCLUSION VOTING: closer one always wins.
    dmap[o] = max( dmap[o], dmap_vote[o] );
    //dmap[o] += dmap_vote[o];
    dmap_weight[o] += 1.0f;
  }
  else {
    dmap_vote[o] = 0.0f;
  }
}

/*
static __global__ void vtv_sr_mask_dmap_device( int W, int H,
						float *dmap, float *dmap_weight )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }

  int o = ox + oy*W;
  float w = dmap_weight[o];
  if ( w == 0.0f ) {
    dmap[o] = 0.0f;
  }
  else {
    //dmap[o] /= w;
    dmap_weight[o] = 1.0f;
  }
}*/



static __global__ void vtv_sr_finalize_vmask_device( int W, int H,
						     float *filter, float *mask )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = ox + oy*W;

  // replace mask with filtered value, but only if nonzero
  if ( mask[o] != 0.0f ) {
    mask[o] = filter[o];
  }
}




static __global__ void vtv_sr_downsample_view_device( int W, int H, //lo-res size
                  int W_hi, // hi-res width
                  float *hi, // input in hi-res
                  const bool *visibility_mask, // visibility weight in hi-res
                  int dsf, // down scale factor
                  float *lo,
                  float invalid_value = 0.) // result : mean of visible elements in *hi
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }

  int o_hi = (ox*dsf) + (oy*dsf)*W_hi;
  float value = 0.0f;
  float sum_of_elems = 0;
  for ( int y=0; y<dsf; y++ ) {
    for ( int x=0; x<dsf; x++ ) {
      if ( visibility_mask[o_hi] ) {
        value += hi[o_hi];
        sum_of_elems += 1.0f;
      }
      o_hi++;
    }
    o_hi += W_hi - dsf;
  }

  int o = ox + oy*W;
  if (sum_of_elems > 0 ) {
    lo[o] = value / sum_of_elems;
  } else {
    lo[o] = invalid_value;
  }
}



// compute transpose of downsampling operation
static __global__ void vtv_sr_ds_transpose_view_device( int W, int H,
							int W_hi,
							float *lo, // input low res image
							float *vmask, //
							float *vmask_lo, // used to check it's not 0, and find "mean value"
							int dsf,
							float *hi ) // output hi-res image : each low-res pixel is "splitted" into its weight distribution using hi-res vmask
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = ox + oy*W;

  float val = lo[o];
  float weight = vmask_lo[o];
  if ( weight != 0.0f ) {
    val /= weight;
  }

  int o_hi = (ox*dsf) + (oy*dsf)*W_hi;
  for ( int y=0; y<dsf; y++ ) {
    for ( int x=0; x<dsf; x++ ) {
      hi[o_hi] = vmask[o_hi] * val;
      o_hi++;
    }
    o_hi += W_hi - dsf;
  }
}


/*
static __global__ void vtv_sr_warp_disparity_map_device( int W, int H,
							 int step_x, int step_y,
							 int base_x, int base_y,
							 float *dmap,
							 float dx, float dy,
							 float *out_depth,
							 int *out_mask )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  ox = ox * step_x + base_x;
  oy = oy * step_y + base_y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = ox + oy*W;

  float disp = dmap[o];
  float wx = float(ox) + dx * disp;
  float wy = float(oy) + dy * disp;

  // find closest pixel
  int px = int( round(wx) );
  int py = int( round(wy) );

  // check if inside domain
  if ( px<1 || py<1 || px>=W-1 || py>=H-1 ) {
    return;
  }

  // check if warp is visible
  // conservative occlusion
  int op = px + py*W;
  if ( disp > out_depth[op] ) {
    out_depth[op] = disp;
    out_mask[op] = 1;
  }

  op = px+1 + py*W;
  if ( disp > out_depth[op] ) {
    out_depth[op] = disp;
    out_mask[op] = 1;
  }
  op = px + (py+1)*W;
  if ( disp > out_depth[op] ) {
    out_depth[op] = disp;
    out_mask[op] = 1;
  }
  op = px+1 + (py+1)*W;
  if ( disp > out_depth[op] ) {
    out_depth[op] = disp;
    out_mask[op] = 1;
  }

  op = px-1 + py*W;
  if ( disp > out_depth[op] ) {
    out_depth[op] = disp;
    out_mask[op] = 1;
  }
  op = px + (py-1)*W;
  if ( disp > out_depth[op] ) {
    out_depth[op] = disp;
    out_mask[op] = 1;
  }
  op = px-1 + (py-1)*W;
  if ( disp > out_depth[op] ) {
    out_depth[op] = disp;
    out_mask[op] = 1;
  }

  op = px-1 + (py+1)*W;
  if ( disp > out_depth[op] ) {
    out_depth[op] = disp;
    out_mask[op] = 1;
  }
  op = px+1 + (py-1)*W;
  if ( disp > out_depth[op] ) {
    out_depth[op] = disp;
    out_mask[op] = 1;
  }

}
*/


static __global__ void vtv_sr_mask_min_filter_float_device( int W, int H, 
							    float *w,
							    float *out )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = ox + oy*W;

  float v = w[ o ];
  float wv;
#define CMP(n) wv=w[n]; if (wv != 0.0f) v = min( v,wv );
  if ( ox>0 ) {
    if ( oy>0 ) {
      CMP(o-W-1)
    }
    if ( oy<H-1 ) {
      CMP(o+W-1)
    }
    CMP(o-1)
  }
  if ( ox<W-1 ) {
    if ( oy>0 ) {
      CMP(o-W+1)
    }
    if ( oy<H-1 ) {
      CMP(o+W+1)
    }
    CMP(o+1)
  }
  if ( oy>0 ) {
    CMP(o-W)
  }
  if ( oy<H-1 ) {
    CMP(o+W)
  }
#undef CMP

  out[o] = v;
}



static __global__ void vtv_sr_mask_min_filter_device( int W, int H, 
						      int *w,
						      int *out )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = ox + oy*W;

  int v = w[ o ];
  if ( ox>0 ) {
    if ( oy>0 ) {
      v = min( v, w[o-W-1] );
      v = min( v, w[o-1] );
    }
    if ( oy<H-1 ) {
      v = min( v, w[o+W-1] );
    }
  }
  if ( ox<W-1 ) {
    if ( oy>0 ) {
      v = min( v, w[o-W+1] );
    }
    if ( oy<H-1 ) {
      v = min( v, w[o+W+1] );
    }
    v = min( v, w[o+1] );
  }
  if ( oy>0 ) {
    v = min( v, w[o-W] );
  }
  if ( oy<H-1 ) {
    v = min( v, w[o+W] );
  }

  out[o] = v;
}



// compute binary visibility : 
// point is visible in target view if depth at warp location
// lies within disparity gradient range
static __global__ void vtv_sr_compute_visibility_mask_device( int W, int H,
                    const float *dmap,
                    float dx,
                    float dy,
                    const float *dmap_warped,
                    bool *out_mask,
                    float disp_threshold )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = ox + oy*W;

  // point is visible in target view if depth at warp location
  // lies within disparity gradient range
  float d = dmap[o];

  // find closest pixel
  float ux = ox + d * dx;
  float uy = oy + d * dy;
  int px = int( floor(ux) );
  int py = int( floor(uy) );  
  // Pixel going outside is not visible
  if ( px<0 || py<0 || px>W-1 || py>H-1 ) {
    out_mask[o] = 0;
    return;
  }
  float udmin = dmap_warped[ px + py*W ];
  float udmax = udmin;

  float td;
  if ( px < W-1 ) {
    td= dmap_warped[ px +1 + py*W ];
    udmin = min( udmin, td );
    udmax = max( udmax, td );
  }

  if ( py < H-1 ) {
    td = dmap_warped[ px + (py+1)*W ];
    udmin = min( udmin, td );
    udmax = max( udmax, td );
  }

  if ( px < W-1 && py < H-1) {
    td = dmap_warped[ px +1 + (py+1)*W ];
    udmin = min( udmin, td );
    udmax = max( udmax, td );
  }

  // Bastian, this breaks the tests but I think it's the good way to filter disparities
  // The final results look nicer, this removes some artifacts at the depth borders.

  // Thanks, included for testing - Bastian.
  {
    // denormalize disparity values into pixel values
    // disp_threshold should be set in pixel units
    float radius = hypotf( dx, dy );
    udmin *= radius;
    udmax *= radius;
    d *= radius;
  }
  
  //if ( udmin - 3.0f * dgradmax - 1.0f <= d && d <= udmax + 3.0f * dgradmax + 1.0f ) {
  if ( udmin - disp_threshold <= d && d <= udmax + disp_threshold ) {
    out_mask[o] = 1;
  }
  else {
    out_mask[o] = 0;
  }
}

/*
static __global__ void vtv_sr_filter_inconsistent_depth_device( int W, int H,
								float *dmapv,
								float dx_vu,
								float dy_vu,
								float *vmask,
								float *dmapu,
								float dx_uv,
								float dy_uv,
								int *umask,
								int *feedback )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = ox + oy*W;
  if ( ox==0 || oy==0 || ox==W-1 || oy==H-1 ) {
    umask[o] = 0;
    return;
  }

  // Temp: allow all
  umask[o] = 1;
  return;
}

static __global__ void vtv_sr_filter_inconsistent_depth_device( int W, int H,
								float *vx,
								float *vy,
								int *vmask,
								float *ux,
								float *uy,
								float *umask,
								int *feedback )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = ox + oy*W;
  if ( ox==0 || oy==0 || ox==W-1 || oy==H-1 ) {
    umask[o] = 0.0f;
    return;
  }

  if ( umask[o] == 0.0f ) {
    return;
  }

  // find closest pixel
  int px = int( round(ux[o]) );
  int py = int( round(uy[o]) );
  if ( px<0 || py<0 || px>=W || py >= H ) {
    umask[o] = 0.0f;
    return;
  }
  
  int op = px + W*py;
  if ( vmask[op] == 0 ) {
    umask[o] = 0.0f;
    return;
  }

  float tx = vx[op];
  float ty = vy[op];
  float dist = hypotf( tx - float(ox), ty - float(oy) );
  if ( dist >= 3.0f ) {
    umask[o] = 0.0f;
    feedback[o] = 1;
  }
}
*/


static __global__ void vtv_sr_init_regularizer_weight_device( int W, int H,
							      float lambda_max,
							      float lambda_min,
							      float mask_max,
							      float *mask,
							      float *weight )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;

  float m = mask[o];
  if ( m == 0.0f ) {
    weight[o] = lambda_max;
  }
  else {
    weight[o] = max( lambda_min, lambda_max - m / mask_max ); 
  }
}


static __global__ void vtv_sr_multiply_mask_device( int W, int H, int *dst, cuflt *src )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;
  if ( src[o] == 0.0f ) {
    dst[o] = 0;
  }
}


static __global__ void vtv_sr_get_view_image_channel_device( int W, int H, int nchannels,
							     unsigned char *view, int n,
							     float *channel )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;
  channel[o] = float(view[ o*nchannels + n ]) / 255.0f;
}


static __global__ void vtv_sr_set_view_image_channel_device( int W, int H,
							     float *channel,
							     unsigned char *view, int n )
{
  // Global thread index
  const int ox = IMUL( blockDim.x, blockIdx.x ) + threadIdx.x;
  const int oy = IMUL( blockDim.y, blockIdx.y ) + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  const int o = IMUL( oy,W ) + ox;
  view[ o*3 + n ] = (unsigned char)( channel[o] * 255.0f );
}

/*
static __global__ void vtv_sr_filter_mask_device( int W, int H,
						  int *m, int *m_out )
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox>=W || oy>=H ) {
    return;
  }
  int o = ox + oy*W;
  if ( ox==0 || oy==0 || ox==W-1 || oy==H-1 ) {
    m_out[o] = 0;
    return;
  }

  // count number of pixels which are "1"
  int count = m[o-1] + m[o] + m[o+1];
  count += m[o-W-1] + m[o-W] + m[o-W+1];
  count += m[o+W-1] + m[o+W] + m[o+W+1];
  if ( count>4 ) {
    m_out[o] = 1;
  }
  else if ( count < 4 ) {
    m_out[o] = 0;
  }
  else {
    m_out[o] = m[o];
  }
}*/
