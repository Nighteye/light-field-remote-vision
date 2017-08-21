/* -*-c++-*- */

// Compute the init image given the TV smooth solution and a mask
static __global__ void vtv_sr_compute_filled_init( int W, int H,
                                                   cuflt *u,
                                                   bool *visibility_mask,
                                                   cuflt *output ) {
    
    // Global thread index
    int px = blockDim.x * blockIdx.x + threadIdx.x;
    int py = blockDim.y * blockIdx.y + threadIdx.y;
    if ( px >= W || py >= H ) {
        return;
    }
    int p = py*W + px;
    
    if ( visibility_mask[p] ) { // not a hole
        output[p] = u[p];
    }
}

// compute the visibility from u
static __global__ void cuda_setup_visibility_mask_device( int W, int H, int R,
                                                          int *index_array,
                                                          int start, int end,
                                                          size_t ks,
                                                          size_t dsf,
                                                          cuflt *warp_tau_x, cuflt *warp_tau_y,
                                                          bool *visibility_mask ) { // 1 if visible, 0 elsewhere
    // Global thread index
    int idx = R * blockIdx.x + threadIdx.x + start;
    if ( idx>=end ) {
        return;
    }
    int m = index_array[idx];
    
    // get location in u, warp value in pixel coordinates
    // warps coordinates: bottom left corner
    // x_centered = x_corner - 0.5
    cuflt px = warp_tau_x[m] - 0.5;
    cuflt py = warp_tau_y[m] - 0.5;
    
    // Compute local convolution
    for ( int j = 0 ; j < ks ; j++ ) {
        for ( int i = 0 ; i < ks ; i++ ) {
            
            int x = (int)floor(px - float(dsf)*0.5 + 0.5) + j;
            int y = (int)floor(py - float(dsf)*0.5 + 0.5) + i;
            if ( 0 <= x && x <= W-1 &&
                 0 <= y && y <= H-1 ) {

                cuflt dx = 1.0;
                if ( j == 0 ) {
                    dx = (cuflt(x) + 0.5) - (px - 0.5*cuflt(dsf));
                }
                else if ( j == ks - 1 ) {
                    dx = (px + 0.5*cuflt(dsf)) - (cuflt(x) - 0.5);
                }
                cuflt dy = 1.0;
                if ( i == 0 ) {
                    dy = (cuflt(y) + 0.5) - (py - 0.5*cuflt(dsf));
                }
                else if ( i == ks - 1 ) {
                    dy = (py + 0.5*cuflt(dsf)) - (cuflt(y) - 0.5);
                }

                if ( dx*dy != 0.0 ) {
                    visibility_mask[x + y*W] = true;
                }
            }
        }
    }
}

// compute the visibility from u
static __global__ void cuda_setup_visibility_mask_device( int W, int H,
                                                          const cuflt* const input,
                                                          bool *visibility_mask ) { // 1 if visible, 0 elsewhere

    // Global thread index
    int ox = blockDim.x * blockIdx.x + threadIdx.x;
    int oy = blockDim.y * blockIdx.y + threadIdx.y;
    if ( ox >= W || oy >= H ) {
        return;
    }
    int o = oy*W + ox;

    if(input[o] < 1.0) {
        visibility_mask[o] = false;
    } else {
        visibility_mask[o] = true;
    }
}

// Compute the gradient of the image using Sobel operator
static __global__ void vtv_sr_compute_gradient_device( int W, int H,
                                                       cuflt *r,
                                                       cuflt *g,
                                                       cuflt *b,
                                                       cuflt *px, cuflt *py,
                                                       const bool* const visibility) {
    
    // Global thread index
    int ox = blockDim.x * blockIdx.x + threadIdx.x;
    int oy = blockDim.y * blockIdx.y + threadIdx.y;
    if ( ox >= W || oy >= H ) {
        return;
    }
    int o = oy*W + ox;
    
    cuflt gradX = 0.0;
    cuflt gradY = 0.0;
    
    if (0 < ox && ox < W-1 && 0 < oy && oy < H-1) { // grad = 0 on image boundaries
        
        int ppo = (oy-1)*W + (ox-1), pco = ppo+1, pno = pco+1;
        int cpo = (oy+0)*W + (ox-1), cco = cpo+1, cno = cco+1;
        int npo = (oy+1)*W + (ox-1), nco = npo+1, nno = nco+1;

        if(visibility != 0) {
            if(!visibility[ppo] || !visibility[pco] || !visibility[pno] ||
                    !visibility[cpo] || !visibility[cco] || !visibility[cno] ||
                    !visibility[npo] || !visibility[nco] || !visibility[nno]) {
                return;
            }
        }
        
        // luminance from rgb
        cuflt Ypp = 0.2126 * r[ppo] + 0.7152 * g[ppo] + 0.0722 * b[ppo];
        cuflt Ypc = 0.2126 * r[pco] + 0.7152 * g[pco] + 0.0722 * b[pco];
        cuflt Ypn = 0.2126 * r[pno] + 0.7152 * g[pno] + 0.0722 * b[pno];

        cuflt Ycp = 0.2126 * r[cpo] + 0.7152 * g[cpo] + 0.0722 * b[cpo];
        //cuflt Ycc = 0.2126 * r[cco] + 0.7152 * g[cco] + 0.0722 * b[cco];
        cuflt Ycn = 0.2126 * r[cno] + 0.7152 * g[cno] + 0.0722 * b[cno];

        cuflt Ynp = 0.2126 * r[npo] + 0.7152 * g[npo] + 0.0722 * b[npo];
        cuflt Ync = 0.2126 * r[nco] + 0.7152 * g[nco] + 0.0722 * b[nco];
        cuflt Ynn = 0.2126 * r[nno] + 0.7152 * g[nno] + 0.0722 * b[nno];

        gradX = (Ypn + 2* Ycn + Ynn - Ypp - 2* Ycp - Ynp)/8.;
        gradY = (Ynp + 2* Ync + Ynn - Ypp - 2* Ypc - Ypn)/8.;
    }
    
    px[o] = gradX;
    py[o] = gradY;
}

// Compute the gradient of a one-channel image using finite differences
static __global__ void vtv_sr_compute_RGB_gradient_device( int W, int H,
                                                           cuflt *input, // one channel of the image
                                                           cuflt *px, cuflt *py ) { // output gradients

    // Global thread index
    int ox = blockDim.x * blockIdx.x + threadIdx.x;
    int oy = blockDim.y * blockIdx.y + threadIdx.y;
    if ( ox >= W || oy >= H ) {
        return;
    }
    int o = oy*W + ox;

    px[o] = 0.0;
    py[o] = 0.0;

    if ( 0 < ox && ox < W-1 &&
         0 < oy && oy < H-1 ) {

        px[o] = 0.5*( input[o+1] - input[o-1] );
        py[o] = 0.5*( input[o+W] - input[o-W] );
    }
}

// compute the angular weights from warped u gradient
static __global__ void vtv_sr_angular_weights_device( int w, int h, // low res
                                                      const cuflt *u_gradient_x, // vi domain, low res
                                                      const cuflt *u_gradient_y,
                                                      cuflt sigma_sensor,
                                                      const cuflt *dpart_x, // vi domain, low res
                                                      const cuflt *dpart_y, // dpart replaces aux_dmap_sigma*dtau/dz
                                                      cuflt ugrad_threshold,
                                                      cuflt *output ) {
    // Global thread index
    int mx = blockDim.x * blockIdx.x + threadIdx.x;
    int my = blockDim.y * blockIdx.y + threadIdx.y;
    if ( mx>=w || my>=h ) {
        return;
    }

    int m = mx + my*w;

    // warped u gradient
    cuflt grad_x = u_gradient_x[m];
    cuflt grad_y = u_gradient_y[m];

    // Threshold the gradient value. Do not threshold the dot product, direction is important
    cuflt norm = sqrt(grad_x * grad_x + grad_y * grad_y);
    if( norm > ugrad_threshold ) {
        grad_x  = grad_x / norm * ugrad_threshold;
        grad_y  = grad_y / norm * ugrad_threshold;
    }

    // structured: float sigma_geom = aux_dmap_sigma * (grad_x * dx + grad_y * dy);
    cuflt sigma_geom = grad_x * dpart_x[m] + grad_y * dpart_y[m];

    // Threshold sigma_geom: do not forget that this magnitude represents a color variance
    //    Our random variable represents a color and lives in [0,1]
    //    The variance of a variable in [0,1] can not be greater than 0.5^2
    if (sigma_geom > 0.5) {
        sigma_geom = 0.5;
    }

    // The maximum weight is achieved when the u gradient is 0
    // Weight = 1./(sigma_sensor * sigma_sensor)
    output[m] = 1.0 / (sigma_geom * sigma_geom + sigma_sensor * sigma_sensor);
}

// compute the deformation weights |det D tau_i|^(-1) in omega_i domain
static __global__ void vtv_sr_gold_deform_weights_device( int W, int H,
                                                          int w, int h,
                                                          const cuflt *warp_tau_x,
                                                          const cuflt *warp_tau_y,
                                                          cuflt *deforma_weights ) {
    // Global thread index
    int mx = blockDim.x * blockIdx.x + threadIdx.x;
    int my = blockDim.y * blockIdx.y + threadIdx.y;
    if ( mx >= w || my >= h ) {
        return;
    }

    int m = mx + my*w;

    // test visibility
    if ( warp_tau_x[m] < 0 || warp_tau_x[m] > W || warp_tau_y[m] < 0 || warp_tau_y[m] > H ) { // pixel not visible from input view vi
        deforma_weights[m] = 0.0;
        return;
    }

    // Neumann condition : gradient = 0 -> weight = 1.0
    if ( mx == 0 || my == 0 || mx == w-1 || my == h-1 ) {
        deforma_weights[m] = 1.0;
        return;
    }
    if ( warp_tau_x[m-1] < 0 || warp_tau_x[m-1] > W || warp_tau_y[m-1] < 0 || warp_tau_y[m-1] > H ||
         warp_tau_x[m+1] < 0 || warp_tau_x[m+1] > W || warp_tau_y[m+1] < 0 || warp_tau_y[m+1] > H ||
         warp_tau_x[m-w] < 0 || warp_tau_x[m-w] > W || warp_tau_y[m-w] < 0 || warp_tau_y[m-w] > H ||
         warp_tau_x[m+w] < 0 || warp_tau_x[m+w] > W || warp_tau_y[m+w] < 0 || warp_tau_y[m+w] > H ) {

        deforma_weights[m] = 1.0;
        return;
    }

    cuflt Dxx = 0.5 * (warp_tau_x[m+1] - warp_tau_x[m-1]);
    cuflt Dyx = 0.5 * (warp_tau_y[m+1] - warp_tau_y[m-1]);
    cuflt Dxy = 0.5 * (warp_tau_x[m+w] - warp_tau_x[m-w]);
    cuflt Dyy = 0.5 * (warp_tau_y[m+w] - warp_tau_y[m-w]);

    cuflt N = fabs(Dxx * Dyy - Dxy * Dyx);

#define MAX_DEFORM_WEIGHT 1.0
    if ( N > 1.0 / MAX_DEFORM_WEIGHT ) {
        deforma_weights[m] = 1.0 / N;
    }
    else {
        deforma_weights[m] = MAX_DEFORM_WEIGHT;
    }
}

__global__ void cuda_convolution_nonsep_device_param( int W, int H, // high res (input)
                                                      int w, int h, // low res (output)
                                                      size_t ks, // size of a kernel
                                                      cuflt *A, // v_i = A*u
                                                      cuflt *warp_tau_x, // vi domain, low res, values high res
                                                      cuflt *warp_tau_y,
                                                      const cuflt *i_r, const cuflt *i_g, const cuflt *i_b,
                                                      cuflt *o_r, cuflt *o_g, cuflt *o_b ) {
    // Global thread index
    int mx = blockDim.x * blockIdx.x + threadIdx.x;
    int my = blockDim.y * blockIdx.y + threadIdx.y;
    if ( mx >= w || my >= h ) {
        return;
    }
    int m = my*w + mx;
    
    o_r[m] = 0.0;
    o_g[m] = 0.0;
    o_b[m] = 0.0;
    
    // test visibility
    if ( warp_tau_x[m] < 0 || warp_tau_y[m] < 0 || warp_tau_x[m] > W || warp_tau_y[m] > H ) {
        return;
    }
    
    // get location in u, warp value in pixel coordinates
    // warps coordinates: bottom left corner
    // x_centered = x_corner - 0.5
    cuflt px = warp_tau_x[m] - 0.5;
    cuflt py = warp_tau_y[m] - 0.5;

    int dsf = W/w;
    
    // Compute local convolution
    for ( int j = 0 ; j < ks ; j++ ) {
        for ( int i = 0 ; i < ks ; i++ ) {
            
            int x = (int)floor(px - float(dsf)*0.5 + 0.5) + j;
            int y = (int)floor(py - float(dsf)*0.5 + 0.5) + i;
            if ( 0 <= x && x <= W-1 &&
                 0 <= y && y <= H-1 ) {
                
                o_r[m] += A[m*ks*ks + ks*i + j] * i_r[y*W + x];
                o_g[m] += A[m*ks*ks + ks*i + j] * i_g[y*W + x];
                o_b[m] += A[m*ks*ks + ks*i + j] * i_b[y*W + x];
            }
        }
    }
}

__global__ void cuda_convolution_nonsep_device_param( int W, int H, // high res (input)
                                                      int w, int h, // low res (output)
                                                      size_t ks, // size of a kernel
                                                      cuflt *A, // v_i = A*u
                                                      cuflt *warp_tau_x, // vi domain, low res, values high res
                                                      cuflt *warp_tau_y,
                                                      const cuflt *I,
                                                      cuflt *O ) {
    // Global thread index
    int mx = blockDim.x * blockIdx.x + threadIdx.x;
    int my = blockDim.y * blockIdx.y + threadIdx.y;
    if ( mx >= w || my >= h ) {
        return;
    }
    int m = my*w + mx;

    O[m] = 0.0;

    // test visibility
    if ( warp_tau_x[m] < 0 || warp_tau_y[m] < 0 || warp_tau_x[m] > W || warp_tau_y[m] > H ) {
        return;
    }

    // get location in u, warp value in pixel coordinates
    // warps coordinates: bottom left corner
    // x_centered = x_corner - 0.5
    cuflt px = warp_tau_x[m] - 0.5;
    cuflt py = warp_tau_y[m] - 0.5;

    int dsf = W/w;

    // Compute local convolution
    for ( int j = 0 ; j < ks ; j++ ) {
        for ( int i = 0 ; i < ks ; i++ ) {

            int x = (int)floor(px - float(dsf)*0.5 + 0.5) + j;
            int y = (int)floor(py - float(dsf)*0.5 + 0.5) + i;
            if ( 0 <= x && x <= W-1 &&
                 0 <= y && y <= H-1 ) {

                O[m] += A[m*ks*ks + ks*i + j] * I[y*W + x];
            }
        }
    }
}

// forward splatting with tau warp and a single input view
static __global__ void cuda_deconvolution_nonsep_device_param( int W, int H, int R,
                                                               const cuflt *I,
                                                               int *index_array,
                                                               int start, int end,
                                                               size_t ks,
                                                               size_t dsf,
                                                               cuflt *A,
                                                               cuflt *warp_tau_x, cuflt *warp_tau_y,
                                                               cuflt *O,
                                                               cuflt *out_weight ) { // has the total weight contributed by u

    // Global thread index
    int idx = R * blockIdx.x + threadIdx.x + start;
    if ( idx>=end ) {
        return;
    }
    int m = index_array[idx];

    // get location in u, warp value in pixel coordinates
    // warps coordinates: bottom left corner
    // x_centered = x_corner - 0.5
    cuflt px = warp_tau_x[m] - 0.5;
    cuflt py = warp_tau_y[m] - 0.5;

    for ( int j = 0 ; j < ks ; j++ ) {
        for ( int i = 0 ; i < ks ; i++ ) {

            int x = (int)floor(px - float(dsf)*0.5 + 0.5) + j;
            int y = (int)floor(py - float(dsf)*0.5 + 0.5) + i;

            if ( 0 <= x && x <= W-1 &&
                 0 <= y && y <= H-1 ) {

                cuflt weight = 1.0;
                if ( A != 0 ) {
                    weight *= A[m*ks*ks + ks*i + j];
                }

                // we don't normalize by the sum of the weights
                out_weight[y*W + x] += weight;

                O[y*W + x] += weight * I[m];
            }
        }
    }
}

// forward splatting with tau warp and a single input view
static __global__ void cuda_deconvolution_nonsep_device_param( int W, int H, int R,
                                                               const cuflt *i_r, const cuflt *i_g, const cuflt *i_b,
                                                               int *index_array,
                                                               int start, int end,
                                                               size_t ks,
                                                               size_t dsf,
                                                               cuflt *A,
                                                               cuflt *warp_tau_x, cuflt *warp_tau_y,
                                                               cuflt *o_r, cuflt *o_g, cuflt *o_b,
                                                               cuflt *out_weight ) // has the total weight contributed by u
{
    // Global thread index
    int idx = R * blockIdx.x + threadIdx.x + start;
    if ( idx>=end ) {
        return;
    }
    int m = index_array[idx];
    
    // get location in u, warp value in pixel coordinates
    // warps coordinates: bottom left corner
    // x_centered = x_corner - 0.5
    cuflt px = warp_tau_x[m] - 0.5;
    cuflt py = warp_tau_y[m] - 0.5;

    for ( int j = 0 ; j < ks ; j++ ) {
        for ( int i = 0 ; i < ks ; i++ ) {
            
            int x = (int)floor(px - float(dsf)*0.5 + 0.5) + j;
            int y = (int)floor(py - float(dsf)*0.5 + 0.5) + i;

            if ( 0 <= x && x <= W-1 &&
                 0 <= y && y <= H-1 ) {

                cuflt weight = 1.0;
                if ( A != 0 ) {
                    weight *= A[m*ks*ks + ks*i + j];
                }

                out_weight[y*W + x] += weight;

                o_r[y*W + x] += weight * i_r[m];
                o_g[y*W + x] += weight * i_g[m];
                o_b[y*W + x] += weight * i_b[m];

            }
        }
    }
}

// weighted forward splatting with tau warp
static __global__ void cuda_weighted_deconvolution_nonsep_device_param( int W, int H, int R,
                                                                        const cuflt *i_r, const cuflt *i_g, const cuflt *i_b,
                                                                        int *index_array,
                                                                        int start, int end,
                                                                        size_t ks,
                                                                        size_t dsf,
                                                                        cuflt *A,
                                                                        cuflt *in_weights,
                                                                        cuflt *warp_tau_x, cuflt *warp_tau_y,
                                                                        cuflt *o_r, cuflt *o_g, cuflt *o_b,
                                                                        cuflt *out_weight ) // has the total weight contributed by u
{
    // Global thread index
    int idx = R * blockIdx.x + threadIdx.x + start;
    if ( idx>=end ) {
        return;
    }
    int m = index_array[idx];

    // get location in u, warp value in pixel coordinates
    // warps coordinates: bottom left corner
    // x_centered = x_corner - 0.5
    cuflt px = warp_tau_x[m] - 0.5;
    cuflt py = warp_tau_y[m] - 0.5;

    for ( int j = 0 ; j < ks ; j++ ) {
        for ( int i = 0 ; i < ks ; i++ ) {

            int x = (int)floor(px - float(dsf)*0.5 + 0.5) + j;
            int y = (int)floor(py - float(dsf)*0.5 + 0.5) + i;

            if ( 0 <= x && x <= W-1 &&
                 0 <= y && y <= H-1 ) {

                if ( i_r == 0 || o_r == 0 ||
                     i_g == 0 || o_g == 0 ||
                     i_b == 0 || o_b == 0 ) {
                    out_weight[y*W + x] += 1.0;
                } else {
                    cuflt weight = in_weights[m];
                    if ( A != 0 ) {
                        weight *= A[m*ks*ks + ks*i + j];
                    }

                    out_weight[y*W + x] += weight;

                    o_r[y*W + x] += weight * i_r[m];
                    o_g[y*W + x] += weight * i_g[m];
                    o_b[y*W + x] += weight * i_b[m];
                }
            }
        }
    }
}

// Box filtering
__global__ void cuda_set_A_box_filtering( int W, int H,
                                          int w, int h,
                                          size_t ks,
                                          cuflt *warp_tau_x,
                                          cuflt *warp_tau_y,
                                          cuflt* A ) {
    
    // Global thread index
    int mx = blockDim.x * blockIdx.x + threadIdx.x;
    int my = blockDim.y * blockIdx.y + threadIdx.y;
    if ( mx >= w || my >= h ) {
        return;
    }
    int m = my*w + mx;

    // get location in u, warp value in pixel coordinates
    // warps coordinates: bottom left corner
    // x_centered = x_corner - 0.5
    cuflt px = warp_tau_x[m] - 0.5;
    cuflt py = warp_tau_y[m] - 0.5;
    
    cuflt norm = 0.0;
    for ( int j = 0 ; j < ks ; j++ ) {
        for ( int i = 0 ; i < ks ; i++ ) {

            int x = (int)floor(px - float(ks)*0.5 + 0.5) + j;
            int y = (int)floor(py - float(ks)*0.5 + 0.5) + i;
            if ( 0 <= x && x <= W-1 &&
                 0 <= y && y <= H-1 ) {

                A[m*ks*ks + ks*i + j] = 1.0;
                norm += 1.0;
            } else {
                A[m*ks*ks + ks*i + j] = 0.0;
            }
        }
    }
    if ( norm != 0 ) { // shouldn't be null anyway
        for ( int i = 0 ; i < ks*ks ; i++ ) {
            
            A[m*ks*ks + i] /= norm;
        }
    }
}

__global__ void cuda_set_A_gaussian( int W, int H,
                                     int w, int h,
                                     size_t ks,
                                     cuflt *warp_tau_x,
                                     cuflt *warp_tau_y,
                                     cuflt* sparse_matrix ) {

    // Global thread index
    int mx = blockDim.x * blockIdx.x + threadIdx.x;
    int my = blockDim.y * blockIdx.y + threadIdx.y;
    if ( mx >= w || my >= h ) {
        return;
    }
    int m = my*w + mx;

    cuflt A = 0.0;
    cuflt B = 0.0;
    cuflt C = 0.0;
    cuflt D = 0.0;
    cuflt N = 0.0;

    if ( mx == 0 || my == 0 || mx == w-1 || my == h-1 ||
         warp_tau_x[m-1] < 0 || warp_tau_x[m-1] > W || warp_tau_y[m-1] < 0 || warp_tau_y[m-1] > H ||
         warp_tau_x[m+1] < 0 || warp_tau_x[m+1] > W || warp_tau_y[m+1] < 0 || warp_tau_y[m+1] > H ||
         warp_tau_x[m-w] < 0 || warp_tau_x[m-w] > W || warp_tau_y[m-w] < 0 || warp_tau_y[m-w] > H ||
         warp_tau_x[m+w] < 0 || warp_tau_x[m+w] > W || warp_tau_y[m+w] < 0 || warp_tau_y[m+w] > H ) {

        N = 1.0;
        A = 1.0;
        B = 0.0;
        C = 0.0;
        D = 1.0;

    } else {

        cuflt Dxx = 0.5 * (warp_tau_x[m+1] - warp_tau_x[m-1]);
        cuflt Dyx = 0.5 * (warp_tau_y[m+1] - warp_tau_y[m-1]);
        cuflt Dxy = 0.5 * (warp_tau_x[m+w] - warp_tau_x[m-w]);
        cuflt Dyy = 0.5 * (warp_tau_y[m+w] - warp_tau_y[m-w]);

        N = fabs(Dxx * Dyy - Dxy * Dyx);

        if ( N == 0.0 ) {

            N = 1.0;
            A = 1.0;
            B = 0.0;
            C = 0.0;
            D = 1.0;

        } else {

            A = Dyy*Dyy + Dyx*Dyx;
            B = -Dyy*Dxy - Dyx*Dxx;
            C = -Dyy*Dxy - Dyx*Dxx;
            D = Dxy*Dxy + Dxx*Dxx;
        }
    }

    // get location in u, warp value in pixel coordinates
    // warps coordinates: bottom left corner
    // x_centered = x_corner - 0.5
    cuflt px = warp_tau_x[m] - 0.5;
    cuflt py = warp_tau_y[m] - 0.5;

    cuflt norm = 0.0;
    for ( int j = 0 ; j < ks ; j++ ) {
        for ( int i = 0 ; i < ks ; i++ ) {

            int x = (int)floor(px - float(ks)*0.5 + 0.5) + j;
            int y = (int)floor(py - float(ks)*0.5 + 0.5) + i;
            if ( 0 <= x && x <= W-1 &&
                 0 <= y && y <= H-1 ) {

                // in pixel coordinates
                cuflt X = x - px;
                cuflt Y = y - py;

                sparse_matrix[m*ks*ks + ks*i + j] = exp(-(A*X*X + (B+C)*X*Y + D*Y*Y)/(72*N*N));
                norm += exp(-(A*X*X + (B+C)*X*Y + D*Y*Y)/(72*N*N));
            } else {
                sparse_matrix[m*ks*ks + ks*i + j] = 0.0;
            }
        }
    }
    if ( norm != 0 ) { // shouldn't be null anyway
        for ( int i = 0 ; i < ks*ks ; i++ ) {

            sparse_matrix[m*ks*ks + i] /= norm;
        }
    }
}

// Bilinear filtering, see "Limits on super-resolution and how to break them"
__global__ void cuda_set_A_bilinear( int W, int H,
                                     int w, int h,
                                     size_t ks,
                                     cuflt *warp_tau_x,
                                     cuflt *warp_tau_y,
                                     cuflt* A ) {

    // Global thread index
    int mx = blockDim.x * blockIdx.x + threadIdx.x;
    int my = blockDim.y * blockIdx.y + threadIdx.y;
    if ( mx >= w || my >= h ) {
        return;
    }
    int m = my*w + mx;

    // get location in u, warp value in pixel coordinates
    // warps coordinates: bottom left corner
    // x_centered = x_corner - 0.5
    cuflt px = warp_tau_x[m] - 0.5;
    cuflt py = warp_tau_y[m] - 0.5;

    int dsf = W/w;

    cuflt norm = 0.0;
    for ( int j = 0 ; j < ks ; j++ ) {
        for ( int i = 0 ; i < ks ; i++ ) {

            int x = (int)floor(px - cuflt(dsf)*0.5 + 0.5) + j;
            int y = (int)floor(py - cuflt(dsf)*0.5 + 0.5) + i;
            if ( 0 <= x && x <= W-1 &&
                 0 <= y && y <= H-1 ) {

                cuflt dx = 1.0;
                if ( j == 0 ) {
                    dx = (cuflt(x) + 0.5) - (px - 0.5*cuflt(dsf));
                }
                else if ( j == ks - 1 ) {
                    dx = (px + 0.5*cuflt(dsf)) - (cuflt(x) - 0.5);
                }
                cuflt dy = 1.0;
                if ( i == 0 ) {
                    dy = (cuflt(y) + 0.5) - (py - 0.5*cuflt(dsf));
                }
                else if ( i == ks - 1 ) {
                    dy = (py + 0.5*cuflt(dsf)) - (cuflt(y) - 0.5);
                }

                A[m*ks*ks + ks*i + j] = dx*dy;
                norm += dx*dy;
            } else {
                A[m*ks*ks + ks*i + j] = 0.0;
            }
        }
    }
    if ( norm != 0 ) { // shouldn't be null anyway
        for ( int i = 0 ; i < ks*ks ; i++ ) {

            A[m*ks*ks + i] /= norm;
        }
    }
}

__global__ void cuda_laplacian_device( int W, int H,
                                       cuflt *input,
                                       cuflt *output,
                                       bool *visibility ) { // for hole filling
    
    // Global thread index
    int ox = blockDim.x * blockIdx.x + threadIdx.x;
    int oy = blockDim.y * blockIdx.y + threadIdx.y;
    if ( ox >= W || oy >= H ) {
        return;
    }
    int o = oy*W + ox;
    
    cuflt weight = 0.0;
    output[o] = 0.0;
    
    if ( visibility != 0 ) { // we mind the holes
        if ( !visibility[o] || !visibility[o-1] || !visibility[o+1] || !visibility[o-W] || !visibility[o+W] ) { // it's a hole
            return;
        }
    }
    
    if ( ox > 0 ) {
        output[o] += input[o-1];
        weight += 1.0;
    }
    if ( oy > 0 ) {
        output[o] += input[o-W];
        weight += 1.0;
    }
    if ( ox < W-1 ) {
        output[o] += input[o+1];
        weight += 1.0;
    }
    if ( oy < H-1 ) {
        output[o] += input[o+W];
        weight += 1.0;
    }
    
    output[o] /= weight;
    output[o] -= input[o];
}

// Compute the divergence of a vector, forward differences
__global__ void cuda_divergence_device( int W, int H,
                                        cuflt *inputX,
                                        cuflt *inputY,
                                        cuflt *output ) {

    // Global thread index
    int ox = blockDim.x * blockIdx.x + threadIdx.x;
    int oy = blockDim.y * blockIdx.y + threadIdx.y;
    if ( ox >= W || oy >= H ) {
        return;
    }
    int o = oy*W + ox;

    output[o] = 0.0;

    if ( ox != 0 && ox != W-1 ) {
        output[o] += (inputX[o+1] - inputX[o-1])/2;
    }
    if ( oy != 0 && oy != H-1 ) {
        output[o] += (inputY[o+W] - inputY[o-W])/2;
    }
}

// Compute the gradient of the luminance, forward differences
__global__ void cuda_gradient_device( int W, int H,
                                      cuflt *r,
                                      cuflt *g,
                                      cuflt *b,
                                      cuflt *gradX,
                                      cuflt *gradY ) {

    // Global thread index
    int ox = blockDim.x * blockIdx.x + threadIdx.x;
    int oy = blockDim.y * blockIdx.y + threadIdx.y;
    if ( ox >= W || oy >= H ) {
        return;
    }
    int o = oy*W + ox;

    gradX[o] = 0.0;
    gradY[o] = 0.0;

    // luminance from rgb
    cuflt Ycc = 0.2126 * r[o] + 0.7152 * g[o] + 0.0722 * b[o];
    cuflt Ync = 0.2126 * r[o + 1] + 0.7152 * g[o + 1] + 0.0722 * b[o + 1];
    cuflt Ycn = 0.2126 * r[o + W] + 0.7152 * g[o + W] + 0.0722 * b[o + W];

    if ( ox < W-1 ) {
        gradX[o] = Ync - Ycc;
    }
    if ( oy < H-1 ) {
        gradY[o] += Ycn - Ycc;
    }
}

// Compute laplacian of u o tau_i
__global__ void cuda_laplacian_of_composition_device( int W, int H,
                                                      int w, int h,
                                                      const cuflt *input,
                                                      const size_t ks, // size of a kernel
                                                      const cuflt *A,
                                                      const cuflt *warp_tau_x, cuflt *warp_tau_y,
                                                      cuflt *output ) {
    
    // Global thread index
    int mx = blockDim.x * blockIdx.x + threadIdx.x;
    int my = blockDim.y * blockIdx.y + threadIdx.y;
    if ( mx >= w || my >= h ) {
        return;
    }
    int m = my*w + mx;
    
    output[m] = 0.0;
    
    // Boundary conditions
    if ( mx <= 0 && w-1 <= mx &&
         my <= 0 && h-1 <= my ) {
        return;
    }
    
    // test visibility
    if ( warp_tau_x[m] < 0 || warp_tau_y[m] < 0 || warp_tau_x[m] > W || warp_tau_y[m] > H ||
         warp_tau_x[m+1] < 0 || warp_tau_y[m+1] < 0 || warp_tau_x[m+1] > W || warp_tau_y[m+1] > H ||
         warp_tau_x[m-1] < 0 || warp_tau_y[m-1] < 0 || warp_tau_x[m-1] > W || warp_tau_y[m-1] > H ||
         warp_tau_x[m+w] < 0 || warp_tau_y[m+w] < 0 || warp_tau_x[m+w] > W || warp_tau_y[m+w] > H ||
         warp_tau_x[m-w] < 0 || warp_tau_y[m-w] < 0 || warp_tau_x[m-w] > W || warp_tau_y[m-w] > H ) {
        return;
    }
    
    // Compute the finite differences of the warps
    cuflt txx = 0.5*(warp_tau_x[m+1] - warp_tau_x[m-1]);
    cuflt txy = 0.5*(warp_tau_x[m+w] - warp_tau_x[m-w]);
    cuflt tyx = 0.5*(warp_tau_y[m+1] - warp_tau_y[m-1]);
    cuflt tyy = 0.5*(warp_tau_y[m+w] - warp_tau_y[m-w]);
    
    // get location in u, warp value in pixel coordinates
    // warps coordinates: bottom left corner
    // x_centered = x_corner - 0.5
    cuflt px = warp_tau_x[m] - 0.5;
    cuflt py = warp_tau_y[m] - 0.5;
    
    int p = py*W + px;
    if ( 1 <= px && px < W-1 &&
         1 <= py && py < H-1 ) {
        
        // Compute the local hessian of u
        cuflt Hxx = 0.25*(input[p-1] - 2*input[p] + input[p+1]);
        cuflt Hyy = 0.25*(input[p-W] - 2*input[p] + input[p+W]);
        cuflt Hxy = 0.25*(input[p-W-1] - input[p-W+1] - input[p+W-1] + input[p+W+1]);
        
        output[m] += (txx*txx + txy*txy)*Hxx + (tyx*tyx + tyy*tyy)*Hyy + 2*(txx*tyx + txy*tyy)*Hxy;
    }
    
    //    // Compute local convolution
    //    for ( int j = 0 ; j < ks ; j++ ) {
    //        for ( int i = 0 ; i < ks ; i++ ) {
    
    //            int x = int(px + 0.5) - ks / 2 + j;
    //            int y = int(py + 0.5) - ks / 2 + i;
    //            if ( 1 <= x && x < W-1 &&
    //                 1 <= y && y < H-1 ) {
    
    //                int p = y*W + x;
    //                // Compute the local hessian of u
    //                cuflt Hxx = 0.25*(input[p-1] - 2*input[p] + input[p+1]);
    //                cuflt Hyy = 0.25*(input[p-W] - 2*input[p] + input[p+W]);
    //                cuflt Hxy = 0.25*(input[p-W-1] - input[p-W+1] - input[p+W-1] + input[p+W+1]);
    
    //                output[m] += A[m*ks*ks + ks*i + j] * ((txx*txx + txy*txy)*Hxx + (tyx*tyx + tyy*tyy)*Hyy + 2*(txx*tyx + txy*tyy)*Hxy);
    //            }
    //        }
    //    }
}

// compute z from Jacobi method x(i+1) = B*x(i) + z
__global__ void cuda_jacobi_z( int W, int H,
                               cuflt *u, // roughly estimated solution
                               cuflt lambda,
                               cuflt *weights, // weights for the data term
                               cuflt *z ) {

    // Global thread index
    int ox = blockDim.x * blockIdx.x + threadIdx.x;
    int oy = blockDim.y * blockIdx.y + threadIdx.y;
    if ( ox >= W || oy >= H ) {
        return;
    }
    int o = oy*W + ox;

    cuflt data_w = lambda * weights[o];

    z[o] = (data_w * u[o] - z[o])/(4 + data_w);
}

// Compute B from Jacobi method x(i+1) = B*x(i) + z
__global__ void cuda_jacobi_B( int W, int H,
                               cuflt *x_in, // current solution x(i)
                               cuflt *z,
                               cuflt lambda,
                               cuflt *weights, // weights for the data term
                               cuflt *x_out ) { // x(i+1)

    // Global thread index
    int ox = blockDim.x * blockIdx.x + threadIdx.x;
    int oy = blockDim.y * blockIdx.y + threadIdx.y;
    if ( ox >= W || oy >= H ) {
        return;
    }
    int o = oy*W + ox;

    x_out[o] = 0.0;

    if ( lambda < 0 ) {

        return;
    }

    // we repeat the value at the boundaries
    if ( ox == 0 ) {
        x_out[o] += x_in[o];
    } else {
        x_out[o] += x_in[o-1];
    }
    if ( oy == 0 ) {
        x_out[o] += x_in[o];
    } else {
        x_out[o] += x_in[o-W];
    }
    if ( ox == W-1 ) {
        x_out[o] += x_in[o];
    } else {
        x_out[o] += x_in[o+1];
    }
    if ( oy == H-1 ) {
        x_out[o] += x_in[o];
    } else {
        x_out[o] += x_in[o+W];
    }

    x_out[o] /= (4 + lambda * weights[o]);
    x_out[o] += z[o];
}




