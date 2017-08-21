/* -*-c++-*- */

// compute the visibility from u
static __global__ void setup_visibility_mask( int W, int H, int R,
                                                          int *index_array,
                                                          int start, int end,
                                                          size_t ks,
                                                          size_t dsf,
                                                          float *warp_tau_x, float *warp_tau_y,
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
    float px = warp_tau_x[m] - 0.5;
    float py = warp_tau_y[m] - 0.5;
    
    // Compute local convolution
    for ( int j = 0 ; j < ks ; j++ ) {
        for ( int i = 0 ; i < ks ; i++ ) {
            
            int x = (int)floor(px - float(dsf)*0.5 + 0.5) + j;
            int y = (int)floor(py - float(dsf)*0.5 + 0.5) + i;
            if ( 0 <= x && x <= W-1 &&
                 0 <= y && y <= H-1 ) {

                float dx = 1.0;
                if ( j == 0 ) {
                    dx = (float(x) + 0.5) - (px - 0.5*float(dsf));
                }
                else if ( j == ks - 1 ) {
                    dx = (px + 0.5*float(dsf)) - (float(x) - 0.5);
                }
                float dy = 1.0;
                if ( i == 0 ) {
                    dy = (float(y) + 0.5) - (py - 0.5*float(dsf));
                }
                else if ( i == ks - 1 ) {
                    dy = (py + 0.5*float(dsf)) - (float(y) - 0.5);
                }

                if ( dx*dy != 0.0 ) {
                    visibility_mask[x + y*W] = true;
                }
            }
        }
    }
}

// Compute the gradient of the image using Sobel operator
static __global__ void compute_gradient( int W, int H,
                                                       float *r,
                                                       float *g,
                                                       float *b,
                                                       float *px, float *py ) {
    
    // Global thread index
    int ox = blockDim.x * blockIdx.x + threadIdx.x;
    int oy = blockDim.y * blockIdx.y + threadIdx.y;
    if ( ox >= W || oy >= H ) {
        return;
    }
    int o = oy*W + ox;
    
    float gradX = 0.0;
    float gradY = 0.0;
    
    if (0 < ox && ox < W-1 && 0 < oy && oy < H-1) { // grad = 0 on image boundaries
        
        int ppo = (oy-1)*W + (ox-1), pco = ppo+1, pno = pco+1;
        int cpo = (oy+0)*W + (ox-1), cco = cpo+1, cno = cco+1;
        int npo = (oy+1)*W + (ox-1), nco = npo+1, nno = nco+1;
        
        // luminance from rgb
        float Ypp = 0.2126 * r[ppo] + 0.7152 * g[ppo] + 0.0722 * b[ppo];
        float Ypc = 0.2126 * r[pco] + 0.7152 * g[pco] + 0.0722 * b[pco];
        float Ypn = 0.2126 * r[pno] + 0.7152 * g[pno] + 0.0722 * b[pno];
        
        float Ycp = 0.2126 * r[cpo] + 0.7152 * g[cpo] + 0.0722 * b[cpo];
        //float Ycc = 0.2126 * r[cco] + 0.7152 * g[cco] + 0.0722 * b[cco];
        float Ycn = 0.2126 * r[cno] + 0.7152 * g[cno] + 0.0722 * b[cno];
        
        float Ynp = 0.2126 * r[npo] + 0.7152 * g[npo] + 0.0722 * b[npo];
        float Ync = 0.2126 * r[nco] + 0.7152 * g[nco] + 0.0722 * b[nco];
        float Ynn = 0.2126 * r[nno] + 0.7152 * g[nno] + 0.0722 * b[nno];
        
        gradX = (Ypn + 2* Ycn + Ynn - Ypp - 2* Ycp - Ynp)/8.;
        gradY = (Ynp + 2* Ync + Ynn - Ypp - 2* Ypc - Ypn)/8.;
    }
    
    px[o] = gradX;
    py[o] = gradY;
}

// compute the angular weights from warped u gradient
static __global__ void angular_weights( int w, int h, // low res
                                                      const float *u_gradient_x, // vi domain, low res
                                                      const float *u_gradient_y,
                                                      float sigma_sensor,
                                                      const float *dpart_x, // vi domain, low res
                                                      const float *dpart_y, // dpart replaces aux_dmap_sigma*dtau/dz
                                                      float ugrad_threshold,
                                                      float *output ) {
    // Global thread index
    int mx = blockDim.x * blockIdx.x + threadIdx.x;
    int my = blockDim.y * blockIdx.y + threadIdx.y;
    if ( mx>=w || my>=h ) {
        return;
    }

    int m = mx + my*w;

    // warped u gradient
    float grad_x = u_gradient_x[m];
    float grad_y = u_gradient_y[m];

    // Threshold the gradient value. Do not threshold the dot product, direction is important
    float norm = sqrt(grad_x * grad_x + grad_y * grad_y);
    if( norm > ugrad_threshold ) {
        grad_x  = grad_x / norm * ugrad_threshold;
        grad_y  = grad_y / norm * ugrad_threshold;
    }

    // structured: float sigma_geom = aux_dmap_sigma * (grad_x * dx + grad_y * dy);
    float sigma_geom = grad_x * dpart_x[m] + grad_y * dpart_y[m];

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
static __global__ void gold_deform_weights( int W, int H,
                                                          int w, int h,
                                                          const float *warp_tau_x,
                                                          const float *warp_tau_y,
                                                          float *deforma_weights ) {
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

    float Dxx = 0.5 * (warp_tau_x[m+1] - warp_tau_x[m-1]);
    float Dyx = 0.5 * (warp_tau_y[m+1] - warp_tau_y[m-1]);
    float Dxy = 0.5 * (warp_tau_x[m+w] - warp_tau_x[m-w]);
    float Dyy = 0.5 * (warp_tau_y[m+w] - warp_tau_y[m-w]);

    float N = fabs(Dxx * Dyy - Dxy * Dyx);

#define MAX_DEFORM_WEIGHT 1.0
    if ( N > 1.0 / MAX_DEFORM_WEIGHT ) {
        deforma_weights[m] = 1.0 / N;
    }
    else {
        deforma_weights[m] = MAX_DEFORM_WEIGHT;
    }
}

__global__ void convolution_nonsep_param( int W, int H, // high res (input)
                                                      int w, int h, // low res (output)
                                                      size_t ks, // size of a kernel
                                                      float *A, // v_i = A*u
                                                      float *warp_tau_x, // vi domain, low res, values high res
                                                      float *warp_tau_y,
                                                      const float *i_r, const float *i_g, const float *i_b,
                                                      float *o_r, float *o_g, float *o_b ) {
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
    float px = warp_tau_x[m] - 0.5;
    float py = warp_tau_y[m] - 0.5;

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

__global__ void convolution_nonsep_param( int W, int H, // high res (input)
                                                      int w, int h, // low res (output)
                                                      size_t ks, // size of a kernel
                                                      float *A, // v_i = A*u
                                                      float *warp_tau_x, // vi domain, low res, values high res
                                                      float *warp_tau_y,
                                                      const float *I,
                                                      float *O ) {
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
    float px = warp_tau_x[m] - 0.5;
    float py = warp_tau_y[m] - 0.5;

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
static __global__ void deconvolution_nonsep_param( int W, int H, int R,
                                                               const float *I,
                                                               int *index_array,
                                                               int start, int end,
                                                               size_t ks,
                                                               size_t dsf,
                                                               float *A,
                                                               float *warp_tau_x, float *warp_tau_y,
                                                               float *O,
                                                               float *out_weight ) { // has the total weight contributed by u

    // Global thread index
    int idx = R * blockIdx.x + threadIdx.x + start;
    if ( idx>=end ) {
        return;
    }
    int m = index_array[idx];

    // get location in u, warp value in pixel coordinates
    // warps coordinates: bottom left corner
    // x_centered = x_corner - 0.5
    float px = warp_tau_x[m] - 0.5;
    float py = warp_tau_y[m] - 0.5;

    for ( int j = 0 ; j < ks ; j++ ) {
        for ( int i = 0 ; i < ks ; i++ ) {

            int x = (int)floor(px - float(dsf)*0.5 + 0.5) + j;
            int y = (int)floor(py - float(dsf)*0.5 + 0.5) + i;

            if ( 0 <= x && x <= W-1 &&
                 0 <= y && y <= H-1 ) {

                float weight = 1.0;
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
static __global__ void deconvolution_nonsep_param( int W, int H, int R,
                                                               const float *i_r, const float *i_g, const float *i_b,
                                                               int *index_array,
                                                               int start, int end,
                                                               size_t ks,
                                                               size_t dsf,
                                                               float *A,
                                                               float *warp_tau_x, float *warp_tau_y,
                                                               float *o_r, float *o_g, float *o_b,
                                                               float *out_weight ) // has the total weight contributed by u
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
    float px = warp_tau_x[m] - 0.5;
    float py = warp_tau_y[m] - 0.5;

    for ( int j = 0 ; j < ks ; j++ ) {
        for ( int i = 0 ; i < ks ; i++ ) {
            
            int x = (int)floor(px - float(dsf)*0.5 + 0.5) + j;
            int y = (int)floor(py - float(dsf)*0.5 + 0.5) + i;

            if ( 0 <= x && x <= W-1 &&
                 0 <= y && y <= H-1 ) {

                float weight = 1.0;
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
static __global__ void weighted_deconvolution_nonsep_param( int W, int H, int R,
                                                                        const float *i_r, const float *i_g, const float *i_b,
                                                                        int *index_array,
                                                                        int start, int end,
                                                                        size_t ks,
                                                                        size_t dsf,
                                                                        float *A,
                                                                        float *in_weights,
                                                                        float *warp_tau_x, float *warp_tau_y,
                                                                        float *o_r, float *o_g, float *o_b,
                                                                        float *out_weight ) // has the total weight contributed by u
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
    float px = warp_tau_x[m] - 0.5;
    float py = warp_tau_y[m] - 0.5;

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
                    float weight = in_weights[m];
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

// Bilinear filtering, see "Limits on super-resolution and how to break them"
__global__ void set_B_bilinear( int W, int H,
                                     int w, int h,
                                     size_t ks,
                                     float *warp_tau_x,
                                     float *warp_tau_y,
                                     float* A ) {

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
    float px = warp_tau_x[m] - 0.5;
    float py = warp_tau_y[m] - 0.5;

    int dsf = W/w;

    float norm = 0.0;
    for ( int j = 0 ; j < ks ; j++ ) {
        for ( int i = 0 ; i < ks ; i++ ) {

            int x = (int)floor(px - float(dsf)*0.5 + 0.5) + j;
            int y = (int)floor(py - float(dsf)*0.5 + 0.5) + i;
            if ( 0 <= x && x <= W-1 &&
                 0 <= y && y <= H-1 ) {

                float dx = 1.0;
                if ( j == 0 ) {
                    dx = (float(x) + 0.5) - (px - 0.5*float(dsf));
                }
                else if ( j == ks - 1 ) {
                    dx = (px + 0.5*float(dsf)) - (float(x) - 0.5);
                }
                float dy = 1.0;
                if ( i == 0 ) {
                    dy = (float(y) + 0.5) - (py - 0.5*float(dsf));
                }
                else if ( i == ks - 1 ) {
                    dy = (py + 0.5*float(dsf)) - (float(y) - 0.5);
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

__global__ void compute_laplacian( int W, int H,
                                       float *input,
                                       float *output,
                                       bool *visibility ) { // for hole filling
    
    // Global thread index
    int ox = blockDim.x * blockIdx.x + threadIdx.x;
    int oy = blockDim.y * blockIdx.y + threadIdx.y;
    if ( ox >= W || oy >= H ) {
        return;
    }
    int o = oy*W + ox;
    
    float weight = 0.0;
    output[o] = 0.0;
    
    if ( visibility != 0 ) { // we're doing hole filling
        if ( visibility[o] ) { // not a hole
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
__global__ void compute_divergence( int W, int H,
                                        float *inputX,
                                        float *inputY,
                                        float *output ) {

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

// compute z from Jacobi method x(i+1) = B*x(i) + z
__global__ void jacobi_z( int W, int H,
                               float *u, // roughly estimated solution
                               float lambda,
                               float *weights, // weights for the data term
                               float *z ) {

    // Global thread index
    int ox = blockDim.x * blockIdx.x + threadIdx.x;
    int oy = blockDim.y * blockIdx.y + threadIdx.y;
    if ( ox >= W || oy >= H ) {
        return;
    }
    int o = oy*W + ox;

    float data_w = lambda * weights[o];

    z[o] = (data_w * u[o] - z[o])/(4 + data_w);
}

// Compute B from Jacobi method x(i+1) = B*x(i) + z
__global__ void jacobi_B( int W, int H,
                               float *x_in, // current solution x(i)
                               float *z,
                               float lambda,
                               float *weights, // weights for the data term
                               float *x_out ) { // x(i+1)

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




