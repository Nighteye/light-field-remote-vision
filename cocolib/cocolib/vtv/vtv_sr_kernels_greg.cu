/* -*-c++-*- */

// backward rendering with warp (either beta or tau) and a single input view (Greg)
static __global__ void vtv_sr_bw_warp_device_unstructured( int W, int H,
                                                           const float *u, // input image
                                                           const float *warp_x,
                                                           const float *warp_y,
                                                           const float *visibility, // this is the other warp,
                                                           // if the value is negative the pixel in the input image is invalid: do not use it for interpolation
                                                           float *v, // output image
                                                           int sampling )
{
    // Global thread index
    int ox = blockDim.x * blockIdx.x + threadIdx.x;
    int oy = blockDim.y * blockIdx.y + threadIdx.y;
    if ( ox>=W || oy>=H ) {
        return;
    }
    int o = ox + oy*W;

    // if pixels goes outside the image, set it to black
    // although its warp should be invalid (-W, -H)
    if ( warp_x[o] < 0 || warp_y[o] < 0 || warp_x[o] > W || warp_y[o] > H ) {
        v[o] = 0.0f;
        return;
    }

    // get location in u
    const float uxv = warp_x[o] - 0.5f;
    const float uyv = warp_y[o] - 0.5f;
    int cx = (int)floor(uxv);
    int cy = (int)floor(uyv);
    int co = cx + cy*W;
    const float dx = uxv - float(cx);
    const float dy = uyv - float(cy);

    switch(sampling) {

    case 0:

        cx = (int)floor(uxv + 0.5f);
        cy = (int)floor(uyv + 0.5f);
        co = cx + cy*W;

        if (visibility[co] < 0) {
          //printf("Asking for invalid input pixel\n");
          v[o] = 0.;
        } else {
          v[o] = u[co];
        }

        break;

    case 1:
    {
        // transpose bilinear sampling
        float mxmym = (1.0f - dx) * (1.0f - dy);
        float mxpym = dx * (1.0f - dy);
        float mxmyp = (1.0f - dx) * dy;
        float mxpyp = dx * dy;

        float r = 0.0f;
        float weight = 1.0f;
        if ( cx >= 0 && cy >= 0  && visibility[co] >= 0) {
            r += u[ co + 0 ] * mxmym ;
        } else {
            weight -= mxmym;
        }
        if ( cx < W-1 && cy >= 0 && visibility[co+1] >= 0) {
            r += u[ co + 1 ] * mxpym ;
        } else {
            weight -= mxpym;
        }
        if ( cy < H-1 && cx >= 0 && visibility[co+W] >= 0) {
            r += u[ co + W ] * mxmyp ;
        } else {
            weight -= mxmyp;
        }
        if ( cx < W-1 && cy < H-1 && visibility[co+W+1] >= 0) {
            r += u[ co + W + 1 ] * mxpyp ;
        } else {
            weight -= mxpyp;
        }

        if (weight > 0) {
        // bilinear sampling (todo: texture lookup)
          v[o] = r/weight;
        } else {
          v[o] = 0;
        }

        break;
    }

    case 2:
    {
        const int px = cx - 1;
        const int nx = cx + 1;
        const int ax = cx + 2;
        const int py = cy - 1;
        const int ny = cy + 1;
        const int ay = cy + 2;

        // Dirichlet boundary conditions: value = 0

        const float Ipp = (px < 0 || py < 0 || px >= W || py >= H) ? 0 : u[px + py*W];
        const float Icp = (cx < 0 || py < 0 || cx >= W || py >= H) ? 0 : u[cx + py*W];
        const float Inp = (nx < 0 || py < 0 || nx >= W || py >= H) ? 0 : u[nx + py*W];
        const float Iap = (ax < 0 || py < 0 || ax >= W || py >= H) ? 0 : u[ax + py*W];

        const float Ipc = (px < 0 || cy < 0 || px >= W || cy >= H) ? 0 : u[px + cy*W];
        const float Icc = (cx < 0 || cy < 0 || cx >= W || cy >= H) ? 0 : u[cx + cy*W];
        const float Inc = (nx < 0 || cy < 0 || nx >= W || cy >= H) ? 0 : u[nx + cy*W];
        const float Iac = (ax < 0 || cy < 0 || ax >= W || cy >= H) ? 0 : u[ax + cy*W];

        const float Ipn = (px < 0 || ny < 0 || px >= W || ny >= H) ? 0 : u[px + ny*W];
        const float Icn = (cx < 0 || ny < 0 || cx >= W || ny >= H) ? 0 : u[cx + ny*W];
        const float Inn = (nx < 0 || ny < 0 || nx >= W || ny >= H) ? 0 : u[nx + ny*W];
        const float Ian = (ax < 0 || ny < 0 || ax >= W || ny >= H) ? 0 : u[ax + ny*W];

        const float Ipa = (px < 0 || ay < 0 || px >= W || ay >= H) ? 0 : u[px + ay*W];
        const float Ica = (cx < 0 || ay < 0 || cx >= W || ay >= H) ? 0 : u[cx + ay*W];
        const float Ina = (nx < 0 || ay < 0 || nx >= W || ay >= H) ? 0 : u[nx + ay*W];
        const float Iaa = (ax < 0 || ay < 0 || ax >= W || ay >= H) ? 0 : u[ax + ay*W];

        const float Ip = Icp + 0.5f*(dx*(-Ipp+Inp) + dx*dx*(2*Ipp-5*Icp+4*Inp-Iap) + dx*dx*dx*(-Ipp+3*Icp-3*Inp+Iap));
        const float Ic = Icc + 0.5f*(dx*(-Ipc+Inc) + dx*dx*(2*Ipc-5*Icc+4*Inc-Iac) + dx*dx*dx*(-Ipc+3*Icc-3*Inc+Iac));
        const float In = Icn + 0.5f*(dx*(-Ipn+Inn) + dx*dx*(2*Ipn-5*Icn+4*Inn-Ian) + dx*dx*dx*(-Ipn+3*Icn-3*Inn+Ian));
        const float Ia = Ica + 0.5f*(dx*(-Ipa+Ina) + dx*dx*(2*Ipa-5*Ica+4*Ina-Iaa) + dx*dx*dx*(-Ipa+3*Ica-3*Ina+Iaa));

        v[o] = Ic + 0.5f*(dy*(-Ip+In) + dy*dy*(2*Ip-5*Ic+4*In-Ia) + dy*dy*dy*(-Ip+3*Ic-3*In+Ia));
        break;
    }
    }
}

// forward splatting with warp (either beta or tau) and a single input view (Greg)
// bilinear contribution (TODO: nearest and bicubic)
static __global__ void vtv_sr_fw_warp_device_unstructured( int W, int H, int R,
                                                           cuflt *u, // input image
                                                           int *index_array,
                                                           int start, int end,
                                                           cuflt *fw_warp_x, cuflt *fw_warp_y,
                                                           cuflt *bw_warp_x, cuflt *bw_warp_y, // to test visibility
                                                           cuflt *out, // u image forward warped and 'auto-blended'
                                                           cuflt *out_weight ) // has the total weight contributed by u
{
    // Global thread index
    int idx = R * blockIdx.x + threadIdx.x + start;
    if ( idx>=end ) {
        return;
    }
    int o = index_array[idx];

    // only warp inside mask pixels
    if ( fw_warp_x[o] < 0 || fw_warp_y[o] < 0 || fw_warp_x[o] > W || fw_warp_y[o] > H ) {
        return;
    }

    // get location in u
    cuflt uv = u[o];
    cuflt uxv = fw_warp_x[o] - 0.5f;
    cuflt uyv = fw_warp_y[o] - 0.5f;
    int cx = (int)floor(uxv);
    int cy = (int)floor(uyv);

    // bilinear interpolation of the gradient
    int co = cx + cy*W;
    cuflt ax = uxv - cuflt(cx);
    cuflt ay = uyv - cuflt(cy);

    cuflt mxmym = (1.0f - ax) * (1.0f - ay);
    cuflt mxpym = ax * (1.0f - ay);
    cuflt mxmyp = (1.0f - ax) * ay;
    cuflt mxpyp = ax * ay;

    // test beta visibility of pixel destination
    if ( cx >= 0 && cy >= 0 ) {
        if ( 0 <= bw_warp_x[ co ] && bw_warp_x[ co ] <= W &&
             0 <= bw_warp_y[ co ] && bw_warp_y[ co ] <= H ) {
          out[ co + 0 ] += uv * mxmym;
          out_weight[ co + 0 ] += mxmym;
        }

    }
    if ( cx < W-1 && cy >= 0  ) {
      if ( 0 <= bw_warp_x[ co +1] && bw_warp_x[ co +1] <= W &&
           0 <= bw_warp_y[ co +1] && bw_warp_y[ co +1] <= H ) {
        out[ co + 1 ] += uv * mxpym;
        out_weight[ co + 1 ] += mxpym;
      }
    }
    if ( cy < H-1 && cx >= 0 ) {
      if ( 0 <= bw_warp_x[ co +W] && bw_warp_x[ co +W] <= W &&
           0 <= bw_warp_y[ co +W] && bw_warp_y[ co +W] <= H ) {
        out[ co + W ] += uv * mxmyp;
        out_weight[ co + W ] += mxmyp;
      }
    }
    if ( cx < W-1 && cy < H-1 ) {
      if ( 0 <= bw_warp_x[ co + W + 1] && bw_warp_x[ co + W + 1] <= W &&
           0 <= bw_warp_y[ co + W + 1] && bw_warp_y[ co + W + 1] <= H ) {
        out[ co + W + 1 ] += uv * mxpyp;
        out_weight[ co + W + 1 ] += mxpyp;
      }
    }
}

// compute the tau warp using the dmap (Greg)
static __global__ void vtv_sr_compute_tau_device( int W, int H,
                                                  const float *dmap, // input dispariy map corresponding to vi
                                                  float dx, float dy, // optical center displacement
                                                  float *tau_warp_x,
                                                  float *tau_warp_y) // output tau warp of view vi
{
    // Global thread index
    int ox = blockDim.x * blockIdx.x + threadIdx.x;
    int oy = blockDim.y * blockIdx.y + threadIdx.y;
    if ( ox>=W || oy>=H ) {
        return;
    }
    int o = ox + oy*W;

    // get location in u
    // warps coordinates: bottom left corner
    // x_corner = x_centered + 0.5
    float d = dmap[o];
    tau_warp_x[o] = float(ox) + d * dx + 0.5f;
    tau_warp_y[o] = float(oy) + d * dy + 0.5f;

    if ( tau_warp_x[o] < 0 || tau_warp_y[o] < 0 || tau_warp_x[o] > W || tau_warp_y[o] > H ) {
        tau_warp_x[o] = -W;
        tau_warp_y[o] = -H;
        return;
    }
}

// compute tau visibility using the visibility mask (Greg)
static __global__ void vtv_sr_set_tau_visibility_device( int W, int H,
                                                         const bool *visibility_mask, // binary mask corresponding to vi
                                                         float *tau_warp_x,
                                                         float *tau_warp_y) // output tau warp of view vi
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
        tau_warp_x[o] = -W;
        tau_warp_y[o] = -H;
    }
}

// compute the visibility mask of beta warp using the dmap. If visible from vi, set to 1. (Greg)
static __global__ void vtv_sr_compute_beta_visibility_accumulate_device( int W, int H, int R,
                                                                         int *index_array,
                                                                         int start, int end,
                                                                         float *dmap,
                                                                         float dx, float dy,
                                                                         bool *visibility, // forward warp only pixels visible from u
                                                                         float *out_x, // beta warp x, forward warped and 'auto-blended'
                                                                         float *out_y) // beta warp y
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
    float w = visibility[o];
    if ( w == 0 ) {
        return;
    }

    // get location in u
    float d = dmap[o];
    float uxv = ox + d * dx;
    float uyv = oy + d * dy;
    int px = (int)floor(uxv);
    int py = (int)floor(uyv);
    if ( px < -1 || py < -1 || px > W-1 || py > H-1 ) {
        return;
    }
    int po = px + py*W;

    float ax = uxv - float(px);
    float ay = uyv - float(py);

    float mxmym = (1.0f - ax) * (1.0f - ay);
    float mxpym = ax * (1.0f - ay);
    float mxmyp = (1.0f - ax) * ay;
    float mxpyp = ax * ay;

    if ( px >= 0 && py >= 0 && mxmym > 0 ) {

        out_x[ po + 0 ] = 1.0f;
        out_y[ po + 0 ] = 1.0f;
    }
    if ( px < W-1 && py >= 0 && mxpym > 0 ) {

        out_x[ po + 1 ] = 1.0f;
        out_y[ po + 1 ] = 1.0f;
    }
    if ( py < H-1 && px >= 0 && mxmyp > 0 ) {

        out_x[ po + W ] = 1.0f;
        out_y[ po + W ] = 1.0f;
    }
    if ( px < W-1 && py < H-1 && mxpyp > 0 ) {

        out_x[ po + W + 1 ] = 1.0f;
        out_y[ po + W + 1 ] = 1.0f;
    }
}

// Finalize the beta visibility by setting the non visible pixels to (-W, -H) and the others to 0.0 (Greg)
static __global__ void vtv_sr_set_beta_visibility_device( int W, int H,
                                                          float *out_x, // beta warp x, forward warped and 'auto-blended'
                                                          float *out_y) // beta warp y
{
    // location in u
    int ox = blockDim.x * blockIdx.x + threadIdx.x;
    int oy = blockDim.y * blockIdx.y + threadIdx.y;
    int o = ox + oy*W;
    if ( ox>=W || oy>=H ) {
        return;
    }

    if ( out_x[ o ] > 0 ) {
        out_x[ o ] = 0.0f;
        out_y[ o ] = 0.0f;
    } else {
        out_x[ o ] = -W;
        out_y[ o ] = -H;
    }
}

// compute the beta warp using the dmap. Each u element contributes to 4 neighboring pixels
// with its corresponding weight vmask = 0 or 1 (visibility) (Greg)
static __global__ void vtv_sr_compute_beta_accumulate_device( int W, int H, int R,
                                                              int *index_array,
                                                              int start, int end,
                                                              float *dmap,
                                                              float dx, float dy,
                                                              bool *visibility, // visible from u
                                                              float *out_x, // beta warp x, forward warped and 'auto-blended'
                                                              float *out_y, // beta warp y
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
    if ( visibility[o] == 0 ) {
        return;
    }

    // get location in u
    float d = dmap[o];
    float uxv = ox + d * dx;
    float uyv = oy + d * dy;
    int px = (int)floor(uxv);
    int py = (int)floor(uyv);
    if ( px < -1 || py < -1 || px > W-1 || py > H-1 ) {
        return;
    }
    int po = px + py*W;
    float ax = uxv - float(px);
    float ay = uyv - float(py);

    float mxmym = (1.0f - ax) * (1.0f - ay);
    float mxpym = ax * (1.0f - ay);
    float mxmyp = (1.0f - ax) * ay;
    float mxpyp = ax * ay;

    // warp coordinates: bottom left corner
    // x_corner = x_centered + 0.5

    if ( px >= 0 && py >= 0 ) {
        out_x[ po + 0 ] += ( float(ox) + 0.5f ) * mxmym;
        out_y[ po + 0 ] += ( float(oy) + 0.5f ) * mxmym;
        out_weight[ po + 0 ] += mxmym;
    }
    if ( px < W-1 && py >= 0 ) {
        out_x[ po + 1 ] += ( float(ox) + 0.5f ) * mxpym;
        out_y[ po + 1 ] += ( float(oy) + 0.5f ) * mxpym;
        out_weight[ po + 1 ] += mxpym;
    }
    if ( py < H-1 && px >= 0 ) {
        out_x[ po + W ] += ( float(ox) + 0.5f ) * mxmyp;
        out_y[ po + W ] += ( float(oy) + 0.5f ) * mxmyp;
        out_weight[ po + W ] += mxmyp;
    }
    if ( px < W-1 && py < H-1 ) {
        out_x[ po + W + 1 ] += ( float(ox) + 0.5f ) * mxpyp;
        out_y[ po + W + 1 ] += ( float(oy) + 0.5f ) * mxpyp;
        out_weight[ po + W + 1 ] += mxpyp;
    }
}

// compute the dpart of a view (Greg)
static __global__ void vtv_sr_compute_dpart_device( int W, int H,
                                                    float dx, float dy, // optical center displacement
                                                    const bool *visibility_mask, // binary mask corresponding to vi
                                                    float *dpart_x,
                                                    float *dpart_y) // output dpart of view vi
{
    // Global thread index
    int ox = blockDim.x * blockIdx.x + threadIdx.x;
    int oy = blockDim.y * blockIdx.y + threadIdx.y;
    if ( ox>=W || oy>=H ) {
        return;
    }
    int o = ox + oy*W;

    // set to 0 for occluded pixels (weight = 0 anyway)
    if ( visibility_mask[o] == 0 ) {
        dpart_x[o] = 0.0f;
        dpart_y[o] = 0.0f;
        return;
    }

    // for simplified camera config, "bayesian view synthesis" says dpart = sigma_dmap*(c-ci), c-ci = (dx, dy)
    // hack: here we store (c-ci) in dpart and sigma_dmap apart so that the product dpart*sigma_dmap=sigma_z*dtau/dz
    dpart_x[o] = dx;
    dpart_y[o] = dy;
}

// compute the deformation weights |det D beta_i|^(-1) (Greg)
static __global__ void vtv_sr_init_beta_mask_weight_device( int W, int H,
                                                            const float *warp_beta_x,
                                                            const float *warp_beta_y,
                                                            float *vmask_weighted ) // output gradient weights
{
    // Global thread index
    int ox = blockDim.x * blockIdx.x + threadIdx.x;
    int oy = blockDim.y * blockIdx.y + threadIdx.y;
    if ( ox>=W || oy>=H ) {
        return;
    }

    int o = ox + oy*W;
    if ( warp_beta_x[o] < 0 || warp_beta_x[o] > W || warp_beta_y[o] < 0 || warp_beta_y[o] > H ) { // pixel not visible from input view vi
        vmask_weighted[o] = 0.0f;
        return;
    }

    // Neumann condition : gradient = 0 -> weight = 1.0
    if ( ox == 0 || oy == 0 || ox == W-1 || oy == H-1 ) {
        vmask_weighted[o] = 1.0f;
        return;
    }

    float Dxx = 0.5f * (warp_beta_x[o+1] - warp_beta_x[o-1]);
    float Dyx = 0.5f * (warp_beta_y[o+1] - warp_beta_y[o-1]);
    float Dxy = 0.5f * (warp_beta_x[o+W] - warp_beta_x[o-W]);
    float Dyy = 0.5f * (warp_beta_y[o+W] - warp_beta_y[o-W]);

    float N = fabs(Dxx * Dyy - Dxy * Dyx);

#define GRAD_W_MAX 20.0f
    if ( N > 1.0f / GRAD_W_MAX ) {
        vmask_weighted[o] = 1.0f / N;
    }
    else {
        vmask_weighted[o] = GRAD_W_MAX;
    }
}

// compute the deformation weights |det D tau_i|^(-1) (Greg)
static __global__ void vtv_sr_init_tau_mask_weight_device( int W, int H,
                                                           const bool *visibility_mask,
                                                           const float *warp_tau_x,
                                                           const float *warp_tau_y,
                                                           float *vmask_weighted ) // output gradient weights
{
    // Global thread index
    int ox = blockDim.x * blockIdx.x + threadIdx.x;
    int oy = blockDim.y * blockIdx.y + threadIdx.y;
    if ( ox>=W || oy>=H ) {
        return;
    }

    int o = ox + oy*W;

    // set weight of occluded pixels to 0
    if ( visibility_mask[o] == 0 ) {
        vmask_weighted[o] = 0;
        return;
    }

    // Neumann condition : gradient = 0 -> weight = 1.0
    if ( ox == 0 || oy == 0 || ox == W-1 || oy == H-1 ) {
        vmask_weighted[o] = 1.0f;
        return;
    }

    float Dxx = 0.5f * (warp_tau_x[o+1] - warp_tau_x[o-1]);
    float Dyx = 0.5f * (warp_tau_y[o+1] - warp_tau_y[o-1]);
    float Dxy = 0.5f * (warp_tau_x[o+W] - warp_tau_x[o-W]);
    float Dyy = 0.5f * (warp_tau_y[o+W] - warp_tau_y[o-W]);

    float N = fabs(Dxx * Dyy - Dxy * Dyx);


#define GRAD_W_MAX 20.0f
    if ( N > 1.0f / GRAD_W_MAX ) {
        vmask_weighted[o] = 1.0f / N;
    }
    else {
        vmask_weighted[o] = GRAD_W_MAX;
    }

}


// compute the angular weights with u gradient (Greg)
static __global__ void vtv_sr_u_gradient_weight_unstructured_device( int W, int H,
                                                                     float *u_gradient_x, //u domain
                                                                     float *u_gradient_y,
                                                                     float *warp_tau_x, //vi domain
                                                                     float *warp_tau_y,
                                                                     float sigma_sensor,
                                                                     float *dpart_x, //vi domain
                                                                     float *dpart_y, // dpart replaces aux_dmap_sigma*dtau/dz
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

    // if outside : weight = 0
    if ( warp_tau_x[o] < 0 || warp_tau_y[o] < 0 || warp_tau_x[o] > W || warp_tau_y[o] > H ) {
        output[o] = 0.;
        return;
    }

    // get location in u
    // warps coordinates: bottom left corner
    // x_centered = x_corner - 0.5f
    float uxv = warp_tau_x[o] - 0.5f;
    float uyv = warp_tau_y[o] - 0.5f;
    int px = (int)floor(uxv);
    int py = (int)floor(uyv);

    // bilinear interpolation of the gradient
    int po = px + py*W;
    float ax = uxv - float(px);
    float ay = uyv - float(py);

    float mxmym = (1.0f - ax) * (1.0f - ay);
    float mxpym = ax * (1.0f - ay);
    float mxmyp = (1.0f - ax) * ay;
    float mxpyp = ax * ay;

    float grad_x = 0.0f;
    float grad_y = 0.0f;
    float weight = 1.0f;

    if ( px >= 0 && py >= 0 ) {
        grad_x += u_gradient_x[ po + 0 ] * mxmym;
        grad_y += u_gradient_y[ po + 0 ] * mxmym;
    } else {
        weight -= mxmym;
    }
    if ( px < W-1 && py >= 0 ) {
        grad_x += u_gradient_x[ po + 1 ] * mxpym;
        grad_y += u_gradient_y[ po + 1 ] * mxpym;
    } else {
        weight -= mxpym;
    }
    if ( py < H-1 && px >= 0 ) {
        grad_x += u_gradient_x[ po + W ] * mxmyp;
        grad_y += u_gradient_y[ po + W ] * mxmyp;
    } else {
        weight -= mxmyp;
    }
    if ( px < W-1 && py < H-1 ) {
        grad_x += u_gradient_x[ po + W + 1 ] * mxpyp;
        grad_y += u_gradient_y[ po + W + 1 ] * mxpyp;
    } else {
        weight -= mxpyp;
    }

    if ( weight != 0.0f) {
        grad_x /= weight;
        grad_y /= weight;
    } else {
        grad_x = 0.0f;
        grad_y = 0.0f;
    }

    // Threshold the gradient value. Do not threshold the dot product, direction is important
    float norm = sqrt(grad_x * grad_x + grad_y * grad_y);
    if( norm > ugrad_threshold ) {
        grad_x  = grad_x / norm * ugrad_threshold;
        grad_y  = grad_y / norm * ugrad_threshold;
    }

    // structured: float sigma_geom = aux_dmap_sigma * (grad_x * dx + grad_y * dy);
    float sigma_geom = grad_x * dpart_x[o] + grad_y * dpart_y[o];

    // Threshold sigma_geom: do not forget that this magnitude represents a color variance
    //    Our random variable represents a color and lives in [0,1]
    //    The variance of a variable in [0,1] can not be greater than 0.5^2
    if (sigma_geom > 0.5) {
      sigma_geom = 0.5;
    }

    // The maximum weight is achieved when the u gradient is 0
    // Weight = 1./(sigma_sensor * sigma_sensor)
    output[o] = 1.0 / (sigma_geom * sigma_geom + sigma_sensor * sigma_sensor);
}

// compute the omega_i visibility masks using the tau warps (Greg)
// actually it just checks whether the warp is valid (visible pixel) or not
static __global__ void vtv_sr_visibility_from_tau_device( int W, int H,
                                                          const float *warp_tau_x,
                                                          const float *warp_tau_y, // input tau warp of view vi
                                                          bool *visibility_mask) // output binary mask corresponding to vi

{
    // Global thread index
    int ox = blockDim.x * blockIdx.x + threadIdx.x;
    int oy = blockDim.y * blockIdx.y + threadIdx.y;
    if ( ox>=W || oy>=H ) {
        return;
    }
    int o = ox + oy*W;

    // let visibility mask to 0 if occluded pixel
    if ( warp_tau_x[o] < 0 || warp_tau_y[o] < 0 || warp_tau_x[o] > W || warp_tau_y[o] > H ) {
        visibility_mask[o] = false;
    } else {
        visibility_mask[o] = true;
    }
}

// compute upsampling operation by bilinear interpolation
// coordinates are pixel centers
static __global__ void vtv_sr_bilinear_upsample_device( int W, int H, // Hi-res size
                                                        int w, int h, // Lo-res size
                                                        float F,      // Scale factor
                                                        float *m,     // lo-res matrix
                                                        float *M,    // hi-res result
                                                        bool *visibility)
{
    // Global thread index
    int ox = blockDim.x * blockIdx.x + threadIdx.x;
    int oy = blockDim.y * blockIdx.y + threadIdx.y;
    if ( ox>=W || oy>=H ) {
        return;
    }
    int o = ox + oy*W;

    if (visibility[o] != 1) {
      M[o]=0.;
      return;
    }
    // coordinates in low res image (float)
    // (F-1)/2 offset
    float ux = (float(ox) - (F-1.0f)/2.0f) / F;
    float uy = (float(oy) - (F-1.0f)/2.0f) / F;

    // bilinear interpolation
    int px = (int)floor(ux);
    int py = (int)floor(uy);
    int po = px + py*w;
    float ax = ux - float(px);
    float ay = uy - float(py);

    float mxmym = (1.0f - ax) * (1.0f - ay);
    float mxpym = ax * (1.0f - ay);
    float mxmyp = (1.0f - ax) * ay;
    float mxpyp = ax * ay;

    float r = 0.0f;
    float weight = 1.0f;
    if ( px >= 0 && py >= 0 ) {
        r += m[ po + 0 ] * mxmym;
    } else {
        weight -= mxmym;
    }
    if ( px < w-1 && py >= 0 ) {
        r += m[ po + 1 ] * mxpym;
    } else {
        weight -= mxpym;
    }
    if ( py < h-1 && px >= 0 ) {
        r += m[ po + w ] * mxmyp;
    } else {
        weight -= mxmyp;
    }
    if ( px < w-1 && py < h-1) {
        r += m[ po + w + 1 ] * mxpyp;
    } else {
        weight -= mxpyp;
    }

    M[o] = r/weight;
}

static __global__ void compute_luminance_device( int W, int H,
                                                 const cuflt *r,
                                                 const cuflt *g,
                                                 const cuflt *b,
                                                 cuflt *lu)
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  int o = oy*W + ox;

  lu[o] =  0.2126f * r[o] + 0.7152f * g[o] + 0.0722f * b[o];
}


static __global__ void vtv_sr_compute_gradient_device( int W, int H,
                                                       cuflt *r,
                                                       cuflt *g,
                                                       cuflt *b,
                                                       cuflt *visibility,
                                                       cuflt *px, cuflt *py )
{
    // Global thread index
    int ox = blockDim.x * blockIdx.x + threadIdx.x;
    int oy = blockDim.y * blockIdx.y + threadIdx.y;
    if ( ox >= W || oy >= H ) {
      return;
    }
    int o = oy*W + ox;

    cuflt gradX = 0.0f;
    cuflt gradY = 0.0f;

    if (false) {
      if (ox < W-1 && oy < H-1) {
        if (visibility[o] > 0 || visibility[o+1] > 0 || visibility[o+W] > 0) {

          // luminance from rgb
          cuflt Yc = 0.2126f * r[o] + 0.7152f * g[o] + 0.0722f * b[o];
          cuflt Yxn = 0.2126f * r[o+1] + 0.7152f * g[o+1] + 0.0722f * b[o+1];
          cuflt Yyn = 0.2126f * r[o+W] + 0.7152f * g[o+W] + 0.0722f * b[o+W];

          // Step for each p equals gradient component of phi
          // Forward differences, Neumann
          // X
          if ( ox < W-1 ) {
              gradX = Yxn - Yc;
          }

          // Y
          if ( oy < H-1 ) {
            gradY = Yyn - Yc;
          }
        }
      }
    } else {
      if (0 < ox && ox < W-1 && 0 < oy && oy < H-1) {
        int ppo = (oy-1)*W + (ox-1), pco = ppo+1, pno = pco+1;
        int cpo = (oy+0)*W + (ox-1), cco = cpo+1, cno = cco+1;
        int npo = (oy-1)*W + (ox-1), nco = npo+1, nno = nco+1;

        if (visibility[ppo] > 0 || visibility[pco] > 0 || visibility[pno] > 0 ||
            visibility[cpo] > 0 || visibility[cco] > 0 || visibility[cno] > 0 ||
            visibility[npo] > 0 || visibility[nco] > 0 || visibility[nno] > 0 ) {

          // luminance from rgb
          cuflt Ypp = 0.2126f * r[ppo] + 0.7152f * g[ppo] + 0.0722f * b[ppo];
          cuflt Ypc = 0.2126f * r[pco] + 0.7152f * g[pco] + 0.0722f * b[pco];
          cuflt Ypn = 0.2126f * r[pno] + 0.7152f * g[pno] + 0.0722f * b[pno];

          cuflt Ycp = 0.2126f * r[cpo] + 0.7152f * g[cpo] + 0.0722f * b[cpo];
          //cuflt Ycc = 0.2126f * r[cco] + 0.7152f * g[cco] + 0.0722f * b[cco];
          cuflt Ycn = 0.2126f * r[cno] + 0.7152f * g[cno] + 0.0722f * b[cno];

          cuflt Ynp = 0.2126f * r[npo] + 0.7152f * g[npo] + 0.0722f * b[npo];
          cuflt Ync = 0.2126f * r[nco] + 0.7152f * g[nco] + 0.0722f * b[nco];
          cuflt Ynn = 0.2126f * r[nno] + 0.7152f * g[nno] + 0.0722f * b[nno];

          gradX = (Ypn + 2* Ycn + Ynn - Ypp - 2* Ycp - Ynp)/8.;
          gradY = (Ynp + 2* Ync + Ynn - Ypp - 2* Ypc - Ypn)/8.;
        }
      }
    }

    px[o] = gradX;
    py[o] = gradY;
}

// blend two warped image. the backward warped image is stored in output. input is the forward warped iamge (splatting).
// deformation weights are |det D g_i|^(-1), where g is the backward warp
static __global__ void vtv_sr_blend_warped_images_device( int W, int H,
                                                       cuflt *deform_weights,
                                                       cuflt *input, // forward warped
                                                       cuflt *output) // backward warped
{
    // Global thread index
    int ox = blockDim.x * blockIdx.x + threadIdx.x;
    int oy = blockDim.y * blockIdx.y + threadIdx.y;
    if ( ox >= W || oy >= H ) {
        return;
    }
    int o = oy*W + ox;

    if ( deform_weights[o] < 1.0 ) { // for image compression zone, we take the forward warped image (splatting)

        output[o] = input[o];
    }
}

static __device__ int bilinear_interpolation (int W, int H,
                                              cuflt *img,
                                              cuflt ux, cuflt uy,
                                              cuflt *result,
                                              cuflt *mask = 0) {
  // bilinear interpolation
  int px = (int)floor(ux);
  int py = (int)floor(uy);
  int po = px + py*W;

  float ax = ux - float(px);
  float ay = uy - float(py);

  float mxmym = (1.0f - ax) * (1.0f - ay);
  float mxpym = ax * (1.0f - ay);
  float mxmyp = (1.0f - ax) * ay;
  float mxpyp = ax * ay;

  float weight = 1.0f;

  *result = 0;

  if ( px >= 0 && py >= 0 ) {
    if (!mask || mask[po+0] > 0) {
      *result += img[ po + 0 ] * mxmym;
    } else {
      weight -= mxmym;
    }
  } else {
    weight -= mxmym;
  }

  if ( px < W-1 && py >= 0 ) {
    if (!mask || mask[po+1] > 0) {
      *result += img[ po + 1 ] * mxpym;
    } else {
      weight -= mxpym;
    }
  } else {
    weight -= mxpym;
  }

  if ( py < H-1 && px >= 0 ) {
    if (!mask || mask[po+W] > 0) {
      *result += img[ po + W ] * mxmyp;
    } else {
      weight -= mxmyp;
    }
  } else {
    weight -= mxmyp;
  }

  if ( px < W-1 && py < H-1) {
    if (!mask || mask[po+W+1] > 0) {
      *result += img[ po + W + 1 ] * mxpyp;
    } else {
      weight -= mxpyp;
    }
  } else {
    weight -= mxpyp;
  }

  if (weight > 0) {
    *result = *result/weight;
    return 0;
  }
  return -1;
}

/*
static __global__ void check_warp_coherence_device( int W, int H,
                                                    cuflt *warp_beta_x,
                                                    cuflt *warp_beta_y,
                                                    int w, int h,
                                                    cuflt *warp_tau_x,
                                                    cuflt *warp_tau_y)
{
    // Global thread index
    int ox = blockDim.x * blockIdx.x + threadIdx.x;
    int oy = blockDim.y * blockIdx.y + threadIdx.y;
    if ( ox >= W || oy >= H ) {
        return;
    }
    // Gamma coordinate
    int o = oy*W + ox;

    if ( warp_beta_x[o] < 0 || W < warp_beta_x[o] ||
         warp_beta_y[o] < 0 || H < warp_beta_y[o]  ) {
        return;
    }
    // Omegai coordinate
    // warps coordinates: bottom left corner
    // x_centered = x_corner - 0.5f
    cuflt i_xv = warp_beta_x[o] - 0.5f;
    cuflt i_yv = warp_beta_y[o] - 0.5f;

    // We are restrictive to avoid warp artifacts
    const float THRESHOLD = 0.5;

    // round for nearest pixel
    int i_px = (int)floor(i_xv+0.5);
    int i_py = (int)floor(i_yv+0.5);
    int po = i_px + i_py * w;

    float warp_x_value = warp_tau_x[po];
    float warp_y_value = warp_tau_y[po];

    int res;
    res = bilinear_interpolation(w,h, warp_tau_x, i_xv, i_yv, &warp_x_value, warp_tau_x);
    if (res < 0) {
      warp_x_value = -1; //set to invalid
    }

    res = bilinear_interpolation(w,h, warp_tau_y, i_xv, i_yv, &warp_y_value, warp_tau_y);
    if (res < 0) {
      warp_y_value = -1; //set to invalid
    }

    bool invalid = false;
    if ( warp_x_value < 0 || w < warp_x_value ||
         warp_y_value < 0 || h < warp_y_value  ) {
      // INVALID
      invalid = true;
    } else if ( fabs( (warp_x_value- 0.5f) - ox) >=  THRESHOLD) {
      invalid = true;
      //printf("Invalidating warp at %d %d %g %g\n", ox, oy, warp_tau_x[po]-0.5, warp_tau_y[po]-0.5);
    } else if ( fabs( (warp_y_value- 0.5f) - oy ) >= THRESHOLD ) {
      //printf("Invalidating warp at %d %d %g %g\n", ox, oy, warp_tau_x[po]-0.5, warp_tau_y[po]-0.5);
      invalid = true;
    }

    if (invalid) {
      warp_beta_x[o] = -W;
      warp_beta_y[o] = -H;
    } else {
      // set integer components so that everytime we fall in the same pixel
      warp_beta_x[o] = i_px + 0.5f;
      warp_beta_y[o] = i_py + 0.5f;
    }
}*/

static __global__ void check_warp_coherence_device( int W, int H,
                                                    cuflt *warp_beta_x,
                                                    cuflt *warp_beta_y,
                                                    cuflt *warp_beta_def_x,
                                                    cuflt *warp_beta_def_y,
                                                    int w, int h,
                                                    cuflt *warp_tau_x,
                                                    cuflt *warp_tau_y)
{
    // Global thread index
    int ox = blockDim.x * blockIdx.x + threadIdx.x;
    int oy = blockDim.y * blockIdx.y + threadIdx.y;
    if ( ox >= W || oy >= H ) {
        return;
    }
    // Gamma coordinate
    int o = oy*W + ox;

    if ( warp_beta_x[o] < 0 || W < warp_beta_x[o] ||
         warp_beta_y[o] < 0 || H < warp_beta_y[o]  ) {
        return;
    }

    // Omegai floating pixel coordinate
    cuflt vi_x = warp_beta_x[o] - 0.5f;
    cuflt vi_y = warp_beta_y[o] - 0.5f;

    // Omegai integer pixel coordinate (round)
    int vi_i = (int)floor(vi_x+0.5);
    int vi_j = (int)floor(vi_y+0.5);
    // int po = vi_i + vi_j * w;*/

    // Gamma floating pixel coordinate
    cuflt u_x;// = warp_tau_x[po] -0.5f;
    cuflt u_y;// = warp_tau_y[po] -0.5f;

    int res;
    res = bilinear_interpolation(w,h, warp_tau_x, vi_x, vi_y, &u_x, warp_tau_x);
    if (res < 0) {
      u_x = -1; //set to invalid
    }

    res = bilinear_interpolation(w,h, warp_tau_y, vi_x, vi_y, &u_y, warp_tau_y);
    if (res < 0) {
      u_y = -1; //set to invalid
    }

    // Test if warp is valid
    if ( u_x < 0 || w < u_x ||
         u_y < 0 || h < u_y  ) {
      // INVALID
      warp_beta_x[o] = -W;
      warp_beta_y[o] = -H;
      return;
    }

    // change warp coordinates to pixel coordinates
    u_x = u_x - 0.5;
    u_y = u_y - 0.5;
    // this are ready to be compared to ox and oy

    // their product is the |det D beta|-1
    float inv_def_x = 1./abs(warp_beta_def_x[o]);
    float inv_def_y = 1./abs(warp_beta_def_y[o]);

    int pix_x = 1353;
    int pix_y = 99;

    // Gamma integer pixel coordinate (round)
    int u_i = (int)floor(u_x+0.5);
    int u_j = (int)floor(u_y+0.5);


    bool invalid = false;

    if (u_i != ox || u_j != oy) {
      invalid = true;
    }

    /*// split cases: contraction or dilatation
    if (inv_def_x > 1) {
      // CONTRACTION: beta maps neighbouring pixels to the same pixel
      // when coming back u_i (and u_j) can be at a distance 0.5 * inv_def of ox (and oy)
      if (abs(u_x - ox) > 0.5 * inv_def_x) {
        //if (ox == pix_x && oy == pix_y) {
          printf("X def >1 invalid :%d, %d to %g %g -> %d %d, back to %g %g -> %d %d\n", pix_x, pix_y, vi_x, vi_y, vi_i, vi_j, u_x, u_y, u_i, u_j);
        //}
        invalid = true;
      }
    } else {
      // DILATATION: backwards should map exactly to the same coordinate
      //if (fabs(u_i - ox) > 0.5 ) {
      if (u_i != ox) {
        //if (ox == pix_x && oy == pix_y) {
          printf("X def < 1 invalid:  %d, %d to %g %g -> %d %d, back to %g %g -> %d %d\n", pix_x, pix_y, vi_x, vi_y, vi_i, vi_j, u_x, u_y, u_i, u_j);
        //}
        invalid = true;
      }
    }

    if (inv_def_y > 1) {
      // CONTRACTION: beta maps neighbouring pixels to the same pixel
      // when coming back u_i (and u_j) can be at a distance 0.5 * inv_def of ox (and oy)
      if (abs(u_y - oy) > 0.5 * inv_def_y) {
        //if (ox == pix_x && oy == pix_x) {
          printf("Y def >1 invalid %d, %d to %g %g -> %d %d, back to %g %g -> %d %d\n", pix_x, pix_y, vi_x, vi_y, vi_i, vi_j, u_x, u_y, u_i, u_j);
        //}
        invalid = true;
      }
    } else {
      // DILATATION: backwards should map exactly to the same coordinate
      //if (fabs(u_j - oy) > 0.5 ) {
      if (u_j != oy) {
        //if (ox == pix_x && oy == pix_y) {
          printf("Y def <1 invalid: %d, %d to %g %g -> %d %d, back to %g %g -> %d %d\n", pix_x, pix_y, vi_x, vi_y, vi_i, vi_j, u_x, u_y, u_i, u_j);
        //}

        invalid = true;
      }
    }*/


    if (invalid) {
      warp_beta_x[o] = -W;
      warp_beta_y[o] = -H;

      //if (ox == pix_x && oy == pix_y) {
        printf("%d %d: def %g %g\n", pix_x, pix_y, inv_def_x, inv_def_y);
        //printf("99, 99 to %g %g -> %d %d, back to %g %g -> %d %d\n", vi_x, vi_y, vi_i, vi_j, u_x, u_y, u_i, u_j);
      //}
    } else {
      // set integer components so that everytime we fall in the same pixel
      warp_beta_x[o] = vi_i + 0.5f;
      warp_beta_y[o] = vi_j + 0.5f;
    }
}

static __global__ void check_warp_perfect_coherence_device( int W, int H,
                                                            cuflt *warp_beta_x,
                                                            cuflt *warp_beta_y,
                                                            int w, int h,
                                                            cuflt *warp_tau_x,
                                                            cuflt *warp_tau_y) {
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  // Gamma coordinate
  int o = oy*W + ox;

  if ( warp_beta_x[o] < 0 || W < warp_beta_x[o] ||
       warp_beta_y[o] < 0 || H < warp_beta_y[o]  ) {
    return;
  }

  // Omegai floating pixel coordinate
  cuflt vi_x = warp_beta_x[o] - 0.5f;
  cuflt vi_y = warp_beta_y[o] - 0.5f;

  // Omegai integer pixel coordinate (round)
  int vi_i = (int)floor(vi_x+0.5);
  int vi_j = (int)floor(vi_y+0.5);
  int po = vi_i + vi_j * w;

  // Gamma floating pixel coordinate
  cuflt u_x = warp_tau_x[po] -0.5f;
  cuflt u_y = warp_tau_y[po] -0.5f;

  int u_i = (int)floor(u_x+0.5);
  int u_j = (int)floor(u_y+0.5);

  if (u_i != ox || u_j != oy) {
    printf("PROOOOBLEM at %d %d -> %d %d \n", ox, oy, u_i, u_j);
  }
}


static __global__ void filter_invalid_device( int W, int H,
                                              cuflt *value, // overwrite value
                                              cuflt *mask) {
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  // Omega_i coordinate
  int o = oy*W + ox;
  if( mask[o] < 0) {
    value[o] = -1.;
  }
}


static __global__ void multiply_with_deformation_weights( int W, int H,
                                                    cuflt *vmask, // overwrite vmask
                                                    cuflt *deform_weight_beta)
{
    // Global thread index
    int ox = blockDim.x * blockIdx.x + threadIdx.x;
    int oy = blockDim.y * blockIdx.y + threadIdx.y;
    if ( ox >= W || oy >= H ) {
        return;
    }
    // Omega_i coordinate
    int o = oy*W + ox;

    cuflt inv_weight = deform_weight_beta[o];

    if (inv_weight == 0.) {
      vmask[o] = 0.;
      return;
    }

#define GRAD_W_MAX 20.0f
    if ( inv_weight > 1.0f / GRAD_W_MAX ) {
      vmask[o] *= 1.0f / inv_weight;
    }
    else {
      vmask[o] *= GRAD_W_MAX;
    }
}

static __global__ void cuda_dot_product_device (int W, int H, cuflt *x1, cuflt *x2, cuflt *y1, cuflt *y2, cuflt *out)
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }
  // Omega_i coordinate
  int o = oy*W + ox;

  out[o] = fabs( x1[o] * y1[o] + x2[o] * y2[o]);
}

// from
// http://stackoverflow.com/questions/2328258/cumulative-normal-distribution-function-in-c-c
static __device__ cuflt CND(double d)
{
    const double       A1 = 0.31938153;
    const double       A2 = -0.356563782;
    const double       A3 = 1.781477937;
    const double       A4 = -1.821255978;
    const double       A5 = 1.330274429;
    const double RSQRT2PI = 0.39894228040143267793994605993438;

    double
    K = 1.0 / (1.0 + 0.2316419 * fabs(d));

    double
    cnd = RSQRT2PI * exp(- 0.5 * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0 - cnd;

    return cnd;
}

// given an image and a vector, compute the color variance along the vector (in both directions)
static __global__ void cuda_probability_epipolar_device (int W, int H, cuflt *part_x, cuflt *part_y,
                                                         cuflt *mask, cuflt *out)
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }

  // Omega_i coordinate
  int o = oy*W + ox;

  if (mask[o] < 0) {
    out[o] = 0;
    return;
  }

  // compute length of the partial vector
  cuflt length = sqrt( part_x[o] * part_x[o] + part_y[o] * part_y[o]);

  // length is sigma_zi * 2.57 (or other parameter used in the computation of the warps
  float sigma = length/2.57;

  // this sigma gives a Normal ditribution N(0,sigma2)
  // now we want the probability of the pixel interval [-0.5, 0.5] in this distribution
  float normalized_interval = 0.5 / sigma;

  double prob = CND(normalized_interval);

  prob = 2*prob -1;

  const cuflt min_proba = 0.000001;
  if (prob < min_proba) {
    out[o] = min_proba;
  } else {
    out[o] = prob;
  }

  return;
}

// given an image and a vector, compute the color variance along the vector (in both directions)
static __global__ void cuda_variance_epipolar_device (int W, int H, cuflt *lu, cuflt *part_x, cuflt *part_y,
                                                      cuflt *mask, cuflt *out)
{
  // Global thread index
  int ox = blockDim.x * blockIdx.x + threadIdx.x;
  int oy = blockDim.y * blockIdx.y + threadIdx.y;
  if ( ox >= W || oy >= H ) {
    return;
  }

  // Omega_i coordinate
  int o = oy*W + ox;

  if (mask[o] < 0) {
    out[o] = -1;
    return;
  }

  // compute length of the partial vector
  cuflt length = sqrt( part_x[o] * part_x[o] + part_y[o] * part_y[o]);
  int nbSteps = floor(length);
  if (nbSteps > 50) {
    out[o] = 0.25;
    return;
  }



  // FIRST STEP TO COMPUTE THE MEAN
  cuflt sum = lu[o];
  int nbSamples = 1;

  // we should use gaussian coefficients, but for now we use a uniform
  for (int i=-nbSteps; i<= nbSteps; ++i) {
    // avoid to miss the center pixel because of the rounding
    if (i == 0) {
      continue;
    }

    cuflt pos_x = ox + part_x[o] * (cuflt)i/ (cuflt)nbSteps;
    cuflt pos_y = oy + part_y[o] * (cuflt)i/ (cuflt)nbSteps;

    // round coordinates
    int pos_i = floor(pos_x+0.5);
    int pos_j = floor(pos_y+0.5);
    if (pos_i <0 || W-1 < pos_i || pos_j < 0 || H-1 < pos_j ) {
      continue;
    }



    // check if position is valid
    int po = pos_i + W * pos_j;
    if (mask[po] < 0) {
      continue;
    } else {
      cuflt value;
      int res = bilinear_interpolation(W,H, lu, pos_x, pos_y, &value, mask);
      if (res == 0) {
        sum += value;
        nbSamples++;
      }
    }
  }

  if (nbSamples == 0) {
    printf("VERY BIG PROBLEM\n");
    // set variance to maximum
    out[o] = 0.3;
    return;
  }

  cuflt mean = sum / (float)nbSamples;

  // SECOND STEP TO COMPUTE THE VARIANCE
  sum = (lu[o] - mean) * (lu[o] - mean);
  nbSamples = 1;

  for (int i=-nbSteps; i<= nbSteps; ++i) {
    // avoid to miss the center pixel because of the rounding
    if (i == 0) {
      continue;
    }

    cuflt pos_x = ox + part_x[o] * (cuflt)i/ (cuflt)nbSteps;
    cuflt pos_y = oy + part_y[o] * (cuflt)i/ (cuflt)nbSteps;

    // round coordinates
    int pos_i = floor(pos_x+0.5);
    int pos_j = floor(pos_y+0.5);
    if (pos_i <0 || W-1 < pos_i || pos_j < 0 || H-1 < pos_j ) {
      continue;
    }

    // check if position is valid
    int po = pos_i + W * pos_j;
    if (mask[po] < 0) {
      continue;
    } else {
      cuflt value;
      int res = bilinear_interpolation(W,H, lu, pos_x, pos_y, &value, mask);
      if (res == 0) {
        sum += (mean-value) * (mean-value);
        nbSamples++;
      }
    }
  }

  if (nbSamples == 0) {
    printf("VERY BIG PROBLEM\n");
    // set variance to maximum
    out[o] = 0.4;
  } else if (nbSamples == 1) {
    out[o] = 0.;
  } else {
    // apply Bessel's correction
    out[o] = sum / (nbSamples-1);
  }
}


static __global__ void cuda_sigma_g_to_omegai_def (int w, int h, cuflt *sigma_g, cuflt* deformation,
                                                   cuflt sigma_sensor, cuflt ugrad_threshold)
{
  // Global thread index
    int ox = blockDim.x * blockIdx.x + threadIdx.x;
    int oy = blockDim.y * blockIdx.y + threadIdx.y;
    if ( ox >= w || oy >= h ) {
      return;
    }
    // Omega_i coordinate
    int o = oy*w + ox;

    cuflt sigma_g_value = sigma_g[o];


    // if invalid value, set weight to zero
    if (sigma_g_value < 0) {
      sigma_g[o] = 0.;
      return;
    }

    cuflt inv_def = deformation[o];
    cuflt def;

#define GRAD_W_MAX 20.0f
    if ( inv_def > 1.0f / GRAD_W_MAX ) {
      def = 1.0f / inv_def;
    }
    else {
      def = GRAD_W_MAX;
    }

    // to switch between wanners equations and ours
    // WANNER
    if (ugrad_threshold < 0.5 ) { // Wanner method
      sigma_g[o] = def/ (sigma_sensor * sigma_sensor);
    }
    // OURS
    else {
      // switch depending on deformation
      if (def <= 1) {
        sigma_g[o] = def/ (sigma_sensor * sigma_sensor + sigma_g_value * sigma_g_value);
      } else {
        sigma_g[o] = 1./ (sigma_sensor * sigma_sensor / def + sigma_g_value * sigma_g_value);
      }
    }
}

