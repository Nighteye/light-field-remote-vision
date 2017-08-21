/* -*-c++-*- */

#include <cuda/cuda_kernels.cuh>
#include <cuda/cuda_helper.h>

#include "solver.cuh"
#include "gradientIBR.cuh"

static const float epsilon = 1e-3;
static const int nbChannels = 3;

void cudaPrintf( float* device, std::string myStr ) {

    float *host = new float;

    CUDA_SAFE_CALL( cudaMemcpy(host, device, sizeof(float), cudaMemcpyDeviceToHost) );

    printf("%s: %f\n", myStr.c_str(), *host);

    delete host;
}

// perform dot product c = <A,B>
__global__ void dotProduct( const int W, const int H, const float *const A, const float *const B, float *const c ) {

    // share data between threads for reduction in dot product
    extern __shared__ float sdata[];

    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;

    if ( ox >= W || oy >= H ) {

        return;
    }

    int o = oy*W + ox;

    // compute new residual magnitude deltaNew = r.r
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    sdata[tid] = A[o] * B[o];

    // Reduction

    for ( int size = blockDim.x/2 ; size > 0 ; size /= 2 ) {

        if ( threadIdx.x < size && ox + size < W ) {

            sdata[tid] += sdata[tid + size];
        }
        __syncthreads();
    }

    for ( int size = blockDim.y/2 ; size > 0 ; size /= 2 ) {

        if ( threadIdx.y < size && oy + size < H ) {

            sdata[tid] += sdata[tid + size*blockDim.x];
        }
        __syncthreads();
    }

    if ( tid == 0 ) {

        atomicAdd(c, sdata[0]);
    }


//    // Kopf's reduction
//    int tid = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
//    sdata[tid] = g*g;

//    __syncthreads();

//    int elsPerBand = blockDim.x * blockDim.y;
//    int size = 2*elsPerBand;

//    if (tid < elsPerBand)
//    {
//        sdata[tid] += sdata[tid+size];
//    }

//    __syncthreads();

//    for (size = size/2; size > 0; size >>= 1)
//    {
//        if (tid < size)
//        {
//            sdata[tid] += sdata[tid+size];
//        }
//        __syncthreads();
//    }

//    if (tid == 0)
//    {
//        rMagn2Array[blockIdx.x + blockIdx.y*gridDim.x] = sdata[0];
//    }
}

void testDotProduct( Data *data ) {

    // check for required data
    assert( data != NULL );
    size_t W = data->_W;
    size_t H = data->_H;
    assert( W*H > 0 );
    dim3 DimBlock = data->_DimBlock; // high res
    dim3 DimGrid = data->_DimGrid;

    float *A, *B, *c;
    CUDA_SAFE_CALL( cudaMalloc((void**)&A, data->_nfbytes_hi) );
    CUDA_SAFE_CALL( cudaMalloc((void**)&B, data->_nfbytes_hi) );
    CUDA_SAFE_CALL( cudaMalloc((void**)&c, sizeof(float)) );

    cuda_set_all_device<<< DimGrid, DimBlock >>>( W, H, A, 6.0 );
    cuda_set_all_device<<< DimGrid, DimBlock >>>( W, H, B, 0.2 );
    cudaMemset( c, 0, sizeof(float) );

    dotProduct<<< DimGrid, DimBlock, nbChannels * DimBlock.x * DimBlock.y * sizeof(float) >>>(W, H, A, B, c);

    std::cout << "dot product results: ";
    cudaPrintf( c, "" );

    CUDA_SAFE_CALL( cudaFree( A ));
    CUDA_SAFE_CALL( cudaFree( B ));
    CUDA_SAFE_CALL( cudaFree( c ));
}

// update the current solution x, the residual r and compute the new residual magnitude
__global__ void updateSolution( const int W, const int H,
                                const float *const directionR, const float *const directionG, const float *const directionB,
                                float *const qR, float *const qG, float *const qB,
                                float *const u_r, float *const u_g, float *const u_b,
                                float *const residualR, float *const residualG, float *const residualB,
                                float *const alphaR, float *const alphaG, float *const alphaB ) {

    // share data between threads for reduction in dot product
    extern __shared__ float sdata[];

    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;

    if ( ox >= W || oy >= H ) {

        return;
    }

    int o = oy*W + ox;

    // use alpha to update the current solution
    u_r[o] += *alphaR * directionR[o];
    u_g[o] += *alphaG * directionG[o];
    u_b[o] += *alphaB * directionB[o];

    // update the residual
    residualR[o] -= *alphaR * qR[o];
    residualG[o] -= *alphaG * qG[o];
    residualB[o] -= *alphaB * qB[o];

    // compute new residual magnitude deltaNew = r.r
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    sdata[tid + 0*blockDim.x*blockDim.y] = residualR[o] * residualR[o];
    sdata[tid + 1*blockDim.x*blockDim.y] = residualG[o] * residualG[o];
    sdata[tid + 2*blockDim.x*blockDim.y] = residualB[o] * residualB[o];

    __syncthreads();

    // Reduction

    for ( int size = blockDim.x/2 ; size > 0 ; size /= 2 ) {

        if ( threadIdx.x < size && ox + size < W ) {

            sdata[tid + 0*blockDim.x*blockDim.y] += sdata[tid + 0*blockDim.x*blockDim.y + size];
            sdata[tid + 1*blockDim.x*blockDim.y] += sdata[tid + 1*blockDim.x*blockDim.y + size];
            sdata[tid + 2*blockDim.x*blockDim.y] += sdata[tid + 2*blockDim.x*blockDim.y + size];
        }
        __syncthreads();
    }

    for ( int size = blockDim.y/2 ; size > 0 ; size /= 2 ) {

        if ( threadIdx.y < size && oy + size < H ) {

            sdata[tid + 0*blockDim.x*blockDim.y] += sdata[tid + 0*blockDim.x*blockDim.y + size*blockDim.x];
            sdata[tid + 1*blockDim.x*blockDim.y] += sdata[tid + 1*blockDim.x*blockDim.y + size*blockDim.x];
            sdata[tid + 2*blockDim.x*blockDim.y] += sdata[tid + 2*blockDim.x*blockDim.y + size*blockDim.x];
        }
        __syncthreads();
    }

    if ( tid == 0 ) {

        atomicAdd(alphaR, sdata[0 + 0*blockDim.x*blockDim.y]);
        atomicAdd(alphaG, sdata[0 + 1*blockDim.x*blockDim.y]);
        atomicAdd(alphaB, sdata[0 + 2*blockDim.x*blockDim.y]);
    }
}

// update the step alpha
__global__ void updateAlpha( const int W, const int H,
                             const float *const directionR, const float *const directionG, const float *const directionB,
                             const float *const lambda,
                             float *const qR, float *const qG, float *const qB,
                             float *const alphaR, float *const alphaG, float *const alphaB ) {

    // share data between threads for reduction in dot product
    extern __shared__ float sdata[];

    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;

    if ( ox >= W || oy >= H ) {

        return;
    }

    int o = oy*W + ox;

    // We compute Ad, so no data b in here

    float datatermR = *lambda * directionR[o];
    float datatermG = *lambda * directionG[o];
    float datatermB = *lambda * directionB[o];

    // we compute Ad with d the current direction
    // gradient term x: minus 2nd order partial derivative of d, along x direction
    float laplacianXR, laplacianXG, laplacianXB;
    if ( ox == 0 ) {

        laplacianXR = directionR[o] - directionR[o+1];
        laplacianXG = directionG[o] - directionG[o+1];
        laplacianXB = directionB[o] - directionB[o+1];

    } else if ( ox < W-1) {

        laplacianXR = 2*directionR[o] - directionR[o-1] - directionR[o+1];
        laplacianXG = 2*directionG[o] - directionG[o-1] - directionG[o+1];
        laplacianXB = 2*directionB[o] - directionB[o-1] - directionB[o+1];

    } else {

        laplacianXR = directionR[o] - directionR[o-1];
        laplacianXG = directionG[o] - directionG[o-1];
        laplacianXB = directionB[o] - directionB[o-1];
    }
    // gradient term y: minus 2nd order partial derivative of d, along y direction
    float laplacianYR, laplacianYG, laplacianYB;
    if ( oy == 0 ) {

        laplacianYR = directionR[o] - directionR[o+W];
        laplacianYG = directionG[o] - directionG[o+W];
        laplacianYB = directionB[o] - directionB[o+W];

    } else if ( oy < H-1) {

        laplacianYR = 2*directionR[o] - directionR[o-W] - directionR[o+W];
        laplacianYG = 2*directionG[o] - directionG[o-W] - directionG[o+W];
        laplacianYB = 2*directionB[o] - directionB[o-W] - directionB[o+W];

    } else {

        laplacianYR = directionR[o] - directionR[o-W];
        laplacianYG = directionG[o] - directionG[o-W];
        laplacianYB = directionB[o] - directionB[o-W];
    }

    // Ad
    qR[o] = datatermR + laplacianXR + laplacianYR;
    qG[o] = datatermG + laplacianXG + laplacianYG;
    qB[o] = datatermB + laplacianXB + laplacianYB;

    // alpha = deltaNew/d.q
    // for now we compute d.q and store it in alpha
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    sdata[tid + 0*blockDim.x*blockDim.y] = directionR[o] * qR[o];
    sdata[tid + 1*blockDim.x*blockDim.y] = directionG[o] * qG[o];
    sdata[tid + 2*blockDim.x*blockDim.y] = directionB[o] * qB[o];

    __syncthreads();

    for ( int size = blockDim.x/2 ; size > 0 ; size /= 2 ) {

        if ( threadIdx.x < size && ox + size < W ) {

            sdata[tid + 0*blockDim.x*blockDim.y] += sdata[tid + 0*blockDim.x*blockDim.y + size];
            sdata[tid + 1*blockDim.x*blockDim.y] += sdata[tid + 1*blockDim.x*blockDim.y + size];
            sdata[tid + 2*blockDim.x*blockDim.y] += sdata[tid + 2*blockDim.x*blockDim.y + size];
        }
        __syncthreads();
    }

    for ( int size = blockDim.y/2 ; size > 0 ; size /= 2 ) {

        if ( threadIdx.y < size && oy + size < H ) {

            sdata[tid + 0*blockDim.x*blockDim.y] += sdata[tid + 0*blockDim.x*blockDim.y + size*blockDim.x];
            sdata[tid + 1*blockDim.x*blockDim.y] += sdata[tid + 1*blockDim.x*blockDim.y + size*blockDim.x];
            sdata[tid + 2*blockDim.x*blockDim.y] += sdata[tid + 2*blockDim.x*blockDim.y + size*blockDim.x];
        }
        __syncthreads();
    }

    if ( tid == 0 ) {

        atomicAdd(alphaR, sdata[0 + 0*blockDim.x*blockDim.y]);
        atomicAdd(alphaG, sdata[0 + 1*blockDim.x*blockDim.y]);
        atomicAdd(alphaB, sdata[0 + 2*blockDim.x*blockDim.y]);
    }
}

// test function: 2D grid dot product A.B = c, A and B two vectors and c a real number
__global__ void dotProduct2D( const int W, const int H,
                              const float *const A, const float *const B,
                              float *C ) {

    // share data between threads for reduction in dot product
    extern __shared__ float sdata[];

    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;

    if ( ox >= W || oy >= H ) {

        return;
    }

    int o = oy*W + ox;

    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    sdata[tid] = A[o] * B[o];

    __syncthreads();

    for ( int size = blockDim.x/2 ; size > 0 ; size /= 2 ) {

        if ( threadIdx.x < size && ox + size < W ) {

            sdata[tid] += sdata[tid + size];
        }
        __syncthreads();
    }

    for ( int size = blockDim.y/2 ; size > 0 ; size /= 2 ) {

        if ( threadIdx.y < size && oy + size < H ) {

            sdata[tid] += sdata[tid + size*blockDim.x];
        }
        __syncthreads();
    }

    if ( tid == 0 ) {

        C[blockIdx.x + blockIdx.y*gridDim.x] = sdata[0];
    }
}

// initialization of conjugate gradient method
__global__ void initConjGrad( const int W, const int H,
                              const float *const u_r, const float *const u_g, const float *const u_b,
                              const float *const gX, const float *const gY,
                              float *const residualR, float *const residualG, float *const residualB,
                              float *const deltaR, float *const deltaG, float *const deltaB ) {

    // share data between threads for reduction in dot product
    extern __shared__ float sdata[];

    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;

    if ( ox >= W || oy >= H ) {

        return;
    }

    int o = oy*W + ox;

    // the data term is zero because we initialize with the data

    // we compute Ax - b with x the initialization
    // gradient term x: divergence of the gradient along x minus 2nd order partial derivative of x
    float laplacianXR, laplacianXG, laplacianXB;
    if ( ox == 0 ) {

        laplacianXR = u_r[o] - u_r[o+1] + gX[o];
        laplacianXG = u_g[o] - u_g[o+1] + gX[o];
        laplacianXB = u_b[o] - u_b[o+1] + gX[o];

    } else if ( ox < W-1) {

        laplacianXR = 2*u_r[o] - u_r[o-1] - u_r[o+1] + gX[o] - gX[o-1];
        laplacianXG = 2*u_g[o] - u_g[o-1] - u_g[o+1] + gX[o] - gX[o-1];
        laplacianXB = 2*u_b[o] - u_b[o-1] - u_b[o+1] + gX[o] - gX[o-1];

    } else {

        laplacianXR = u_r[o] - u_r[o-1] - gX[o-1];
        laplacianXG = u_g[o] - u_g[o-1] - gX[o-1];
        laplacianXB = u_b[o] - u_b[o-1] - gX[o-1];
    }
    // gradient term y: divergence of the gradient along y minus 2nd order partial derivative of y
    float laplacianYR, laplacianYG, laplacianYB;
    if ( oy == 0 ) {

        laplacianYR = u_r[o] - u_r[o+W] + gY[o];
        laplacianYG = u_g[o] - u_g[o+W] + gY[o];
        laplacianYB = u_b[o] - u_b[o+W] + gY[o];

    } else if ( oy < H-1) {

        laplacianYR = 2*u_r[o] - u_r[o-W] - u_r[o+W] + gY[o] - gY[o-W];
        laplacianYG = 2*u_g[o] - u_g[o-W] - u_g[o+W] + gY[o] - gY[o-W];
        laplacianYB = 2*u_b[o] - u_b[o-W] - u_b[o+W] + gY[o] - gY[o-W];

    } else {    // r = b - Ax
        residualR[o] = - (laplacianXR + laplacianYR);
        residualG[o] = - (laplacianXG + laplacianYG);
        residualB[o] = - (laplacianXB + laplacianYB);

        // deltaNew = r.r
        int tid = threadIdx.x + threadIdx.y * blockDim.x;
        sdata[tid + 0*blockDim.x*blockDim.y] = residualR[o] * residualR[o];
        sdata[tid + 1*blockDim.x*blockDim.y] = residualG[o] * residualG[o];
        sdata[tid + 2*blockDim.x*blockDim.y] = residualB[o] * residualB[o];

        __syncthreads();

        for ( int size = blockDim.x/2 ; size > 0 ; size /= 2 ) {

            if ( threadIdx.x < size && ox + size < W ) {

                sdata[tid + 0*blockDim.x*blockDim.y] += sdata[tid + 0*blockDim.x*blockDim.y + size];
                sdata[tid + 1*blockDim.x*blockDim.y] += sdata[tid + 1*blockDim.x*blockDim.y + size];
                sdata[tid + 2*blockDim.x*blockDim.y] += sdata[tid + 2*blockDim.x*blockDim.y + size];
            }
            __syncthreads();
        }

        for ( int size = blockDim.y/2 ; size > 0 ; size /= 2 ) {

            if ( threadIdx.y < size && oy + size < H ) {

                sdata[tid + 0*blockDim.x*blockDim.y] += sdata[tid + 0*blockDim.x*blockDim.y + size*blockDim.x];
                sdata[tid + 1*blockDim.x*blockDim.y] += sdata[tid + 1*blockDim.x*blockDim.y + size*blockDim.x];
                sdata[tid + 2*blockDim.x*blockDim.y] += sdata[tid + 2*blockDim.x*blockDim.y + size*blockDim.x];
            }
            __syncthreads();
        }

        if ( tid == 0 ) {

            atomicAdd(deltaR, sdata[0 + 0*blockDim.x*blockDim.y]);
            atomicAdd(deltaG, sdata[0 + 1*blockDim.x*blockDim.y]);
            atomicAdd(deltaB, sdata[0 + 2*blockDim.x*blockDim.y]);
        }

        laplacianYR = u_r[o] - u_r[o-W] - gY[o-W];
        laplacianYG = u_g[o] - u_g[o-W] - gY[o-W];
        laplacianYB = u_b[o] - u_b[o-W] - gY[o-W];
    }

    // r = b - Ax
    residualR[o] = - (laplacianXR + laplacianYR);
    residualG[o] = - (laplacianXG + laplacianYG);
    residualB[o] = - (laplacianXB + laplacianYB);

    // deltaNew = r.r
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    sdata[tid + 0*blockDim.x*blockDim.y] = residualR[o] * residualR[o];
    sdata[tid + 1*blockDim.x*blockDim.y] = residualG[o] * residualG[o];
    sdata[tid + 2*blockDim.x*blockDim.y] = residualB[o] * residualB[o];

    __syncthreads();

    for ( int size = blockDim.x/2 ; size > 0 ; size /= 2 ) {

        if ( threadIdx.x < size && ox + size < W ) {

            sdata[tid + 0*blockDim.x*blockDim.y] += sdata[tid + 0*blockDim.x*blockDim.y + size];
            sdata[tid + 1*blockDim.x*blockDim.y] += sdata[tid + 1*blockDim.x*blockDim.y + size];
            sdata[tid + 2*blockDim.x*blockDim.y] += sdata[tid + 2*blockDim.x*blockDim.y + size];
        }
        __syncthreads();
    }

    for ( int size = blockDim.y/2 ; size > 0 ; size /= 2 ) {

        if ( threadIdx.y < size && oy + size < H ) {

            sdata[tid + 0*blockDim.x*blockDim.y] += sdata[tid + 0*blockDim.x*blockDim.y + size*blockDim.x];
            sdata[tid + 1*blockDim.x*blockDim.y] += sdata[tid + 1*blockDim.x*blockDim.y + size*blockDim.x];
            sdata[tid + 2*blockDim.x*blockDim.y] += sdata[tid + 2*blockDim.x*blockDim.y + size*blockDim.x];
        }
        __syncthreads();
    }

    if ( tid == 0 ) {

        atomicAdd(deltaR, sdata[0 + 0*blockDim.x*blockDim.y]);
        atomicAdd(deltaG, sdata[0 + 1*blockDim.x*blockDim.y]);
        atomicAdd(deltaB, sdata[0 + 2*blockDim.x*blockDim.y]);
    }
}

// compute beta and use it to update the search direction
__global__ void updateDir( const int W, const int H,
                           const float beta,
                           const float *const residual, float *const direction ) {

    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;

    if ( ox >= W || oy >= H ) {

        return;
    }

    int o = oy*W + ox;

    direction[o] = residual[o] + beta * direction[o];
}

// Perform Poisson integration with conjugate gradient method
void possion_conj_grad( Data* data ) {

    // check for required data
    assert( data != NULL );
    size_t W = data->_W;
    size_t H = data->_H;
    assert( W*H > 0 );
    dim3 DimBlock = data->_DimBlock; // high res
    dim3 DimGrid = data->_DimGrid;

    testDotProduct(data);

    const unsigned int nIter = 5;
    float energy = 0.0;
    float previous_energy = 1.0;
    const float lambda = 1.0;

    // init GPU arrays
    std::vector< float* > dResidual, dDirection, dQ, deltaNew, deltaOld, dAlpha;
    float *dLambda;

    for ( int i = 0 ; i < nbChannels ; ++i ) {

        float *tmp = NULL;
        CUDA_SAFE_CALL( cudaMalloc((void**)&tmp, data->_nfbytes_hi) );
        dResidual.push_back( tmp );
        CUDA_SAFE_CALL( cudaMalloc((void**)&tmp, data->_nfbytes_hi) );
        dDirection.push_back( tmp );
        CUDA_SAFE_CALL( cudaMalloc((void**)&tmp, data->_nfbytes_hi) );
        dQ.push_back( tmp );
        CUDA_SAFE_CALL( cudaMalloc((void**)&tmp, sizeof(float)) );
        deltaNew.push_back( tmp );
        CUDA_SAFE_CALL( cudaMalloc((void**)&tmp, sizeof(float)) );
        deltaOld.push_back( tmp );
        CUDA_SAFE_CALL( cudaMalloc((void**)&tmp, sizeof(float)) );
        dAlpha.push_back( tmp );

        CUDA_SAFE_CALL( cudaMemset( dResidual[i], 0, data->_nfbytes_hi ));
        CUDA_SAFE_CALL( cudaMemset( dDirection[i], 0, data->_nfbytes_hi ));
        CUDA_SAFE_CALL( cudaMemset( dQ[i], 0, data->_nfbytes_hi ));
        CUDA_SAFE_CALL( cudaMemset( deltaNew[i], 0, sizeof(float) ));
        CUDA_SAFE_CALL( cudaMemset( deltaOld[i], 0, sizeof(float) ));
        CUDA_SAFE_CALL( cudaMemset( dAlpha[i], 0, sizeof(float) ));
    }

    CUDA_SAFE_CALL( cudaMalloc((void**)&dLambda, sizeof(float)) );
    CUDA_SAFE_CALL( cudaMemcpy(dLambda, &lambda, sizeof(float), cudaMemcpyHostToDevice) );

    // initialization
    initConjGrad<<< DimGrid, DimBlock, nbChannels * DimBlock.x * DimBlock.y * sizeof(float) >>>( W, H,
                                                                                    data->_U[0], data->_U[1], data->_U[2],
            data->_u_grad_x, data->_u_grad_y,
            dResidual[0], dResidual[1], dResidual[2],
            deltaNew[0], deltaNew[1], deltaNew[2] );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    for ( int i = 0 ; i < nbChannels ; ++i ) {

        CUDA_SAFE_CALL( cudaMemcpy(dDirection[i], dResidual[i], data->_nfbytes_hi, cudaMemcpyDeviceToDevice) );
        CUDA_SAFE_CALL( cudaMemcpy(deltaOld[i], deltaNew[i], sizeof(float), cudaMemcpyDeviceToDevice) );
    }

    TRACE("Perform Poisson integration of gradient  [");

    for ( unsigned int iter = 0 ; iter < nIter ; ++iter ) {

//        if ( (iter%(nIter/10)) == 0 ) {
//            TRACE( "." );
//        }

        // compute the step size
        updateAlpha<<< DimGrid, DimBlock, nbChannels * DimBlock.x * DimBlock.y * sizeof(float) >>>( W, H,
                dDirection[0], dDirection[1], dDirection[2],
                dLambda,
                dQ[0], dQ[1], dQ[2],
                dAlpha[0], dAlpha[1], dAlpha[2] );
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        coco::write_pfm_image_signed( W, H, dQ[0], dQ[1], dQ[2], data->_outdir + "/dQ_%02lu.pfm", iter );
        coco::write_pfm_image_signed( W, H, dDirection[0], dDirection[1], dDirection[2], data->_outdir + "/dDirection_%02lu.pfm", iter );
        coco::write_pfm_image_signed( W, H, dResidual[0], dResidual[1], dResidual[2], data->_outdir + "/residual_%02lu.pfm", iter );

        // save the current residual magnitude in deltaOld
        for ( int i = 0 ; i < nbChannels ; ++i ) {

            CUDA_SAFE_CALL( cudaMemcpy(deltaOld[i], deltaNew[i], sizeof(float), cudaMemcpyDeviceToDevice) );
            CUDA_SAFE_CALL( cudaThreadSynchronize() );
        }

        cudaPrintf( deltaNew[0], "deltaNew[0]" );
        cudaPrintf( deltaNew[1], "deltaNew[1]" );
        cudaPrintf( deltaNew[2], "deltaNew[2]" );
        cudaPrintf( dAlpha[0], "dAlpha[0]" );
        cudaPrintf( dAlpha[1], "dAlpha[1]" );
        cudaPrintf( dAlpha[2], "dAlpha[2]" );

        // currently alpha = d.q, but actually alpha should be delta / d.q
        for ( int i = 0 ; i < nbChannels ; ++i ) {

            float *alphaHost = new float;
            float *deltaHost = new float;
            CUDA_SAFE_CALL( cudaMemcpy(alphaHost, dAlpha[i], sizeof(float), cudaMemcpyDeviceToHost) );
            CUDA_SAFE_CALL( cudaMemcpy(deltaHost, deltaNew[i], sizeof(float), cudaMemcpyDeviceToHost) );

            assert( *alphaHost != 0 );
            *alphaHost = *deltaHost / *alphaHost;

            CUDA_SAFE_CALL( cudaMemcpy(dAlpha[i], alphaHost, sizeof(float), cudaMemcpyHostToDevice) );

            delete alphaHost;
            delete deltaHost;
        }

        cudaPrintf( dAlpha[0], "dAlpha[0]" );
        cudaPrintf( dAlpha[1], "dAlpha[1]" );
        cudaPrintf( dAlpha[2], "dAlpha[2]" );

        // update the current solution x, the residual r and compute the new residual magnitude
        updateSolution<<< DimGrid, DimBlock, nbChannels * DimBlock.x * DimBlock.y * sizeof(float) >>>( W, H,
                                                                                          dDirection[0], dDirection[1], dDirection[2],
                                                                                          dQ[0], dQ[1], dQ[2],
                                                                                          data->_U[0], data->_U[1], data->_U[2],
                                                                                          dResidual[0], dResidual[1], dResidual[2],
                                                                                          dAlpha[0], dAlpha[1], dAlpha[2] );
        CUDA_SAFE_CALL( cudaThreadSynchronize() );

        // compute beta
        std::vector< float > beta;

        for ( int i = 0 ; i < nbChannels ; ++i ) {

            float *deltaNewHost = new float;
            float *deltaOldHost = new float;
            CUDA_SAFE_CALL( cudaMemcpy(deltaNewHost, deltaNew[i], sizeof(float), cudaMemcpyDeviceToHost) );
            CUDA_SAFE_CALL( cudaMemcpy(deltaOldHost, deltaOld[i], sizeof(float), cudaMemcpyDeviceToHost) );

            assert( *deltaOldHost != 0 );
            float tmp = *deltaNewHost / *deltaOldHost;
            beta.push_back(tmp);

            delete deltaNewHost;
            delete deltaOldHost;
        }

        TRACE("beta " << 0 << ": " << beta[0] << std::endl);

        // update the search direction
        for ( int i = 0 ; i < nbChannels ; ++i ) {

            updateDir<<< DimGrid, DimBlock >>>( W, H, beta[i], dResidual[i], dDirection[i]);
            CUDA_SAFE_CALL( cudaThreadSynchronize() );
        }

        std::cout << iter << std::endl;
        coco::write_pfm_image_signed( W, H, data->_U[0], data->_U[1], data->_U[2], data->_outdir + "/u_iter_%02lu.pfm", iter );
        CUDA_SAFE_CALL( cudaThreadSynchronize() );
    }
    if ( abs(previous_energy - energy) >= epsilon ) {

        TRACE( "] maximum number of iterations reached" << std::endl );
    }

    for ( int i = 0 ; i < nbChannels ; ++i ) {

        CUDA_SAFE_CALL( cudaFree( dResidual[i] ));
        CUDA_SAFE_CALL( cudaFree( dDirection[i] ));
        CUDA_SAFE_CALL( cudaFree( dQ[i] ));
        CUDA_SAFE_CALL( cudaFree( deltaNew[i] ));
        CUDA_SAFE_CALL( cudaFree( deltaOld[i] ));
        CUDA_SAFE_CALL( cudaFree( dAlpha[i] ));
    }

    CUDA_SAFE_CALL( cudaFree( dLambda ));
}


