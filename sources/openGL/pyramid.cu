#include "pyramid.cuh"

// cuda includes
#include <cuda/cuda_helper.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "../helper_cuda.h"

#include <iostream>
#include <cstdio>

GPUPyramid::GPUPyramid( int W, int H, int pyramidHeight, Texture *texture ) :

    _W( W ), _H( H ), _pyramidHeight(pyramidHeight) {

    init(texture);
}

GPUPyramid::~GPUPyramid() {

    for(uint i = 0 ; i < _gaussianPyramidTex.size() ; ++i) {
        if(_gaussianPyramidTex[i] != 0) {
            delete _gaussianPyramidTex[i];
            _gaussianPyramidTex[i] = 0;
        }
    }
    for(uint i = 0 ; i < _gaussianPyramidArray.size() ; ++i) {
        if(_gaussianPyramidArray[i] != 0) {
            CUDA_SAFE_CALL( cudaFree( _gaussianPyramidArray[i] ));
            _gaussianPyramidArray[i] = 0;
        }
    }

    for(uint i = 0 ; i < _laplacianPyramidTex.size() ; ++i) {
        if(_laplacianPyramidTex[i] != 0) {
            delete _laplacianPyramidTex[i];
            _laplacianPyramidTex[i] = 0;
        }
    }
    for(uint i = 0 ; i < _laplacianPyramidArray.size() ; ++i) {
        if(_laplacianPyramidArray[i] != 0) {
            CUDA_SAFE_CALL( cudaFree( _laplacianPyramidArray[i] ));
            _laplacianPyramidArray[i] = 0;
        }
    }
}

void GPUPyramid::init(Texture *texture) {

    cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());

    const uint nbChannels = 3;
    assert(texture->getInternalFormat() == GL_RGB);

    unsigned char *charBuffer = new unsigned char[nbChannels*_W*_H];
    float *floatBuffer;

    CUDA_SAFE_CALL( cudaMalloc( &floatBuffer, nbChannels*_W*_H*sizeof(float) ));
    CUDA_SAFE_CALL( cudaMemset( floatBuffer, 0, nbChannels*_W*_H*sizeof(float) ));

    // TODO get texture from openGL to CUDA
    cudaGraphicsResource *resData;
    checkCudaErrors( cudaGraphicsGLRegisterImage(&resData, texture->getID(), GL_TEXTURE_RECTANGLE, cudaGraphicsRegisterFlagsReadOnly) );
    glBindTexture(GL_TEXTURE_RECTANGLE, texture->getID());
    glGetTexImage( GL_TEXTURE_RECTANGLE, 0, texture->getFormat(), texture->getType(), charBuffer );
    glBindTexture(GL_TEXTURE_RECTANGLE, 0);

    for(uint i = 0 ; i < nbChannels*_W*_H ; ++i) {

        floatBuffer[i] = (float)charBuffer[i] / 255.0f;
    }
    delete[] charBuffer;
    _gaussianPyramidArray.push_back(floatBuffer);

    Texture* originalTexture = new Texture(0, _W, _H, GL_RGB, GL_FLOAT, GL_RGB32F, false);

    // TODO load openGL texture from CUDA
    originalTexture->loadFromData(_gaussianPyramidArray[0]);
    _gaussianPyramidTex.push_back(originalTexture);


    // for every scale. don't go to max scale
    // uint s = 0;
    // while(_W / (uint)pow(2.0, (double)s) > 0 && _H / (uint)pow(2.0, (double)s) > 0) {
    for(uint s = 1 ; s <= (uint)_pyramidHeight ; ++s) {

        float *gaussianArray;
        Texture* gaussianTex = new Texture(0, _W, _H, GL_RGB, GL_FLOAT, GL_RGB32F, false);

        CUDA_SAFE_CALL( cudaMalloc( &gaussianArray, nbChannels*_W*_H*sizeof(float) ));
        CUDA_SAFE_CALL( cudaMemset( gaussianArray, 0, nbChannels*_W*_H*sizeof(float) ));
        _gaussianPyramidArray.push_back(gaussianArray);

        // TODO oddHDC(_W, _H, nbChannels, s);

        // TODO load openGL texture from CUDA
        gaussianTex->loadFromData(_gaussianPyramidArray[s]);
        _gaussianPyramidTex.push_back(gaussianTex);

        float *laplacianArray;
        Texture* laplacianTex = new Texture(0, _W, _H, GL_RGB, GL_FLOAT, GL_RGB32F, false);

        CUDA_SAFE_CALL( cudaMalloc( &laplacianArray, nbChannels*_W*_H*sizeof(float) ));
        CUDA_SAFE_CALL( cudaMemset( laplacianArray, 0, nbChannels*_W*_H*sizeof(float) ));
        _laplacianPyramidArray.push_back(laplacianArray);

        // TODO dog(_W, _H, nbChannels, s-1);

        // TODO load openGL texture from CUDA
        laplacianTex->loadFromData(_laplacianPyramidArray[s-1]);
        _laplacianPyramidTex.push_back(laplacianTex);
    }
}
