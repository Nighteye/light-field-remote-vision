#ifndef PYRAMID_CUH
#define PYRAMID_CUH

#include <string>
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <vector>

#include "texture.h"

#define checkGLErrors() {\
    GLenum error = glGetError(); \
    if(error != GL_NO_ERROR) { \
    std::cout << "GL_ERROR :" << __FILE__ << " "<< __LINE__ << " " << error << std::endl; \
    } \
    }

struct GPUPyramid {

    GPUPyramid( int W, int H, int pyramidHeight, Texture *texture );
    ~GPUPyramid();
    void init(Texture *texture);

    std::vector< Texture* > _gaussianPyramidTex;
    std::vector< float* > _gaussianPyramidArray;
    std::vector< Texture* > _laplacianPyramidTex;
    std::vector< float* > _laplacianPyramidArray;

    int _W;
    int _H;
    int _pyramidHeight;
};

#endif /* #ifndef PYRAMID_CUH */
