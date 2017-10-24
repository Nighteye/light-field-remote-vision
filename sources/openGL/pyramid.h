#ifndef PYRAMID_H
#define PYRAMID_H

#include <string>
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <vector>

#include "texture.h"

#define INVALID_DEPTH 1000.0f

#define checkGLErrors() {\
    GLenum error = glGetError(); \
    if(error != GL_NO_ERROR) { \
    std::cout << "GL_ERROR :" << __FILE__ << " "<< __LINE__ << " " << error << std::endl; \
    } \
    }

// Perform push pull Laplacian hole-filling, RGB float images
void pushPull(int W, int H, std::vector<cv::Point3f>& image, const std::vector<float>& weights);

// Perform push pull as it was implemented by Gortler '96
void pushPullComplete(int W, int H, std::vector<cv::Point3f>& image, const std::vector<float>& weights);

// DEPTH PYRAMID EXPAND AND REDUCE OPERATIONS (ON DEPTH MAPS)

// filter depth map with oddHDC, don't downsample, size of kernel depends on scale
void oddHDCDepth(int W, int H, int nbChannels, uint scale, const float* const input, float* const output);
// filter depth map with oddHDC, downsample by 2
void oddHDCDepthReduce(int hiW, int hiH, int loW, int loH, int nbChannels, const float* const input, float* const output);
// filter depth map with gaussian, don't downsample, size of kernel depends on scale
void oddGaussianDepth(int W, int H, int nbChannels, const float* const input, float* const output);

void oddHDCDepthExpand(int loW, int loH, int hiW, int hiH, int nbChannels, float *input, float *output);

class Pyramid {

public:

    Pyramid( int W, int H );
    ~Pyramid();

    std::vector< Texture* > _gaussianPyramidTex;
    std::vector< float* > _gaussianPyramidArray;
    std::vector< Texture* > _laplacianPyramidTex;
    std::vector< float* > _laplacianPyramidArray;

    void reduce(int W, int H, int nbChannels, uint scale);
    void createTestImage(int W, int H, int nbChannels);
    void oddHDC(int W, int H, int nbChannels, uint scale);
    void oddHDCreduced(int W, int H, int nbChannels, uint scale);
    void dog(int W, int H, int nbChannels, uint outputScale);
    void dogReduced(int W, int H, int nbChannels, uint outputScale);
    void gaussianToLaplacian(int nbChannels, uint outputScale);
    void collapse(int nbChannels, uint outputScale);
    void sum(int W, int H, int nbChannels, float *input1, float *input2, float *output);
    void collapse(int W, int H, int nbChannels, uint inscale1, uint inscale2, uint outscale);
    void addLevelContribution(int W, int H, int nbChannels, uint scale, float* contribution);
    void sumDetails(int W, int H, int nbChannels, uint inScale1, uint inScale2, float* output);

private:

    int _W;
    int _H;
};

// CLASSIC PYRAMID EXPAND AND REDUCE OPERATIONS (ON IMAGES)

// the one that was in Pyramid class, now outside so we can use it everywhere
void oddHDCexpanded(int inW, int inH, int outW, int outH, int nbChannels, float *input, float *output);

// gaussian filter to create image pyramid
void reduce(int W, int H, int w, int h, int nbChannels, const float* const input, float* const output);
// expand image to compute the laplacian only
void expand(int W, int H, int w, int h, int nbChannels, const float* const input, float* const output);
// gaussian filter to create image pyramid
void reduceGaussian(int W, int H, int w, int h, int nbChannels, const float* const input, float* const output);
// expand image to compute the laplacian only
void expandGaussian(int W, int H, int w, int h, int nbChannels, const float* const input, float* const output);
// compute laplacian L(s-1) = G(s-1) - expand(G(s))
// output is initially expand(s), input is G(s-1)
void computeLaplacian(int W, int H, int nbChannels, const float* const input, float* const output);
// classic reduce for RGB images, with visibility
void oddHDCReduceRGB(int W, int H, int w, int h,
                     const std::vector<cv::Point3f>& inputImage, std::vector<cv::Point3f>& outputImage,
                     const std::vector<bool>& inputVisibility, std::vector<bool>& outputVisibility);
// reduce for RGB images, with weights, as implemented by Gortler
void oddHDCReduceGortler(int W, int H, int w, int h,
                     const std::vector<cv::Point3f>& inputImage, std::vector<cv::Point3f>& outputImage,
                     const std::vector<float>& inputWeight, std::vector<float>& outputWeight);
// classic reduce for boolean images
void oddHDCReduceBool(int W, int H, int w, int h, const std::vector<bool>& input, std::vector<bool>& output);
// classic expand for RGB images, with visibility test
void oddHDCExpandRGB(int loW, int loH, int hiW, int hiH, const std::vector<cv::Point3f>& input, std::vector<cv::Point3f>& output, const std::vector<bool>& visibility);

#endif /* #ifndef PYRAMID_H */
