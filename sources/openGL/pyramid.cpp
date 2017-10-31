#include "pyramid.h"
#include "../lfScene.h"

#include <iostream>
#include <cstdio>
#include <algorithm>

// Perform push pull Laplacian hole-filling, RGB float images
void pushPull(int W, int H, std::vector<cv::Point3f>& image, const std::vector<float>& weights) {

    // HACK
    uint pyramidHeight = 7;

    std::vector< std::vector<cv::Point3f> > imagePyramid(pyramidHeight + 1);
    std::vector< std::vector<bool> > visibilityPyramid(pyramidHeight + 1);

    // init scale 0
    imagePyramid[0] = image;
    visibilityPyramid[0].resize(W*H);
    for(int i = 0 ; i < W*H ; ++i) {
        visibilityPyramid[0][i] = (weights[i] > 0);
    }

    for(uint s = 1 ; s <= (uint)pyramidHeight ; ++s) {

        uint Wscale = (uint)W / (uint)pow(2.0, (double)(s - 1));
        uint Hscale = (uint)H / (uint)pow(2.0, (double)(s - 1));
        uint wscale = (uint)W / (uint)pow(2.0, (double)s);
        uint hscale = (uint)H / (uint)pow(2.0, (double)s);

        imagePyramid[s].assign(wscale*hscale, cv::Point3f(0.0f, 0.0f, 0.0f));
        visibilityPyramid[s].assign(wscale*hscale, false);

        oddHDCReduceRGB(Wscale, Hscale, wscale, hscale,
                        imagePyramid[s - 1], imagePyramid[s],
                visibilityPyramid[s - 1], visibilityPyramid[s]);

        std::cout << "Writing reduced image " << s << " in " << "out/IBR_optical/transcut/obj1_scn1/reducedImage" + std::to_string(s) + ".pfm" << std::endl;
        savePFM(imagePyramid[s], wscale, hscale, "out/IBR_optical/transcut/obj1_scn1/reducedImage" + std::to_string(s) + ".pfm");
        std::cout << "Writing reduced weight " << s << " in " << "out/IBR_optical/transcut/obj1_scn1/reducedWeight" + std::to_string(s) + ".pfm" << std::endl;
        savePFM(visibilityPyramid[s], wscale, hscale, "out/IBR_optical/transcut/obj1_scn1/reducedWeight" + std::to_string(s) + ".pfm");
    }

    std::cout << "Writing reduced image " << 0 << " in " << "out/IBR_optical/transcut/obj1_scn1/reducedImage" + std::to_string(0) + ".pfm" << std::endl;
    savePFM(imagePyramid[0], W, H, "out/IBR_optical/transcut/obj1_scn1/reducedImage" + std::to_string(0) + ".pfm");
    std::cout << "Writing reduced weight " << 0 << " in " << "out/IBR_optical/transcut/obj1_scn1/reducedWeight" + std::to_string(0) + ".pfm" << std::endl;
    savePFM(visibilityPyramid[0], W, H, "out/IBR_optical/transcut/obj1_scn1/reducedWeight" + std::to_string(0) + ".pfm");

    for(uint s = (uint)pyramidHeight ; 0 < s ; --s) {

        uint Wscale = (uint)W / (uint)pow(2.0, (double)(s - 1));
        uint Hscale = (uint)H / (uint)pow(2.0, (double)(s - 1));
        uint wscale = (uint)W / (uint)pow(2.0, (double)s);
        uint hscale = (uint)H / (uint)pow(2.0, (double)s);

        oddHDCExpandRGB(wscale, hscale, Wscale, Hscale, imagePyramid[s], imagePyramid[s-1], visibilityPyramid[s-1]);

        std::cout << "Writing expanded image " << s << " in " << "out/IBR_optical/transcut/obj1_scn1/expandedImage" + std::to_string(s) + ".pfm" << std::endl;
        savePFM(imagePyramid[s-1], Wscale, Hscale, "out/IBR_optical/transcut/obj1_scn1/expandedImage" + std::to_string(s) + ".pfm");
        std::cout << "Writing expanded weight " << s << " in " << "out/IBR_optical/transcut/obj1_scn1/expandedWeight" + std::to_string(s) + ".pfm" << std::endl;
        savePFM(visibilityPyramid[s-1], Wscale, Hscale, "out/IBR_optical/transcut/obj1_scn1/expandedWeight" + std::to_string(s) + ".pfm");
    }

    image = imagePyramid[0];
}

// Perform push pull as it was implemented by Gortler '96
void pushPullGortler(int W, int H, std::vector<cv::Point3f>& image, const std::vector<float>& weights) {

    // HACK
    uint pyramidHeight = 8;

    std::vector< std::vector<cv::Point3f> > imagePyramid(pyramidHeight + 1);
    std::vector< std::vector<float> > weightPyramid(pyramidHeight + 1);

    // init scale 0
    imagePyramid[0] = image;
    weightPyramid[0].resize(W*H);
    for(uint i = 0 ; i < weightPyramid[0].size() ; ++i) { // pre-filter saturated weights
        weightPyramid[0][i] = (weights[i] > 0.0f) ? 1.0f : 0.0f;
//        weightPyramid[0][i] = std::min(weights[i], 1.0f);
//        weightPyramid[0][i] = weights[i];
        imagePyramid[0][i] = weightPyramid[0][i] * image[i];
    }

    for(uint s = 1 ; s <= (uint)pyramidHeight ; ++s) { // PULL

        uint Wscale = (uint)W / (uint)pow(2.0, (double)(s - 1));
        uint Hscale = (uint)H / (uint)pow(2.0, (double)(s - 1));
        uint wscale = (uint)W / (uint)pow(2.0, (double)s);
        uint hscale = (uint)H / (uint)pow(2.0, (double)s);

        imagePyramid[s].assign(wscale*hscale, cv::Point3f(0.0f, 0.0f, 0.0f));
        weightPyramid[s].assign(wscale*hscale, 0.0f);

        oddHDCReduceGortler(Wscale, Hscale, wscale, hscale, imagePyramid[s - 1], imagePyramid[s], weightPyramid[s - 1], weightPyramid[s]);

        std::cout << "Writing reduced image " << s << " in " << "out/IBR_optical/transcut/obj1_scn1/reducedImageGortler" + std::to_string(s) + ".pfm" << std::endl;
        savePFM(imagePyramid[s], wscale, hscale, "out/IBR_optical/transcut/obj1_scn1/reducedImageGortler" + std::to_string(s) + ".pfm");
        std::cout << "Writing reduced weight " << s << " in " << "out/IBR_optical/transcut/obj1_scn1/reducedWeightGortler" + std::to_string(s) + ".pfm" << std::endl;
        savePFM(weightPyramid[s], wscale, hscale, "out/IBR_optical/transcut/obj1_scn1/reducedWeightGortler" + std::to_string(s) + ".pfm");
    }

    std::cout << "Writing reduced image " << 0 << " in " << "out/IBR_optical/transcut/obj1_scn1/reducedImageGortler" + std::to_string(0) + ".pfm" << std::endl;
    savePFM(imagePyramid[0], W, H, "out/IBR_optical/transcut/obj1_scn1/reducedImageGortler" + std::to_string(0) + ".pfm");
    std::cout << "Writing reduced weight " << 0 << " in " << "out/IBR_optical/transcut/obj1_scn1/reducedWeightGortler" + std::to_string(0) + ".pfm" << std::endl;
    savePFM(weightPyramid[0], W, H, "out/IBR_optical/transcut/obj1_scn1/reducedWeightGortler" + std::to_string(0) + ".pfm");

    for(uint s = (uint)pyramidHeight ; 0 < s ; --s) { // PUSH

        uint Wscale = (uint)W / (uint)pow(2.0, (double)(s - 1));
        uint Hscale = (uint)H / (uint)pow(2.0, (double)(s - 1));
        uint wscale = (uint)W / (uint)pow(2.0, (double)s);
        uint hscale = (uint)H / (uint)pow(2.0, (double)s);

        oddHDCExpandGortler(wscale, hscale, Wscale, Hscale, imagePyramid[s], imagePyramid[s-1], weightPyramid[s], weightPyramid[s-1]);

        std::cout << "Writing expanded image " << s << " in " << "out/IBR_optical/transcut/obj1_scn1/expandedImageGortler" + std::to_string(s) + ".pfm" << std::endl;
        savePFM(imagePyramid[s-1], Wscale, Hscale, "out/IBR_optical/transcut/obj1_scn1/expandedImageGortler" + std::to_string(s) + ".pfm");
        std::cout << "Writing expanded weight " << s << " in " << "out/IBR_optical/transcut/obj1_scn1/expandedWeightGortler" + std::to_string(s) + ".pfm" << std::endl;
        savePFM(weightPyramid[s-1], Wscale, Hscale, "out/IBR_optical/transcut/obj1_scn1/expandedWeightGortler" + std::to_string(s) + ".pfm");
    }

    image = imagePyramid[0];

    for(uint i = 0 ; i < weightPyramid[0].size() ; ++i) { // pre-filter saturated weights
        if(weightPyramid[0][i] > 0.0f) {
            image[i] = imagePyramid[0][i] / weightPyramid[0][i];
        }
    }
}

void oddHDCDepth(int W, int H, int nbChannels, uint scale, const float* const input, float* const output) {

    const int kernelRadius = 2;
    const float a = 0.4f;
    const int r = (int)pow((double)2.0, (double)(scale-1));
    const float weightThreshold = 0.5f;
    const float invalidDepth = 50.0f;
    float kernel[kernelRadius+1] = {a, 0.25f, 0.25f - 0.5f*a};

    float *tempArray = new float[nbChannels*W*H];

    for(int i = 0 ; i < H ; ++i) {
        for(int j = 0 ; j < W ; ++j) {
            for(int c = 0 ; c < nbChannels ; ++c) {

                uint idx = i*W*nbChannels+j*nbChannels + c;
                float weight = 0.0;
                tempArray[idx] = 0.0;

                if(input[i*W*nbChannels + j*nbChannels + 0] < invalidDepth) {
                    tempArray[idx] += kernel[0]*input[idx];
                    weight += kernel[0];
                }

                for(int k = 1 ; k <= kernelRadius ; ++k) {

                    if(i-r*k < 0) {
                        if(input[j*nbChannels + 0] < invalidDepth) {
                            tempArray[idx] += kernel[k]*input[j*nbChannels + c];
                            weight += kernel[k];
                        }
                    } else {
                        if(input[(i-r*k)*W*nbChannels+j*nbChannels + 0] < invalidDepth) {
                            tempArray[idx] += kernel[k]*input[(i-r*k)*W*nbChannels+j*nbChannels + c];
                            weight += kernel[k];
                        }
                    }

                    if(H-1 < i+r*k) {
                        if(input[(H-1)*W*nbChannels+j*nbChannels + 0] < invalidDepth) {
                            tempArray[idx] += kernel[k]*input[(H-1)*W*nbChannels+j*nbChannels + c];
                            weight += kernel[k];
                        }
                    } else {
                        if(input[(i+r*k)*W*nbChannels+j*nbChannels + 0] < invalidDepth) {
                            tempArray[idx] += kernel[k]*input[(i+r*k)*W*nbChannels+j*nbChannels + c];
                            weight += kernel[k];
                        }
                    }
                }

                if(weight <= weightThreshold) {
                    tempArray[idx] = INVALID_DEPTH;
                } else {
                    tempArray[idx] /= weight;
                }
            }
        }
    }

    for(int i = 0 ; i < H ; ++i) {
        for(int j = 0 ; j < W ; ++j) {
            for(int c = 0 ; c < nbChannels ; ++c) {

                uint idx = i*W*nbChannels+j*nbChannels + c;
                float weight = 0.0;
                output[idx] = 0.0;

                if(tempArray[i*W*nbChannels + j*nbChannels + 0] < invalidDepth) {
                    output[idx] += kernel[0]*tempArray[idx];
                    weight += kernel[0];
                }

                for(int k = 1 ; k <= kernelRadius ; ++k) {

                    if(j-r*k < 0) {
                        if(tempArray[i*W*nbChannels + 0] < invalidDepth) {
                            output[idx] += kernel[k]*tempArray[i*W*nbChannels + c];
                            weight += kernel[k];
                        }
                    } else {
                        if(tempArray[i*W*nbChannels + (j-r*k)*nbChannels + 0] < invalidDepth) {
                            output[idx] += kernel[k]*tempArray[i*W*nbChannels + (j-r*k)*nbChannels + c];
                            weight += kernel[k];
                        }
                    }

                    if(W-1 < j+r*k) {
                        if(tempArray[i*W*nbChannels + (W-1)*nbChannels + 0] < invalidDepth) {
                            output[idx] += kernel[k]*tempArray[i*W*nbChannels + (W-1)*nbChannels + c];
                            weight += kernel[k];
                        }
                    } else {
                        if(tempArray[i*W*nbChannels + (j+r*k)*nbChannels + 0] < invalidDepth) {
                            output[idx] += kernel[k]*tempArray[i*W*nbChannels + (j+r*k)*nbChannels + c];
                            weight += kernel[k];
                        }
                    }
                }

                if(weight <= weightThreshold) {
                    output[idx] = INVALID_DEPTH;
                } else {
                    output[idx] /= weight;
                }
            }
        }
    }

    delete[] tempArray;
}

void oddHDCDepthReduce(int hiW, int hiH, int loW, int loH,
                       int nbChannels, const float* const input, float* const output) {

    const int kernelRadius = 2;
    const float a = 0.4f;
    const float weightThreshold = 0.5f;
    const float invalidDepth = 50.0f;
    float kernel[kernelRadius+1] = {a, 0.25f, 0.25f - 0.5f*a};

    float *tempArray = new float[nbChannels*hiW*loH];

    for(int i = 0 ; i < loH ; ++i) { // we reduce vertically
        for(int j = 0 ; j < hiW ; ++j) {
            for(int c = 0 ; c < nbChannels ; ++c) {

                uint idx = i*hiW*nbChannels+j*nbChannels + c;
                float weight = 0.0;
                tempArray[idx] = 0.0;

                if(input[2*i*hiW*nbChannels + j*nbChannels + 0] < invalidDepth) {
                    tempArray[idx] += kernel[0]*input[2*i*hiW*nbChannels+j*nbChannels + c];
                    weight += kernel[0];
                }

                for(int k = 1 ; k <= kernelRadius ; ++k) {

                    if(2*i-k < 0) {
                        if(input[j*nbChannels + 0] < invalidDepth) {
                            tempArray[idx] += kernel[k]*input[j*nbChannels + c];
                            weight += kernel[k];
                        }
                    } else {
                        if(input[(2*i-k)*hiW*nbChannels+j*nbChannels + 0] < invalidDepth) {
                            tempArray[idx] += kernel[k]*input[(2*i-k)*hiW*nbChannels+j*nbChannels + c];
                            weight += kernel[k];
                        }
                    }

                    if(hiH-1 < 2*i+k) {
                        if(input[(hiH-1)*hiW*nbChannels+j*nbChannels + 0] < invalidDepth) {
                            tempArray[idx] += kernel[k]*input[(hiH-1)*hiW*nbChannels+j*nbChannels + c];
                            weight += kernel[k];
                        }
                    } else {
                        if(input[(2*i+k)*hiW*nbChannels+j*nbChannels + 0] < invalidDepth) {
                            tempArray[idx] += kernel[k]*input[(2*i+k)*hiW*nbChannels+j*nbChannels + c];
                            weight += kernel[k];
                        }
                    }
                }

                if(weight <= weightThreshold) {
                    tempArray[idx] = INVALID_DEPTH;
                } else {
                    tempArray[idx] /= weight;
                }
            }
        }
    }

    for(int i = 0 ; i < loH ; ++i) {
        for(int j = 0 ; j < loW ; ++j) { // we reduce horizontally
            for(int c = 0 ; c < nbChannels ; ++c) {

                uint idx = i*loW*nbChannels+j*nbChannels + c;
                float weight = 0.0;
                output[idx] = 0.0;

                if(tempArray[i*hiW*nbChannels + 2*j*nbChannels + 0] < invalidDepth) {
                    output[idx] += kernel[0]*tempArray[i*hiW*nbChannels+2*j*nbChannels + c];
                    weight += kernel[0];
                }

                for(int k = 1 ; k <= kernelRadius ; ++k) {

                    if(2*j-k < 0) {
                        if(tempArray[i*hiW*nbChannels + 0] < invalidDepth) {
                            output[idx] += kernel[k]*tempArray[i*hiW*nbChannels + c];
                            weight += kernel[k];
                        }
                    } else {
                        if(tempArray[i*hiW*nbChannels + (2*j-k)*nbChannels + 0] < invalidDepth) {
                            output[idx] += kernel[k]*tempArray[i*hiW*nbChannels + (2*j-k)*nbChannels + c];
                            weight += kernel[k];
                        }
                    }

                    if(hiW-1 < 2*j+k) {
                        if(tempArray[i*hiW*nbChannels + (hiW-1)*nbChannels + 0] < invalidDepth) {
                            output[idx] += kernel[k]*tempArray[i*hiW*nbChannels + (hiW-1)*nbChannels + c];
                            weight += kernel[k];
                        }
                    } else {
                        if(tempArray[i*hiW*nbChannels + (2*j+k)*nbChannels + 0] < invalidDepth) {
                            output[idx] += kernel[k]*tempArray[i*hiW*nbChannels + (2*j+k)*nbChannels + c];
                            weight += kernel[k];
                        }
                    }
                }

                if(weight <= weightThreshold) {
                    output[idx] = INVALID_DEPTH;
                } else {
                    output[idx] /= weight;
                }
            }
        }
    }

    delete[] tempArray;
}

void oddGaussianDepth(int W, int H, int nbChannels, const float* const input, float* const output) {

    const int kernelRadius = 4;
    const float weightThreshold = 0.5f;
    const float invalidDepth = 50.0f;
    float kernel[kernelRadius+1] = {0.398943469356098f, 0.241971445656601f, 0.053991127420704f, 0.004431861620031f, 0.000133830624615};

    float *tempArray = new float[nbChannels*W*H];

    for(int i = 0 ; i < H ; ++i) {
        for(int j = 0 ; j < W ; ++j) {
            for(int c = 0 ; c < nbChannels ; ++c) {

                uint idx = i*W*nbChannels+j*nbChannels + c;
                float weight = 0.0;
                tempArray[idx] = 0.0;

                if(input[i*W*nbChannels + j*nbChannels + 0] < invalidDepth) {
                    tempArray[idx] += kernel[0]*input[idx];
                    weight += kernel[0];
                }

                for(int k = 1 ; k <= kernelRadius ; ++k) {

                    if(i-k < 0) {
                        if(input[j*nbChannels + 0] < invalidDepth) {
                            tempArray[idx] += kernel[k]*input[j*nbChannels + c];
                            weight += kernel[k];
                        }
                    } else {
                        if(input[(i-k)*W*nbChannels+j*nbChannels + 0] < invalidDepth) {
                            tempArray[idx] += kernel[k]*input[(i-k)*W*nbChannels+j*nbChannels + c];
                            weight += kernel[k];
                        }
                    }

                    if(H-1 < i+k) {
                        if(input[(H-1)*W*nbChannels+j*nbChannels + 0] < invalidDepth) {
                            tempArray[idx] += kernel[k]*input[(H-1)*W*nbChannels+j*nbChannels + c];
                            weight += kernel[k];
                        }
                    } else {
                        if(input[(i+k)*W*nbChannels+j*nbChannels + 0] < invalidDepth) {
                            tempArray[idx] += kernel[k]*input[(i+k)*W*nbChannels+j*nbChannels + c];
                            weight += kernel[k];
                        }
                    }
                }

                if(weight <= weightThreshold) {
                    tempArray[idx] = INVALID_DEPTH;
                } else {
                    tempArray[idx] /= weight;
                }
            }
        }
    }

    for(int i = 0 ; i < H ; ++i) {
        for(int j = 0 ; j < W ; ++j) {
            for(int c = 0 ; c < nbChannels ; ++c) {

                uint idx = i*W*nbChannels+j*nbChannels + c;
                float weight = 0.0;
                output[idx] = 0.0;

                if(tempArray[i*W*nbChannels + j*nbChannels + 0] < invalidDepth) {
                    output[idx] += kernel[0]*tempArray[idx];
                    weight += kernel[0];
                }

                for(int k = 1 ; k <= kernelRadius ; ++k) {

                    if(j-k < 0) {
                        if(tempArray[i*W*nbChannels + 0] < invalidDepth) {
                            output[idx] += kernel[k]*tempArray[i*W*nbChannels + c];
                            weight += kernel[k];
                        }
                    } else {
                        if(tempArray[i*W*nbChannels + (j-k)*nbChannels + 0] < invalidDepth) {
                            output[idx] += kernel[k]*tempArray[i*W*nbChannels + (j-k)*nbChannels + c];
                            weight += kernel[k];
                        }
                    }

                    if(W-1 < j+k) {
                        if(tempArray[i*W*nbChannels + (W-1)*nbChannels + 0] < invalidDepth) {
                            output[idx] += kernel[k]*tempArray[i*W*nbChannels + (W-1)*nbChannels + c];
                            weight += kernel[k];
                        }
                    } else {
                        if(tempArray[i*W*nbChannels + (j+k)*nbChannels + 0] < invalidDepth) {
                            output[idx] += kernel[k]*tempArray[i*W*nbChannels + (j+k)*nbChannels + c];
                            weight += kernel[k];
                        }
                    }
                }

                if(weight <= weightThreshold) {
                    output[idx] = INVALID_DEPTH;
                } else {
                    output[idx] /= weight;
                }
            }
        }
    }

    delete[] tempArray;
}

void oddHDCDepthExpand(int loW, int loH, int hiW, int hiH, int nbChannels, float *input, float *output) {

    const int kernelRadius = 2;
    const float a = 0.4f;
    const float weightThreshold = 0.5f;
    const float invalidDepth = 50.0f;
    float kernel[kernelRadius+1] = {a, 0.25f, 0.25f - 0.5f*a};

    float *tempArray = new float[nbChannels*loW*hiH];

    for(int i = 0 ; i < hiH ; ++i) {
        for(int j = 0 ; j < loW ; ++j) {
            for(int c = 0 ; c < nbChannels ; ++c) {

                uint o = i*loW*nbChannels + j*nbChannels + c;
                float weight = 0.0;
                tempArray[o] = 0.0;

                if(i%2 == 0) {

                    if((i-2)/2 < 0) {
                        if(input[j*nbChannels + 0] < invalidDepth) {
                            tempArray[o] += 2*kernel[2]*input[j*nbChannels + c];
                            weight += 2*kernel[2];
                        }
                    } else {
                        if(input[(i-2)/2*loW*nbChannels + j*nbChannels + 0] < invalidDepth) {
                            tempArray[o] += 2*kernel[2]*input[(i-2)/2*loW*nbChannels + j*nbChannels + c];
                            weight += 2*kernel[2];
                        }
                    }

                    if(loH-1 < i/2) {
                        if(input[(loH-1)*loW*nbChannels + j*nbChannels + 0] < invalidDepth) {
                            tempArray[o] += 2*kernel[0]*input[(loH-1)*loW*nbChannels + j*nbChannels + c];
                            weight += 2*kernel[0];
                        }
                    } else {
                        if(input[i/2*loW*nbChannels + j*nbChannels + 0] < invalidDepth) {
                            tempArray[o] += 2*kernel[0]*input[i/2*loW*nbChannels + j*nbChannels + c];
                            weight += 2*kernel[0];
                        }
                    }

                    if(loH-1 < (i+2)/2) {
                        if(input[(loH-1)*loW*nbChannels + j*nbChannels + 0] < invalidDepth) {
                            tempArray[o] += 2*kernel[2]*input[(loH-1)*loW*nbChannels + j*nbChannels + c];
                            weight += 2*kernel[2];
                        }
                    } else {
                        if(input[(i+2)/2*loW*nbChannels + j*nbChannels + 0] < invalidDepth) {
                            tempArray[o] += 2*kernel[2]*input[(i+2)/2*loW*nbChannels + j*nbChannels + c];
                            weight += 2*kernel[2];
                        }
                    }

                } else {

                    if((i-1)/2 < 0) {
                        if(input[j*nbChannels + 0] < invalidDepth) {
                            tempArray[o] += 2*kernel[1]*input[j*nbChannels + c];
                            weight += 2*kernel[1];
                        }
                    } else {
                        if(input[(i-1)/2*loW*nbChannels + j*nbChannels + 0] < invalidDepth) {
                            tempArray[o] += 2*kernel[1]*input[(i-1)/2*loW*nbChannels + j*nbChannels + c];
                            weight += 2*kernel[1];
                        }
                    }

                    if(loH-1 < (i+1)/2) {
                        if(input[(loH-1)*loW*nbChannels + j*nbChannels + 0] < invalidDepth) {
                            tempArray[o] += 2*kernel[1]*input[(loH-1)*loW*nbChannels + j*nbChannels + c];
                            weight += 2*kernel[1];
                        }
                    } else {
                        if(input[(i+1)/2*loW*nbChannels + j*nbChannels + 0] < invalidDepth) {
                            tempArray[o] += 2*kernel[1]*input[(i+1)/2*loW*nbChannels + j*nbChannels + c];
                            weight += 2*kernel[1];
                        }
                    }
                }

                if(weight <= weightThreshold) {
                    tempArray[o] = INVALID_DEPTH;
                } else {
                    tempArray[o] /= weight;
                }
            }
        }
    }

    for(int i = 0 ; i < hiH ; ++i) {
        for(int j = 0 ; j < hiW ; ++j) {
            for(int c = 0 ; c < nbChannels ; ++c) {

                uint o = i*hiW*nbChannels+j*nbChannels + c;
                float weight = 0.0;
                output[o] = 0.0;

                if(j%2 == 0) {

                    if((j-2)/2 < 0) {
                        if(tempArray[i*loW*nbChannels + 0] < invalidDepth) {
                            output[o] += 2*kernel[2]*tempArray[i*loW*nbChannels + c];
                            weight += 2*kernel[2];
                        }
                    } else {
                        if(tempArray[i*loW*nbChannels + (j-2)/2*nbChannels + 0] < invalidDepth) {
                            output[o] += 2*kernel[2]*tempArray[i*loW*nbChannels + (j-2)/2*nbChannels + c];
                            weight += 2*kernel[2];
                        }
                    }

                    if(loW-1 < j/2) {
                        if(tempArray[i*loW*nbChannels + (loW-1)*nbChannels + 0] < invalidDepth) {
                            output[o] += 2*kernel[0]*tempArray[i*loW*nbChannels + (loW-1)*nbChannels + c];
                            weight += 2*kernel[0];
                        }
                    } else {
                        if(tempArray[i*loW*nbChannels + j/2*nbChannels + 0] < invalidDepth) {
                            output[o] += 2*kernel[0]*tempArray[i*loW*nbChannels + j/2*nbChannels + c];
                            weight += 2*kernel[0];
                        }
                    }

                    if(loW-1 < (j+2)/2) {
                        if(tempArray[i*loW*nbChannels + (loW-1)*nbChannels + 0] < invalidDepth) {
                            output[o] += 2*kernel[2]*tempArray[i*loW*nbChannels + (loW-1)*nbChannels + c];
                            weight += 2*kernel[2];
                        }
                    } else {
                        if(tempArray[i*loW*nbChannels + (j+2)/2*nbChannels + 0] < invalidDepth) {
                            output[o] += 2*kernel[2]*tempArray[i*loW*nbChannels + (j+2)/2*nbChannels + c];
                            weight += 2*kernel[2];
                        }
                    }

                } else {

                    if((j-1)/2 < 0) {
                        if(tempArray[i*loW*nbChannels + 0] < invalidDepth) {
                            output[o] += 2*kernel[1]*tempArray[i*loW*nbChannels + c];
                            weight += 2*kernel[1];
                        }
                    } else {
                        if(tempArray[i*loW*nbChannels + (j-1)/2*nbChannels + 0] < invalidDepth) {
                            output[o] += 2*kernel[1]*tempArray[i*loW*nbChannels + (j-1)/2*nbChannels + c];
                            weight += 2*kernel[1];
                        }
                    }

                    if(loW-1 < (j+1)/2) {
                        if(tempArray[i*loW*nbChannels + (loW-1)*nbChannels + 0] < invalidDepth) {
                            output[o] += 2*kernel[1]*tempArray[i*loW*nbChannels + (loW-1)*nbChannels + c];
                            weight += 2*kernel[1];
                        }
                    } else {
                        if(tempArray[i*loW*nbChannels + (j+1)/2*nbChannels + 0] < invalidDepth) {
                            output[o] += 2*kernel[1]*tempArray[i*loW*nbChannels + (j+1)/2*nbChannels + c];
                            weight += 2*kernel[1];
                        }
                    }
                }

                if(weight <= weightThreshold) {
                    output[o] = INVALID_DEPTH;
                } else {
                    output[o] /= weight;
                }
            }
        }
    }

    delete[] tempArray;
}

Pyramid::Pyramid( int W, int H ) :

    _W( W ), _H( H ) {
}

Pyramid::~Pyramid() {

    for(uint i = 0 ; i < _gaussianPyramidTex.size() ; ++i) {
        if(_gaussianPyramidTex[i] != 0) {
            delete _gaussianPyramidTex[i];
            _gaussianPyramidTex[i] = 0;
        }
    }
    for(uint i = 0 ; i < _gaussianPyramidArray.size() ; ++i) {
        if(_gaussianPyramidArray[i] != 0) {
            delete[] _gaussianPyramidArray[i];
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
            delete[] _laplacianPyramidArray[i];
            _laplacianPyramidArray[i] = 0;
        }
    }
}

void Pyramid::reduce(int W, int H, int nbChannels, uint scale) {

    const int kernelRadius = 4;
    float kernel[kernelRadius+1] = {0.398943469356098f, 0.241971445656601f, 0.053991127420704f, 0.004431861620031f, 0.000133830624615f};

    float *input = _gaussianPyramidArray[scale-1];
    float *output = _gaussianPyramidArray[scale];

    float *tempArray = new float[nbChannels*W*H];

    for(int i = 0 ; i < H ; ++i) {
        for(int j = 0 ; j < W ; ++j) {
            for(int c = 0 ; c < nbChannels ; ++c) {

                tempArray[i*W*nbChannels+j*nbChannels + c] = kernel[0]*input[i*W*nbChannels + j*nbChannels + c];

                for(int k = 1 ; k <= kernelRadius ; ++k) {

                    if(i-k < 0) {
                        tempArray[i*W*nbChannels+j*nbChannels + c] += kernel[k]*input[j*nbChannels + c];
                    } else {
                        tempArray[i*W*nbChannels+j*nbChannels + c] += kernel[k]*input[(i-k)*W*nbChannels+j*nbChannels + c];
                    }

                    if(H-1 < i+k) {
                        tempArray[i*W*nbChannels+j*nbChannels + c] += kernel[k]*input[(H-1)*W*nbChannels+j*nbChannels + c];
                    } else {
                        tempArray[i*W*nbChannels+j*nbChannels + c] += kernel[k]*input[(i+k)*W*nbChannels+j*nbChannels + c];
                    }
                }
            }
        }
    }

    for(int i = 0 ; i < H ; ++i) {
        for(int j = 0 ; j < W ; ++j) {
            for(int c = 0 ; c < nbChannels ; ++c) {

                output[i*W*nbChannels+j*nbChannels + c] = kernel[0]*tempArray[i*W*nbChannels + j*nbChannels + c];

                for(int k = 1 ; k <= kernelRadius ; ++k) {

                    if(j-k < 0) {
                        output[i*W*nbChannels+j*nbChannels + c] += kernel[k]*tempArray[i*W*nbChannels + c];
                    } else {
                        output[i*W*nbChannels+j*nbChannels + c] += kernel[k]*tempArray[i*W*nbChannels + (j-k)*nbChannels + c];
                    }

                    if(W-1 < j+k) {
                        output[i*W*nbChannels+j*nbChannels + c] += kernel[k]*tempArray[i*W*nbChannels + (W-1)*nbChannels + c];
                    } else {
                        output[i*W*nbChannels+j*nbChannels + c] += kernel[k]*tempArray[i*W*nbChannels + (j+k)*nbChannels + c];
                    }
                }
            }
        }
    }

    delete[] tempArray;
}

void Pyramid::createTestImage(int W, int H, int nbChannels) {

    float *output = _gaussianPyramidArray[0];

    for(int i = 0 ; i < H ; ++i) {
        for(int j = 0 ; j < W ; ++j) {
            for(int c = 0 ; c < nbChannels ; ++c) {

                uint idx = i*W*nbChannels+j*nbChannels + c;

                if(j < W/2) {

                    output[idx] = float(j) / float(W);
                } else {

                    output[idx] = 0.0;
                }
            }
        }
    }
}

void Pyramid::oddHDC(int W, int H, int nbChannels, uint scale) {

    const int kernelRadius = 2;
    const float a = 0.4f;
    const int r = (int)pow((double)2.0, (double)(scale-1));
    float kernel[kernelRadius+1] = {a, 0.25f, 0.25f - 0.5f*a};

    float *input = _gaussianPyramidArray[scale-1];
    float *output = _gaussianPyramidArray[scale];

    float *tempArray = new float[nbChannels*W*H];

    for(int i = 0 ; i < H ; ++i) {
        for(int j = 0 ; j < W ; ++j) {
            for(int c = 0 ; c < nbChannels ; ++c) {

                uint idx = i*W*nbChannels+j*nbChannels + c;
                tempArray[idx] = 0.0;

                tempArray[idx] += kernel[0]*input[i*W*nbChannels + j*nbChannels + c];

                for(int k = 1 ; k <= kernelRadius ; ++k) {

                    if(i-r*k < 0) {
                        tempArray[idx] += kernel[k]*input[j*nbChannels + c];
                    } else {
                        tempArray[idx] += kernel[k]*input[(i-r*k)*W*nbChannels+j*nbChannels + c];
                    }

                    if(H-1 < i+r*k) {
                        tempArray[idx] += kernel[k]*input[(H-1)*W*nbChannels+j*nbChannels + c];
                    } else {
                        tempArray[idx] += kernel[k]*input[(i+r*k)*W*nbChannels+j*nbChannels + c];
                    }
                }
            }
        }
    }

    for(int i = 0 ; i < H ; ++i) {
        for(int j = 0 ; j < W ; ++j) {
            for(int c = 0 ; c < nbChannels ; ++c) {

                uint idx = i*W*nbChannels+j*nbChannels + c;
                output[idx] = 0.0;

                output[idx] += kernel[0]*tempArray[i*W*nbChannels + j*nbChannels + c];

                for(int k = 1 ; k <= kernelRadius ; ++k) {

                    if(j-r*k < 0) {
                        output[idx] += kernel[k]*tempArray[i*W*nbChannels + c];
                    } else {
                        output[idx] += kernel[k]*tempArray[i*W*nbChannels + (j-r*k)*nbChannels + c];
                    }

                    if(W-1 < j+r*k) {
                        output[idx] += kernel[k]*tempArray[i*W*nbChannels + (W-1)*nbChannels + c];
                    } else {
                        output[idx] += kernel[k]*tempArray[i*W*nbChannels + (j+r*k)*nbChannels + c];
                    }
                }
            }
        }
    }

    delete[] tempArray;
}

void Pyramid::oddHDCreduced(int W, int H, int nbChannels, uint scale) {

    const int kernelRadius = 2;
    const float a = 0.4f;
    const int inR = (int)pow((double)2.0, (double)(scale-1));
    const int outR = (int)pow((double)2.0, (double)(scale));
    const int inW = W/inR;
    const int inH = H/inR;
    const int outW = W/outR;
    const int outH = H/outR;
    float kernel[kernelRadius+1] = {a, 0.25f, 0.25f - 0.5f*a};

    float *input = _gaussianPyramidArray[scale-1];
    float *output = _gaussianPyramidArray[scale];

    float *tempArray = new float[nbChannels*inW*outH];

    for(int i = 0 ; i < outH ; ++i) {
        for(int j = 0 ; j < inW ; ++j) {
            for(int c = 0 ; c < nbChannels ; ++c) {

                float weight = 0.0;
                uint o = i*inW*nbChannels + j*nbChannels + c;
                tempArray[o] = 0.0;

                if(input[2*i*inW*nbChannels + j*nbChannels + c] != 0.0) {
                    weight += kernel[0];
                    tempArray[o] += kernel[0]*input[2*i*inW*nbChannels + j*nbChannels + c];
                }

                for(int k = 1 ; k <= kernelRadius ; ++k) {

                    if(2*i-k < 0) {
                        if(input[j*nbChannels + c] != 0.0) {
                            weight += kernel[k];
                            tempArray[o] += kernel[k]*input[j*nbChannels + c];
                        }
                    } else {
                        if(input[(2*i-k)*inW*nbChannels + j*nbChannels + c] != 0.0) {
                            weight += kernel[k];
                            tempArray[o] += kernel[k]*input[(2*i-k)*inW*nbChannels+j*nbChannels + c];
                        }
                    }

                    if(inH-1 < 2*i+k) {
                        if(input[(inH-1)*inW*nbChannels + j*nbChannels + c] != 0.0) {
                            weight += kernel[k];
                            tempArray[o] += kernel[k]*input[(inH-1)*inW*nbChannels + j*nbChannels + c];
                        }
                    } else {
                        if(input[(2*i+k)*inW*nbChannels + j*nbChannels + c] != 0.0) {
                            weight += kernel[k];
                            tempArray[o] += kernel[k]*input[(2*i+k)*inW*nbChannels + j*nbChannels + c];
                        }
                    }
                }

                if(weight != 0.0) {
                    tempArray[o] /= weight;
                }
            }
        }
    }

    for(int i = 0 ; i < outH ; ++i) {
        for(int j = 0 ; j < outW ; ++j) {
            for(int c = 0 ; c < nbChannels ; ++c) {

                float weight = 0.0;
                uint o = i*outW*nbChannels + j*nbChannels + c;
                output[o] = 0.0;

                if(tempArray[i*inW*nbChannels + 2*j*nbChannels + c] != 0.0) {
                    weight += kernel[0];
                    output[o] += kernel[0]*tempArray[i*inW*nbChannels + 2*j*nbChannels + c];
                }

                for(int k = 1 ; k <= kernelRadius ; ++k) {

                    if(2*j-k < 0) {
                        if(tempArray[i*inW*nbChannels + c] != 0.0) {
                            weight += kernel[k];
                            output[o] += kernel[k]*tempArray[i*inW*nbChannels + c];
                        }
                    } else {
                        if(tempArray[i*inW*nbChannels + (2*j-k)*nbChannels + c] != 0.0) {
                            weight += kernel[k];
                            output[o] += kernel[k]*tempArray[i*inW*nbChannels + (2*j-k)*nbChannels + c];
                        }
                    }

                    if(inW-1 < 2*j+k) {
                        if(tempArray[i*inW*nbChannels + (inW-1)*nbChannels + c] != 0.0) {
                            weight += kernel[k];
                            output[o] += kernel[k]*tempArray[i*inW*nbChannels + (inW-1)*nbChannels + c];
                        }
                    } else {
                        if(tempArray[i*inW*nbChannels + (2*j+k)*nbChannels + c] != 0.0) {
                            weight += kernel[k];
                            output[o] += kernel[k]*tempArray[i*inW*nbChannels + (2*j+k)*nbChannels + c];
                        }
                    }
                }

                if(weight != 0.0) {
                    output[o] /= weight;
                }
            }
        }
    }

    delete[] tempArray;
}

void oddHDCexpanded(int inW, int inH, int outW, int outH, int nbChannels, float *input, float *output) {

    const int kernelRadius = 2;
    const float a = 0.4f;

    float kernel[kernelRadius+1] = {a, 0.25f, 0.25f - 0.5f*a};

    float *tempArray = new float[nbChannels*inW*outH];

    for(int i = 0 ; i < outH ; ++i) {
        for(int j = 0 ; j < inW ; ++j) {
            for(int c = 0 ; c < nbChannels ; ++c) {

                uint o = i*inW*nbChannels + j*nbChannels + c;
                tempArray[o] = 0.0;

                if(i%2 == 0) {

                    if((i-2)/2 < 0) {
                        tempArray[o] += 2*kernel[2]*input[j*nbChannels + c];
                    } else {
                        tempArray[o] += 2*kernel[2]*input[(i-2)/2*inW*nbChannels + j*nbChannels + c];
                    }

                    if(inH-1 < i/2) {
                        tempArray[o] += 2*kernel[0]*input[(inH-1)*inW*nbChannels + j*nbChannels + c];
                    } else {
                        tempArray[o] += 2*kernel[0]*input[i/2*inW*nbChannels + j*nbChannels + c];
                    }

                    if(inH-1 < (i+2)/2) {
                        tempArray[o] += 2*kernel[2]*input[(inH-1)*inW*nbChannels + j*nbChannels + c];
                    } else {
                        tempArray[o] += 2*kernel[2]*input[(i+2)/2*inW*nbChannels + j*nbChannels + c];
                    }

                } else {

                    if((i-1)/2 < 0) {
                        tempArray[o] += 2*kernel[1]*input[j*nbChannels + c];
                    } else {
                        tempArray[o] += 2*kernel[1]*input[(i-1)/2*inW*nbChannels + j*nbChannels + c];
                    }

                    if(inH-1 < (i+1)/2) {
                        tempArray[o] += 2*kernel[1]*input[(inH-1)*inW*nbChannels + j*nbChannels + c];
                    } else {
                        tempArray[o] += 2*kernel[1]*input[(i+1)/2*inW*nbChannels + j*nbChannels + c];
                    }
                }
            }
        }
    }

    for(int i = 0 ; i < outH ; ++i) {
        for(int j = 0 ; j < outW ; ++j) {
            for(int c = 0 ; c < nbChannels ; ++c) {

                uint o = i*outW*nbChannels+j*nbChannels + c;
                output[o] = 0.0;

                if(j%2 == 0) {

                    if((j-2)/2 < 0) {
                        output[o] += 2*kernel[2]*tempArray[i*inW*nbChannels + c];
                    } else {
                        output[o] += 2*kernel[2]*tempArray[i*inW*nbChannels + (j-2)/2*nbChannels + c];
                    }

                    if(inW-1 < j/2) {
                        output[o] += 2*kernel[0]*tempArray[i*inW*nbChannels + (inW-1)*nbChannels + c];
                    } else {
                        output[o] += 2*kernel[0]*tempArray[i*inW*nbChannels + j/2*nbChannels + c];
                    }

                    if(inW-1 < (j+2)/2) {
                        output[o] += 2*kernel[2]*tempArray[i*inW*nbChannels + (inW-1)*nbChannels + c];
                    } else {
                        output[o] += 2*kernel[2]*tempArray[i*inW*nbChannels + (j+2)/2*nbChannels + c];
                    }

                } else {

                    if((j-1)/2 < 0) {
                        output[o] += 2*kernel[1]*tempArray[i*inW*nbChannels + c];
                    } else {
                        output[o] += 2*kernel[1]*tempArray[i*inW*nbChannels + (j-1)/2*nbChannels + c];
                    }

                    if(inW-1 < (j+1)/2) {
                        output[o] += 2*kernel[1]*tempArray[i*inW*nbChannels + (inW-1)*nbChannels + c];
                    } else {
                        output[o] += 2*kernel[1]*tempArray[i*inW*nbChannels + (j+1)/2*nbChannels + c];
                    }
                }
            }
        }
    }

    delete[] tempArray;
}

void Pyramid::dog(int W, int H, int nbChannels, uint outputScale) {

    float *input1 = _gaussianPyramidArray[outputScale];
    float *input2 = _gaussianPyramidArray[outputScale+1];
    float *output = _laplacianPyramidArray[outputScale];

    for(int i = 0 ; i < H ; ++i) {
        for(int j = 0 ; j < W ; ++j) {
            for(int c = 0 ; c < nbChannels ; ++c) {

                uint o = i*W*nbChannels+j*nbChannels + c;

                output[o] = input1[o] - input2[o];
            }
        }
    }
}

void Pyramid::dogReduced(int W, int H, int nbChannels, uint outputScale) {

    float *input = _gaussianPyramidArray[outputScale];
    float *output = _laplacianPyramidArray[outputScale];

    for(int i = 0 ; i < H ; ++i) {
        for(int j = 0 ; j < W ; ++j) {
            for(int c = 0 ; c < nbChannels ; ++c) {
                if(c != 3) {
                    uint o = i*W*nbChannels+j*nbChannels + c;
                    output[o] = input[o] - output[o];
                }
            }
        }
    }
}

void Pyramid::gaussianToLaplacian(int nbChannels, uint outputScale) {

    const int inR = (int)pow((double)2.0, (double)(outputScale+1));
    const int outR = (int)pow((double)2.0, (double)(outputScale));
    const int inW = _W/inR;
    const int inH = _H/inR;
    const int outW = _W/outR;
    const int outH = _H/outR;

    oddHDCexpanded(inW, inH, outW, outH, nbChannels, _gaussianPyramidArray[outputScale+1], _laplacianPyramidArray[outputScale]);
    dogReduced(outW, outH, nbChannels, outputScale);
}

void Pyramid::collapse(int nbChannels, uint outputScale) {

    float *laplacian = _laplacianPyramidArray[outputScale];
    float *highScaleGaussian = _gaussianPyramidArray[outputScale+1];
    float *expandedGaussian = _gaussianPyramidArray[outputScale];
    float *output = _gaussianPyramidArray[outputScale];

    const int inR = (int)pow((double)2.0, (double)(outputScale+1));
    const int outR = (int)pow((double)2.0, (double)(outputScale));
    const int inW = _W/inR;
    const int inH = _H/inR;
    const int outW = _W/outR;
    const int outH = _H/outR;

    oddHDCexpanded(inW, inH, outW, outH, nbChannels, highScaleGaussian, expandedGaussian);

    sum(outW, outH, nbChannels, laplacian, expandedGaussian, output);
}

void Pyramid::sum(int W, int H, int nbChannels, float *input1, float *input2, float *output) {

    for(int i = 0 ; i < H ; ++i) {
        for(int j = 0 ; j < W ; ++j) {
            for(int c = 0 ; c < nbChannels ; ++c) {

                uint o = i*W*nbChannels + j*nbChannels + c;

                output[o] = input1[o] + input2[o];
            }
        }
    }
}

void Pyramid::collapse(int W, int H, int nbChannels, uint inscale1, uint inscale2, uint outscale) {

    float *input1 = _gaussianPyramidArray[inscale1];
    float *input2 = _laplacianPyramidArray[inscale2];
    float *output = _gaussianPyramidArray[outscale];

    for(int i = 0 ; i < H ; ++i) {
        for(int j = 0 ; j < W ; ++j) {
            for(int c = 0 ; c < nbChannels ; ++c) {

                output[i*W*nbChannels+j*nbChannels + c] = input1[i*W*nbChannels+j*nbChannels + c] + input2[i*W*nbChannels+j*nbChannels + c];
            }
        }
    }
}

void Pyramid::addLevelContribution(int W, int H, int nbChannels, uint scale, float* contribution) {

    float *level = _laplacianPyramidArray[scale];

    for(int i = 0 ; i < H ; ++i) {
        for(int j = 0 ; j < W ; ++j) {

            uint oa = i*W*nbChannels+j*nbChannels + nbChannels-1;
            float alpha = contribution[oa];

            for(int c = 0 ; c < (nbChannels-1) ; ++c) {

                uint o = i*W*nbChannels+j*nbChannels + c;

                level[o] += contribution[o] * alpha;
                level[oa] += alpha;
            }
        }
    }
}

void Pyramid::sumDetails(int W, int H, int nbChannels, uint inScale1, uint inScale2, float* output) {

    for(int i = 0 ; i < H ; ++i) {
        for(int j = 0 ; j < W ; ++j) {
            for(int c = 0 ; c < nbChannels ; ++c) {

                output[i*W*nbChannels+j*nbChannels + c] = 0;

                for(uint s = inScale1 ; s <= inScale2 ; ++s ) {

                    output[i*W*nbChannels+j*nbChannels + c] += _laplacianPyramidArray[s][i*W*nbChannels+j*nbChannels + c];
                }
            }
        }
    }
}

// CLASSIC PYRAMID EXPAND AND REDUCE OPERATIONS (ON IMAGES)

void reduce(int W, int H, int w, int h, int nbChannels, const float* const input, float* const output) {

    const float a = 0.4f;
    const uint kernelSize = 5;

    float kernel[kernelSize] = {0.25f - 0.5f*a, 0.25f, a, 0.25f, 0.25f - 0.5f*a};

    float *tempArray = new float[nbChannels*W*h];

    for(int i = 0 ; i < h ; ++i) {
        for(int j = 0 ; j < W ; ++j) {

            float weight = 0.0;

            tempArray[i*W*nbChannels+j*nbChannels + 0] = 0;
            tempArray[i*W*nbChannels+j*nbChannels + 1] = 0;
            tempArray[i*W*nbChannels+j*nbChannels + 2] = 0;

            if(2*i-2 < 0) {
                if(input[j*nbChannels + 0] != 0.0 || input[j*nbChannels + 1] != 0.0 || input[j*nbChannels + 2] != 0.0) {
                    tempArray[i*W*nbChannels+j*nbChannels + 0] += kernel[0]*input[j*nbChannels + 0];
                    tempArray[i*W*nbChannels+j*nbChannels + 1] += kernel[0]*input[j*nbChannels + 1];
                    tempArray[i*W*nbChannels+j*nbChannels + 2] += kernel[0]*input[j*nbChannels + 2];
                    weight += kernel[0];
                }
            } else {
                if(input[(2*i-2)*W*nbChannels+j*nbChannels + 0] != 0.0 || input[(2*i-2)*W*nbChannels+j*nbChannels + 1] != 0.0 || input[(2*i-2)*W*nbChannels+j*nbChannels + 2] != 0.0) {
                    tempArray[i*W*nbChannels+j*nbChannels + 0] += kernel[0]*input[(2*i-2)*W*nbChannels+j*nbChannels + 0];
                    tempArray[i*W*nbChannels+j*nbChannels + 1] += kernel[0]*input[(2*i-2)*W*nbChannels+j*nbChannels + 1];
                    tempArray[i*W*nbChannels+j*nbChannels + 2] += kernel[0]*input[(2*i-2)*W*nbChannels+j*nbChannels + 2];
                    weight += kernel[0];
                }
            }

            if(2*i-1 < 0) {
                if(input[j*nbChannels + 0] != 0.0 || input[j*nbChannels + 1] != 0.0 || input[j*nbChannels + 2] != 0.0) {
                    tempArray[i*W*nbChannels+j*nbChannels + 0] += kernel[1]*input[j*nbChannels + 0];
                    tempArray[i*W*nbChannels+j*nbChannels + 1] += kernel[1]*input[j*nbChannels + 1];
                    tempArray[i*W*nbChannels+j*nbChannels + 2] += kernel[1]*input[j*nbChannels + 2];
                    weight += kernel[1];
                }
            } else {
                if(input[(2*i-1)*W*nbChannels+j*nbChannels + 0] != 0.0 || input[(2*i-1)*W*nbChannels+j*nbChannels + 1] != 0.0 || input[(2*i-1)*W*nbChannels+j*nbChannels + 2] != 0.0) {
                    tempArray[i*W*nbChannels+j*nbChannels + 0] += kernel[1]*input[(2*i-1)*W*nbChannels+j*nbChannels + 0];
                    tempArray[i*W*nbChannels+j*nbChannels + 1] += kernel[1]*input[(2*i-1)*W*nbChannels+j*nbChannels + 1];
                    tempArray[i*W*nbChannels+j*nbChannels + 2] += kernel[1]*input[(2*i-1)*W*nbChannels+j*nbChannels + 2];
                    weight += kernel[1];
                }
            }

            if(input[(2*i)*W*nbChannels+j*nbChannels + 0] != 0.0 || input[(2*i)*W*nbChannels+j*nbChannels + 1] != 0.0 || input[(2*i)*W*nbChannels+j*nbChannels + 2] != 0.0) {
                tempArray[i*W*nbChannels+j*nbChannels + 0] += kernel[2]*input[(2*i)*W*nbChannels+j*nbChannels + 0];
                tempArray[i*W*nbChannels+j*nbChannels + 1] += kernel[2]*input[(2*i)*W*nbChannels+j*nbChannels + 1];
                tempArray[i*W*nbChannels+j*nbChannels + 2] += kernel[2]*input[(2*i)*W*nbChannels+j*nbChannels + 2];
                weight += kernel[2];
            }

            if(H-1 < 2*i+1) {
                if(input[(H-1)*W*nbChannels+j*nbChannels + 0] != 0.0 || input[(H-1)*W*nbChannels+j*nbChannels + 1] != 0.0 || input[(H-1)*W*nbChannels+j*nbChannels + 2] != 0.0) {
                    tempArray[i*W*nbChannels+j*nbChannels + 0] += kernel[3]*input[(H-1)*W*nbChannels+j*nbChannels + 0];
                    tempArray[i*W*nbChannels+j*nbChannels + 1] += kernel[3]*input[(H-1)*W*nbChannels+j*nbChannels + 1];
                    tempArray[i*W*nbChannels+j*nbChannels + 2] += kernel[3]*input[(H-1)*W*nbChannels+j*nbChannels + 2];
                    weight += kernel[3];
                }
            } else {
                if(input[(2*i+1)*W*nbChannels+j*nbChannels + 0] != 0.0 || input[(2*i+1)*W*nbChannels+j*nbChannels + 1] != 0.0 || input[(2*i+1)*W*nbChannels+j*nbChannels + 2] != 0.0) {
                    tempArray[i*W*nbChannels+j*nbChannels + 0] += kernel[3]*input[(2*i+1)*W*nbChannels+j*nbChannels + 0];
                    tempArray[i*W*nbChannels+j*nbChannels + 1] += kernel[3]*input[(2*i+1)*W*nbChannels+j*nbChannels + 1];
                    tempArray[i*W*nbChannels+j*nbChannels + 2] += kernel[3]*input[(2*i+1)*W*nbChannels+j*nbChannels + 2];
                    weight += kernel[3];
                }
            }

            if(H-1 < 2*i+2) {
                if(input[(H-1)*W*nbChannels+j*nbChannels + 0] != 0.0 || input[(H-1)*W*nbChannels+j*nbChannels + 1] != 0.0 || input[(H-1)*W*nbChannels+j*nbChannels + 2] != 0.0) {
                    tempArray[i*W*nbChannels+j*nbChannels + 0] += kernel[4]*input[(H-1)*W*nbChannels+j*nbChannels + 0];
                    tempArray[i*W*nbChannels+j*nbChannels + 1] += kernel[4]*input[(H-1)*W*nbChannels+j*nbChannels + 1];
                    tempArray[i*W*nbChannels+j*nbChannels + 2] += kernel[4]*input[(H-1)*W*nbChannels+j*nbChannels + 2];
                    weight += kernel[4];
                }
            } else {
                if(input[(2*i+2)*W*nbChannels+j*nbChannels + 0] != 0.0 || input[(2*i+2)*W*nbChannels+j*nbChannels + 1] != 0.0 || input[(2*i+2)*W*nbChannels+j*nbChannels + 2] != 0.0) {
                    tempArray[i*W*nbChannels+j*nbChannels + 0] += kernel[4]*input[(2*i+2)*W*nbChannels+j*nbChannels + 0];
                    tempArray[i*W*nbChannels+j*nbChannels + 1] += kernel[4]*input[(2*i+2)*W*nbChannels+j*nbChannels + 1];
                    tempArray[i*W*nbChannels+j*nbChannels + 2] += kernel[4]*input[(2*i+2)*W*nbChannels+j*nbChannels + 2];
                    weight += kernel[4];
                }
            }

            if(weight == 0) {
                tempArray[i*W*nbChannels+j*nbChannels + 0] = 0.0;
                tempArray[i*W*nbChannels+j*nbChannels + 1] = 0.0;
                tempArray[i*W*nbChannels+j*nbChannels + 2] = 0.0;
            } else {
                tempArray[i*W*nbChannels+j*nbChannels + 0] /= weight;
                tempArray[i*W*nbChannels+j*nbChannels + 1] /= weight;
                tempArray[i*W*nbChannels+j*nbChannels + 2] /= weight;
            }
        }
    }

    for(int i = 0 ; i < h ; ++i) {
        for(int j = 0 ; j < w ; ++j) {

            float weight = 0.0;

            output[i*w*nbChannels+j*nbChannels + 0] = 0;
            output[i*w*nbChannels+j*nbChannels + 1] = 0;
            output[i*w*nbChannels+j*nbChannels + 2] = 0;

            if(2*j-2 < 0) {
                if(tempArray[i*W*nbChannels + 0] != 0.0 || tempArray[i*W*nbChannels + 1] != 0.0 || tempArray[i*W*nbChannels + 2] != 0.0) {
                    output[i*w*nbChannels+j*nbChannels + 0] += kernel[0]*tempArray[i*W*nbChannels + 0];
                    output[i*w*nbChannels+j*nbChannels + 1] += kernel[0]*tempArray[i*W*nbChannels + 1];
                    output[i*w*nbChannels+j*nbChannels + 2] += kernel[0]*tempArray[i*W*nbChannels + 2];
                    weight += kernel[0];
                }
            } else {
                if(tempArray[i*W*nbChannels+(2*j-2)*nbChannels + 0] != 0.0 || tempArray[i*W*nbChannels+(2*j-2)*nbChannels + 1] != 0.0 || tempArray[i*W*nbChannels+(2*j-2)*nbChannels + 2] != 0.0) {
                    output[i*w*nbChannels+j*nbChannels + 0] += kernel[0]*tempArray[i*W*nbChannels+(2*j-2)*nbChannels + 0];
                    output[i*w*nbChannels+j*nbChannels + 1] += kernel[0]*tempArray[i*W*nbChannels+(2*j-2)*nbChannels + 1];
                    output[i*w*nbChannels+j*nbChannels + 2] += kernel[0]*tempArray[i*W*nbChannels+(2*j-2)*nbChannels + 2];
                    weight += kernel[0];
                }
            }

            if(2*j-1 < 0) {
                if(tempArray[i*W*nbChannels + 0] != 0.0 || tempArray[i*W*nbChannels + 1] != 0.0 || tempArray[i*W*nbChannels + 2] != 0.0) {
                    output[i*w*nbChannels+j*nbChannels + 0] += kernel[1]*tempArray[i*W*nbChannels + 0];
                    output[i*w*nbChannels+j*nbChannels + 1] += kernel[1]*tempArray[i*W*nbChannels + 1];
                    output[i*w*nbChannels+j*nbChannels + 2] += kernel[1]*tempArray[i*W*nbChannels + 2];
                    weight += kernel[1];
                }
            } else {
                if(tempArray[i*W*nbChannels+(2*j-1)*nbChannels + 0] != 0.0 || tempArray[i*W*nbChannels+(2*j-1)*nbChannels + 1] != 0.0 || tempArray[i*W*nbChannels+(2*j-1)*nbChannels + 2] != 0.0) {
                    output[i*w*nbChannels+j*nbChannels + 0] += kernel[1]*tempArray[i*W*nbChannels+(2*j-1)*nbChannels + 0];
                    output[i*w*nbChannels+j*nbChannels + 1] += kernel[1]*tempArray[i*W*nbChannels+(2*j-1)*nbChannels + 1];
                    output[i*w*nbChannels+j*nbChannels + 2] += kernel[1]*tempArray[i*W*nbChannels+(2*j-1)*nbChannels + 2];
                    weight += kernel[1];
                }
            }

            if(tempArray[i*W*nbChannels+(2*j)*nbChannels + 0] != 0.0 || tempArray[i*W*nbChannels+(2*j)*nbChannels + 1] != 0.0 || tempArray[i*W*nbChannels+(2*j)*nbChannels + 2] != 0.0) {
                output[i*w*nbChannels+j*nbChannels + 0] += kernel[2]*tempArray[i*W*nbChannels+(2*j)*nbChannels + 0];
                output[i*w*nbChannels+j*nbChannels + 1] += kernel[2]*tempArray[i*W*nbChannels+(2*j)*nbChannels + 1];
                output[i*w*nbChannels+j*nbChannels + 2] += kernel[2]*tempArray[i*W*nbChannels+(2*j)*nbChannels + 2];
                weight += kernel[2];
            }

            if(W-1 < 2*j+1) {
                if(tempArray[i*W*nbChannels+(W-1)*nbChannels + 0] != 0.0 || tempArray[i*W*nbChannels+(W-1)*nbChannels + 1] != 0.0 || tempArray[i*W*nbChannels+(W-1)*nbChannels + 2] != 0.0) {
                    output[i*w*nbChannels+j*nbChannels + 0] += kernel[3]*tempArray[i*W*nbChannels+(W-1)*nbChannels + 0];
                    output[i*w*nbChannels+j*nbChannels + 1] += kernel[3]*tempArray[i*W*nbChannels+(W-1)*nbChannels + 1];
                    output[i*w*nbChannels+j*nbChannels + 2] += kernel[3]*tempArray[i*W*nbChannels+(W-1)*nbChannels + 2];
                    weight += kernel[3];
                }
            } else {
                if(tempArray[i*W*nbChannels+(2*j+1)*nbChannels + 0] != 0.0 || tempArray[i*W*nbChannels+(2*j+1)*nbChannels + 1] != 0.0 || tempArray[i*W*nbChannels+(2*j+1)*nbChannels + 2] != 0.0) {
                    output[i*w*nbChannels+j*nbChannels + 0] += kernel[3]*tempArray[i*W*nbChannels+(2*j+1)*nbChannels + 0];
                    output[i*w*nbChannels+j*nbChannels + 1] += kernel[3]*tempArray[i*W*nbChannels+(2*j+1)*nbChannels + 1];
                    output[i*w*nbChannels+j*nbChannels + 2] += kernel[3]*tempArray[i*W*nbChannels+(2*j+1)*nbChannels + 2];
                    weight += kernel[3];
                }
            }

            if(W-1 < 2*j+2) {
                if(tempArray[i*W*nbChannels+(W-1)*nbChannels + 0] != 0.0 || tempArray[i*W*nbChannels+(W-1)*nbChannels + 1] != 0.0 || tempArray[i*W*nbChannels+(W-1)*nbChannels + 2] != 0.0) {
                    output[i*w*nbChannels+j*nbChannels + 0] += kernel[4]*tempArray[i*W*nbChannels+(W-1)*nbChannels + 0];
                    output[i*w*nbChannels+j*nbChannels + 1] += kernel[4]*tempArray[i*W*nbChannels+(W-1)*nbChannels + 1];
                    output[i*w*nbChannels+j*nbChannels + 1] += kernel[4]*tempArray[i*W*nbChannels+(W-1)*nbChannels + 2];
                    weight += kernel[4];
                }
            } else {
                if(tempArray[i*W*nbChannels+(2*j+2)*nbChannels + 0] != 0.0 || tempArray[i*W*nbChannels+(2*j+2)*nbChannels + 1] != 0.0 || tempArray[i*W*nbChannels+(2*j+2)*nbChannels + 2] != 0.0) {
                    output[i*w*nbChannels+j*nbChannels + 0] += kernel[4]*tempArray[i*W*nbChannels+(2*j+2)*nbChannels + 0];
                    output[i*w*nbChannels+j*nbChannels + 1] += kernel[4]*tempArray[i*W*nbChannels+(2*j+2)*nbChannels + 1];
                    output[i*w*nbChannels+j*nbChannels + 2] += kernel[4]*tempArray[i*W*nbChannels+(2*j+2)*nbChannels + 2];
                    weight += kernel[4];
                }
            }

            if(weight == 0) {
                output[i*w*nbChannels+j*nbChannels + 0] = 0.0;
                output[i*w*nbChannels+j*nbChannels + 1] = 0.0;
                output[i*w*nbChannels+j*nbChannels + 2] = 0.0;
            } else {
                output[i*w*nbChannels+j*nbChannels + 0] /= weight;
                output[i*w*nbChannels+j*nbChannels + 1] /= weight;
                output[i*w*nbChannels+j*nbChannels + 2] /= weight;
            }
        }
    }

    delete[] tempArray;
}

void expand(int W, int H, int w, int h, int nbChannels, const float* const input, float* const output) {

    const float a = 0.4;
    const uint kernelSize = 5;

    float kernel[kernelSize] = {0.25f - 0.5f*a, 0.25f, a, 0.25f, 0.25f - 0.5f*a};

    float *tempArray = new float[nbChannels*w*H];

    for(int i = 0 ; i < H ; ++i) {
        for(int j = 0 ; j < w ; ++j) {
            for(int c = 0 ; c < nbChannels ; ++c) {

                tempArray[i*w*nbChannels+j*nbChannels + c] = 0;

                if(i%2 == 0) {

                    if((i-2)/2 < 0) {
                        tempArray[i*w*nbChannels+j*nbChannels + c] += 2*kernel[0]*input[j*nbChannels + c];
                    } else {
                        tempArray[i*w*nbChannels+j*nbChannels + c] += 2*kernel[0]*input[(i-2)/2*w*nbChannels+j*nbChannels + c];
                    }
                    if(h-1 < i/2) {
                        tempArray[i*w*nbChannels+j*nbChannels + c] += 2*kernel[2]*input[(h-1)*w*nbChannels+j*nbChannels + c];
                    } else {
                        tempArray[i*w*nbChannels+j*nbChannels + c] += 2*kernel[2]*input[i/2*w*nbChannels+j*nbChannels + c];
                    }
                    if(h-1 < (i+2)/2) {
                        tempArray[i*w*nbChannels+j*nbChannels + c] += 2*kernel[4]*input[(h-1)*w*nbChannels+j*nbChannels + c];
                    } else {
                        tempArray[i*w*nbChannels+j*nbChannels + c] += 2*kernel[4]*input[(i+2)/2*w*nbChannels+j*nbChannels + c];
                    }

                } else {

                    if((i-1)/2 < 0) {
                        tempArray[i*w*nbChannels+j*nbChannels + c] += 2*kernel[1]*input[j*nbChannels + c];
                    } else {
                        tempArray[i*w*nbChannels+j*nbChannels + c] += 2*kernel[1]*input[(i-1)/2*w*nbChannels+j*nbChannels + c];
                    }
                    if(h-1 < (i+1)/2) {
                        tempArray[i*w*nbChannels+j*nbChannels + c] += 2*kernel[3]*input[(h-1)*w*nbChannels+j*nbChannels + c];
                    } else {
                        tempArray[i*w*nbChannels+j*nbChannels + c] += 2*kernel[3]*input[(i+1)/2*w*nbChannels+j*nbChannels + c];
                    }
                }
            }
        }
    }

    for(int i = 0 ; i < H ; ++i) {
        for(int j = 0 ; j < W ; ++j) {
            for(int c = 0 ; c < nbChannels ; ++c) {

                output[i*W*nbChannels+j*nbChannels + c] = 0;

                if(j%2 == 0) {

                    if((j-2)/2 < 0) {
                        output[i*W*nbChannels+j*nbChannels + c] += 2*kernel[0]*tempArray[i*w*nbChannels + c];
                    } else {
                        output[i*W*nbChannels+j*nbChannels + c] += 2*kernel[0]*tempArray[i*w*nbChannels+(j-2)/2*nbChannels + c];
                    }
                    if(w-1 < j/2) {
                        output[i*W*nbChannels+j*nbChannels + c] += 2*kernel[2]*tempArray[i*w*nbChannels+(w-1)*nbChannels + c];
                    } else {
                        output[i*W*nbChannels+j*nbChannels + c] += 2*kernel[2]*tempArray[i*w*nbChannels+j/2*nbChannels + c];
                    }
                    output[i*W*nbChannels+j*nbChannels + c] += 2*kernel[2]*tempArray[i*w*nbChannels+j/2*nbChannels + c];
                    if(w-1 < (j+2)/2) {
                        output[i*W*nbChannels+j*nbChannels + c] += 2*kernel[4]*tempArray[i*w*nbChannels+(w-1)*nbChannels + c];
                    } else {
                        output[i*W*nbChannels+j*nbChannels + c] += 2*kernel[4]*tempArray[i*w*nbChannels+(j+2)/2*nbChannels + c];
                    }

                } else {

                    if((j-1)/2 < 0) {
                        output[i*W*nbChannels+j*nbChannels + c] += 2*kernel[1]*tempArray[i*w*nbChannels + c];
                    } else {
                        output[i*W*nbChannels+j*nbChannels + c] += 2*kernel[1]*tempArray[i*w*nbChannels+(j-1)/2*nbChannels + c];
                    }
                    if(w-1 < (j+1)/2) {
                        output[i*W*nbChannels+j*nbChannels + c] += 2*kernel[3]*tempArray[i*w*nbChannels+(w-1)*nbChannels + c];
                    } else {
                        output[i*W*nbChannels+j*nbChannels + c] += 2*kernel[3]*tempArray[i*w*nbChannels+(j+1)/2*nbChannels + c];
                    }
                }
            }
        }
    }

    delete[] tempArray;
}

void reduceGaussian(int W, int H, int w, int h, int nbChannels, const float* const input, float* const output) {

    const int kernelSize = 8;
    //    float kernel[kernelSize] = {0.000133830624615f, 0.004431861620031f, 0.053991127420704f, 0.241971445656601f, 0.398943469356098f, 0.241971445656601f, 0.053991127420704f, 0.004431861620031f, 0.000133830624615};
    float kernel[kernelSize] = {0.000872710786526f, 0.017528864726030f, 0.129521764811203f, 0.352076659676240f, 0.352076659676240f, 0.129521764811203f, 0.017528864726030f, 0.000872710786526f};

    float *tempArray = new float[nbChannels*W*h];

    for(int i = 0 ; i < h ; ++i) {
        for(int j = 0 ; j < W ; ++j) {
            for(int c = 0 ; c < nbChannels ; ++c) {

                // tempArray[i*W*nbChannels+j*nbChannels + c] = kernel[kernelSize/2]*input[(2*i)*W*nbChannels+j*nbChannels + c];
                tempArray[i*W*nbChannels+j*nbChannels + c] = 0;

                for(int k = 0 ; k < kernelSize/2 ; ++k) {

                    if(2*i-k < 0) {
                        tempArray[i*W*nbChannels+j*nbChannels + c] += kernel[kernelSize/2-1-k]*input[j*nbChannels + c];
                    } else {
                        tempArray[i*W*nbChannels+j*nbChannels + c] += kernel[kernelSize/2-1-k]*input[(2*i-k)*W*nbChannels+j*nbChannels + c];
                    }

                    if(H-1 < 2*i+1+k) {
                        tempArray[i*W*nbChannels+j*nbChannels + c] += kernel[kernelSize/2+k]*input[(H-1)*W*nbChannels+j*nbChannels + c];
                    } else {
                        tempArray[i*W*nbChannels+j*nbChannels + c] += kernel[kernelSize/2+k]*input[(2*i+1+k)*W*nbChannels+j*nbChannels + c];
                    }
                }
            }
        }
    }

    for(int i = 0 ; i < h ; ++i) {
        for(int j = 0 ; j < w ; ++j) {
            for(int c = 0 ; c < nbChannels ; ++c) {

                // output[i*w*nbChannels+j*nbChannels + c] = kernel[kernelSize/2]*tempArray[i*W*nbChannels+(2*j)*nbChannels + c];
                output[i*w*nbChannels+j*nbChannels + c] = 0;

                for(int k = 0 ; k < kernelSize/2 ; ++k) {

                    if(2*j-k < 0) {
                        output[i*w*nbChannels+j*nbChannels + c] += kernel[kernelSize/2-1-k]*tempArray[i*W*nbChannels + c];
                    } else {
                        output[i*w*nbChannels+j*nbChannels + c] += kernel[kernelSize/2-1-k]*tempArray[i*W*nbChannels+(2*j-k)*nbChannels + c];
                    }

                    if(W-1 < 2*j+1+k) {
                        output[i*w*nbChannels+j*nbChannels + c] += kernel[kernelSize/2+k]*tempArray[i*W*nbChannels+(W-1)*nbChannels + c];
                    } else {
                        output[i*w*nbChannels+j*nbChannels + c] += kernel[kernelSize/2+k]*tempArray[i*W*nbChannels+(2*j+1+k)*nbChannels + c];
                    }
                }
            }
        }
    }

    delete[] tempArray;
}

void expandGaussian(int W, int H, int w, int h, int nbChannels, const float* const input, float* const output) {

    const int kernelSize = 8;
    float kernel[kernelSize] = {0.000872710786526f, 0.017528864726030f, 0.129521764811203f, 0.352076659676240f, 0.352076659676240f, 0.129521764811203f, 0.017528864726030f, 0.000872710786526f};
    //    float kernel[kernelSize] = {0.000133830624615f,
    //                                0.004431861620031f,
    //                                0.053991127420704f,
    //                                0.241971445656601f,
    //                                0.398943469356098f,
    //                                0.241971445656601f,
    //                                0.053991127420704f,
    //                                0.004431861620031f,
    //                                0.000133830624615};

    float totalWeight = kernel[0] + kernel[2] + kernel[4] + kernel[6];

    float *tempArray = new float[nbChannels*w*H];

    for(int i = 0 ; i < H ; ++i) {
        for(int j = 0 ; j < w ; ++j) {
            for(int c = 0 ; c < nbChannels ; ++c) {

                tempArray[i*w*nbChannels+j*nbChannels + c] = 0;

                if(i%2 == 0) {

                    //                    tempArray[i*w*nbChannels+j*nbChannels + c] = kernel[kernelSize/2]*input[i/2*w*nbChannels+j*nbChannels + c];

                    for(int k = -4 ; k <= 2 ; k += 2) {

                        if((i+k)/2 < 0) {
                            tempArray[i*w*nbChannels+j*nbChannels + c] += kernel[kernelSize/2+k]*input[j*nbChannels + c];
                        } else if(h-1 < (i+k)/2) {
                            tempArray[i*w*nbChannels+j*nbChannels + c] += kernel[kernelSize/2+k]*input[(h-1)*w*nbChannels+j*nbChannels + c];
                        } else {
                            tempArray[i*w*nbChannels+j*nbChannels + c] += kernel[kernelSize/2+k]*input[(i+k)/2*w*nbChannels+j*nbChannels + c];
                        }

                        //                        if((i-k)/2 < 0) {
                        //                            tempArray[i*w*nbChannels+j*nbChannels + c] += kernel[kernelSize/2-2-k]*input[j*nbChannels + c];
                        //                        } else {
                        //                            tempArray[i*w*nbChannels+j*nbChannels + c] += kernel[kernelSize/2-1-k]*input[(i-k)/2*w*nbChannels+j*nbChannels + c];
                        //                        }

                        //                        if(h-1 < (i+k)/2) {
                        //                            tempArray[i*w*nbChannels+j*nbChannels + c] += kernel[kernelSize/2+k]*input[(h-1)*w*nbChannels+j*nbChannels + c];
                        //                        } else {
                        //                            tempArray[i*w*nbChannels+j*nbChannels + c] += kernel[kernelSize/2+k]*input[(i+k)/2*w*nbChannels+j*nbChannels + c];
                        //                        }
                    }

                } else {

                    for(int k = -3 ; k <= 3 ; k += 2) {

                        if((i+k)/2 < 0) {
                            tempArray[i*w*nbChannels+j*nbChannels + c] += kernel[kernelSize/2+k]*input[j*nbChannels + c];
                        } else if(h-1 < (i+k)/2) {
                            tempArray[i*w*nbChannels+j*nbChannels + c] += kernel[kernelSize/2+k]*input[(h-1)*w*nbChannels+j*nbChannels + c];
                        } else {
                            tempArray[i*w*nbChannels+j*nbChannels + c] += kernel[kernelSize/2+k]*input[(i+k)/2*w*nbChannels+j*nbChannels + c];
                        }

                        //                        if((i-k)/2 < 0) {
                        //                            tempArray[i*w*nbChannels+j*nbChannels + c] += kernel[kernelSize/2-k]*input[j*nbChannels + c];
                        //                        } else {
                        //                            tempArray[i*w*nbChannels+j*nbChannels + c] += kernel[kernelSize/2-k]*input[(i-k)/2*w*nbChannels+j*nbChannels + c];
                        //                        }

                        //                        if(h-1 < (i+k)/2) {
                        //                            tempArray[i*w*nbChannels+j*nbChannels + c] += kernel[kernelSize/2+k]*input[(h-1)*w*nbChannels+j*nbChannels + c];
                        //                        } else {
                        //                            tempArray[i*w*nbChannels+j*nbChannels + c] += kernel[kernelSize/2+k]*input[(i+k)/2*w*nbChannels+j*nbChannels + c];
                        //                        }
                    }
                }

                tempArray[i*w*nbChannels+j*nbChannels + c] /= totalWeight;
            }
        }
    }

    for(int i = 0 ; i < H ; ++i) {
        for(int j = 0 ; j < W ; ++j) {
            for(int c = 0 ; c < nbChannels ; ++c) {

                output[i*W*nbChannels+j*nbChannels + c] = 0;

                if(j%2 == 0) {

                    //                    output[i*W*nbChannels+j*nbChannels + c] = kernel[kernelSize/2]*tempArray[i*w*nbChannels+j/2*nbChannels + c];

                    for(int k = -4 ; k <= 2 ; k += 2) {

                        if((j+k)/2 < 0) {
                            output[i*W*nbChannels+j*nbChannels + c] += kernel[kernelSize/2+k]*tempArray[i*w*nbChannels + c];
                        } else if(w-1 < (j+k)/2) {
                            output[i*W*nbChannels+j*nbChannels + c] += kernel[kernelSize/2+k]*tempArray[i*w*nbChannels+(w-1)*nbChannels + c];
                        } else {
                            output[i*W*nbChannels+j*nbChannels + c] += kernel[kernelSize/2+k]*tempArray[i*w*nbChannels+(j+k)/2*nbChannels + c];
                        }

                        //                        if((j-k)/2 < 0) {
                        //                            output[i*W*nbChannels+j*nbChannels + c] += kernel[kernelSize/2-k]*tempArray[i*w*nbChannels + c];
                        //                        } else {
                        //                            output[i*W*nbChannels+j*nbChannels + c] += kernel[kernelSize/2-k]*tempArray[i*w*nbChannels+(j-k)/2*nbChannels + c];
                        //                        }

                        //                        if(w-1 < (j+k)/2) {
                        //                            output[i*W*nbChannels+j*nbChannels + c] += kernel[kernelSize/2+k]*tempArray[i*w*nbChannels+(w-1)*nbChannels + c];
                        //                        } else {
                        //                            output[i*W*nbChannels+j*nbChannels + c] += kernel[kernelSize/2+k]*tempArray[i*w*nbChannels+(j+k)/2*nbChannels + c];
                        //                        }
                    }

                } else {

                    for(int k = -3 ; k <= 3 ; k += 2) {

                        if((j+k)/2 < 0) {
                            output[i*W*nbChannels+j*nbChannels + c] += kernel[kernelSize/2+k]*tempArray[i*w*nbChannels + c];
                        } else if(w-1 < (j+k)/2) {
                            output[i*W*nbChannels+j*nbChannels + c] += kernel[kernelSize/2+k]*tempArray[i*w*nbChannels+(w-1)*nbChannels + c];
                        } else {
                            output[i*W*nbChannels+j*nbChannels + c] += kernel[kernelSize/2+k]*tempArray[i*w*nbChannels+(j+k)/2*nbChannels + c];
                        }

                        //                        if((j-k)/2 < 0) {
                        //                            output[i*W*nbChannels+j*nbChannels + c] += kernel[kernelSize/2-k]*tempArray[i*w*nbChannels + c];
                        //                        } else {
                        //                            output[i*W*nbChannels+j*nbChannels + c] += kernel[kernelSize/2-k]*tempArray[i*w*nbChannels+(j-k)/2*nbChannels + c];
                        //                        }

                        //                        if(w-1 < (j+k)/2) {
                        //                            output[i*W*nbChannels+j*nbChannels + c] += kernel[kernelSize/2+k]*tempArray[i*w*nbChannels+(w-1)*nbChannels + c];
                        //                        } else {
                        //                            output[i*W*nbChannels+j*nbChannels + c] += kernel[kernelSize/2+k]*tempArray[i*w*nbChannels+(j+k)/2*nbChannels + c];
                        //                        }
                    }
                }

                output[i*W*nbChannels+j*nbChannels + c] /= totalWeight;
            }
        }
    }

    delete[] tempArray;
}

void computeLaplacian(int W, int H, int nbChannels, const float* const input, float* const output) {

    for(int i = 0 ; i < H ; ++i) {
        for(int j = 0 ; j < W ; ++j) {
            for(int c = 0 ; c < nbChannels ; ++c) {

                output[i*W*nbChannels+j*nbChannels + c] = input[i*W*nbChannels+j*nbChannels + c] - output[i*W*nbChannels+j*nbChannels + c];
            }
        }
    }
}

void oddHDCReduceRGB(int W, int H, int w, int h,
                     const std::vector<cv::Point3f>& inputImage, std::vector<cv::Point3f>& outputImage,
                     const std::vector<bool>& inputVisibility, std::vector<bool>& outputVisibility) {

    const int radius = 2; // radius
    const float a = 0.4f;
    float kernel[radius + 1] = {a, 0.25f, 0.25f - 0.5f*a};

    std::vector<cv::Point3f> tempArrayImage(W*H);
    std::vector<bool> tempArrayVisibility(W*H);

    for(int i = 0 ; i < h ; ++i) {
        for(int j = 0 ; j < W ; ++j) {

            uint idx = i*W + j;
            float weight = 0.0f;
            tempArrayImage[idx] = cv::Point3f(0.0f, 0.0f, 0.0f);
            tempArrayVisibility[i*W+j] = false;

            if(inputVisibility[2*i*W + j]) {
                tempArrayImage[idx] += kernel[0]*inputImage[2*i*W + j];
                tempArrayVisibility[idx] = tempArrayVisibility[idx] || inputVisibility[2*i*W + j];
                weight += kernel[0];
            }

            for(int k = 1 ; k <= radius ; ++k) {

                if(2*i - k < 0) {
                    if(inputVisibility[j]) {
                        tempArrayImage[idx] += kernel[k]*inputImage[j];
                        tempArrayVisibility[idx] = tempArrayVisibility[idx] || inputVisibility[j];
                        weight += kernel[k];
                    }
                } else {
                    if(inputVisibility[(2*i - k)*W + j]) {
                        tempArrayImage[idx] += kernel[k]*inputImage[(2*i - k)*W + j];
                        tempArrayVisibility[idx] = tempArrayVisibility[idx] || inputVisibility[(2*i - k)*W + j];
                        weight += kernel[k];
                    }
                }

                if(H - 1 < 2*i + k) {
                    if(inputVisibility[(H - 1)*W + j]) {
                        tempArrayImage[idx] += kernel[k]*inputImage[(H - 1)*W + j];
                        tempArrayVisibility[idx] = tempArrayVisibility[idx] || inputVisibility[(H - 1)*W + j];
                        weight += kernel[k];
                    }
                } else {
                    if(inputVisibility[(2*i + k)*W + j]) {
                        tempArrayImage[idx] += kernel[k]*inputImage[(2*i + k)*W + j];
                        tempArrayVisibility[idx] = tempArrayVisibility[idx] || inputVisibility[(2*i + k)*W + j];
                        weight += kernel[k];
                    }
                }
            }

            if(weight > 0) {

                tempArrayImage[idx] /= weight;
            }
        }
    }

    for(int i = 0 ; i < h ; ++i) {
        for(int j = 0 ; j < w ; ++j) {

            uint idx = i*w + j;
            float weight = 0.0f;
            outputImage[idx] = cv::Point3f(0.0f, 0.0f, 0.0f);
            outputVisibility[idx] = false;

            if(tempArrayVisibility[i*W + 2*j]) {
                outputImage[idx] += kernel[0]*tempArrayImage[i*W + 2*j];
                outputVisibility[idx] = outputVisibility[idx] || tempArrayVisibility[i*W + 2*j];
                weight += kernel[0];
            }

            for(int k = 1 ; k <= radius ; ++k) {

                if(2*j - k < 0) {
                    if(tempArrayVisibility[i*W]) {
                        outputImage[idx] += kernel[k]*tempArrayImage[i*W];
                        outputVisibility[idx] = outputVisibility[idx] || tempArrayVisibility[i*W];
                        weight += kernel[k];
                    }
                } else {
                    if(tempArrayVisibility[i*W + (2*j - k)]) {
                        outputImage[idx] += kernel[k]*tempArrayImage[i*W + (2*j - k)];
                        outputVisibility[idx] = outputVisibility[idx] || tempArrayVisibility[i*W + (2*j - k)];
                        weight += kernel[k];
                    }
                }

                if(W - 1 < 2*j + k) {
                    if(tempArrayVisibility[i*W + (W - 1)]) {
                        outputImage[idx] += kernel[k]*tempArrayImage[i*W + (W - 1)];
                        outputVisibility[idx] = outputVisibility[idx] || tempArrayVisibility[i*W + (W - 1)];
                        weight += kernel[k];
                    }
                } else {
                    if(tempArrayVisibility[i*W + (2*j + k)]) {
                        outputImage[idx] += kernel[k]*tempArrayImage[i*W + (2*j + k)];
                        outputVisibility[idx] = outputVisibility[idx] || tempArrayVisibility[i*W + (2*j + k)];
                        weight += kernel[k];
                    }
                }
            }

            if(weight > 0) {

                outputImage[idx] /= weight;
            }
        }
    }
}

void oddHDCReduceGortler(int W, int H, int w, int h,
                         const std::vector<cv::Point3f>& inputImage, std::vector<cv::Point3f>& outputImage,
                         const std::vector<float>& inputWeight, std::vector<float>& outputWeight) {

    const int radius = 2; // radius
    const float a = 0.4f;
    float kernel[radius + 1] = {a, 0.25f, 0.25f - 0.5f*a};

    std::vector<cv::Point3f> tempArrayImage(W*H);
    std::vector<float> tempArrayWeight(W*H);

    for(int i = 0 ; i < h ; ++i) {
        for(int j = 0 ; j < W ; ++j) {

            uint idx = i*W + j;

            if(H - 1 < 2*i) {
                tempArrayImage[idx] = kernel[0]*inputImage[(H - 1)*W + j];
                tempArrayWeight[i*W + j] = kernel[0]*inputWeight[(H - 1)*W + j];
            } else {
                tempArrayImage[idx] = kernel[0]*inputImage[2*i*W + j];
                tempArrayWeight[i*W + j] = kernel[0]*inputWeight[2*i*W + j];
            }

            for(int k = 1 ; k <= radius ; ++k) {

                if(2*i - k < 0) {
                    tempArrayImage[idx] += kernel[k]*inputImage[j];
                    tempArrayWeight[idx] += kernel[k]*inputWeight[j];
                } else {
                    tempArrayImage[idx] += kernel[k]*inputImage[(2*i - k)*W + j];
                    tempArrayWeight[idx] += kernel[k]*inputWeight[(2*i - k)*W + j];
                }

                if(H - 1 < 2*i + k) {
                    tempArrayImage[idx] += kernel[k]*inputImage[(H - 1)*W + j];
                    tempArrayWeight[idx] += kernel[k]*inputWeight[(H - 1)*W + j];
                } else {
                    tempArrayImage[idx] += kernel[k]*inputImage[(2*i + k)*W + j];
                    tempArrayWeight[idx] += kernel[k]*inputWeight[(2*i + k)*W + j];
                }
            }
        }
    }

    for(int i = 0 ; i < h ; ++i) {
        for(int j = 0 ; j < w ; ++j) {

            uint idx = i*w + j;

            if(W - 1 < 2*j) {
                outputImage[idx] = kernel[0]*tempArrayImage[i*W + (W - 1)];
                outputWeight[idx] = kernel[0]*tempArrayWeight[i*W + (W - 1)];
            } else {
                outputImage[idx] = kernel[0]*tempArrayImage[i*W + 2*j];
                outputWeight[idx] = kernel[0]*tempArrayWeight[i*W + 2*j];
            }

            for(int k = 1 ; k <= radius ; ++k) {

                if(2*j - k < 0) {
                    outputImage[idx] += kernel[k]*tempArrayImage[i*W];
                    outputWeight[idx] += kernel[k]*tempArrayWeight[i*W];
                } else {
                    outputImage[idx] += kernel[k]*tempArrayImage[i*W + (2*j - k)];
                    outputWeight[idx] += kernel[k]*tempArrayWeight[i*W + (2*j - k)];
                }

                if(W - 1 < 2*j + k) {
                    outputImage[idx] += kernel[k]*tempArrayImage[i*W + (W - 1)];
                    outputWeight[idx] += kernel[k]*tempArrayWeight[i*W + (W - 1)];
                } else {
                    outputImage[idx] += kernel[k]*tempArrayImage[i*W + (2*j + k)];
                    outputWeight[idx] += kernel[k]*tempArrayWeight[i*W + (2*j + k)];
                }
            }

            if(outputWeight[idx] > 0.0f) {

                outputImage[idx] /= outputWeight[idx];
            }

            outputWeight[idx] = std::min(outputWeight[idx], 1.0f);
//            outputWeight[idx] = (outputWeight[idx] > 0.0f) ? 1.0f : 0.0f;

            outputImage[idx] *= outputWeight[idx];
        }
    }
}

void oddHDCReduceBool(int W, int H, int w, int h, const std::vector<bool>& input, std::vector<bool>& output) {

    std::vector<bool> tempArray(W*H);

    for(int i = 0 ; i < h ; ++i) {
        for(int j = 0 ; j < W ; ++j) {

            tempArray[i*W+j] = false;

            if(2*i-2 < 0) {
                tempArray[i*W+j] = tempArray[i*W+j] || input[j];
            } else {
                tempArray[i*W+j] = tempArray[i*W+j] || input[(2*i-2)*W+j];
            }

            if(2*i-1 < 0) {
                tempArray[i*W+j] = tempArray[i*W+j] || input[j];
            } else {
                tempArray[i*W+j] = tempArray[i*W+j] || input[(2*i-1)*W+j];
            }

            if(input[(2*i)*W+j] != 0.0) {
                tempArray[i*W+j] = tempArray[i*W+j] || input[(2*i)*W+j];
            }

            if(H-1 < 2*i+1) {
                tempArray[i*W+j] = tempArray[i*W+j] || input[(H-1)*W+j];
            } else {
                tempArray[i*W+j] = tempArray[i*W+j] || input[(2*i+1)*W+j];
            }

            if(H-1 < 2*i+2) {
                tempArray[i*W+j] = tempArray[i*W+j] || input[(H-1)*W+j];
            } else {
                tempArray[i*W+j] = tempArray[i*W+j] || input[(2*i+2)*W+j];
            }
        }
    }

    for(int i = 0 ; i < h ; ++i) {
        for(int j = 0 ; j < w ; ++j) {

            output[i*w+j] = false;

            if(2*j-2 < 0) {
                output[i*w+j] = output[i*w+j] || tempArray[i*W];
            } else {
                output[i*w+j] = output[i*w+j] || tempArray[i*W+(2*j-2)];
            }

            if(2*j-1 < 0) {
                output[i*w+j] = output[i*w+j] || tempArray[i*W];
            } else {
                output[i*w+j] = output[i*w+j] || tempArray[i*W+(2*j-1)];
            }

            if(tempArray[i*W+(2*j)] != 0.0) {
                output[i*w+j] = output[i*w+j] || tempArray[i*W+(2*j)];
            }

            if(W-1 < 2*j+1) {
                output[i*w+j] = output[i*w+j] || tempArray[i*W+(W-1)];
            } else {
                output[i*w+j] = output[i*w+j] || tempArray[i*W+(2*j+1)];
            }

            if(W-1 < 2*j+2) {
                output[i*w+j] = output[i*w+j] || tempArray[i*W+(W-1)];
            } else {
                output[i*w+j] = output[i*w+j] || tempArray[i*W+(2*j+2)];
            }
        }
    }
}

void oddHDCExpandRGB(int loW, int loH, int hiW, int hiH, const std::vector<cv::Point3f>& input, std::vector<cv::Point3f>& output, const std::vector<bool>& visibility) {

    const int radius = 2; // radius
    const float a = 0.4f;
    float kernel[radius + 1] = {a, 0.25f, 0.25f - 0.5f*a};

    std::vector<cv::Point3f> tempArray(loW*hiH);

    for(int i = 0 ; i < hiH ; ++i) {
        for(int j = 0 ; j < loW ; ++j) {

            uint o = i*loW + j;
            tempArray[o] = cv::Point3f(0.0f, 0.0f, 0.0f);

            if(i%2 == 0) {

                if((i - 2)/2 < 0) {
                    tempArray[o] += 2*kernel[2]*input[j];
                } else {
                    tempArray[o] += 2*kernel[2]*input[(i - 2)/2*loW + j];
                }

                if(loH - 1 < i/2) {
                    tempArray[o] += 2*kernel[0]*input[(loH - 1)*loW + j];
                } else {
                    tempArray[o] += 2*kernel[0]*input[i/2*loW + j];
                }

                if(loH - 1 < (i + 2)/2) {
                    tempArray[o] += 2*kernel[2]*input[(loH - 1)*loW + j];
                } else {
                    tempArray[o] += 2*kernel[2]*input[(i + 2)/2*loW + j];
                }

            } else {

                if((i - 1)/2 < 0) {
                    tempArray[o] += 2*kernel[1]*input[j];
                } else {
                    tempArray[o] += 2*kernel[1]*input[(i - 1)/2*loW + j];
                }

                if(loH - 1 < (i + 1)/2) {
                    tempArray[o] += 2*kernel[1]*input[(loH - 1)*loW + j];
                } else {
                    tempArray[o] += 2*kernel[1]*input[(i + 1)/2*loW + j];
                }
            }
        }
    }

    for(int i = 0 ; i < hiH ; ++i) {
        for(int j = 0 ; j < hiW ; ++j) {

            uint o = i*hiW + j;

            if(!visibility[o]) {

                output[o] = cv::Point3f(0.0f, 0.0f, 0.0f);

                if(j%2 == 0) {

                    if((j - 2)/2 < 0) {
                        output[o] += 2*kernel[2]*tempArray[i*loW];
                    } else {
                        output[o] += 2*kernel[2]*tempArray[i*loW + (j - 2)/2];
                    }

                    if(loW - 1 < j/2) {
                        output[o] += 2*kernel[0]*tempArray[i*loW + (loW - 1)];
                    } else {
                        output[o] += 2*kernel[0]*tempArray[i*loW + j/2];
                    }

                    if(loW - 1 < (j + 2)/2) {
                        output[o] += 2*kernel[2]*tempArray[i*loW + (loW - 1)];
                    } else {
                        output[o] += 2*kernel[2]*tempArray[i*loW + (j + 2)/2];
                    }

                } else {

                    if((j - 1)/2 < 0) {
                        output[o] += 2*kernel[1]*tempArray[i*loW];
                    } else {
                        output[o] += 2*kernel[1]*tempArray[i*loW + (j - 1)/2];
                    }

                    if(loW - 1 < (j + 1)/2) {
                        output[o] += 2*kernel[1]*tempArray[i*loW + (loW - 1)];
                    } else {
                        output[o] += 2*kernel[1]*tempArray[i*loW + (j + 1)/2];
                    }
                }
            }
        }
    }
}

void oddHDCExpandGortler(int loW, int loH, int hiW, int hiH,
                         const std::vector<cv::Point3f>& inputImage, std::vector<cv::Point3f>& outputImage,
                         const std::vector<float>& inputWeight, std::vector<float>& outputWeight) {

    const int radius = 2; // radius
    const float a = 0.4f;
    float kernel[radius + 1] = {a, 0.25f, 0.25f - 0.5f*a};

    std::vector<cv::Point3f> tempArrayImage(loW*hiH);
    std::vector<float> tempArrayWeight(loW*hiH);

    for(int i = 0 ; i < hiH ; ++i) {
        for(int j = 0 ; j < loW ; ++j) {

            uint o = i*loW + j;
            tempArrayImage[o] = cv::Point3f(0.0f, 0.0f, 0.0f);
            tempArrayWeight[o] = 0.0f;

            if(i%2 == 0) {

                if((i - 2)/2 < 0) {
                    tempArrayImage[o] += 2*kernel[2]*inputImage[j];
                    tempArrayWeight[o] += 2*kernel[2]*inputWeight[j];
                } else {
                    tempArrayImage[o] += 2*kernel[2]*inputImage[(i - 2)/2*loW + j];
                    tempArrayWeight[o] += 2*kernel[2]*inputWeight[(i - 2)/2*loW + j];
                }

                if(loH - 1 < i/2) {
                    tempArrayImage[o] += 2*kernel[0]*inputImage[(loH - 1)*loW + j];
                    tempArrayWeight[o] += 2*kernel[0]*inputWeight[(loH - 1)*loW + j];
                } else {
                    tempArrayImage[o] += 2*kernel[0]*inputImage[i/2*loW + j];
                    tempArrayWeight[o] += 2*kernel[0]*inputWeight[i/2*loW + j];
                }

                if(loH - 1 < (i + 2)/2) {
                    tempArrayImage[o] += 2*kernel[2]*inputImage[(loH - 1)*loW + j];
                    tempArrayWeight[o] += 2*kernel[2]*inputWeight[(loH - 1)*loW + j];
                } else {
                    tempArrayImage[o] += 2*kernel[2]*inputImage[(i + 2)/2*loW + j];
                    tempArrayWeight[o] += 2*kernel[2]*inputWeight[(i + 2)/2*loW + j];
                }

            } else {

                if((i - 1)/2 < 0) {
                    tempArrayImage[o] += 2*kernel[1]*inputImage[j];
                    tempArrayWeight[o] += 2*kernel[1]*inputWeight[j];
                } else {
                    tempArrayImage[o] += 2*kernel[1]*inputImage[(i - 1)/2*loW + j];
                    tempArrayWeight[o] += 2*kernel[1]*inputWeight[(i - 1)/2*loW + j];
                }

                if(loH - 1 < (i + 1)/2) {
                    tempArrayImage[o] += 2*kernel[1]*inputImage[(loH - 1)*loW + j];
                    tempArrayWeight[o] += 2*kernel[1]*inputWeight[(loH - 1)*loW + j];
                } else {
                    tempArrayImage[o] += 2*kernel[1]*inputImage[(i + 1)/2*loW + j];
                    tempArrayWeight[o] += 2*kernel[1]*inputWeight[(i + 1)/2*loW + j];
                }
            }
        }
    }

    for(int i = 0 ; i < hiH ; ++i) {
        for(int j = 0 ; j < hiW ; ++j) {

            uint o = i*hiW + j;

            cv::Point3f tmpOutputImage = cv::Point3f(0.0f, 0.0f, 0.0f);
            float tmpOutputWeight = 0.0f;

            if(j%2 == 0) {

                if((j - 2)/2 < 0) {
                    tmpOutputImage += 2*kernel[2]*tempArrayImage[i*loW];
                    tmpOutputWeight += 2*kernel[2]*tempArrayWeight[i*loW];
                } else {
                    tmpOutputImage += 2*kernel[2]*tempArrayImage[i*loW + (j - 2)/2];
                    tmpOutputWeight += 2*kernel[2]*tempArrayWeight[i*loW + (j - 2)/2];
                }

                if(loW - 1 < j/2) {
                    tmpOutputImage += 2*kernel[0]*tempArrayImage[i*loW + (loW - 1)];
                    tmpOutputWeight += 2*kernel[0]*tempArrayWeight[i*loW + (loW - 1)];
                } else {
                    tmpOutputImage += 2*kernel[0]*tempArrayImage[i*loW + j/2];
                    tmpOutputWeight += 2*kernel[0]*tempArrayWeight[i*loW + j/2];
                }

                if(loW - 1 < (j + 2)/2) {
                    tmpOutputImage += 2*kernel[2]*tempArrayImage[i*loW + (loW - 1)];
                    tmpOutputWeight += 2*kernel[2]*tempArrayWeight[i*loW + (loW - 1)];
                } else {
                    tmpOutputImage += 2*kernel[2]*tempArrayImage[i*loW + (j + 2)/2];
                    tmpOutputWeight += 2*kernel[2]*tempArrayWeight[i*loW + (j + 2)/2];
                }

            } else {

                if((j - 1)/2 < 0) {
                    tmpOutputImage += 2*kernel[1]*tempArrayImage[i*loW];
                    tmpOutputWeight += 2*kernel[1]*tempArrayWeight[i*loW];
                } else {
                    tmpOutputImage += 2*kernel[1]*tempArrayImage[i*loW + (j - 1)/2];
                    tmpOutputWeight += 2*kernel[1]*tempArrayWeight[i*loW + (j - 1)/2];
                }

                if(loW - 1 < (j + 1)/2) {
                    tmpOutputImage += 2*kernel[1]*tempArrayImage[i*loW + (loW - 1)];
                    tmpOutputWeight += 2*kernel[1]*tempArrayWeight[i*loW + (loW - 1)];
                } else {
                    tmpOutputImage += 2*kernel[1]*tempArrayImage[i*loW + (j + 1)/2];
                    tmpOutputWeight += 2*kernel[1]*tempArrayWeight[i*loW + (j + 1)/2];
                }
            }

            // normalization
            if(tmpOutputWeight > 0.0f) {

                tmpOutputImage /= tmpOutputWeight;
            }

            // compositing
            outputImage[o] = (1.0f - outputWeight[o]) * tmpOutputImage + outputImage[o];
            outputWeight[o] = (1.0f - outputWeight[o]) * tmpOutputWeight + outputWeight[o];
            outputImage[o] *= outputWeight[o];
        }
    }
}


