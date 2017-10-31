#include "lfScene.h"
#include "opticalFlow.h"
#include "triangulation.h"
#include "openGL/pyramid.h"
#include "ply_io.h"
#include <iostream>
#include <fstream>

// backward warping of one view, bilinear interpolation
// convention: 0 coordinate is at the center
void bilinearInterpolation( int width, int height,
                            const std::vector<cv::Point3f>& inputImage,
                            cv::Point3f& outputColor,
                            const cv::Point2f& dest ) {

    // get location in input image
    int cx = (int)floor(dest.x);
    int cy = (int)floor(dest.y);
    int co = cx + cy*width;
    const float dx = dest.x - float(cx); // distance to the center
    const float dy = dest.y - float(cy);

    // transpose bilinear sampling
    float mxmym = (1.0f - dx) * (1.0f - dy);
    float mxpym = dx * (1.0f - dy);
    float mxmyp = (1.0f - dx) * dy;
    float mxpyp = dx * dy;
    float weight = 0.0f;

    if(0 <= cx && cx < width &&
            0 <= cy && cy < height) {
        outputColor += inputImage[co + 0] * mxmym ;
        weight += mxmym;
    }
    if(0 <= cx && cx < width - 1 &&
            0 <= cy && cy < height) {
        outputColor += inputImage[co + 1] * mxpym ;
        weight += mxpym;
    }
    if(0 <= cx && cx < width &&
            0 <= cy && cy < height - 1) {
        outputColor += inputImage[co + width] * mxmyp ;
        weight += mxmyp;
    }
    if(0 <= cx && cx < width - 1 &&
            0 <= cy && cy < height - 1) {
        outputColor += inputImage[co + width + 1] * mxpyp ;
        weight += mxpyp;
    }

    if(weight > 0) {
        outputColor /= weight;
    }
}

InputCam::InputCam( uint W, uint H, std::string outdir ) :
    
    _W(W), _H(H), _outdir(outdir), _texture(W*H) {
    
}

InputCam::~InputCam( ) {
    
}

const std::vector<cv::Point3f>&  InputCam::getTextureRGB() const {
    
    return _texture;
}

PinholeCamera InputCam::getPinholeCamera() const {
    
    return _pinholeCamera;
}

LFScene::LFScene(bool unitTest,
                 std::string outdir,
                 std::string winTitle,
                 std::string mveName, std::string imageName, std::string cameraName,
                 int windowW1, int windowW2, int windowH1, int windowH2,
                 uint camWidth, uint camHeight,
                 int sMin, int sMax, int sRmv,
                 int tMin, int tMax, int tRmv,
                 bool stanfordConfig) :
    
    _unitTest(unitTest),
    _outdir(outdir),
    _windowTitle(winTitle),
    _mveName(mveName), _imageName(imageName), _cameraName(cameraName),
    _windowW1(windowW1), _windowW2(windowW2), _windowH1(windowH1), _windowH2(windowH2),
    _camWidth(camWidth), _camHeight(camHeight),
    _sMin(sMin), _sMax(sMax), _sRmv(sRmv),
    _tMin(tMin), _tMax(tMax), _tRmv(tRmv),
    _stanfordConfig(stanfordConfig),
    _flowedLightField(camWidth*camHeight) {
    
    _mveRdrIdx = 17*_tRmv + _sRmv;
    _S = _sMax - _sMin + 1;
    _T = _tMax - _tMin + 1;
    _centralS = _S/2 + _sMin;
    _centralT = _T/2 + _tMin;
    _nbCameras = _S*_T;
    
    if(_sRmv >= 0 && _tRmv >= 0) {
        _renderIndex = _S*(_tRmv - _tMin) + (_sRmv - _sMin);
    } else {
        _renderIndex = -1;
    }
    _centralIndex = _nbCameras/2;
    
    _windowWidth = windowW2 - windowW1;
    _windowHeight = windowH2 - windowH1;

    if(stanfordConfig) {

        // EXAMPLE WITH STANFORD DATASET

        /* (6,  6) -> (4, 4)
         * (6,  6) -> (6, 4)
         * (8,  6) -> (8, 4)
         * (10, 6) -> (10, 4)
         * (10, 6) -> (12, 4)
         *
         * (6,  6) -> (4, 6)
         * (8,  8) -> (6, 6)
         * (8,  8) -> (8, 6)
         * (8,  8) -> (10, 6)
         * (10, 6) -> (12, 6)
         *
         * (6,  8) -> (4, 8)
         * (8,  8) -> (6, 8)
         * central flow, don't count
         * (8,  8) -> (10, 8)
         * (10, 8) -> (12, 8)
         *
         * (6, 10) -> (4, 10)
         * (8,  8) -> (6, 10)
         * (8,  8) -> (8, 10)
         * (8,  8) -> (10, 10)
         * (10, 10) -> (12, 10)
         *
         * (6,  10) -> (4, 12)
         * (6,  10) -> (6, 12)
         * (8,  10) -> (8, 12)
         * (10, 10) -> (10, 12)
         * (10, 10) -> (12, 12)
         * */

        _sIndicesRight = {_sMin, _sMin + 2, _sMin + 4, _sMin + 6, _sMin + 8,
                          _sMin, _sMin + 2, _sMin + 4, _sMin + 6, _sMin + 8,
                          _sMin, _sMin + 2, _sMin + 4, _sMin + 6, _sMin + 8,
                          _sMin, _sMin + 2, _sMin + 4, _sMin + 6, _sMin + 8,
                          _sMin, _sMin + 2, _sMin + 4, _sMin + 6, _sMin + 8};

        _tIndicesRight = {_tMin, _tMin, _tMin, _tMin, _tMin,
                          _tMin + 2, _tMin + 2, _tMin + 2, _tMin + 2, _tMin + 2,
                          _tMin + 4, _tMin + 4, _tMin + 4, _tMin + 4, _tMin + 4,
                          _tMin + 6, _tMin + 6, _tMin + 6, _tMin + 6, _tMin + 6,
                          _tMin + 8, _tMin + 8, _tMin + 8, _tMin + 8, _tMin + 8};

        _sIndicesLeft = {_sMin + 2, _sMin + 2, _sMin + 4, _sMin + 6, _sMin + 6,
                         _sMin + 2, _sMin + 4, _sMin + 4, _sMin + 4, _sMin + 6,
                         _sMin + 2, _sMin + 4, _sMin + 4, _sMin + 4, _sMin + 6,
                         _sMin + 2, _sMin + 4, _sMin + 4, _sMin + 4, _sMin + 6,
                         _sMin + 2, _sMin + 2, _sMin + 4, _sMin + 6, _sMin + 6};

        _tIndicesLeft = {_tMin + 2, _tMin + 2, _tMin + 2, _tMin + 2, _tMin + 2,
                         _tMin + 2, _tMin + 4, _tMin + 4, _tMin + 4, _tMin + 2,
                         _tMin + 4, _tMin + 4, _tMin + 4, _tMin + 4, _tMin + 4,
                         _tMin + 6, _tMin + 4, _tMin + 4, _tMin + 4, _tMin + 6,
                         _tMin + 6, _tMin + 6, _tMin + 6, _tMin + 6, _tMin + 6};


    } else { // TOLF dataset

        _sIndicesRight = {_sMin, _sMin + 1, _sMin + 2, _sMin + 3, _sMin + 4,
                          _sMin, _sMin + 1, _sMin + 2, _sMin + 3, _sMin + 4,
                          _sMin, _sMin + 1, _sMin + 2, _sMin + 3, _sMin + 4,
                          _sMin, _sMin + 1, _sMin + 2, _sMin + 3, _sMin + 4,
                          _sMin, _sMin + 1, _sMin + 2, _sMin + 3, _sMin + 4};

        _tIndicesRight = {_tMin, _tMin, _tMin, _tMin, _tMin,
                          _tMin + 1, _tMin + 1, _tMin + 1, _tMin + 1, _tMin + 1,
                          _tMin + 2, _tMin + 2, _tMin + 2, _tMin + 2, _tMin + 2,
                          _tMin + 3, _tMin + 3, _tMin + 3, _tMin + 3, _tMin + 3,
                          _tMin + 4, _tMin + 4, _tMin + 4, _tMin + 4, _tMin + 4};

        _sIndicesLeft = {_sMin + 1, _sMin + 1, _sMin + 2, _sMin + 3, _sMin + 3,
                         _sMin + 1, _sMin + 2, _sMin + 2, _sMin + 2, _sMin + 3,
                         _sMin + 1, _sMin + 2, _sMin + 2, _sMin + 2, _sMin + 3,
                         _sMin + 1, _sMin + 2, _sMin + 2, _sMin + 2, _sMin + 3,
                         _sMin + 1, _sMin + 1, _sMin + 2, _sMin + 3, _sMin + 3};

        _tIndicesLeft = {_tMin + 1, _tMin + 1, _tMin + 1, _tMin + 1, _tMin + 1,
                         _tMin + 1, _tMin + 2, _tMin + 2, _tMin + 2, _tMin + 1,
                         _tMin + 2, _tMin + 2, _tMin + 2, _tMin + 2, _tMin + 2,
                         _tMin + 3, _tMin + 2, _tMin + 2, _tMin + 2, _tMin + 3,
                         _tMin + 3, _tMin + 3, _tMin + 3, _tMin + 3, _tMin + 3};
    }
}

LFScene::~LFScene() {
    
    for( uint i = 0 ; i < _nbCameras ; ++i ) {
        if(i < _vCam.size()){
            if(_vCam[i] != 0) {
                delete _vCam[i];
                _vCam[i] = 0;
            }
        }
    }
}

void LFScene::save1fMap(const std::vector<float>& map, const std::string& name, int arg1) {

    char temp[500];
    std::fill_n(temp, 500, 0);
    sprintf( temp, name.c_str(), arg1 );
    std::cout << "Saving map " << temp << std::endl;
    savePFM(map, _camWidth, _camHeight, temp);
}

void LFScene::save2fMap(const std::vector<cv::Point2f>& map, const std::string& name, int arg1) {

    char temp[500];
    std::fill_n(temp, 500, 0);
    sprintf( temp, name.c_str(), arg1 );
    std::cout << "Saving map " << temp << std::endl;
    savePFM(map, _camWidth, _camHeight, temp);
}

void LFScene::save3fMap(const std::vector<cv::Point3f>& map, const std::string& name, int arg1) {

    char temp[500];
    std::fill_n(temp, 500, 0);
    sprintf( temp, name.c_str(), arg1 );
    std::cout << "Saving map " << temp << std::endl;
    savePFM(map, _camWidth, _camHeight, temp);
}

void LFScene::save1uMap(const std::vector<float>& map, const std::string& name, int arg1) {

    char temp[500];
    std::fill_n(temp, 500, 0);
    sprintf( temp, name.c_str(), arg1 );
    std::cout << "Saving map " << temp << std::endl;
    savePNG(map, _camWidth, _camHeight, temp);
}

void LFScene::save2uMap(const std::vector<cv::Point2f>& map, const std::string& name, int arg1) {

    char temp[500];
    std::fill_n(temp, 500, 0);
    sprintf( temp, name.c_str(), arg1 );
    std::cout << "Saving map " << temp << std::endl;
    savePNG(map, _camWidth, _camHeight, temp);
}

void LFScene::save3uMap(const std::vector<cv::Point3f>& map, const std::string& name, int arg1) {

    char temp[500];
    std::fill_n(temp, 500, 0);
    sprintf( temp, name.c_str(), arg1 );
    std::cout << "Saving map " << temp << std::endl;
    savePNG(map, _camWidth, _camHeight, temp);
}

void LFScene::load1fMap(std::vector<float>& map, const std::string& name, int arg1) {

    char temp[500];
    std::fill_n(temp, 500, 0);
    sprintf( temp, name.c_str(), arg1 );
    std::cout << "Loading map " << temp << std::endl;
    loadPFM(map, _camWidth, _camHeight, std::string(temp));
}

void LFScene::load2fMap(std::vector<cv::Point2f>& map, const std::string& name, int arg1) {

    char temp[500];
    std::fill_n(temp, 500, 0);
    sprintf( temp, name.c_str(), arg1 );
    std::cout << "Loading map " << temp << std::endl;
    loadPFM(map, _camWidth, _camHeight, std::string(temp));
}

void LFScene::load3fMap(std::vector<cv::Point3f>& map, const std::string& name, int arg1) {

    char temp[500];
    std::fill_n(temp, 500, 0);
    sprintf( temp, name.c_str(), arg1 );
    std::cout << "Loading map " << temp << std::endl;
    loadPFM(map, _camWidth, _camHeight, std::string(temp));
}

void LFScene::computeVisibilityMask(std::vector<cv::Point2f>& flowLtoR, std::vector<cv::Point2f>& flowRtoL,
                                    std::vector<bool>& visibilityMask)
{
    const float epsilon = 1.0;
    
    for(uint y = 0 ; y < _camHeight ; ++y)
    {
        for(uint x = 0 ; x < _camWidth ; ++x)
        {
            uint pixelL = y*_camWidth + x;
            float xL = (float)x;
            float yL = (float)y;
            
            float xR = xL + flowLtoR[pixelL].x;
            float yR = yL + flowLtoR[pixelL].y;
            
            if(0 <= xR && xR < _camWidth &&
                    0 <= yR && yR < _camHeight)
            {
                uint pixelR = (uint)yR*_camWidth + (uint)xR;
                
                float xRL = xR + flowRtoL[pixelR].x;
                float yRL = yR + flowRtoL[pixelR].y;
                
                if(0 <= xR && xR < _camWidth &&
                        0 <= yR && yR < _camHeight &&
                        abs(xRL - xL) < epsilon &&
                        abs(yRL - yL) < epsilon)
                {
                    visibilityMask[pixelL] = true;
                }
                else
                {
                    visibilityMask[pixelL] = false;
                }
            }
            else
            {
                visibilityMask[pixelL] = false;
            }
        }
    }
}

void LFScene::computePerPixelCorresp(std::string flowAlg)
{
    std::string leftImageName = "";
    std::string rightImageName = "";
    const std::string flowLtoRName = _outdir + "/flow%02luto%02lu.pfm";
    const std::string flowRtoLName = _outdir + "/flow%02luto%02lu.pfm";
    const std::string visibilityMaskLName = _outdir + "/visibilityMask%02luto%02lu.pfm";
    const std::string visibilityMaskRName = _outdir + "/visibilityMask%02luto%02lu.pfm";
    char tempCharArray[500];
    const uint imageSize = _camHeight*_camWidth;
    
    for(int sl = _sMin ; sl < _sMax ; ++sl)
    {
        int sr = sl + 1;
        
        // we don't take into account the view to remove in the reconstruction,
        // compute the flow with new right view when ignoring it
        if(sr == _sRmv) {
            
            ++sr;
        }
        
        if(sl == _sRmv) {
            
            continue;
        }
        
        std::cout << "Computing flow between view " << sl << " and " << sr << std::endl;
        
        std::vector<cv::Point2f> currentFlowLtoR(imageSize);
        std::vector<cv::Point2f> currentFlowRtoL(imageSize);
        std::vector<bool> currentVisibilityMaskl(imageSize);
        std::vector<bool> currentVisibilityMaskr(imageSize);
        
        memset(tempCharArray, 0, sizeof(tempCharArray));
        sprintf( tempCharArray, _imageName.c_str(), sl );
        leftImageName = std::string(tempCharArray);
        memset(tempCharArray, 0, sizeof(tempCharArray));
        sprintf( tempCharArray, _imageName.c_str(), sr );
        rightImageName = std::string(tempCharArray);
        
        // testOpticalFlow(leftImageName, rightImageName);
        computeOpticalFlow(leftImageName,
                           rightImageName,
                           flowAlg,
                           &currentFlowLtoR,
                           &currentFlowRtoL);
        
        memset(tempCharArray, 0, sizeof(tempCharArray));
        sprintf( tempCharArray, flowLtoRName.c_str(), sl, sr );
        savePFM(currentFlowLtoR, _camWidth, _camHeight, std::string(tempCharArray));
        memset(tempCharArray, 0, sizeof(tempCharArray));
        sprintf( tempCharArray, flowRtoLName.c_str(), sr, sl );
        savePFM(currentFlowRtoL, _camWidth, _camHeight, std::string(tempCharArray));
        
        computeVisibilityMask(currentFlowLtoR, currentFlowRtoL, currentVisibilityMaskl);
        computeVisibilityMask(currentFlowRtoL, currentFlowLtoR, currentVisibilityMaskr);
        
        memset(tempCharArray, 0, sizeof(tempCharArray));
        sprintf( tempCharArray, visibilityMaskLName.c_str(), sl, sr );
        savePFM(currentVisibilityMaskl, _camWidth, _camHeight, std::string(tempCharArray));
        memset(tempCharArray, 0, sizeof(tempCharArray));
        sprintf( tempCharArray, visibilityMaskRName.c_str(), sr, sl );
        savePFM(currentVisibilityMaskr, _camWidth, _camHeight, std::string(tempCharArray));
    }
}

void LFScene::computeFlowedLightfield()
{
    // load optical flow
    const std::string flowLtoRName = _outdir + "/flow%02luto%02lu.pfm";
    const std::string visibilityMaskLName = _outdir + "/visibilityMask%02luto%02lu.pfm";
    char flowLtoRNameChar[500];
    char visibilityMaskLNameChar[500];
    const uint imageSize = _camHeight*_camWidth;
    //    uint nbInputViews = 0;
    uint nbInputFlows = 0;
    //    const bool verbose = false;
    const bool backVisibility = false; // load external backward visibility
    
    std::vector<std::vector<cv::Point2f> > flowLtoR;
    // visibilityMaskl : pixels of l view that are not visible from r view
    std::vector<std::vector<bool> > visibilityMaskl;
    
    // loading optical flow and visibility masks (optional)
    // counting the number of input OPTICAL FLOWS depending on whether we remove the target view or not
    for(int sl = _sMin ; sl < _sMax ; ++sl)
    {
        int sr = sl + 1;
        
        // we ignore the view and take the next one to the right
        if(sr == _sRmv) {
            
            ++sr;
        }
        
        if(sl == _sRmv) {
            
            continue;
        }
        
        // TODO: remove the push_back calls
        
        sprintf( flowLtoRNameChar, flowLtoRName.c_str(), sl, sr );
        sprintf( visibilityMaskLNameChar, visibilityMaskLName.c_str(), sl, sr );
        
        std::cout << "Loading flow " << flowLtoRNameChar << " and visibility map " << visibilityMaskLNameChar << std::endl;
        
        std::vector<cv::Point2f> currentFlowLtoR(imageSize);
        std::vector<bool> currentVisibilityMaskl(imageSize);
        
        loadPFM(currentFlowLtoR, _camWidth, _camHeight, std::string(flowLtoRNameChar));
        
        if(backVisibility) {
            loadPFM(currentVisibilityMaskl, _camWidth, _camHeight, std::string(visibilityMaskLNameChar));
        } else {
            std::fill(currentVisibilityMaskl.begin(), currentVisibilityMaskl.end(), 1.0);
        }
        
        flowLtoR.push_back(currentFlowLtoR);
        visibilityMaskl.push_back(currentVisibilityMaskl);
        ++nbInputFlows;
    }
    
    //    // counting the number of input VIEWS depending on whether we remove the target view or not
    //    if(_sMin <= _sRmv && _sRmv <= _sMax) {
    //        nbInputViews = _sMax - _sMin;
    //    } else {
    //        nbInputViews = _sMax - _sMin + 1;
    //    }
    
    for(uint y = 0 ; y < _camHeight ; ++y)
    {
        for(uint x = 0 ; x < _camWidth ; ++x)
        {
            const uint initialIndex = y*_camWidth + x;
            cv::Point2f currentPoint((float)x, (float)y);
            
            for(uint k = 0 ; k < nbInputFlows ; ++k)
            {
                const uint currentIndex = (uint)currentPoint.y*_camWidth + (uint)currentPoint.x;
                
                if(0 <= currentPoint.x && currentPoint.x < _camWidth &&
                        0 <= currentPoint.y && currentPoint.y < _camHeight)
                {
                    if(visibilityMaskl[k][currentIndex])
                    {
                        float xL = currentPoint.x;
                        float yL = currentPoint.y;
                        float xR = xL + flowLtoR[k][currentIndex].x;
                        float yR = yL + flowLtoR[k][currentIndex].y;
                        
                        currentPoint.x = xR;
                        currentPoint.y = yR;
                        _flowedLightField[initialIndex].push_back(currentPoint);
                    }
                    else
                    {
                        break;
                    }
                }
                else
                {
                    break;
                }
            }
        }
    }
    
    //    std::vector<cv::Point3f> centerData(nbInputViews);
    
    //    uint k = 0;
    //    for(int s = _sMin ; s < _sMax ; ++s)
    //    {
    //        if(s == _sRmv)
    //        {
    //            continue;
    //        }
    
    //        PinholeCamera v_k = _vCam[s - _sMin]->getPinholeCamera();
    
    //        centerData[k].x = (float)v_k._C[0];
    //        centerData[k].y = (float)v_k._C[1];
    //        centerData[k].z = (float)v_k._C[2];
    
    //        ++k;
    //    }
    
    //    std::vector<float> trace(imageSize);
    //    std::vector<float> determinant(imageSize);
    //    std::vector<float> eigenRatio(imageSize);
    //    std::vector< cv::Point2f > eigenValues(imageSize);
    //    std::vector< cv::Point3f > eigenVector(imageSize);
    //    std::vector< cv::Point2f > targetImage(imageSize); // target image point (warp)
    
    for(uint y = 0 ; y < _camHeight ; ++y)
    {
        for(uint x = 0 ; x < _camWidth ; ++x)
        {
            //            uint idx = y*_camWidth + x;
            //            computeCovariance(_flowedLightField[idx], centerData, targetCenter,
            //                              &trace[idx], &determinant[idx], &eigenRatio[idx], &eigenValues[idx], &eigenVector[idx],
            //                              &targetImage[idx]);
            
            //            const bool verbose2 = false;
            
            //            if(verbose2)
            //            {
            //                if(x == 500 && y == 200)
            //                {
            //                    std::cout << "Flowed lightfield at point " << cv::Point2f(x, y) << " of size " << flowedLightField[idx].size() << ": " << flowedLightField[idx] << std::endl;
            
            //                    std::cout << "imageData: " << _flowedLightField[idx] << std::endl;
            //                    std::cout << "nbSamples: " << _flowedLightField[idx].size() << std::endl;
            
            //                    std::cout << "trace: " << trace[idx] << std::endl;
            //                    std::cout << "determinant: " << determinant[idx] << std::endl;
            //                    std::cout << "eigenRatio: " << eigenRatio[idx] << std::endl;
            //                    std::cout << "eigenValues: " << eigenValues[idx] << std::endl;
            //                    std::cout << "eigenVector: " << eigenVector[idx] << std::endl;
            
            //                    std::cout << "target image point: " << targetImage[idx] << std::endl;
            //                }
            //            }
            
            const bool verbose3 = false;
            if(verbose3)
            {
                if(x == 500 && y == 200)
                {
                    // test triangulation
                    
                    const int nbSamples = 5;
                    
                    std::vector<cv::Mat> R(nbSamples), R_transp(nbSamples), K(nbSamples), K_inv(nbSamples);
                    std::vector<cv::Point3f> C(nbSamples), t(nbSamples);
                    
                    for(int k = 0 ; k < nbSamples ; ++k) {
                        
                        PinholeCamera v_k = _vCam[k]->getPinholeCamera();
                        glm::mat3 glmK = v_k._K;
                        glm::mat3 glmR = v_k._R;
                        glm::vec3 glmC = v_k._C;
                        glm::vec3 glmt = v_k._t;
                        K[k] = (cv::Mat_<float>(3,3) << glmK[0][0], glmK[1][0], glmK[2][0],
                                glmK[0][1], glmK[1][1], glmK[2][1],
                                glmK[0][2], glmK[1][2], glmK[2][2]);
                        K_inv[k] = K[k].inv();
                        R[k] = (cv::Mat_<float>(3,3) << glmR[0][0], glmR[1][0], glmR[2][0],
                                glmR[0][1], glmR[1][1], glmR[2][1],
                                glmR[0][2], glmR[1][2], glmR[2][2]);
                        R_transp[k] = (cv::Mat_<float>(3,3) << glmR[0][0], glmR[0][1], glmR[0][2],
                                glmR[1][0], glmR[1][1], glmR[1][2],
                                glmR[2][0], glmR[2][1], glmR[2][2]);
                        C[k] = cv::Point3f((float)glmC[0], (float)glmC[1], (float)glmC[2]);
                        t[k] = cv::Point3f((float)glmt[0], (float)glmt[1], (float)glmt[2]);
                    }
                    
                    std::vector<float> point3DLF(3), point3DClassic(3);
                    float finalCostLF(0.0), finalCostClassic(0.0);
                    testLFTriangulation(nbSamples, K_inv, R_transp, C, point3DLF, finalCostLF);
                    testClassicTriangulation(nbSamples, K, R, t, point3DClassic, finalCostClassic);
                    
                    std::cout << "point3DLF: (" << point3DLF[0] << ", " << point3DLF[1] << ", " << point3DLF[2] << ")" << std::endl
                              << "point3DClassic: (" << point3DClassic[0] << ", " << point3DClassic[1] << ", " << point3DClassic[2] << ")" << std::endl;
                    
                    std::cout << "Check LF triangulation" << std::endl;
                    reprojectionCheck(point3DLF, nbSamples, K, R, t);
                    std::cout << "Check classic triangulation" << std::endl;
                    reprojectionCheck(point3DClassic, nbSamples, K, R, t);
                }
            }
        }
    }
    
    //    if(verbose) {
    
    //        std::cout << "Saving trace " << _outdir + "/trace.pfm" << std::endl;
    //        std::cout << "Saving trace " << _outdir + "/determinant.pfm" << std::endl;
    //        std::cout << "Saving trace " << _outdir + "/eigenRatio.pfm" << std::endl;
    //        std::cout << "Saving trace " << _outdir + "/eigenValues.pfm" << std::endl;
    //        std::cout << "Saving trace " << _outdir + "/eigenVector.pfm" << std::endl;
    //        std::cout << "Saving trace " << _outdir + "/targetImage.pfm" << std::endl;
    
    //        savePFM(trace, _camWidth, _camHeight, _outdir + "/trace.pfm");
    //        savePFM(determinant, _camWidth, _camHeight, _outdir + "/determinant.pfm");
    //        savePFM(eigenRatio, _camWidth, _camHeight, _outdir + "/eigenRatio.pfm");
    //        savePFM(eigenValues, _camWidth, _camHeight, _outdir + "/eigenValues.pfm");
    //        savePFM(eigenVector, _camWidth, _camHeight, _outdir + "/eigenVector.pfm");
    //        savePFM(targetImage, _camWidth, _camHeight, _outdir + "/targetImage.pfm");
    //    }
}

void LFScene::computePerPixelCorrespStarConfig(std::string flowAlg) {
    
    std::string leftImageName = "";
    std::string rightImageName = "";
    const std::string flowLtoRName = _outdir + "/flow%02luto%02lu.pfm";
    
    char tempCharArray[500];
    int mveIndexLeft = _centralT*17 + _centralS; // HACK, TODO: stanford LF range as parameter
    sprintf( tempCharArray, _imageName.c_str(), mveIndexLeft );
    leftImageName = std::string(tempCharArray);
    
    const uint imageSize = _camHeight*_camWidth;
    
    OpticalFlow opticalFlow(9);
    
    std::vector<std::vector<cv::Point2f> > flowLtoR(8);
    
    uint flowIdx = 0;
    for(int t = _tMin ; t <= _tMax ; ++t) {
        
        for(int s = _sMin ; s <= _sMax ; ++s) {
            
            uint viewIndexCurrent = _S*(t - _tMin) + (s - _sMin);
            if(viewIndexCurrent == _centralIndex) {
                std::cout << "central view, pass" << std::endl;
                continue;
            }
            //            if(viewIndexCurrent != 0) {
            //                std::cout << "central view, pass" << std::endl;
            //                continue;
            //            }
            
            flowLtoR[flowIdx].resize(imageSize);
            
            // stanford images
            int mveIndexRight = t*17 + s; // HACK, TODO: stanford LF range as parameter
            sprintf( tempCharArray, _imageName.c_str(), mveIndexRight );
            rightImageName = std::string(tempCharArray);
            
            std::cout << "Computing flow between view " << leftImageName << " and " << rightImageName << std::endl;
            
            opticalFlow.pushThread(flowIdx,
                                   leftImageName,
                                   rightImageName,
                                   flowAlg,
                                   flowLtoR[flowIdx]);
            
            ++flowIdx;
        }
    }
    
    opticalFlow.join();
    
    // save flows
    flowIdx = 0;
    for(int t = _tMin ; t <= _tMax ; ++t) {
        
        for(int s = _sMin ; s <= _sMax ; ++s) {
            
            uint viewIndexCurrent = _S*(t - _tMin) + (s - _sMin);
            if(viewIndexCurrent == _centralIndex) {
                std::cout << "central view, pass" << std::endl;
                continue;
            }
            
            std::cout << "Save flow left to right between " << _centralIndex << " and " << viewIndexCurrent << " in " << tempCharArray << std::endl;
            memset(tempCharArray, 0, sizeof(tempCharArray));
            sprintf( tempCharArray, flowLtoRName.c_str(), _centralIndex, viewIndexCurrent );
            savePFM(flowLtoR[flowIdx], _camWidth, _camHeight, std::string(tempCharArray));
            
            ++flowIdx;
        }
    }
}

// run optical flow on custom config
void LFScene::computePerPixelCorrespCustomConfig(std::string flowAlg) {

    // STAR CONFIG EXAMPLE

    /* (1, 1) -> (0, 0)
     * (1, 1) -> (1, 0)
     * (2, 1) -> (2, 0)
     * (3, 1) -> (3, 0)
     * (3, 1) -> (4, 0)
     *
     * (1, 1) -> (0, 1)
     * (2, 2) -> (1, 1)
     * (2, 2) -> (2, 1)
     * (2, 2) -> (3, 1)
     * (3, 1) -> (4, 1)
     *
     * (1, 2) -> (0, 2)
     * (2, 2) -> (1, 2)
     * central flow, don't count
     * (2, 2) -> (3, 2)
     * (3, 2) -> (4, 2)
     *
     * (1, 3) -> (0, 3)
     * (2, 2) -> (1, 3)
     * (2, 2) -> (2, 3)
     * (2, 2) -> (3, 3)
     * (3, 3) -> (4, 3)
     *
     * (1, 3) -> (0, 4)
     * (1, 3) -> (1, 4)
     * (2, 3) -> (2, 4)
     * (3, 3) -> (3, 4)
     * (3, 3) -> (4, 4)
     * */

    uint range(0);
    if(_stanfordConfig) { // HACK, TODO: stanford LF range as parameter
        range = 17;
    } else {
        range = 5;
    }
    
    assert(_nbCameras == _sIndicesLeft.size());
    // nbFlows = _nbCameras - 1;
    
    std::string leftImageName = "";
    std::string rightImageName = "";
    const std::string flowLtoRName = _outdir + "/flow%02lu.pfm";
    
    char tempCharArray[500];
    
    const uint imageSize = _camHeight*_camWidth;
    
    std::vector<std::vector<cv::Point2f> > flowLtoR(_nbCameras);
    
    OpticalFlow* opticalFlow = new OpticalFlow(9);
    uint opticalFlowIdx = 0;
    
    for(uint viewIndex = 0 ; viewIndex < 8 ; ++viewIndex) {
        
        flowLtoR[viewIndex].resize(imageSize);
        
        // stanford images
        int sLeft = _sIndicesLeft[viewIndex];
        int tLeft = _tIndicesLeft[viewIndex];
        int sRight = _sIndicesRight[viewIndex];
        int tRight = _tIndicesRight[viewIndex];
        
        int imageIndexLeft = tLeft*range + sLeft;
        memset(tempCharArray, 0, sizeof(tempCharArray));
        sprintf( tempCharArray, _imageName.c_str(), imageIndexLeft );
        leftImageName = std::string(tempCharArray);
        int imageIndexRight = tRight*range + sRight;
        memset(tempCharArray, 0, sizeof(tempCharArray));
        sprintf( tempCharArray, _imageName.c_str(), imageIndexRight );
        rightImageName = std::string(tempCharArray);
        
        std::cout << "Computing flow between view " << leftImageName << " and " << rightImageName << std::endl;
        
        opticalFlow->pushThread(opticalFlowIdx,
                                leftImageName,
                                rightImageName,
                                flowAlg,
                                flowLtoR[viewIndex]);
        
        ++opticalFlowIdx;
    }
    opticalFlow->join();
    
    delete opticalFlow;
    opticalFlow = new OpticalFlow(9);
    opticalFlowIdx = 0;
    
    for(uint viewIndex = 8 ; viewIndex < 17 ; ++viewIndex) {
        
        if(viewIndex == _nbCameras/2) {
            std::cout << "central flow, don't compute" << std::endl;
            continue;
        }
        
        flowLtoR[viewIndex].resize(imageSize);
        
        // stanford images
        int sLeft = _sIndicesLeft[viewIndex];
        int tLeft = _tIndicesLeft[viewIndex];
        int sRight = _sIndicesRight[viewIndex];
        int tRight = _tIndicesRight[viewIndex];
        
        int imageIndexLeft = tLeft*range + sLeft;
        memset(tempCharArray, 0, sizeof(tempCharArray));
        sprintf( tempCharArray, _imageName.c_str(), imageIndexLeft );
        leftImageName = std::string(tempCharArray);
        int imageIndexRight = tRight*range + sRight;
        memset(tempCharArray, 0, sizeof(tempCharArray));
        sprintf( tempCharArray, _imageName.c_str(), imageIndexRight );
        rightImageName = std::string(tempCharArray);
        
        std::cout << "Computing flow between view " << leftImageName << " and " << rightImageName << std::endl;
        
        opticalFlow->pushThread(opticalFlowIdx,
                                leftImageName,
                                rightImageName,
                                flowAlg,
                                flowLtoR[viewIndex]);
        
        ++opticalFlowIdx;
    }
    opticalFlow->join();
    
    delete opticalFlow;
    opticalFlow = new OpticalFlow(9);
    opticalFlowIdx = 0;
    
    for(uint viewIndex = 17 ; viewIndex < 25 ; ++viewIndex) {
        
        flowLtoR[viewIndex].resize(imageSize);
        
        // stanford images
        int sLeft = _sIndicesLeft[viewIndex];
        int tLeft = _tIndicesLeft[viewIndex];
        int sRight = _sIndicesRight[viewIndex];
        int tRight = _tIndicesRight[viewIndex];
        
        int imageIndexLeft = tLeft*range + sLeft;
        memset(tempCharArray, 0, sizeof(tempCharArray));
        sprintf( tempCharArray, _imageName.c_str(), imageIndexLeft );
        leftImageName = std::string(tempCharArray);
        int imageIndexRight = tRight*range + sRight;
        memset(tempCharArray, 0, sizeof(tempCharArray));
        sprintf( tempCharArray, _imageName.c_str(), imageIndexRight );
        rightImageName = std::string(tempCharArray);
        
        std::cout << "Computing flow between view " << leftImageName << " and " << rightImageName << std::endl;
        
        opticalFlow->pushThread(opticalFlowIdx,
                                leftImageName,
                                rightImageName,
                                flowAlg,
                                flowLtoR[viewIndex]);
        
        ++opticalFlowIdx;
    }
    opticalFlow->join();
    
    delete opticalFlow;
    
    // SAVE ALL OPTICAL FLOWS
    for(uint viewIndex = 0 ; viewIndex < 25 ; ++viewIndex) {
        
        if(viewIndex == _nbCameras/2) {
            std::cout << "central flow, don't compute" << std::endl;
            continue;
        }
        
        // stanford images
        int sLeft = _sIndicesLeft[viewIndex];
        int tLeft = _tIndicesLeft[viewIndex];
        int sRight = _sIndicesRight[viewIndex];
        int tRight = _tIndicesRight[viewIndex];

        save2fMap(flowLtoR[viewIndex], flowLtoRName, viewIndex);
        std::cout << "Save flow left to right between (" << sLeft << ", " << tLeft << ") and (" << sRight << ", " << tRight << ") in " << tempCharArray << std::endl;
    }
}

// TODO: change mve indices before using this function!!!
void LFScene::computePerPixelCorrespBandConfig(std::string flowAlg) {
    
    std::string leftImageName = "";
    std::string rightImageName = "";
    int viewIndexLeft = 0;
    int viewIndexRight = 0;
    const std::string flowLtoRName = _outdir + "/flow%02luto%02lu.pfm";
    const uint imageSize = _camHeight*_camWidth;
    std::vector<cv::Point2f> currentFlowLtoR(imageSize);
    std::vector<cv::Point2f> currentFlowRtoL(imageSize);
    char tempCharArray[500];
    int t = _centralT;
    
    // INIT (central view)
    
    viewIndexLeft = _S*(t - _tMin) + (_centralS - _sMin);
    sprintf( tempCharArray, _imageName.c_str(), viewIndexLeft );
    leftImageName = std::string(tempCharArray);
    
    viewIndexRight = _S*(t + 1 - _tMin) + (_centralS - _sMin);
    sprintf( tempCharArray, _imageName.c_str(), viewIndexRight );
    rightImageName = std::string(tempCharArray);
    
    std::cout << "Computing flow between view (" << _centralS << ", " << t << ") and ("
              << _centralS << ", " << t + 1 << ")." << std::endl;
    
    computeOpticalFlow(leftImageName,
                       rightImageName,
                       flowAlg,
                       &currentFlowLtoR,
                       &currentFlowRtoL);
    
    memset(tempCharArray, 0, sizeof(tempCharArray));
    sprintf( tempCharArray, flowLtoRName.c_str(), viewIndexLeft, viewIndexRight );
    savePFM(currentFlowLtoR, _camWidth, _camHeight, std::string(tempCharArray));
    
    // -----------------------------------------------------------------------------------
    
    viewIndexRight = _S*(t - 1 - _tMin) + (_centralS - _sMin);
    sprintf( tempCharArray, _imageName.c_str(), viewIndexRight );
    rightImageName = std::string(tempCharArray);
    
    std::cout << "Computing flow between view (" << _centralS << ", " << t << ") and ("
              << _centralS << ", " << t - 1 << ")." << std::endl;
    
    computeOpticalFlow(leftImageName,
                       rightImageName,
                       flowAlg,
                       &currentFlowLtoR,
                       &currentFlowRtoL);
    
    memset(tempCharArray, 0, sizeof(tempCharArray));
    sprintf( tempCharArray, flowLtoRName.c_str(), viewIndexLeft, viewIndexRight );
    savePFM(currentFlowLtoR, _camWidth, _camHeight, std::string(tempCharArray));
    
    // -----------------------------------------------------------------------------------
    
    if(_centralS < _sMax) {
        
        viewIndexRight = _S*(t  - _tMin) + (_centralS + 1 - _sMin);
        sprintf( tempCharArray, _imageName.c_str(), viewIndexRight );
        rightImageName = std::string(tempCharArray);
        
        std::cout << "Computing flow between view (" << _centralS << ", " << t << ") and ("
                  << _centralS + 1 << ", " << t << ")." << std::endl;
        
        computeOpticalFlow(leftImageName,
                           rightImageName,
                           flowAlg,
                           &currentFlowLtoR,
                           &currentFlowRtoL);
        
        memset(tempCharArray, 0, sizeof(tempCharArray));
        sprintf( tempCharArray, flowLtoRName.c_str(), viewIndexLeft, viewIndexRight );
        savePFM(currentFlowLtoR, _camWidth, _camHeight, std::string(tempCharArray));
    }
    
    if(_centralS > _sMin) {
        
        viewIndexRight = _S*(t  - _tMin) + (_centralS - 1 - _sMin);
        sprintf( tempCharArray, _imageName.c_str(), viewIndexRight );
        rightImageName = std::string(tempCharArray);
        
        std::cout << "Computing flow between view (" << _centralS << ", " << t << ") and ("
                  << _centralS - 1 << ", " << t << ")." << std::endl;
        
        computeOpticalFlow(leftImageName,
                           rightImageName,
                           flowAlg,
                           &currentFlowLtoR,
                           &currentFlowRtoL);
        
        memset(tempCharArray, 0, sizeof(tempCharArray));
        sprintf( tempCharArray, flowLtoRName.c_str(), viewIndexLeft, viewIndexRight );
        savePFM(currentFlowLtoR, _camWidth, _camHeight, std::string(tempCharArray));
    }
    
    for(int s = _centralS + 1 ; s <= _sMax ; ++s) {
        
        viewIndexLeft = _S*(t - _tMin) + (s - _sMin);
        sprintf( tempCharArray, _imageName.c_str(), viewIndexLeft );
        leftImageName = std::string(tempCharArray);
        
        viewIndexRight = _S*(t + 1 - _tMin) + (s - _sMin);
        sprintf( tempCharArray, _imageName.c_str(), viewIndexRight );
        rightImageName = std::string(tempCharArray);
        
        std::cout << "Computing flow between view (" << s << ", " << t << ") and ("
                  << s << ", " << t + 1 << ")." << std::endl;
        
        computeOpticalFlow(leftImageName,
                           rightImageName,
                           flowAlg,
                           &currentFlowLtoR,
                           &currentFlowRtoL);
        
        memset(tempCharArray, 0, sizeof(tempCharArray));
        sprintf( tempCharArray, flowLtoRName.c_str(), viewIndexLeft, viewIndexRight );
        savePFM(currentFlowLtoR, _camWidth, _camHeight, std::string(tempCharArray));
        
        // -----------------------------------------------------------------------------------
        
        viewIndexRight = _S*(t - 1 - _tMin) + (s - _sMin);
        sprintf( tempCharArray, _imageName.c_str(), viewIndexRight );
        rightImageName = std::string(tempCharArray);
        
        std::cout << "Computing flow between view (" << s << ", " << t << ") and ("
                  << s << ", " << t - 1 << ")." << std::endl;
        
        computeOpticalFlow(leftImageName,
                           rightImageName,
                           flowAlg,
                           &currentFlowLtoR,
                           &currentFlowRtoL);
        
        memset(tempCharArray, 0, sizeof(tempCharArray));
        sprintf( tempCharArray, flowLtoRName.c_str(), viewIndexLeft, viewIndexRight );
        savePFM(currentFlowLtoR, _camWidth, _camHeight, std::string(tempCharArray));
        
        // -----------------------------------------------------------------------------------
        
        if(s < _sMax) {
            
            viewIndexRight = _S*(t  - _tMin) + (s + 1 - _sMin);
            sprintf( tempCharArray, _imageName.c_str(), viewIndexRight );
            rightImageName = std::string(tempCharArray);
            
            std::cout << "Computing flow between view (" << s << ", " << t << ") and ("
                      << s + 1 << ", " << t << ")." << std::endl;
            
            computeOpticalFlow(leftImageName,
                               rightImageName,
                               flowAlg,
                               &currentFlowLtoR,
                               &currentFlowRtoL);
            
            memset(tempCharArray, 0, sizeof(tempCharArray));
            sprintf( tempCharArray, flowLtoRName.c_str(), viewIndexLeft, viewIndexRight );
            savePFM(currentFlowLtoR, _camWidth, _camHeight, std::string(tempCharArray));
        }
    }
    
    for(int s = _centralS - 1 ; s >=_sMin ; --s) {
        
        viewIndexLeft = _S*(t - _tMin) + (s - _sMin);
        sprintf( tempCharArray, _imageName.c_str(), viewIndexLeft );
        leftImageName = std::string(tempCharArray);
        
        viewIndexRight = _S*(t + 1 - _tMin) + (s - _sMin);
        sprintf( tempCharArray, _imageName.c_str(), viewIndexRight );
        rightImageName = std::string(tempCharArray);
        
        std::cout << "Computing flow between view (" << s << ", " << t << ") and ("
                  << s << ", " << t + 1 << ")." << std::endl;
        
        computeOpticalFlow(leftImageName,
                           rightImageName,
                           flowAlg,
                           &currentFlowLtoR,
                           &currentFlowRtoL);
        
        memset(tempCharArray, 0, sizeof(tempCharArray));
        sprintf( tempCharArray, flowLtoRName.c_str(), viewIndexLeft, viewIndexRight );
        savePFM(currentFlowLtoR, _camWidth, _camHeight, std::string(tempCharArray));
        
        // -----------------------------------------------------------------------------------
        
        viewIndexRight = _S*(t - 1 - _tMin) + (s - _sMin);
        sprintf( tempCharArray, _imageName.c_str(), viewIndexRight );
        rightImageName = std::string(tempCharArray);
        
        std::cout << "Computing flow between view (" << s << ", " << t << ") and ("
                  << s << ", " << t - 1 << ")." << std::endl;
        
        computeOpticalFlow(leftImageName,
                           rightImageName,
                           flowAlg,
                           &currentFlowLtoR,
                           &currentFlowRtoL);
        
        memset(tempCharArray, 0, sizeof(tempCharArray));
        sprintf( tempCharArray, flowLtoRName.c_str(), viewIndexLeft, viewIndexRight );
        savePFM(currentFlowLtoR, _camWidth, _camHeight, std::string(tempCharArray));
        
        // -----------------------------------------------------------------------------------
        
        if(s > _sMin) {
            
            viewIndexRight = _S*(t  - _tMin) + (s - 1 - _sMin);
            sprintf( tempCharArray, _imageName.c_str(), viewIndexRight );
            rightImageName = std::string(tempCharArray);
            
            std::cout << "Computing flow between view (" << s << ", " << t << ") and ("
                      << s - 1 << ", " << t << ")." << std::endl;
            
            computeOpticalFlow(leftImageName,
                               rightImageName,
                               flowAlg,
                               &currentFlowLtoR,
                               &currentFlowRtoL);
            
            memset(tempCharArray, 0, sizeof(tempCharArray));
            sprintf( tempCharArray, flowLtoRName.c_str(), viewIndexLeft, viewIndexRight );
            savePFM(currentFlowLtoR, _camWidth, _camHeight, std::string(tempCharArray));
        }
    }
}

void LFScene::computeFlowedLFStarConfig() {
    
    // load optical flow
    const std::string flowLtoRName = _outdir + "/flow%02luto%02lu.pfm";
    const std::string visibilityMaskLName = _outdir + "/visibilityMask%02luto%02lu.pfm";
    char flowLtoRNameChar[500];
    char visibilityMaskLNameChar[500];
    const uint imageSize = _camHeight*_camWidth;
    //    uint nbInputViews = 0;
    //    const bool verbose = false;
    const bool backVisibility = false; // take backward visibility into account, TODO: what if true?
    
    std::vector<std::vector<cv::Point2f> > flowLtoR(_vCam.size());
    // visibilityMaskl : pixels of l view that are not visible from r view
    std::vector<std::vector<bool> > visibilityMaskl(_vCam.size());
    
    // loading the optical flows
    // for all views except the central one
    for(int t = _tMin ; t <= _tMax ; ++t) {
        
        for(int s = _sMin ; s <= _sMax ; ++s) {
            
            if(s == _centralS && t == _centralT) {
                
                continue;
            }
            
            int k = _S*(t - _tMin) + (s - _sMin); // current view index
            
            sprintf( flowLtoRNameChar, flowLtoRName.c_str(), _centralIndex, k );
            sprintf( visibilityMaskLNameChar, visibilityMaskLName.c_str(), _centralIndex, k );
            
            std::cout << "Loading flow " << flowLtoRNameChar << " and visibility map " << visibilityMaskLNameChar << std::endl;
            
            std::vector<cv::Point2f> currentFlowLtoR(imageSize);
            std::vector<bool> currentVisibilityMaskl(imageSize);
            
            loadPFM(currentFlowLtoR, _camWidth, _camHeight, std::string(flowLtoRNameChar));
            
            if(backVisibility) {
                loadPFM(currentVisibilityMaskl, _camWidth, _camHeight, std::string(visibilityMaskLNameChar));
            } else {
                std::fill(currentVisibilityMaskl.begin(), currentVisibilityMaskl.end(), 1.0);
            }
            
            flowLtoR[k] = currentFlowLtoR;
            visibilityMaskl[k] = currentVisibilityMaskl;
        }
    }
    
    // number of samples for each flow is nbCameras = nbSurrounding + centralView
    // we don't remove any input view
    
    // computing the LF flow
    for(uint y = 0 ; y < _camHeight ; ++y)
    {
        for(uint x = 0 ; x < _camWidth ; ++x)
        {
            const uint idx = y*_camWidth + x;
            // size of _flowedLightField[idx] equals the number of surrounding views plsu the central view, even if not all views contribute
            // so don't pushback, but fill with empty point instead
            _flowedLightField[idx].resize(_vCam.size());
            
            for(uint k = 0 ; k < _vCam.size() ; ++k)
            {
                if(k == _centralIndex) {
                    
                    _flowedLightField[idx][k].x = x;
                    _flowedLightField[idx][k].y = y;
                    
                } else {
                    
                    // TODO: if visible
                    // convention: 0 coordinate is at the center
                    cv::Point2f xL = cv::Point2f((float)x, (float)y);
                    cv::Point2f xR = cv::Point2f(xL.x + flowLtoR[k][idx].x, xL.y + flowLtoR[k][idx].y);
                    
                    _flowedLightField[idx][k] = xR;
                }
            }
        }
    }
}

// no visibility calculation
void LFScene::computeFlowedLFCustomConfig() {
    
    // load optical flow
    const std::string flowLtoRName = _outdir + "/flow%02lu.pfm";
    char flowLtoRNameChar[500];
    const uint imageSize = _camHeight*_camWidth;
    
    std::vector<std::vector<cv::Point2f> > flowLtoR(_vCam.size());
    
    // loading the optical flows
    // for all views except the central one
    
    for(uint viewIndex = 0 ; viewIndex < _nbCameras ; ++viewIndex) {
        
        if(viewIndex == _nbCameras/2) {
            std::cout << "central flow, don't load" << std::endl;
            continue;
        }
        
        std::fill_n(flowLtoRNameChar, 500, 0.0);
        sprintf( flowLtoRNameChar, flowLtoRName.c_str(), viewIndex );
        std::cout << "Loading flow " << flowLtoRNameChar << std::endl;
        std::vector<cv::Point2f> currentFlowLtoR(imageSize);
        loadPFM(currentFlowLtoR, _camWidth, _camHeight, std::string(flowLtoRNameChar));
        flowLtoR[viewIndex] = currentFlowLtoR;
    }
    
    // number of samples for each flow is nbCameras - centralView
    // we don't remove any input view
    
    // computing the LF flow
    for(uint y = 0 ; y < _camHeight ; ++y)
    {
        for(uint x = 0 ; x < _camWidth ; ++x)
        {
            const uint idx1 = y*_camWidth + x;
            
            _flowedLightField[idx1].resize(_vCam.size());
            
            // TODO: if visible
            // convention: 0 coordinate is at the center
            cv::Point2f x2 = cv::Point2f(-1.0, -1.0); // invalid value
            for(uint k = 0 ; k < _vCam.size() ; ++k) {
                _flowedLightField[idx1][k] = x2;
            }
            int idx2 = 0;
            
            x2 = cv::Point2f((float)x + flowLtoR[6][idx1].x, (float)y + flowLtoR[6][idx1].y);
            _flowedLightField[idx1][6] = x2;
            if(0 <= x2.x && x2.x < _camWidth &&
                    0 <= x2.y && x2.y < _camHeight) {
                idx2 = (int)(x2.y)*_camWidth + (int)(x2.x);
                _flowedLightField[idx1][0] = cv::Point2f(x2.x + flowLtoR[0][idx2].x, x2.y + flowLtoR[0][idx2].y);
                _flowedLightField[idx1][1] = cv::Point2f(x2.x + flowLtoR[1][idx2].x, x2.y + flowLtoR[1][idx2].y);
                _flowedLightField[idx1][5] = cv::Point2f(x2.x + flowLtoR[5][idx2].x, x2.y + flowLtoR[5][idx2].y);
            }
            
            x2 = cv::Point2f((float)x + flowLtoR[8][idx1].x, (float)y + flowLtoR[8][idx1].y);
            _flowedLightField[idx1][8] = x2;
            if(0 <= x2.x && x2.x < _camWidth &&
                    0 <= x2.y && x2.y < _camHeight) {
                idx2 = (int)(x2.y)*_camWidth + (int)(x2.x);
                _flowedLightField[idx1][3] = cv::Point2f(x2.x + flowLtoR[3][idx2].x, x2.y + flowLtoR[3][idx2].y);
                _flowedLightField[idx1][4] = cv::Point2f(x2.x + flowLtoR[4][idx2].x, x2.y + flowLtoR[4][idx2].y);
                _flowedLightField[idx1][9] = cv::Point2f(x2.x + flowLtoR[9][idx2].x, x2.y + flowLtoR[9][idx2].y);
            }
            
            x2 = cv::Point2f((float)x + flowLtoR[16][idx1].x, (float)y + flowLtoR[16][idx1].y);
            _flowedLightField[idx1][16] = x2;
            if(0 <= x2.x && x2.x < _camWidth &&
                    0 <= x2.y && x2.y < _camHeight) {
                idx2 = (int)(x2.y)*_camWidth + (int)(x2.x);
                _flowedLightField[idx1][15] = cv::Point2f(x2.x + flowLtoR[15][idx2].x, x2.y + flowLtoR[15][idx2].y);
                _flowedLightField[idx1][20] = cv::Point2f(x2.x + flowLtoR[20][idx2].x, x2.y + flowLtoR[20][idx2].y);
                _flowedLightField[idx1][21] = cv::Point2f(x2.x + flowLtoR[21][idx2].x, x2.y + flowLtoR[21][idx2].y);
            }
            
            x2 = cv::Point2f((float)x + flowLtoR[18][idx1].x, (float)y + flowLtoR[18][idx1].y);
            _flowedLightField[idx1][18] = x2;
            if(0 <= x2.x && x2.x < _camWidth &&
                    0 <= x2.y && x2.y < _camHeight) {
                idx2 = (int)(x2.y)*_camWidth + (int)(x2.x);
                _flowedLightField[idx1][23] = cv::Point2f(x2.x + flowLtoR[23][idx2].x, x2.y + flowLtoR[23][idx2].y);
                _flowedLightField[idx1][24] = cv::Point2f(x2.x + flowLtoR[24][idx2].x, x2.y + flowLtoR[24][idx2].y);
                _flowedLightField[idx1][19] = cv::Point2f(x2.x + flowLtoR[19][idx2].x, x2.y + flowLtoR[19][idx2].y);
            }
            
            x2 = cv::Point2f((float)x + flowLtoR[7][idx1].x, (float)y + flowLtoR[7][idx1].y);
            _flowedLightField[idx1][7] = x2;
            if(0 <= x2.x && x2.x < _camWidth &&
                    0 <= x2.y && x2.y < _camHeight) {
                idx2 = (int)(x2.y)*_camWidth + (int)(x2.x);
                _flowedLightField[idx1][2] = cv::Point2f(x2.x + flowLtoR[2][idx2].x, x2.y + flowLtoR[2][idx2].y);
            }
            
            x2 = cv::Point2f((float)x + flowLtoR[13][idx1].x, (float)y + flowLtoR[13][idx1].y);
            _flowedLightField[idx1][13] = x2;
            if(0 <= x2.x && x2.x < _camWidth &&
                    0 <= x2.y && x2.y < _camHeight) {
                idx2 = (int)(x2.y)*_camWidth + (int)(x2.x);
                _flowedLightField[idx1][14] = cv::Point2f(x2.x + flowLtoR[14][idx2].x, x2.y + flowLtoR[14][idx2].y);
            }
            
            x2 = cv::Point2f((float)x + flowLtoR[11][idx1].x, (float)y + flowLtoR[11][idx1].y);
            _flowedLightField[idx1][11] = x2;
            if(0 <= x2.x && x2.x < _camWidth &&
                    0 <= x2.y && x2.y < _camHeight) {
                idx2 = (int)(x2.y)*_camWidth + (int)(x2.x);
                _flowedLightField[idx1][10] = cv::Point2f(x2.x + flowLtoR[10][idx2].x, x2.y + flowLtoR[10][idx2].y);
            }
            
            x2 = cv::Point2f((float)x + flowLtoR[17][idx1].x, (float)y + flowLtoR[17][idx1].y);
            _flowedLightField[idx1][17] = x2;
            if(0 <= x2.x && x2.x < _camWidth &&
                    0 <= x2.y && x2.y < _camHeight) {
                idx2 = (int)(x2.y)*_camWidth + (int)(x2.x);
                _flowedLightField[idx1][22] = cv::Point2f(x2.x + flowLtoR[22][idx2].x, x2.y + flowLtoR[22][idx2].y);
            }
        }
    }
}

// to test the triangulated 3D point
// print some info
void LFScene::testTriangulation(uint x, uint y) {
    
    // target cam
    
    cv::Mat targetR, targetR_transp, targetK, targetK_inv;
    cv::Point3f targetC, targett;
    
    PinholeCamera targetCam = _vCam[_centralIndex]->getPinholeCamera();
    glm::mat3 glmTargetK = targetCam._K;
    glm::mat3 glmTargetR = targetCam._R;
    glm::vec3 glmTargetC = targetCam._C;
    glm::vec3 glmTargett = targetCam._t;
    targetK = (cv::Mat_<float>(3,3) << glmTargetK[0][0], glmTargetK[1][0], glmTargetK[2][0],
            glmTargetK[0][1], glmTargetK[1][1], glmTargetK[2][1],
            glmTargetK[0][2], glmTargetK[1][2], glmTargetK[2][2]);
    targetK_inv = targetK.inv();
    targetR = (cv::Mat_<float>(3,3) << glmTargetR[0][0], glmTargetR[1][0], glmTargetR[2][0],
            glmTargetR[0][1], glmTargetR[1][1], glmTargetR[2][1],
            glmTargetR[0][2], glmTargetR[1][2], glmTargetR[2][2]);
    targetR_transp = (cv::Mat_<float>(3,3) << glmTargetR[0][0], glmTargetR[0][1], glmTargetR[0][2],
            glmTargetR[1][0], glmTargetR[1][1], glmTargetR[1][2],
            glmTargetR[2][0], glmTargetR[2][1], glmTargetR[2][2]);
    targetC = cv::Point3f((float)glmTargetC[0], (float)glmTargetC[1], (float)glmTargetC[2]);
    targett = cv::Point3f((float)glmTargett[0], (float)glmTargett[1], (float)glmTargett[2]);
    
    const uint idx = y*_camWidth + x;
    std::vector<cv::Point2f> flow = _flowedLightField[idx];
    
    const int nbSamples = flow.size();
    
    std::vector<cv::Mat> R(nbSamples), R_transp(nbSamples), K(nbSamples), K_inv(nbSamples);
    std::vector<cv::Point3f> C(nbSamples), t(nbSamples);
    
    for(uint camIdx = 0 ; camIdx < _vCam.size() ; ++camIdx) {
        
        PinholeCamera v_k = _vCam[camIdx]->getPinholeCamera();
        glm::mat3 glmK = v_k._K;
        glm::mat3 glmR = v_k._R;
        glm::vec3 glmC = v_k._C;
        glm::vec3 glmt = v_k._t;
        K[camIdx] = (cv::Mat_<float>(3,3) << glmK[0][0], glmK[1][0], glmK[2][0],
                glmK[0][1], glmK[1][1], glmK[2][1],
                glmK[0][2], glmK[1][2], glmK[2][2]);
        K_inv[camIdx] = K[camIdx].inv();
        R[camIdx] = (cv::Mat_<float>(3,3) << glmR[0][0], glmR[1][0], glmR[2][0],
                glmR[0][1], glmR[1][1], glmR[2][1],
                glmR[0][2], glmR[1][2], glmR[2][2]);
        R_transp[camIdx] = (cv::Mat_<float>(3,3) << glmR[0][0], glmR[0][1], glmR[0][2],
                glmR[1][0], glmR[1][1], glmR[1][2],
                glmR[2][0], glmR[2][1], glmR[2][2]);
        C[camIdx] = cv::Point3f((float)glmC[0], (float)glmC[1], (float)glmC[2]);
        t[camIdx] = cv::Point3f((float)glmt[0], (float)glmt[1], (float)glmt[2]);
    }
    
    std::vector<float> parameters(3);
    float finalCostLF(0.0);
    float conditionNumber(0.0);
    triangulationLF(nbSamples, flow, K_inv, R_transp, C, parameters, conditionNumber, finalCostLF);
    
    // compute 3D point
    std::vector<float> point3DLF(3);
    point3DLF[0] = parameters[1]/(1-parameters[0]);
    point3DLF[1] = parameters[2]/(1-parameters[0]);
    point3DLF[2] = 1.0/(1-parameters[0]);
    
    // render point
    cv::Mat point3D = (cv::Mat_<float>(3, 1) << point3DLF[0], point3DLF[1], point3DLF[2]);
    cv::Mat point2D = targetK*(targetR*point3D + cv::Mat(targett));
    
    // PRINT
    // --------------------------------------------------------------------------------------------
    
    // target cam
    std::cout << "targetK: " << targetK << std::endl;
    std::cout << "targetR: " << targetR << std::endl;
    std::cout << "targett: " << targett << std::endl;
    std::cout << "targetC: " << targetC << std::endl << std::endl;
    
    // input cams
    for(int k = 0 ; k < nbSamples ; ++k) {
        
        std::cout << "Sample: " << k << std::endl;
        std::cout << "K: " << K[k] << std::endl;
        std::cout << "R: " << R[k] << std::endl;
        std::cout << "t: " << t[k] << std::endl;
        std::cout << "C: " << C[k] << std::endl;
        std::cout << "K_inv: " << K_inv[k] << std::endl;
        std::cout << "R_transp: " << R_transp[k] << std::endl << std::endl;
    }
    
    std::cout << "point3D: " << point3D << std::endl << std::endl;
    
    std::cout << x << ", " << y << ": (" << (float)point2D.at<float>(0, 0)/(float)point2D.at<float>(2, 0) << ", "
              << (float)point2D.at<float>(1, 0)/(float)point2D.at<float>(2, 0) << ", "
              << (float)point2D.at<float>(2, 0) << ")" << std::endl << std::endl;
    
    assert(false);
}

//    std::vector<float> parameters3(3); // (a, bu, bv)
//    float finalCostLF(0.0);
//    float conditionNumber(0.0);
//    triangulationLF(flow.size(), flow, K_inv, R_transp, C, parameters3, finalCostLF, conditionNumber, verbose);

//    // LF TRIANGULATION VALIDATION
//    // comparing to classic triangulation
//    std::vector<float> point3DLF(3), point3DClassic(3);

//    triangulationClassic(flow.size(), flow, K, R, t, point3DClassic, finalCostClassic, verbose);

//    // compute 3D point for comparison with classic triangulation (reprojection error)
//    point3DLF[0] = parameters3[1]/(1 - parameters3[0]);
//    point3DLF[1] = parameters3[2]/(1 - parameters3[0]);
//    point3DLF[2] = 1.0/(1 - parameters3[0]);
//    cv::Mat point3D = (cv::Mat_<float>(3, 1) << point3DLF[0], point3DLF[1], point3DLF[2]);

//    if(verbose) {
//        std::cout << "3D point with LF method: " << point3D << std::endl;
//    }

//    // compute reprojection error
//    reprojectionCompute(point3DLF, flow.size(), flow, K, R, t, reprojErrorLF);

//    reprojectionCompute(point3DClassic, flow.size(), flow, K, R, t, reprojErrorClassic);

//    // render 3D point (optional)
//    points.push_back(point3DLF); // to save 3D point cloud
//    cv::Mat point2D = targetK*(targetR*point3D + cv::Mat(targett));
//    targetDepth = (double)point2D.at<float>(2, 0);

//    // save target depth map
//    std::string targetDepthMapName = _outdir + "/targetDepthMap.pfm";
//    std::cout << "Writing target depth map " << targetDepthMapName << std::endl;
//    savePFM(targetDepthMap, _camWidth, _camHeight, targetDepthMapName);

//    // save reprojection error maps (LF and classic)
//    std::string reprojErrorLFMapName = _outdir + "/reprojErrorLFMap.pfm";
//    std::cout << "Writing reprojection error map (curve fitting method) " << reprojErrorLFMapName << std::endl;
//    savePFM(reprojErrorLFMap, _camWidth, _camHeight, reprojErrorLFMapName);

//    std::string reprojErrorClassicMapName = _outdir + "/reprojErrorClassicMap.pfm";
//    std::cout << "Writing reprojection error map (classic triangulation method) " << reprojErrorClassicMapName << std::endl;
//    savePFM(reprojErrorClassicMap, _camWidth, _camHeight, reprojErrorClassicMapName);

//    std::string finalCostClassicMapName = _outdir + "/finalCostClassicMap.pfm";
//    std::cout << "Writing parameter map (classic triangulation method) " << finalCostClassicMapName << std::endl;
//    savePFM(finalCostClassicMap, _camWidth, _camHeight, finalCostClassicMapName);

//    // save point cloud
//    std::string plyName = _outdir + "/point_cloud.ply";
//    std::cout << "Writing ply file in " << plyName << std::endl;
//    writePly(plyName.c_str(), points);

void LFScene::curveFitting() {

    const bool verbose = false;
    const uint nbPixels = _camWidth*_camHeight;

    assert(_windowWidth > 0 && _windowHeight > 0);

    // INIT MAPS

    std::vector<cv::Point3f> parameter3Map(nbPixels);
    std::vector<cv::Point2f> parameterAlpha2Map(nbPixels);
    std::vector<cv::Point2f> parameterBeta2Map(nbPixels);
    std::vector<cv::Point2f > parameter6AlphauMap(nbPixels);
    std::vector<cv::Point2f > parameter6AlphavMap(nbPixels);
    std::vector<cv::Point2f> parameter6BetaMap(nbPixels);
    std::vector<float> finalCost3Map(nbPixels);
    std::vector<float> finalCost4Map(nbPixels);
    std::vector<float> finalCost6Map(nbPixels);
    std::vector<float> conditionNumber3Map(nbPixels);
    std::vector<float> conditionNumber4Map(nbPixels);
    std::vector<float> conditionNumber6Map(nbPixels);

    for(uint y = 0 ; y < (uint)_camHeight ; ++y) {

        for(uint x = 0 ; x < (uint)_camWidth ; ++x) {

            const uint idx = y*_camWidth + x;

            parameter3Map[idx].x = 0.0; parameter3Map[idx].y = 0.0; parameter3Map[idx].z = 0.0;
            parameterAlpha2Map[idx].x = 0.0; parameterAlpha2Map[idx].y = 0.0;
            parameterBeta2Map[idx].x = 0.0; parameterBeta2Map[idx].y = 0.0;
            parameter6AlphauMap[idx].x = 0.0; parameter6AlphauMap[idx].y = 0.0;
            parameter6AlphavMap[idx].x = 0.0; parameter6AlphavMap[idx].y = 0.0;
            parameter6BetaMap[idx].x = 0.0; parameter6BetaMap[idx].y = 0.0;
            finalCost3Map[idx] = 0.0;
            finalCost4Map[idx] = 0.0;
            finalCost6Map[idx] = 0.0;
            conditionNumber3Map[idx] = 0.0;
            conditionNumber4Map[idx] = 0.0;
            conditionNumber6Map[idx] = 0.0;
        }
    }

    Uint32 beginingLoop(0), endLoop(0), elapsedTime(0);
    beginingLoop = SDL_GetTicks();

    std::cout << "[";

    for(uint y = _windowH1 ; y < _windowH2 ; ++y) {

        for(uint x = _windowW1 ; x < _windowW2 ; ++x) {

            if(verbose) {
                if(x != 8*_camWidth/16 || y != 21*_camHeight/32) {
                    continue;
                }
            }

            const uint idx = y*_camWidth + x;
            std::vector<cv::Point2f> flow;
            std::vector<cv::Mat> R, R_transp, K, K_inv;
            std::vector<cv::Point3f> C, t;

            for(int camIdx = 0 ; camIdx < (int)_vCam.size() ; ++camIdx) {

                if(camIdx == _renderIndex) { // remove target view (custom)

                    continue;
                }

                if(_flowedLightField[idx][camIdx].x >= 0 && _flowedLightField[idx][camIdx].y >= 0) {

                    flow.push_back(_flowedLightField[idx][camIdx]);

                    PinholeCamera v_k = _vCam[camIdx]->getPinholeCamera();
                    glm::mat3 glmK = v_k._K;
                    glm::mat3 glmR = v_k._R;
                    glm::vec3 glmC = v_k._C;
                    glm::vec3 glmt = v_k._t;
                    K.push_back((cv::Mat_<float>(3,3) << glmK[0][0], glmK[1][0], glmK[2][0],
                            glmK[0][1], glmK[1][1], glmK[2][1],
                            glmK[0][2], glmK[1][2], glmK[2][2]));
                    K_inv.push_back(K.back().inv());
                    R.push_back((cv::Mat_<float>(3,3) << glmR[0][0], glmR[1][0], glmR[2][0],
                            glmR[0][1], glmR[1][1], glmR[2][1],
                            glmR[0][2], glmR[1][2], glmR[2][2]));
                    R_transp.push_back((cv::Mat_<float>(3,3) << glmR[0][0], glmR[0][1], glmR[0][2],
                            glmR[1][0], glmR[1][1], glmR[1][2],
                            glmR[2][0], glmR[2][1], glmR[2][2]));
                    C.push_back(cv::Point3f((float)glmC[0], (float)glmC[1], (float)glmC[2]));
                    t.push_back(cv::Point3f((float)glmt[0], (float)glmt[1], (float)glmt[2]));
                }
            }

            const uint nbSamples = flow.size();

            //            assert(nbSamples >= 8);
            assert(R.size() == R_transp.size()); // nb samples
            assert(R_transp.size() == K.size());
            assert(K.size() == K_inv.size());
            assert(K_inv.size() == C.size());
            assert(C.size() == t.size());
            assert(t.size() == nbSamples);

            //                if(x == 3363 && y == 15) {
            //                    testTriangulation(x, y);
            //                }

            std::vector<float> parameters3(3); // (a, bu, bv)
            std::vector<float> parameters4(4); // (au, av, bu, bv)
            std::vector<float> parameters6(6); // (aus, aut, avs, avt, bu, bv)
            float finalCostLF(0.0);
            float conditionNumber(0.0);

            if(nbSamples >= 2) {

                triangulationLF(nbSamples, flow, K_inv, R_transp, C, parameters3, finalCostLF, conditionNumber, verbose);
                finalCost3Map[idx] = (float)finalCostLF;
                conditionNumber3Map[idx] = (float)conditionNumber;
                triangulationLF(nbSamples, flow, K_inv, R_transp, C, parameters4, finalCostLF, conditionNumber, verbose);
                finalCost4Map[idx] = (float)finalCostLF;
                conditionNumber4Map[idx] = (float)conditionNumber;
                triangulationLF(nbSamples, flow, K_inv, R_transp, C, parameters6, finalCostLF, conditionNumber, verbose);
                finalCost6Map[idx] = (float)finalCostLF;
                conditionNumber6Map[idx] = (float)conditionNumber;
            }

            parameter3Map[idx].x = (float)parameters3[0]; // a
            parameter3Map[idx].y = (float)parameters3[1]; // bu
            parameter3Map[idx].z = (float)parameters3[2]; // bv

            parameterAlpha2Map[idx].x = (float)parameters4[0]; // au
            parameterAlpha2Map[idx].y = (float)parameters4[1]; // av
            parameterBeta2Map[idx].x = (float)parameters4[2]; // bu
            parameterBeta2Map[idx].y = (float)parameters4[3]; // bv

            parameter6AlphauMap[idx].x = (float)parameters6[0]; // aus
            parameter6AlphauMap[idx].y = (float)parameters6[1]; // aut
            parameter6AlphavMap[idx].x = (float)parameters6[2]; // avs
            parameter6AlphavMap[idx].y = (float)parameters6[3]; // avt
            parameter6BetaMap[idx].x = (float)parameters6[4]; // bu
            parameter6BetaMap[idx].y = (float)parameters6[5]; // bv
        }
    }

    std::cout << "]";

    endLoop = SDL_GetTicks();
    elapsedTime = endLoop - beginingLoop;
    std::cout << "elapsedTime: " << elapsedTime << std::endl;

    if(_renderIndex >= 0) {

        save3fMap(parameter3Map, _outdir + "/model_3g_IHM_%02lu.pfm", _renderIndex);
        save2fMap(parameterAlpha2Map, _outdir + "/model_4g_IHM_%02lu_a.pfm", _renderIndex);
        save2fMap(parameterBeta2Map, _outdir + "/model_4g_IHM_%02lu_b.pfm", _renderIndex);
        save2fMap(parameter6AlphauMap, _outdir + "/model_6g_IHM_%02lu_au.pfm", _renderIndex);
        save2fMap(parameter6AlphavMap, _outdir + "/model_6g_IHM_%02lu_av.pfm", _renderIndex);
        save2fMap(parameter6BetaMap, _outdir + "/model_6g_IHM_%02lu_b.pfm", _renderIndex);

        // SAVE FINAL COST MAPS
        save1fMap(finalCost3Map, _outdir + "/finalCost_3g_IHM_%02lu.pfm", _renderIndex);
        save1fMap(finalCost4Map, _outdir + "/finalCost_4g_IHM_%02lu.pfm", _renderIndex);
        save1fMap(finalCost6Map, _outdir + "/finalCost_6g_IHM_%02lu.pfm", _renderIndex);

        // SAVE CONDITION NUMBER MAPS
        save1fMap(conditionNumber3Map, _outdir + "/conditionNumber_3g_IHM_%02lu.pfm", _renderIndex);
        save1fMap(conditionNumber4Map, _outdir + "/conditionNumber_4g_IHM_%02lu.pfm", _renderIndex);
        save1fMap(conditionNumber6Map, _outdir + "/conditionNumber_6g_IHM_%02lu.pfm", _renderIndex);

    } else {

        save3fMap(parameter3Map, _outdir + "/model_3g_IHM_allViews.pfm", _renderIndex);
        save2fMap(parameterAlpha2Map, _outdir + "/model_4g_IHM_allViews_a.pfm", _renderIndex);
        save2fMap(parameterBeta2Map, _outdir + "/model_4g_IHM_allViews_b.pfm", _renderIndex);
        save2fMap(parameter6AlphauMap, _outdir + "/model_6g_IHM_allViews_au.pfm", _renderIndex);
        save2fMap(parameter6AlphavMap, _outdir + "/model_6g_IHM_allViews_av.pfm", _renderIndex);
        save2fMap(parameter6BetaMap, _outdir + "/model_6g_IHM_allViews_b.pfm", _renderIndex);

        // SAVE FINAL COST MAPS
        save1fMap(finalCost3Map, _outdir + "/finalCost_3g_IHM_allViews.pfm", _renderIndex);
        save1fMap(finalCost4Map, _outdir + "/finalCost_4g_IHM_allViews.pfm", _renderIndex);
        save1fMap(finalCost6Map, _outdir + "/finalCost_6g_IHM_allViews.pfm", _renderIndex);

        // SAVE CONDITION NUMBER MAPS
        save1fMap(conditionNumber3Map, _outdir + "/conditionNumber_3g_IHM_allViews.pfm", _renderIndex);
        save1fMap(conditionNumber4Map, _outdir + "/conditionNumber_4g_IHM_allViews.pfm", _renderIndex);
        save1fMap(conditionNumber6Map, _outdir + "/conditionNumber_6g_IHM_allViews.pfm", _renderIndex);
    }
}

void LFScene::curveFittingColor() {

    const uint nbPixels = _camWidth*_camHeight;

    std::vector<cv::Point3f> parameterS9pMap(nbPixels);
    std::vector<cv::Point3f> parameterT9pMap(nbPixels);
    std::vector<cv::Point3f> parameter09pMap(nbPixels);
    std::vector<float> finalCost9pMap(nbPixels);
    const bool verbose = false;

    // C = [rs, rt ; gs, gt ; bs, bt]
    // I0 = [r0, g0, b0]
    // I = C*[s, t] + I0
    // 9 parameter model

    for(uint y = 0 ; y < (uint)_camHeight ; ++y) {

        for(uint x = 0 ; x < (uint)_camWidth ; ++x) {

            const uint idx = y*_camWidth + x;

            parameterS9pMap[idx] = cv::Point3f(0.0f, 0.0f, 0.0f); // [rs, gs, bs]
            parameterT9pMap[idx] = cv::Point3f(0.0f, 0.0f, 0.0f); // [rt, gt, bt]
            parameter09pMap[idx] = cv::Point3f(0.0f, 0.0f, 0.0f); // [r0, g0, b0]
            finalCost9pMap[idx] = 0.0;
        }
    }

    std::cout << "Optimize color" << std::endl;

    for(uint y = _windowH1 ; y < _windowH2 ; ++y) {

        for(uint x = _windowW1 ; x < _windowW2 ; ++x) {

            if(verbose) {
                if(x != 8*_camWidth/16 || y != 21*_camHeight/32) {
                    continue;
                }
            }

            const uint idx = y*_camWidth + x;
            std::vector<cv::Point2f> flow;
            std::vector<cv::Point3f> colorSampleSet;
            std::vector<cv::Mat> R, R_transp, K, K_inv;
            std::vector<cv::Point3f> C, t;

            // fill the flow, the color sample set and the camera parameters of the set
            for(int k = 0 ; k < (int)_vCam.size() ; ++k) {

                if(k == _renderIndex) { // remove target view (custom)

                    continue;
                }

                if(_flowedLightField[idx][k].x >= 0 && _flowedLightField[idx][k].y >= 0 &&
                        _flowedLightField[idx][k].x < _camWidth && _flowedLightField[idx][k].y < _camHeight) {

                    cv::Point3f color(0.0f, 0.0f, 0.0f);
                    bilinearInterpolation(_camWidth, _camHeight, _vCam[k]->getTextureRGB(), color, _flowedLightField[idx][k]);
                    colorSampleSet.push_back(color);

                    flow.push_back(_flowedLightField[idx][k]);

                    PinholeCamera v_k = _vCam[k]->getPinholeCamera();
                    glm::mat3 glmK = v_k._K;
                    glm::mat3 glmR = v_k._R;
                    glm::vec3 glmC = v_k._C;
                    glm::vec3 glmt = v_k._t;
                    K.push_back((cv::Mat_<float>(3,3) << glmK[0][0], glmK[1][0], glmK[2][0],
                            glmK[0][1], glmK[1][1], glmK[2][1],
                            glmK[0][2], glmK[1][2], glmK[2][2]));
                    K_inv.push_back(K.back().inv());
                    R.push_back((cv::Mat_<float>(3,3) << glmR[0][0], glmR[1][0], glmR[2][0],
                            glmR[0][1], glmR[1][1], glmR[2][1],
                            glmR[0][2], glmR[1][2], glmR[2][2]));
                    R_transp.push_back((cv::Mat_<float>(3,3) << glmR[0][0], glmR[0][1], glmR[0][2],
                            glmR[1][0], glmR[1][1], glmR[1][2],
                            glmR[2][0], glmR[2][1], glmR[2][2]));
                    C.push_back(cv::Point3f((float)glmC[0], (float)glmC[1], (float)glmC[2]));
                    t.push_back(cv::Point3f((float)glmt[0], (float)glmt[1], (float)glmt[2]));
                }
            }

            std::vector<float> parameter(9); // [rs, rt, r0 ; gs, gt, g0 ; bs, bt, b0]
            float finalCost9p(0.0);

            colorRegression(flow.size(), flow, colorSampleSet, K_inv, R_transp, C, parameter, finalCost9p, verbose);
            finalCost9pMap[idx] = (float)finalCost9p;

            parameterS9pMap[idx].x = (float)parameter[0]; // rs
            parameterS9pMap[idx].y = (float)parameter[3]; // gs
            parameterS9pMap[idx].z = (float)parameter[6]; // bs
            parameterT9pMap[idx].x = (float)parameter[1]; // rt
            parameterT9pMap[idx].y = (float)parameter[4]; // gt
            parameterT9pMap[idx].z = (float)parameter[7]; // bt
            parameter09pMap[idx].x = (float)parameter[2]; // r0
            parameter09pMap[idx].y = (float)parameter[5]; // g0
            parameter09pMap[idx].z = (float)parameter[8]; // b0
        }
    }

    // SAVE PARAMETER MAPS (LINEAR METHOD ONLY) AND FINAL COST MAPS
    if(_renderIndex >= 0) {

        save3fMap(parameterS9pMap, _outdir + "/model_9p_LIN_%02lu.pfm", _renderIndex);
        save3fMap(parameterT9pMap, _outdir + "/model_9p_LIN_%02lu.pfm", _renderIndex);
        save3fMap(parameter09pMap, _outdir + "/model_9p_LIN_%02lu.pfm", _renderIndex);

        save1fMap(finalCost9pMap, _outdir + "/finalCost_9p_LIN_%02lu.pfm", _renderIndex);

    } else {

        save3fMap(parameterS9pMap, _outdir + "/model_9p_LIN_allViews.pfm", _renderIndex);
        save3fMap(parameterT9pMap, _outdir + "/model_9p_LIN_allViews.pfm", _renderIndex);
        save3fMap(parameter09pMap, _outdir + "/model_9p_LIN_allViews.pfm", _renderIndex);

        save1fMap(finalCost9pMap, _outdir + "/finalCost_9p_LIN_allViews.pfm", _renderIndex);
    }
}

// backward warping of one view, accumulate in outputImage and weight
// convention: 0 coordinate is at the center
void backwardWarping( int width, int height,
                      const std::vector<cv::Point3f>& inputImage,
                      std::vector<cv::Point3f>& outputImage,
                      float& weight,
                      const uint ox, const uint oy,
                      const cv::Point2f& dest,
                      int sampling ) {
    
    const uint o = ox + oy*width;
    
    // get location in input image
    int cx = (int)floor(dest.x);
    int cy = (int)floor(dest.y);
    int co = cx + cy*width;
    const float dx = dest.x - float(cx); // distance to the center
    const float dy = dest.y - float(cy);
    
    switch(sampling) {
    
    case 0: // nearest
        
        cx = (int)floor(dest.x + 0.5);
        cy = (int)floor(dest.y + 0.5);
        co = cx + cy*width;
        
        if(0 <= cx && cx < width && 0 <= cy && cy < height) {
            
            outputImage[o] += inputImage[co];
            weight += 1.0f;
        }
        
        break;
        
    case 1: // bilinear
    {
        // transpose bilinear sampling
        float mxmym = (1.0f - dx) * (1.0f - dy);
        float mxpym = dx * (1.0f - dy);
        float mxmyp = (1.0f - dx) * dy;
        float mxpyp = dx * dy;
        
        if(0 <= cx && cx < width &&
                0 <= cy && cy < height) {
            outputImage[o] += inputImage[co + 0] * mxmym ;
            weight += mxmym;
        }
        if(0 <= cx && cx < width - 1 &&
                0 <= cy && cy < height) {
            outputImage[o] += inputImage[co + 1] * mxpym ;
            weight += mxpym;
        }
        if(0 <= cx && cx < width &&
                0 <= cy && cy < height - 1) {
            outputImage[o] += inputImage[co + width] * mxmyp ;
            weight += mxmyp;
        }
        if(0 <= cx && cx < width - 1 &&
                0 <= cy && cy < height - 1) {
            outputImage[o] += inputImage[co + width + 1] * mxpyp ;
            weight += mxpyp;
        }
        
        break;
    }
        
    case 2: // bicubic
    {
        const int px = cx - 1;
        const int nx = cx + 1;
        const int ax = cx + 2;
        const int py = cy - 1;
        const int ny = cy + 1;
        const int ay = cy + 2;
        
        // Dirichlet boundary conditions: value = 0
        
        const cv::Point3f Ipp = (px < 0 || py < 0 || px >= width || py >= height) ? cv::Point3f(0.0, 0.0, 0.0) : inputImage[px + py*width];
        const cv::Point3f Icp = (cx < 0 || py < 0 || cx >= width || py >= height) ? cv::Point3f(0.0, 0.0, 0.0) : inputImage[cx + py*width];
        const cv::Point3f Inp = (nx < 0 || py < 0 || nx >= width || py >= height) ? cv::Point3f(0.0, 0.0, 0.0) : inputImage[nx + py*width];
        const cv::Point3f Iap = (ax < 0 || py < 0 || ax >= width || py >= height) ? cv::Point3f(0.0, 0.0, 0.0) : inputImage[ax + py*width];
        
        const cv::Point3f Ipc = (px < 0 || cy < 0 || px >= width || cy >= height) ? cv::Point3f(0.0, 0.0, 0.0) : inputImage[px + cy*width];
        const cv::Point3f Icc = (cx < 0 || cy < 0 || cx >= width || cy >= height) ? cv::Point3f(0.0, 0.0, 0.0) : inputImage[cx + cy*width];
        const cv::Point3f Inc = (nx < 0 || cy < 0 || nx >= width || cy >= height) ? cv::Point3f(0.0, 0.0, 0.0) : inputImage[nx + cy*width];
        const cv::Point3f Iac = (ax < 0 || cy < 0 || ax >= width || cy >= height) ? cv::Point3f(0.0, 0.0, 0.0) : inputImage[ax + cy*width];
        
        const cv::Point3f Ipn = (px < 0 || ny < 0 || px >= width || ny >= height) ? cv::Point3f(0.0, 0.0, 0.0) : inputImage[px + ny*width];
        const cv::Point3f Icn = (cx < 0 || ny < 0 || cx >= width || ny >= height) ? cv::Point3f(0.0, 0.0, 0.0) : inputImage[cx + ny*width];
        const cv::Point3f Inn = (nx < 0 || ny < 0 || nx >= width || ny >= height) ? cv::Point3f(0.0, 0.0, 0.0) : inputImage[nx + ny*width];
        const cv::Point3f Ian = (ax < 0 || ny < 0 || ax >= width || ny >= height) ? cv::Point3f(0.0, 0.0, 0.0) : inputImage[ax + ny*width];
        
        const cv::Point3f Ipa = (px < 0 || ay < 0 || px >= width || ay >= height) ? cv::Point3f(0.0, 0.0, 0.0) : inputImage[px + ay*width];
        const cv::Point3f Ica = (cx < 0 || ay < 0 || cx >= width || ay >= height) ? cv::Point3f(0.0, 0.0, 0.0) : inputImage[cx + ay*width];
        const cv::Point3f Ina = (nx < 0 || ay < 0 || nx >= width || ay >= height) ? cv::Point3f(0.0, 0.0, 0.0) : inputImage[nx + ay*width];
        const cv::Point3f Iaa = (ax < 0 || ay < 0 || ax >= width || ay >= height) ? cv::Point3f(0.0, 0.0, 0.0) : inputImage[ax + ay*width];
        
        const cv::Point3f Ip = Icp + 0.5f*(dx*(-Ipp+Inp) + dx*dx*(2*Ipp-5*Icp+4*Inp-Iap) + dx*dx*dx*(-Ipp+3*Icp-3*Inp+Iap));
        const cv::Point3f Ic = Icc + 0.5f*(dx*(-Ipc+Inc) + dx*dx*(2*Ipc-5*Icc+4*Inc-Iac) + dx*dx*dx*(-Ipc+3*Icc-3*Inc+Iac));
        const cv::Point3f In = Icn + 0.5f*(dx*(-Ipn+Inn) + dx*dx*(2*Ipn-5*Icn+4*Inn-Ian) + dx*dx*dx*(-Ipn+3*Icn-3*Inn+Ian));
        const cv::Point3f Ia = Ica + 0.5f*(dx*(-Ipa+Ina) + dx*dx*(2*Ipa-5*Ica+4*Ina-Iaa) + dx*dx*dx*(-Ipa+3*Ica-3*Ina+Iaa));
        
        outputImage[o] += Ic + 0.5f*(dy*(-Ip+In) + dy*dy*(2*Ip-5*Ic+4*In-Ia) + dy*dy*dy*(-Ip+3*Ic-3*In+Ia));
        weight += 1.0;
        break;
    }
    }
}

// forward warping (splatting)
// bilinear contribution (TODO: nearest and bicubic)
// convention: 0 coordinate is at the center
void projectSplat( int width, int height,
                   const cv::Point3f& inputColor,
                   std::vector<cv::Point3f>& outputImage,
                   std::vector<float>& weight, // sum of all contributions to output image
                   const cv::Point2f& dest ) {
    
    // get location in output image
    const int cx = (int)floor(dest.x);
    const int cy = (int)floor(dest.y);
    const int co = cx + cy*width;
    const float dx = dest.x - float(cx); // distance to the center
    const float dy = dest.y - float(cy);
    
    float mxmym = (1.0f - dx) * (1.0f - dy);
    float mxpym = dx * (1.0f - dy);
    float mxmyp = (1.0f - dx) * dy;
    float mxpyp = dx * dy;
    
    // test beta visibility of pixel destination
    if (cx >= 0 && cy >= 0) {
        
        outputImage[co + 0] += inputColor * mxmym;
        weight[co + 0] += mxmym;
    }
    if (cx < width - 1 && cy >= 0) {
        
        outputImage[co + 1] += inputColor * mxpym;
        weight[co + 1] += mxpym;
    }
    if (cy < height - 1 && cx >= 0) {
        
        outputImage[co + width] += inputColor * mxmyp;
        weight[co + width] += mxmyp;
    }
    if (cx < width - 1 && cy < height - 1) {
        
        outputImage[co + width + 1] += inputColor * mxpyp;
        weight[co + width + 1] += mxpyp;
    }
}

void LFScene::loadTargetView(cv::Mat &targetK, cv::Mat &targetR, cv::Point3f &targetC, std::string targetCameraName) {

    std::cout << "Loading view " << targetCameraName << std::endl;

    std::ifstream in( targetCameraName.c_str(), std::ifstream::in );
    assert( in.is_open() );
    assert( in );

    std::string tmp;

    float focal_length = 0.0;
    float pixel_aspect = 0.0;
    float principal_point[] = {0.0, 0.0};
    glm::mat3 R(1.0);
    glm::vec3 t(0.0);
    glm::mat3 K(0.0);
    uint nbMaxWordsHeader = 100;

    uint count = 0; // for safety
    while( strcmp( "[camera]", tmp.c_str() ) && strcmp( "[view]", tmp.c_str() ) && count < nbMaxWordsHeader ) {

        in >> tmp;
        ++count;
    }

    assert( strcmp( "[view]", tmp.c_str() ) && count < nbMaxWordsHeader);

    in >> tmp >> tmp >> focal_length
            >> tmp >> tmp >> pixel_aspect
            >> tmp >> tmp >> principal_point[0] >> principal_point[1]
            >> tmp >> tmp >> R[0][0] >> R[1][0] >> R[2][0] >> R[0][1] >> R[1][1] >> R[2][1] >> R[0][2] >> R[1][2] >> R[2][2]
            >> tmp >> tmp >> t[0] >> t[1] >> t[2];

    in.close();

    assert( pixel_aspect != 0 );

    // focal_length = f1 in pixels divided by larger side
    // pixel_aspect = pixel width divided by pixel height
    // principal_point is also normalized and independent of the image size
    if( _camWidth >= _camHeight ) {
        K[0][0] = _camWidth * focal_length;
    } else {
        K[0][0] = _camHeight * focal_length;
    }
    K[1][1] = K[0][0] / pixel_aspect;
    K[2][2] = 1.0;
    K[2][0] = _camWidth * principal_point[0];
    K[2][1] = _camHeight * principal_point[1];

    glm::vec3 C = -glm::transpose(R) * t;

    targetK = (cv::Mat_<float>(3,3) << K[0][0], K[1][0], K[2][0],
            K[0][1], K[1][1], K[2][1],
            K[0][2], K[1][2], K[2][2]);
    targetR = (cv::Mat_<float>(3,3) << R[0][0], R[1][0], R[2][0],
            R[0][1], R[1][1], R[2][1],
            R[0][2], R[1][2], R[2][2]);
    targetC = cv::Point3f((float)C[0], (float)C[1], (float)C[2]);
}

void LFScene::loadTargetView(cv::Mat &targetK, cv::Mat &targetR, cv::Point3f &targetC) {
    
    char targetCameraNameChar[500];
    memset(targetCameraNameChar, 0, sizeof(targetCameraNameChar));
    sprintf( targetCameraNameChar, _cameraName.c_str(), _mveRdrIdx );
    std::cout << "Loading view " << targetCameraNameChar << std::endl;

    std::ifstream in( targetCameraNameChar, std::ifstream::in );
    assert( in.is_open() );
    assert( in );
    
    std::string tmp;
    
    float focal_length = 0.0;
    float pixel_aspect = 0.0;
    float principal_point[] = {0.0, 0.0};
    glm::mat3 R(1.0);
    glm::vec3 t(0.0);
    glm::mat3 K(0.0);
    uint nbMaxWordsHeader = 100;
    
    uint count = 0; // for safety
    while( strcmp( "[camera]", tmp.c_str() ) && strcmp( "[view]", tmp.c_str() ) && count < nbMaxWordsHeader ) {
        
        in >> tmp;
        ++count;
    }
    
    assert( strcmp( "[view]", tmp.c_str() ) && count < nbMaxWordsHeader);
    
    in >> tmp >> tmp >> focal_length
            >> tmp >> tmp >> pixel_aspect
            >> tmp >> tmp >> principal_point[0] >> principal_point[1]
            >> tmp >> tmp >> R[0][0] >> R[1][0] >> R[2][0] >> R[0][1] >> R[1][1] >> R[2][1] >> R[0][2] >> R[1][2] >> R[2][2]
            >> tmp >> tmp >> t[0] >> t[1] >> t[2];
    
    in.close();
    
    assert( pixel_aspect != 0 );
    
    // focal_length = f1 in pixels divided by larger side
    // pixel_aspect = pixel width divided by pixel height
    // principal_point is also normalized and independent of the image size
    if( _camWidth >= _camHeight ) {
        K[0][0] = _camWidth * focal_length;
    } else {
        K[0][0] = _camHeight * focal_length;
    }
    K[1][1] = K[0][0] / pixel_aspect;
    K[2][2] = 1.0;
    K[2][0] = _camWidth * principal_point[0];
    K[2][1] = _camHeight * principal_point[1];
    
    glm::vec3 C = -glm::transpose(R) * t;
    
    targetK = (cv::Mat_<float>(3,3) << K[0][0], K[1][0], K[2][0],
            K[0][1], K[1][1], K[2][1],
            K[0][2], K[1][2], K[2][2]);
    targetR = (cv::Mat_<float>(3,3) << R[0][0], R[1][0], R[2][0],
            R[0][1], R[1][1], R[2][1],
            R[0][2], R[1][2], R[2][2]);
    targetC = cv::Point3f((float)C[0], (float)C[1], (float)C[2]);
}

// Import target camera parameters (translation) and set fixed focal length (hack)
// Read the camera translation vectors from XML file (TOLF format)
void LFScene::loadTargetTranslation(cv::Mat &targetK, cv::Mat &targetR, cv::Point3f &targetC) {

    char targetCameraNameChar[500];
    memset(targetCameraNameChar, 0, sizeof(targetCameraNameChar));
    sprintf( targetCameraNameChar, _cameraName.c_str(), _mveRdrIdx );
    std::cout << "Loading view " << targetCameraNameChar << std::endl;

    std::ifstream in( targetCameraNameChar, std::ifstream::in );
    assert( in.is_open() );
    assert( in );

    std::string tmp;

    const uint viewIndex = 12; // HACK, TODO: variable for target view number

    glm::mat3 R(1.0);
    glm::vec3 t(0.0);
    glm::mat3 K(0.0);
    uint nbMaxWordsHeader = 100;

    uint count = 0; // for safety
    while( strcmp( "<data>", tmp.c_str() ) && strcmp( "[view]", tmp.c_str() ) && count < nbMaxWordsHeader ) {

        in >> tmp;
        ++count;
    }

    assert( strcmp( "</data>", tmp.c_str() ) && count < nbMaxWordsHeader);

    for(uint i = 0 ; i < viewIndex ; ++i) {

        for(uint j = 0 ; j < 6 ; ++j) {
            in >> tmp;
        }
    }

    in >> tmp >> tmp >> tmp >> t[0] >> t[1] >> t[2];
    in.close();

    K[0][0] = 905.44060602527395;
    K[1][1] = 907.68102844590101;
    K[2][2] = 1.0;
    K[2][0] = 308.29227330174598;
    K[2][1] = 252.10539485267773;

    glm::vec3 C = -glm::transpose(R) * t;

    targetK = (cv::Mat_<float>(3,3) << K[0][0], K[1][0], K[2][0],
            K[0][1], K[1][1], K[2][1],
            K[0][2], K[1][2], K[2][2]);
    targetR = (cv::Mat_<float>(3,3) << R[0][0], R[1][0], R[2][0],
            R[0][1], R[1][1], R[2][1],
            R[0][2], R[1][2], R[2][2]);
    targetC = cv::Point3f((float)C[0], (float)C[1], (float)C[2]);
}

void LFScene::bic() {

    const uint nbPixels = _camWidth*_camHeight;

    std::vector<float> finalCost3Map(nbPixels);
    std::vector<float> finalCost4Map(nbPixels);
    std::vector<float> finalCost6Map(nbPixels);

    if(_renderIndex >= 0) {

        // LOAD FINAL COST MAPS
        load1fMap(finalCost3Map, _outdir + "/finalCost_3g_IHM_%02lu.pfm", _renderIndex);
        load1fMap(finalCost4Map, _outdir + "/finalCost_4g_IHM_%02lu.pfm", _renderIndex);
        load1fMap(finalCost6Map, _outdir + "/finalCost_6g_IHM_%02lu.pfm", _renderIndex);

    } else {

        // LOAD FINAL COST MAPS
        load1fMap(finalCost3Map, _outdir + "/finalCost_3g_IHM_allViews.pfm", _renderIndex);
        load1fMap(finalCost4Map, _outdir + "/finalCost_4g_IHM_allViews.pfm", _renderIndex);
        load1fMap(finalCost6Map, _outdir + "/finalCost_6g_IHM_allViews.pfm", _renderIndex);
    }

    // OUTPUT BIC

    std::vector<float> outputBIC3param(nbPixels);
    std::vector<float> outputBIC4param(nbPixels);
    std::vector<float> outputBIC6param(nbPixels);
    std::vector<float> selectedModel(nbPixels);

    for(uint i = 0 ; i < nbPixels ; ++i) {

        outputBIC3param[i] = finalCost3Map[i] + 3*log(_flowedLightField[i].size());
        outputBIC4param[i] = finalCost4Map[i] + 4*log(_flowedLightField[i].size());
        outputBIC6param[i] = finalCost6Map[i] + 6*log(_flowedLightField[i].size());

        if(outputBIC3param[i] < outputBIC4param[i] && outputBIC3param[i] < outputBIC6param[i]) {

            selectedModel[i] = 0.25;

        } else if(outputBIC4param[i] < outputBIC3param[i] && outputBIC4param[i] < outputBIC6param[i]) {

            selectedModel[i] = 0.50;

        } else {

            selectedModel[i] = 0.75;
        }
    }

    if(_renderIndex >= 0) {

        save1fMap(selectedModel, _outdir + "/selectedModel_g_IHM_%02lu.pfm", _renderIndex);

    } else {

        save1fMap(selectedModel, _outdir + "/selectedModel_g_IHM_allViews.pfm", _renderIndex);
    }
}

void LFScene::renderLightFlow() {
    
    const uint nbPixels = _camWidth*_camHeight;
    
    // TARGET CAM PARAMETERS
    
    cv::Mat targetR, targetK;
    cv::Point3f targetC;
    loadTargetView(targetK, targetR, targetC);

    // LOAD POSITION MODEL PARAMETERS
    // DLT: with DLT, for initialization only (linear method THEN non-linear method)
    
    std::vector<cv::Point3f> map3param(nbPixels);
    std::cout << "Load parameters from " << std::string(_outdir + "/parameter3MapDLT.pfm") << std::endl;
    loadPFM(map3param, _camWidth, _camHeight, std::string(_outdir + "/parameter3MapDLT.pfm"));
    
    std::vector<cv::Point2f> mapAlpha4param(nbPixels);
    std::cout << "Load parameters from " << std::string(_outdir + "/parameterAlpha2MapDLT.pfm") << std::endl;
    loadPFM(mapAlpha4param, _camWidth, _camHeight, std::string(_outdir + "/parameterAlpha2MapDLT.pfm"));
    std::vector<cv::Point2f> mapBeta4param(nbPixels);
    std::cout << "Load parameters from " << std::string(_outdir + "/parameterBeta2MapDLT.pfm") << std::endl;
    loadPFM(mapBeta4param, _camWidth, _camHeight, std::string(_outdir + "/parameterBeta2MapDLT.pfm"));
    
    std::vector<cv::Point2f> mapAlphau6param(nbPixels);
    std::cout << "Load parameters from " << std::string(_outdir + "/parameter6AlphauMapDLT.pfm") << std::endl;
    loadPFM(mapAlphau6param, _camWidth, _camHeight, std::string(_outdir + "/parameter6AlphauMapDLT.pfm"));
    std::vector<cv::Point2f> mapAlphav6param(nbPixels);
    std::cout << "Load parameters from " << std::string(_outdir + "/parameter6AlphavMapDLT.pfm") << std::endl;
    loadPFM(mapAlphav6param, _camWidth, _camHeight, std::string(_outdir + "/parameter6AlphavMapDLT.pfm"));
    std::vector<cv::Point2f> mapBeta6param(nbPixels);
    std::cout << "Load parameters from " << std::string(_outdir + "/parameter6BetaMapDLT.pfm") << std::endl;
    loadPFM(mapBeta6param, _camWidth, _camHeight, std::string(_outdir + "/parameter6BetaMapDLT.pfm"));

    // LOAD COLOR MODEL PARAMETERS

    std::vector<cv::Point3f> parameterSMap(nbPixels);
    std::cout << "Load parameters from " << std::string(_outdir + "/parameterSMap.pfm") << std::endl;
    loadPFM(parameterSMap, _camWidth, _camHeight, std::string(_outdir + "/parameterSMap.pfm"));
    std::vector<cv::Point3f> parameterTMap(nbPixels);
    std::cout << "Load parameters from " << std::string(_outdir + "/parameterTMap.pfm") << std::endl;
    loadPFM(parameterTMap, _camWidth, _camHeight, std::string(_outdir + "/parameterTMap.pfm"));
    std::vector<cv::Point3f> parameter0Map(nbPixels);
    std::cout << "Load parameters from " << std::string(_outdir + "/parameter0Map.pfm") << std::endl;
    loadPFM(parameter0Map, _camWidth, _camHeight, std::string(_outdir + "/parameter0Map.pfm"));

    // OUTPUT IMAGE
    
    std::vector<cv::Point3f> colorMap(nbPixels);
    
    std::vector<cv::Point3f> outputImage3param(nbPixels);
    std::vector<float> reprojError3param(nbPixels); // reprojection error of the central view
    std::vector<float> weightMap3param(nbPixels); // splat contribution (for normalizatiNULLon)
    
    std::vector<cv::Point3f> outputImage4param(nbPixels);
    std::vector<float> reprojError4param(nbPixels); // reprojection error of the central view
    std::vector<float> weightMap4param(nbPixels); // splat contribution (for normalization)
    
    std::vector<cv::Point3f> outputImage6param(nbPixels);
    std::vector<float> reprojError6param(nbPixels); // reprojection error of the central view
    std::vector<float> weightMap6param(nbPixels); // splat contribution (for normalization)
    
    for(uint i = 0 ; i < outputImage3param.size() ; ++i) {
        
        colorMap[i] = cv::Point3f(0.0, 0.0, 0.0);
        
        outputImage3param[i] = cv::Point3f(0.0, 0.0, 0.0);
        reprojError3param[i] = 0.0;
        weightMap3param[i] = 0.0;
        
        outputImage4param[i] = cv::Point3f(0.0, 0.0, 0.0);
        reprojError4param[i] = 0.0;
        weightMap4param[i] = 0.0;
        
        outputImage6param[i] = cv::Point3f(0.0, 0.0, 0.0);
        reprojError6param[i] = 0.0;
        weightMap6param[i] = 0.0;
    }
    
    std::cout << "RENDERING " << std::endl;

    // ------------------------ SPLATTING ------------------------
    
    // TODO: handle visibility (compute z-buffer)
    
    std::cout << "Blending step" << std::endl;
    for(uint y = _windowH1 ; y < _windowH2 ; ++y) {
        for(uint x = _windowW1 ; x < _windowW2 ; ++x) {
            
            const uint idx = y*_camWidth + x;
            
            // find position (splat destination and size/orientation)
            
            // 3 PARAMETERS
            cv::Point2f destPoint3param = cv::Point2f(0.0, 0.0);
            const cv::Point3f parameters = map3param[idx];
            cv::Point3f color(0.0f, 0.0f, 0.0f);

            //            color = colorMap[idx]
            //            splatProjection3param(destPoint3param, parameters, targetK, targetR, targetC);
            splatProjection3param2(destPoint3param, color, parameters, parameterSMap[idx], parameterTMap[idx], parameter0Map[idx], targetK, targetR, targetC);

            // interpolation (splatting)
            if(0.0 <= destPoint3param.x && destPoint3param.x < (float)_camWidth &&
                    0.0 <= destPoint3param.y && destPoint3param.y < (float)_camHeight) {
                
                projectSplat(_camWidth, _camHeight, color, outputImage3param, weightMap3param, destPoint3param);
                reprojError3param[idx] = sqrt((destPoint3param.x - x)*(destPoint3param.x - x) + (destPoint3param.y - y)*(destPoint3param.y - y));
            }
            
            // 4 PARAMETERS
            cv::Point2f destPoint4param = cv::Point2f(0.0, 0.0);
            const cv::Point2f alpha4param = mapAlpha4param[idx];
            const cv::Point2f beta4param = mapBeta4param[idx];
            color = cv::Point3f (0.0f, 0.0f, 0.0f);

            //            color = colorMap[idx]
            //            splatProjection4param(destPoint4param, alpha4param, beta4param, targetK, targetR, targetC);
            splatProjection4param2(destPoint4param, color, alpha4param, beta4param, parameterSMap[idx], parameterTMap[idx], parameter0Map[idx], targetK, targetR, targetC);
            
            // interpolation (splatting)
            if(0.0 <= destPoint4param.x && destPoint4param.x < (float)_camWidth &&
                    0.0 <= destPoint4param.y && destPoint4param.y < (float)_camHeight) {
                
                projectSplat(_camWidth, _camHeight, color, outputImage4param, weightMap4param, destPoint4param);
                reprojError4param[idx] = sqrt((destPoint4param.x - x)*(destPoint4param.x - x) + (destPoint4param.y - y)*(destPoint4param.y - y));
            }
            
            // 6 PARAMETERS
            cv::Point2f destPoint6param = cv::Point2f(0.0, 0.0);
            const cv::Point2f alphau6param = mapAlphau6param[idx];
            const cv::Point2f alphav6param = mapAlphav6param[idx];
            const cv::Point2f beta6param = mapBeta6param[idx];
            color = cv::Point3f (0.0f, 0.0f, 0.0f); // TODO, 3 color maps, one for each model

            //            color = colorMap[idx]
            //            splatProjection6param(destPoint6param, alphau6param, alphav6param, beta6param, targetK, targetR, targetC);
            splatProjection6param2(destPoint6param, color, alphau6param, alphav6param, beta6param, parameterSMap[idx], parameterTMap[idx], parameter0Map[idx], targetK, targetR, targetC);
            
            // interpolation (splatting)
            if(0.0 <= destPoint6param.x && destPoint6param.x < (float)_camWidth &&
                    0.0 <= destPoint6param.y && destPoint6param.y < (float)_camHeight) {
                
                projectSplat(_camWidth, _camHeight, color, outputImage6param, weightMap6param, destPoint6param);
                reprojError6param[idx] = sqrt((destPoint6param.x - x)*(destPoint6param.x - x) + (destPoint6param.y - y)*(destPoint6param.y - y));
            }
        }
    }
    
    std::cout << "Normalization step" << std::endl;
    for(uint y = 0 ; y < _camHeight ; ++y) {
        for(uint x = 0 ; x < _camWidth ; ++x) {
            
            const uint idx = y*_camWidth + x;
            
            if(weightMap3param[idx] != 0) {
                outputImage3param[idx] /= weightMap3param[idx];
            }
            
            if(weightMap4param[idx] != 0) {
                outputImage4param[idx] /= weightMap4param[idx];
            }
            
            if(weightMap6param[idx] != 0) {
                outputImage6param[idx] /= weightMap6param[idx];
            }
        }
    }
    
    std::cout << "Hole filling" << std::endl;
    pushPull(_camWidth, _camHeight, outputImage3param, weightMap3param);
    pushPull(_camWidth, _camHeight, outputImage4param, weightMap4param);
    pushPull(_camWidth, _camHeight, outputImage6param, weightMap6param);
    
    //    // SAVE PFM OUTMUT FILES

    //    std::string colorMapName = _outdir + "/colorMap.pfm";
    //    std::cout << "Writing color map " << colorMapName << std::endl;
    //    savePFM(colorMap, _camWidth, _camHeight, colorMapName);
    
    //    std::string outputImage3paramName = _outdir + "/outputImage3param.pfm";
    //    std::cout << "Writing output image " << outputImage3paramName << std::endl;
    //    savePFM(outputImage3param, _camWidth, _camHeight, outputImage3paramName);
    
    //    std::string weightMap3paramName = _outdir + "/weightMap3param.pfm";
    //    std::cout << "Writing splat weights " << weightMap3paramName << std::endl;
    //    savePFM(weightMap3param, _camWidth, _camHeight, weightMap3paramName);
    
    //    std::string reprojError3paramName = _outdir + "/reprojError3param.pfm";
    //    std::cout << "Writing reprojection error " << reprojError3paramName << std::endl;
    //    savePFM(reprojError3param, _camWidth, _camHeight, reprojError3paramName);
    
    //    std::string outputImage4paramName = _outdir + "/outputImage4param.pfm";
    //    std::cout << "Writing output image " << outputImage4paramName << std::endl;
    //    savePFM(outputImage4param, _camWidth, _camHeight, outputImage4paramName);
    
    //    std::string weightMap4paramName = _outdir + "/weightMap4param.pfm";
    //    std::cout << "Writing splat weights " << weightMap4paramName << std::endl;
    //    savePFM(weightMap4param, _camWidth, _camHeight, weightMap4paramName);
    
    //    std::string reprojError4paramName = _outdir + "/reprojError4param.pfm";
    //    std::cout << "Writing reprojection error " << reprojError4paramName << std::endl;
    //    savePFM(reprojError4param, _camWidth, _camHeight, reprojError4paramName);
    
    //    std::string outputImage6paramName = _outdir + "/outputImage6param.pfm";
    //    std::cout << "Writing output image " << outputImage6paramName << std::endl;
    //    savePFM(outputImage6param, _camWidth, _camHeight, outputImage6paramName);
    
    //    std::string weightMap6paramName = _outdir + "/weightMap6param.pfm";
    //    std::cout << "Writing splat weights " << weightMap6paramName << std::endl;
    //    savePFM(weightMap6param, _camWidth, _camHeight, weightMap6paramName);
    
    //    std::string reprojError6paramName = _outdir + "/reprojError6param.pfm";
    //    std::cout << "Writing reprojection error " << reprojError6paramName << std::endl;
    //    savePFM(reprojError6param, _camWidth, _camHeight, reprojError6paramName);

    // SAVE PNG OUTMUT FILES

    char tempCharArray[500];
    std::string outputImage3paramName = _outdir + "/output3param_%02lu_%02lu.png";
    std::string outputImage4paramName = _outdir + "/output4param_%02lu_%02lu.png";
    std::string outputImage6paramName = _outdir + "/output6param_%02lu_%02lu.png";

    memset(tempCharArray, 0, sizeof(tempCharArray));
    sprintf( tempCharArray, outputImage3paramName.c_str(), _sRmv, _tRmv );
    std::cout << "Save output image " << tempCharArray << std::endl;
    savePNG(outputImage3param, _camWidth, _camHeight, tempCharArray);

    memset(tempCharArray, 0, sizeof(tempCharArray));
    sprintf( tempCharArray, outputImage4paramName.c_str(), _sRmv, _tRmv );
    std::cout << "Save output image " << tempCharArray << std::endl;
    savePNG(outputImage4param, _camWidth, _camHeight, tempCharArray);

    memset(tempCharArray, 0, sizeof(tempCharArray));
    sprintf( tempCharArray, outputImage6paramName.c_str(), _sRmv, _tRmv );
    std::cout << "Save output image " << tempCharArray << std::endl;
    savePNG(outputImage6param, _camWidth, _camHeight, tempCharArray);
}

void LFScene::renderLightFlowLambertianModel() {

    const uint nbPixels = _camWidth*_camHeight;

    std::vector<cv::Point3f> map3param(nbPixels);
    std::cout << "Load parameters from " << std::string(_outdir + "/parameter3MapDLT.pfm") << std::endl;
    loadPFM(map3param, _camWidth, _camHeight, std::string(_outdir + "/parameter3MapDLT.pfm"));

    std::vector<cv::Point2f> mapAlpha4param(nbPixels);
    std::cout << "Load parameters from " << std::string(_outdir + "/parameterAlpha2MapDLT.pfm") << std::endl;
    loadPFM(mapAlpha4param, _camWidth, _camHeight, std::string(_outdir + "/parameterAlpha2MapDLT.pfm"));
    std::vector<cv::Point2f> mapBeta4param(nbPixels);
    std::cout << "Load parameters from " << std::string(_outdir + "/parameterBeta2MapDLT.pfm") << std::endl;
    loadPFM(mapBeta4param, _camWidth, _camHeight, std::string(_outdir + "/parameterBeta2MapDLT.pfm"));

    std::vector<cv::Point2f> mapAlphau6param(nbPixels);
    std::cout << "Load parameters from " << std::string(_outdir + "/parameter6AlphauMapDLT.pfm") << std::endl;
    loadPFM(mapAlphau6param, _camWidth, _camHeight, std::string(_outdir + "/parameter6AlphauMapDLT.pfm"));
    std::vector<cv::Point2f> mapAlphav6param(nbPixels);
    std::cout << "Load parameters from " << std::string(_outdir + "/parameter6AlphavMapDLT.pfm") << std::endl;
    loadPFM(mapAlphav6param, _camWidth, _camHeight, std::string(_outdir + "/parameter6AlphavMapDLT.pfm"));
    std::vector<cv::Point2f> mapBeta6param(nbPixels);
    std::cout << "Load parameters from " << std::string(_outdir + "/parameter6BetaMapDLT.pfm") << std::endl;
    loadPFM(mapBeta6param, _camWidth, _camHeight, std::string(_outdir + "/parameter6BetaMapDLT.pfm"));

    // AVERAGE COLOR IMAGE
    // TODO: interpolate color like position

    std::vector<cv::Point3f> colorMap(nbPixels);

    for(uint i = 0 ; i < colorMap.size() ; ++i) {

        colorMap[i] = cv::Point3f(0.0, 0.0, 0.0);
    }

    std::cout << "RENDERING " << std::endl;
    // for every light flow (set of parameters)
    // compute the color map (constant weight average)
    // compute the splat destination
    // assign the splat destination the computed average color

    std::cout << "Compute color map (average) " << std::endl;
    for(uint y = _windowH1 ; y < _windowH2 ; ++y) {
        for(uint x = _windowW1 ; x < _windowW2 ; ++x) {

            const uint idx = y*_camWidth + x;

            std::vector<cv::Point2f> flow = _flowedLightField[idx];

            // always as many samples as the number of cameras
            assert(flow.size() == _vCam.size());
            assert(flow.size() >= 2); // nbSamples

            float weight = 0.0;

            // find splat color (average color)
            for(uint k = 0 ; k < _vCam.size() ; ++k) {

                backwardWarping(_camWidth, _camHeight, _vCam[k]->getTextureRGB(), colorMap, weight, x, y, flow[k], 1);
            }

            if(weight != 0) {
                colorMap[idx] /= weight;
            }
        }
    }

    // TARGET CAM PARAMETERS

    cv::Mat targetR, targetK;
    cv::Point3f targetC;
    if(_stanfordConfig) {
        loadTargetView(targetK, targetR, targetC);
    } else {
        loadTargetTranslation(targetK, targetR, targetC);
    }

    // OUTPUT IMAGE

    std::vector<cv::Point3f> outputImage3param(nbPixels);
    std::vector<float> weightMap3param(nbPixels); // splat contribution (for normalization)

    std::vector<cv::Point3f> outputImage4param(nbPixels);
    std::vector<float> weightMap4param(nbPixels); // splat contribution (for normalization)

    std::vector<cv::Point3f> outputImage6param(nbPixels);
    std::vector<float> weightMap6param(nbPixels); // splat contribution (for normalization)

    for(uint i = 0 ; i < outputImage3param.size() ; ++i) {

        outputImage3param[i] = cv::Point3f(0.0, 0.0, 0.0);
        weightMap3param[i] = 0.0;

        outputImage4param[i] = cv::Point3f(0.0, 0.0, 0.0);
        weightMap4param[i] = 0.0;

        outputImage6param[i] = cv::Point3f(0.0, 0.0, 0.0);
        weightMap6param[i] = 0.0;
    }

    // TODO: handle visibility (compute z-buffer)

    std::cout << "Blending step" << std::endl;
    for(uint y = _windowH1 ; y < _windowH2 ; ++y) {
        for(uint x = _windowW1 ; x < _windowW2 ; ++x) {

            const uint idx = y*_camWidth + x;

            // find position (splat destination and size/orientation)

            // 3 PARAMETERS
            cv::Point2f destPoint3param = cv::Point2f(0.0, 0.0);
            const cv::Point3f parameters = map3param[idx];

            splatProjection3param(destPoint3param, parameters, targetK, targetR, targetC);

            // interpolation (splatting)
            if(0.0 <= destPoint3param.x && destPoint3param.x < (float)_camWidth &&
                    0.0 <= destPoint3param.y && destPoint3param.y < (float)_camHeight) {

                projectSplat(_camWidth, _camHeight, colorMap[idx], outputImage3param, weightMap3param, destPoint3param);
            }

            // 4 PARAMETERS
            cv::Point2f destPoint4param = cv::Point2f(0.0, 0.0);
            const cv::Point2f alpha4param = mapAlpha4param[idx];
            const cv::Point2f beta4param = mapBeta4param[idx];

            splatProjection4param(destPoint4param, alpha4param, beta4param, targetK, targetR, targetC);

            // interpolation (splatting)
            if(0.0 <= destPoint4param.x && destPoint4param.x < (float)_camWidth &&
                    0.0 <= destPoint4param.y && destPoint4param.y < (float)_camHeight) {

                projectSplat(_camWidth, _camHeight, colorMap[idx], outputImage4param, weightMap4param, destPoint4param);
            }

            // 6 PARAMETERS
            cv::Point2f destPoint6param = cv::Point2f(0.0, 0.0);
            const cv::Point2f alphau6param = mapAlphau6param[idx];
            const cv::Point2f alphav6param = mapAlphav6param[idx];
            const cv::Point2f beta6param = mapBeta6param[idx];

            splatProjection6param(destPoint6param, alphau6param, alphav6param, beta6param, targetK, targetR, targetC);

            // interpolation (splatting)
            if(0.0 <= destPoint6param.x && destPoint6param.x < (float)_camWidth &&
                    0.0 <= destPoint6param.y && destPoint6param.y < (float)_camHeight) {

                projectSplat(_camWidth, _camHeight, colorMap[idx], outputImage6param, weightMap6param, destPoint6param);
            }
        }
    }

    std::cout << "Normalization step" << std::endl;
    for(uint y = _windowH1 ; y < _windowH2 ; ++y) {
        for(uint x = _windowW1 ; x < _windowW2 ; ++x) {

            const uint idx = y*_camWidth + x;

            if(weightMap3param[idx] != 0) {
                outputImage3param[idx] /= weightMap3param[idx];
            }

            if(weightMap4param[idx] != 0) {
                outputImage4param[idx] /= weightMap4param[idx];
            }

            if(weightMap6param[idx] != 0) {
                outputImage6param[idx] /= weightMap6param[idx];
            }
        }
    }

    std::cout << "Hole filling" << std::endl;
    pushPull(_camWidth, _camHeight, outputImage3param, weightMap3param);
    pushPull(_camWidth, _camHeight, outputImage4param, weightMap4param);
    pushPull(_camWidth, _camHeight, outputImage6param, weightMap6param);

    // SAVE PNG OUTMUT FILES

    char tempCharArray[500];
    std::string outputImage3paramName = _outdir + "/output3paramLamb_%02lu_%02lu.png";
    std::string outputImage4paramName = _outdir + "/output4paramLamb_%02lu_%02lu.png";
    std::string outputImage6paramName = _outdir + "/output6paramLamb_%02lu_%02lu.png";

    memset(tempCharArray, 0, sizeof(tempCharArray));
    sprintf( tempCharArray, outputImage3paramName.c_str(), _sRmv, _tRmv );
    std::cout << "Save output image " << tempCharArray << std::endl;
    savePNG(outputImage3param, _camWidth, _camHeight, tempCharArray);

    memset(tempCharArray, 0, sizeof(tempCharArray));
    sprintf( tempCharArray, outputImage4paramName.c_str(), _sRmv, _tRmv );
    std::cout << "Save output image " << tempCharArray << std::endl;
    savePNG(outputImage4param, _camWidth, _camHeight, tempCharArray);

    memset(tempCharArray, 0, sizeof(tempCharArray));
    sprintf( tempCharArray, outputImage6paramName.c_str(), _sRmv, _tRmv );
    std::cout << "Save output image " << tempCharArray << std::endl;
    savePNG(outputImage6param, _camWidth, _camHeight, tempCharArray);
}

void LFScene::renderLightFlowVideo() {

    const uint nbPixels = _camWidth*_camHeight;

    // TARGET CAM PARAMETERS

    cv::Mat targetR, targetR_transp, targetK, targetK_inv;
    cv::Point3f targetC, targett;
    
    // LOAD POSITION MODEL PARAMETERS

    std::vector<cv::Point3f> map3param(nbPixels);
    std::cout << "Load parameters from " << std::string(_outdir + "/parameter3MapDLT.pfm") << std::endl;
    loadPFM(map3param, _camWidth, _camHeight, std::string(_outdir + "/parameter3MapDLT.pfm"));

    std::vector<cv::Point2f> mapAlpha4param(nbPixels);
    std::cout << "Load parameters from " << std::string(_outdir + "/parameterAlpha2MapDLT.pfm") << std::endl;
    loadPFM(mapAlpha4param, _camWidth, _camHeight, std::string(_outdir + "/parameterAlpha2MapDLT.pfm"));
    std::vector<cv::Point2f> mapBeta4param(nbPixels);
    std::cout << "Load parameters from " << std::string(_outdir + "/parameterBeta2MapDLT.pfm") << std::endl;
    loadPFM(mapBeta4param, _camWidth, _camHeight, std::string(_outdir + "/parameterBeta2MapDLT.pfm"));

    std::vector<cv::Point2f> mapAlphau6param(nbPixels);
    std::cout << "Load parameters from " << std::string(_outdir + "/parameter6AlphauMapDLT.pfm") << std::endl;
    loadPFM(mapAlphau6param, _camWidth, _camHeight, std::string(_outdir + "/parameter6AlphauMapDLT.pfm"));
    std::vector<cv::Point2f> mapAlphav6param(nbPixels);
    std::cout << "Load parameters from " << std::string(_outdir + "/parameter6AlphavMapDLT.pfm") << std::endl;
    loadPFM(mapAlphav6param, _camWidth, _camHeight, std::string(_outdir + "/parameter6AlphavMapDLT.pfm"));
    std::vector<cv::Point2f> mapBeta6param(nbPixels);
    std::cout << "Load parameters from " << std::string(_outdir + "/parameter6BetaMapDLT.pfm") << std::endl;
    loadPFM(mapBeta6param, _camWidth, _camHeight, std::string(_outdir + "/parameter6BetaMapDLT.pfm"));

    // LOAD COLOR MODEL PARAMETERS

    std::vector<cv::Point3f> parameterSMap(nbPixels);
    std::cout << "Load parameters from " << std::string(_outdir + "/parameterSMap.pfm") << std::endl;
    loadPFM(parameterSMap, _camWidth, _camHeight, std::string(_outdir + "/parameterSMap.pfm"));
    std::vector<cv::Point3f> parameterTMap(nbPixels);
    std::cout << "Load parameters from " << std::string(_outdir + "/parameterTMap.pfm") << std::endl;
    loadPFM(parameterTMap, _camWidth, _camHeight, std::string(_outdir + "/parameterTMap.pfm"));
    std::vector<cv::Point3f> parameter0Map(nbPixels);
    std::cout << "Load parameters from " << std::string(_outdir + "/parameter0Map.pfm") << std::endl;
    loadPFM(parameter0Map, _camWidth, _camHeight, std::string(_outdir + "/parameter0Map.pfm"));

    // OUTPUT IMAGE

    std::vector<cv::Point3f> outputImage3param(nbPixels);
    std::vector<float> reprojError3param(nbPixels); // reprojection error of the central view
    std::vector<float> weightMap3param(nbPixels); // splat contribution (for normalizatiNULLon)

    std::vector<cv::Point3f> outputImage4param(nbPixels);
    std::vector<float> reprojError4param(nbPixels); // reprojection error of the central view
    std::vector<float> weightMap4param(nbPixels); // splat contribution (for normalization)

    std::vector<cv::Point3f> outputImage6param(nbPixels);
    std::vector<float> reprojError6param(nbPixels); // reprojection error of the central view
    std::vector<float> weightMap6param(nbPixels); // splat contribution (for normalization)
    
    std::cout << "RENDERING " << std::endl;
    
    const int firstFrame = 0;
    const int lastFrame = 199;
    for(int frame = firstFrame ; frame <= lastFrame ; ++frame) {

        // TARGET CAM PARAMETERS

        std::string targetCameraName = _outdir + "/%08i.ini";
        char targetCameraNameChar[500];
        memset(targetCameraNameChar, 0, sizeof(targetCameraNameChar));
        sprintf( targetCameraNameChar, targetCameraName.c_str(), frame );
        loadTargetView(targetK, targetR, targetC, std::string(targetCameraNameChar));
        
        //        // update target camera
        //        float step = 100;
        //        PinholeCamera pinholeCamera1 = _vCam[_centralIndex - 1]->getPinholeCamera();
        //        PinholeCamera pinholeCamera2 = _vCam[_centralIndex + 1]->getPinholeCamera();
        
        //        PinholeCamera targetCam = _vCam[_centralIndex]->getPinholeCamera(); // same K and R as central camera
        //        glm::mat3 glmTargetK = targetCam._K;
        //        glm::mat3 glmTargetR = targetCam._R;
        //        glm::vec3 glmTargetC = pinholeCamera1._C + (float)frame * (pinholeCamera2._C - pinholeCamera1._C) / step;
        //        glm::vec3 glmTargett = -glmTargetR * glmTargetC;
        
        //        targetK = (cv::Mat_<float>(3,3) << glmTargetK[0][0], glmTargetK[1][0], glmTargetK[2][0],
        //                glmTargetK[0][1], glmTargetK[1][1], glmTargetK[2][1],
        //                glmTargetK[0][2], glmTargetK[1][2], glmTargetK[2][2]);
        //        targetK_inv = targetK.inv();
        //        targetR = (cv::Mat_<float>(3,3) << glmTargetR[0][0], glmTargetR[1][0], glmTargetR[2][0],
        //                glmTargetR[0][1], glmTargetR[1][1], glmTargetR[2][1],
        //                glmTargetR[0][2], glmTargetR[1][2], glmTargetR[2][2]);
        //        targetR_transp = (cv::Mat_<float>(3,3) << glmTargetR[0][0], glmTargetR[0][1], glmTargetR[0][2],
        //                glmTargetR[1][0], glmTargetR[1][1], glmTargetR[1][2],
        //                glmTargetR[2][0], glmTargetR[2][1], glmTargetR[2][2]);
        //        targetC = cv::Point3f((float)glmTargetC[0], (float)glmTargetC[1], (float)glmTargetC[2]);
        //        targett = cv::Point3f((float)glmTargett[0], (float)glmTargett[1], (float)glmTargett[2]);
        
        // init buffers
        for(uint i = 0 ; i < outputImage3param.size() ; ++i) {

            outputImage3param[i] = cv::Point3f(0.0, 0.0, 0.0);
            reprojError3param[i] = 0.0;
            weightMap3param[i] = 0.0;

            outputImage4param[i] = cv::Point3f(0.0, 0.0, 0.0);
            reprojError4param[i] = 0.0;
            weightMap4param[i] = 0.0;

            outputImage6param[i] = cv::Point3f(0.0, 0.0, 0.0);
            reprojError6param[i] = 0.0;
            weightMap6param[i] = 0.0;
        }
        
        // TODO: handle visibility (compute z-buffer)
        
        std::cout << "Blending step" << std::endl;
        for(uint y = _windowH1 ; y < _windowH2 ; ++y) {
            for(uint x = _windowW1 ; x < _windowW2 ; ++x) {

                const uint idx = y*_camWidth + x;

                // find position (splat destination and size/orientation)

                // 3 PARAMETERS
                cv::Point2f destPoint3param = cv::Point2f(0.0, 0.0);
                const cv::Point3f parameters = map3param[idx];
                cv::Point3f color(0.0f, 0.0f, 0.0f);

                splatProjection3param2(destPoint3param, color, parameters, parameterSMap[idx], parameterTMap[idx], parameter0Map[idx], targetK, targetR, targetC);

                // interpolation (splatting)
                if(0.0 <= destPoint3param.x && destPoint3param.x < (float)_camWidth &&
                        0.0 <= destPoint3param.y && destPoint3param.y < (float)_camHeight) {

                    projectSplat(_camWidth, _camHeight, color, outputImage3param, weightMap3param, destPoint3param);
                    reprojError3param[idx] = sqrt((destPoint3param.x - x)*(destPoint3param.x - x) + (destPoint3param.y - y)*(destPoint3param.y - y));
                }

                // 4 PARAMETERS
                cv::Point2f destPoint4param = cv::Point2f(0.0, 0.0);
                const cv::Point2f alpha4param = mapAlpha4param[idx];
                const cv::Point2f beta4param = mapBeta4param[idx];
                color = cv::Point3f (0.0f, 0.0f, 0.0f);

                splatProjection4param2(destPoint4param, color, alpha4param, beta4param, parameterSMap[idx], parameterTMap[idx], parameter0Map[idx], targetK, targetR, targetC);

                // interpolation (splatting)
                if(0.0 <= destPoint4param.x && destPoint4param.x < (float)_camWidth &&
                        0.0 <= destPoint4param.y && destPoint4param.y < (float)_camHeight) {

                    projectSplat(_camWidth, _camHeight, color, outputImage4param, weightMap4param, destPoint4param);
                    reprojError4param[idx] = sqrt((destPoint4param.x - x)*(destPoint4param.x - x) + (destPoint4param.y - y)*(destPoint4param.y - y));
                }

                // 6 PARAMETERS
                cv::Point2f destPoint6param = cv::Point2f(0.0, 0.0);
                const cv::Point2f alphau6param = mapAlphau6param[idx];
                const cv::Point2f alphav6param = mapAlphav6param[idx];
                const cv::Point2f beta6param = mapBeta6param[idx];
                color = cv::Point3f (0.0f, 0.0f, 0.0f);

                splatProjection6param2(destPoint6param, color, alphau6param, alphav6param, beta6param, parameterSMap[idx], parameterTMap[idx], parameter0Map[idx], targetK, targetR, targetC);

                // interpolation (splatting)
                if(0.0 <= destPoint6param.x && destPoint6param.x < (float)_camWidth &&
                        0.0 <= destPoint6param.y && destPoint6param.y < (float)_camHeight) {

                    projectSplat(_camWidth, _camHeight, color, outputImage6param, weightMap6param, destPoint6param);
                    reprojError6param[idx] = sqrt((destPoint6param.x - x)*(destPoint6param.x - x) + (destPoint6param.y - y)*(destPoint6param.y - y));
                }
            }
        }

        std::cout << "Normalization step" << std::endl;
        for(uint y = 0 ; y < _camHeight ; ++y) {
            for(uint x = 0 ; x < _camWidth ; ++x) {

                const uint idx = y*_camWidth + x;

                if(weightMap3param[idx] != 0) {
                    outputImage3param[idx] /= weightMap3param[idx];
                }

                if(weightMap4param[idx] != 0) {
                    outputImage4param[idx] /= weightMap4param[idx];
                }

                if(weightMap6param[idx] != 0) {
                    outputImage6param[idx] /= weightMap6param[idx];
                }
            }
        }

        std::cout << "Hole filling" << std::endl;
        pushPull(_camWidth, _camHeight, outputImage3param, weightMap3param);
        pushPull(_camWidth, _camHeight, outputImage4param, weightMap4param);
        pushPull(_camWidth, _camHeight, outputImage6param, weightMap6param);

        // SAVE PNG OUTMUT FILES

        char tempCharArray[500];
        std::string outputImage3paramName = _outdir + "/output3param_%02lu_%02lu_%03lu.png";
        std::string outputImage4paramName = _outdir + "/output4param_%02lu_%02lu_%03lu.png";
        std::string outputImage6paramName = _outdir + "/output6param_%02lu_%02lu_%03lu.png";

        memset(tempCharArray, 0, sizeof(tempCharArray));
        sprintf( tempCharArray, outputImage3paramName.c_str(), _sRmv, _tRmv, frame );
        std::cout << "Save output image " << tempCharArray << std::endl;
        savePNG(outputImage3param, _camWidth, _camHeight, tempCharArray);

        memset(tempCharArray, 0, sizeof(tempCharArray));
        sprintf( tempCharArray, outputImage4paramName.c_str(), _sRmv, _tRmv, frame );
        std::cout << "Save output image " << tempCharArray << std::endl;
        savePNG(outputImage4param, _camWidth, _camHeight, tempCharArray);

        memset(tempCharArray, 0, sizeof(tempCharArray));
        sprintf( tempCharArray, outputImage6paramName.c_str(), _sRmv, _tRmv, frame );
        std::cout << "Save output image " << tempCharArray << std::endl;
        savePNG(outputImage6param, _camWidth, _camHeight, tempCharArray);
    }
}

void LFScene::renderLightFlowLambertianVideo() {

    const uint nbPixels = _camWidth*_camHeight;

    // TARGET CAM PARAMETERS

    cv::Mat targetR, targetK;
    cv::Point3f targetC;

    // LOAD POSITION MODEL PARAMETERS

    std::vector<cv::Point3f> map3param(nbPixels);
    std::vector<cv::Point2f> mapAlpha4param(nbPixels);
    std::vector<cv::Point2f> mapBeta4param(nbPixels);
    std::vector<cv::Point2f> mapAlphau6param(nbPixels);
    std::vector<cv::Point2f> mapAlphav6param(nbPixels);
    std::vector<cv::Point2f> mapBeta6param(nbPixels);

    if(_renderIndex >= 0) {

        load3fMap(map3param, _outdir + "/model_3g_IHM_%02lu.pfm", _renderIndex);
        load2fMap(mapAlpha4param, _outdir + "/model_4g_IHM_%02lu_a.pfm", _renderIndex);
        load2fMap(mapBeta4param, _outdir + "/model_4g_IHM_%02lu_b.pfm", _renderIndex);
        load2fMap(mapAlphau6param, _outdir + "/model_6g_IHM_%02lu_au.pfm", _renderIndex);
        load2fMap(mapAlphav6param, _outdir + "/model_6g_IHM_%02lu_av.pfm", _renderIndex);
        load2fMap(mapBeta6param, _outdir + "/model_6g_IHM_%02lu_b.pfm", _renderIndex);

    } else {

        load3fMap(map3param, _outdir + "/model_3g_IHM_allViews.pfm", _renderIndex);
        load2fMap(mapAlpha4param, _outdir + "/model_4g_IHM_allViews_a.pfm", _renderIndex);
        load2fMap(mapBeta4param, _outdir + "/model_4g_IHM_allViews_b.pfm", _renderIndex);
        load2fMap(mapAlphau6param, _outdir + "/model_6g_IHM_allViews_au.pfm", _renderIndex);
        load2fMap(mapAlphav6param, _outdir + "/model_6g_IHM_allViews_av.pfm", _renderIndex);
        load2fMap(mapBeta6param, _outdir + "/model_6g_IHM_allViews_b.pfm", _renderIndex);
    }

    // for every light flow (set of parameters)
    // compute the color map (constant weight average)
    // compute the splat destination
    // assign the splat destination the computed average color

    // AVERAGE COLOR IMAGE
    // TODO: interpolate color like position

    std::vector<cv::Point3f> colorMap(nbPixels);

    for(uint i = 0 ; i < colorMap.size() ; ++i) {

        colorMap[i] = cv::Point3f(0.0, 0.0, 0.0);
    }

    std::cout << "Compute color map (average) " << std::endl;
    for(uint y = _windowH1 ; y < _windowH2 ; ++y) {
        for(uint x = _windowW1 ; x < _windowW2 ; ++x) {

            const uint idx = y*_camWidth + x;

            std::vector<cv::Point2f> flow = _flowedLightField[idx];

            // always as many samples as the number of cameras
            assert(flow.size() == _vCam.size());
            assert(flow.size() >= 2); // nbSamples

            float weight = 0.0;

            // find splat color (average color)
            for(uint k = 0 ; k < _vCam.size() ; ++k) {

                backwardWarping(_camWidth, _camHeight, _vCam[k]->getTextureRGB(), colorMap, weight, x, y, flow[k], 1);
            }

            if(weight != 0) {
                colorMap[idx] /= weight;
            }
        }
    }

    // OUTPUT IMAGE

    std::vector<cv::Point3f> outputImage3param(nbPixels);
    std::vector<float> weightMap3param(nbPixels); // splat contribution (for normalization)

    std::vector<cv::Point3f> outputImage4param(nbPixels);
    std::vector<float> weightMap4param(nbPixels); // splat contribution (for normalization)

    std::vector<cv::Point3f> outputImage6param(nbPixels);
    std::vector<float> weightMap6param(nbPixels); // splat contribution (for normalization)

    std::cout << "RENDERING " << std::endl;

    const int firstFrame = 0;
    int lastFrame = 50;
    if(_renderIndex >= 0) {
        lastFrame = 0;
    }
    for(int frame = firstFrame ; frame <= lastFrame ; ++frame) {

        PinholeCamera targetCam;

        if(_renderIndex >= 0) {

            targetCam = _vCam[_renderIndex]->getPinholeCamera();

        } else {

            // TARGET CAM PARAMETERS

            //        std::string targetCameraName = _outdir + "/%08i.ini";
            //        char targetCameraNameChar[500];
            //        memset(targetCameraNameChar, 0, sizeof(targetCameraNameChar));
            //        sprintf( targetCameraNameChar, targetCameraName.c_str(), frame );
            //        loadTargetView(targetK, targetR, targetC, std::string(targetCameraNameChar));

            // INPUT VIEWS
            //        targetCam = _vCam[frame]->getPinholeCamera();

            // ZOOM
            //        targetCam = _vCam[12]->getPinholeCamera();
            //        targetCam._K[0][0] += (float)frame * 600.0f;
            //        targetCam._K[1][1] += (float)frame * 600.0f;

            // PANNING
            targetCam = _vCam[_centralIndex]->getPinholeCamera();
            const float step = 50;
            PinholeCamera pinholeCamera1 = _vCam[_centralIndex - 2]->getPinholeCamera();
            PinholeCamera pinholeCamera2 = _vCam[_centralIndex + 2]->getPinholeCamera();
            targetCam._C = pinholeCamera1._C + (float)frame * (pinholeCamera2._C - pinholeCamera1._C) / step;
            targetCam._C = targetCam._C + (float)frame * 2.5f * glm::vec3(0, 0, 1) + (float)frame * 0.5f * glm::vec3(0, 1, 0);
        }

        targetK = (cv::Mat_<float>(3,3) << targetCam._K[0][0], targetCam._K[1][0], targetCam._K[2][0],
                targetCam._K[0][1], targetCam._K[1][1], targetCam._K[2][1],
                targetCam._K[0][2], targetCam._K[1][2], targetCam._K[2][2]);
        targetR = (cv::Mat_<float>(3,3) << targetCam._R[0][0], targetCam._R[1][0], targetCam._R[2][0],
                targetCam._R[0][1], targetCam._R[1][1], targetCam._R[2][1],
                targetCam._R[0][2], targetCam._R[1][2], targetCam._R[2][2]);
        targetC = cv::Point3f((float)targetCam._C[0], (float)targetCam._C[1], (float)targetCam._C[2]);

        //        targetC.z = 0.0;

        // init buffers
        for(uint i = 0 ; i < nbPixels ; ++i) {

            outputImage3param[i] = cv::Point3f(0.0, 0.0, 0.0);
            weightMap3param[i] = 0.0;

            outputImage4param[i] = cv::Point3f(0.0, 0.0, 0.0);
            weightMap4param[i] = 0.0;

            outputImage6param[i] = cv::Point3f(0.0, 0.0, 0.0);
            weightMap6param[i] = 0.0;
        }

        std::cout << "Blending step" << std::endl;
        for(uint y = _windowH1 ; y < _windowH2 ; ++y) {
            for(uint x = _windowW1 ; x < _windowW2 ; ++x) {

                const uint idx = y*_camWidth + x;

                // find position (splat destination and size/orientation)

                // 3 PARAMETERS
                cv::Point2f destPoint3param = cv::Point2f(0.0, 0.0);
                const cv::Point3f parameters = map3param[idx];

                splatProjection3param(destPoint3param, parameters, targetK, targetR, targetC);

                // interpolation (splatting)
                if(0.0 <= destPoint3param.x && destPoint3param.x < (float)_camWidth &&
                        0.0 <= destPoint3param.y && destPoint3param.y < (float)_camHeight) {

                    projectSplat(_camWidth, _camHeight, colorMap[idx], outputImage3param, weightMap3param, destPoint3param);
                }

                // 4 PARAMETERS
                cv::Point2f destPoint4param = cv::Point2f(0.0, 0.0);
                const cv::Point2f alpha4param = mapAlpha4param[idx];
                const cv::Point2f beta4param = mapBeta4param[idx];

                splatProjection4param(destPoint4param, alpha4param, beta4param, targetK, targetR, targetC);

                // interpolation (splatting)
                if(0.0 <= destPoint4param.x && destPoint4param.x < (float)_camWidth &&
                        0.0 <= destPoint4param.y && destPoint4param.y < (float)_camHeight) {

                    projectSplat(_camWidth, _camHeight, colorMap[idx], outputImage4param, weightMap4param, destPoint4param);
                }

                // 6 PARAMETERS
                cv::Point2f destPoint6param = cv::Point2f(0.0, 0.0);
                const cv::Point2f alphau6param = mapAlphau6param[idx];
                const cv::Point2f alphav6param = mapAlphav6param[idx];
                const cv::Point2f beta6param = mapBeta6param[idx];

                splatProjection6param(destPoint6param, alphau6param, alphav6param, beta6param, targetK, targetR, targetC);

                // interpolation (splatting)
                if(0.0 <= destPoint6param.x && destPoint6param.x < (float)_camWidth &&
                        0.0 <= destPoint6param.y && destPoint6param.y < (float)_camHeight) {

                    projectSplat(_camWidth, _camHeight, colorMap[idx], outputImage6param, weightMap6param, destPoint6param);
                }
            }
        }

        std::cout << "Normalization step" << std::endl;
        for(uint i = 0 ; i < nbPixels ; ++i) {

            if(weightMap3param[i] != 0) {
                outputImage3param[i] /= weightMap3param[i];
            }

            if(weightMap4param[i] != 0) {
                outputImage4param[i] /= weightMap4param[i];
            }

            if(weightMap6param[i] != 0) {
                outputImage6param[i] /= weightMap6param[i];
            }
        }

//        std::cout << "Hole filling" << std::endl;
//        pushPull(_camWidth, _camHeight, outputImage3param, weightMap3param);
//        pushPull(_camWidth, _camHeight, outputImage4param, weightMap4param);
//        pushPull(_camWidth, _camHeight, outputImage6param, weightMap6param);

        std::cout << "PULL/PUSH" << std::endl;
        pushPullGortler(_camWidth, _camHeight, outputImage3param, weightMap3param);
        pushPullGortler(_camWidth, _camHeight, outputImage4param, weightMap4param);
        pushPullGortler(_camWidth, _camHeight, outputImage6param, weightMap6param);

        // SAVE PNG OUTMUT FILES

        if(_renderIndex >= 0) {

            save3uMap(outputImage3param, _outdir + "/3g_IBR_%02lu.png", _renderIndex);
            save3uMap(outputImage4param, _outdir + "/4g_IBR_%02lu.png", _renderIndex);
            save3uMap(outputImage6param, _outdir + "/6g_IBR_%02lu.png", _renderIndex);

        } else {

            save3uMap(outputImage3param, _outdir + "/3g_IBR_panning_%03lu.png", frame);
            save3uMap(outputImage4param, _outdir + "/4g_IBR_panning_%03lu.png", frame);
            save3uMap(outputImage6param, _outdir + "/6g_IBR_panning_%03lu.png", frame);

            //            save3uMap(outputImage3param, _outdir + "/3g_IBR_zooming_%03lu.png", frame);
            //            save3uMap(outputImage4param, _outdir + "/4g_IBR_zooming_%03lu.png", frame);
            //            save3uMap(outputImage6param, _outdir + "/6g_IBR_zooming_%03lu.png", frame);
        }
    }
}






