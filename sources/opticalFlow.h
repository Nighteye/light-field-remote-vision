#ifndef OPTICAL_FLOW_H
#define OPTICAL_FLOW_H

#include <string>
#include <vector>
#include <thread>

#include "optical_flow/CvUtil.h"
#include "optical_flow/NovelView.h"

void prepareOpticalFlow(std::string leftImageName,
                        std::string rightImageName,
                        std::vector<cv::Point2f>* flowLtoRvec,
                        surround360::optical_flow::NovelViewGenerator* novelViewGen);

void computeOpticalFlow(std::string leftImageName,
                        std::string rightImageName,
                        std::string flowAlg,
                        std::vector<cv::Point2f>* flowLtoRvec,
                        std::vector<cv::Point2f>* flowRtoLvec);

void testOpticalFlow(std::string leftImageName, std::string rightImageName, std::string outdir);

void computeCovariance(std::vector<cv::Point2f> &data, std::vector<cv::Point3f> &centerData, cv::Point3f targetCenter,
                       float *trace, float *determinant, float *eigenRatio, cv::Point2f *eigenValues, cv::Point3f *eigenVector,
                       cv::Point2f *targetImage);

void computePerPixelCorrespBandConfig(std::string mveName,
                                      std::string flowAlg);

class OpticalFlow {

private:

    uint _nbImages;
    double _startOpticalFlowTime;
    double _opticalFlowRuntime;
    std::vector<surround360::optical_flow::NovelViewGenerator*> _novelViewGenerators;
    std::vector<std::thread> _threads;

public:

    OpticalFlow( uint nbImages );
    ~OpticalFlow();

    void pushThread(uint flowIdx,
                    std::string leftImageName,
                    std::string rightImageName,
                    std::string flowAlg,
                    std::vector<cv::Point2f>& flowLtoRvec);

    void join();
};

#endif /* #ifndef OPTICAL_FLOW_H */
