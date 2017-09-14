#ifndef TRIANGULATION_H
#define TRIANGULATION_H

#include <vector>

#include "optical_flow/CvUtil.h"

// compute the destination image point from flow and novel view camera parameters
void splatProjection3param(cv::Point2f &imagePoint, const cv::Point3f &parameters,
                           const cv::Mat &K, const cv::Mat &R, const cv::Point3f &C);

void splatProjection4param(cv::Point2f &imagePoint, const cv::Point2f &alpha, const cv::Point2f &beta,
                           const cv::Mat &K, const cv::Mat &R, const cv::Point3f &C);

void splatProjection6param(cv::Point2f &imagePoint, const cv::Point2f &alphau, const cv::Point2f &alphav, const cv::Point2f &beta,
                           const cv::Mat &K, const cv::Mat &R, const cv::Point3f &C);

// same with color interpolation
void splatProjection3param2(cv::Point2f &imagePoint, cv::Point3f &color,
                            const cv::Point3f &parameters,
                            const cv::Point3f& paramS, const cv::Point3f& paramT, const cv::Point3f& param0,
                            const cv::Mat &K, const cv::Mat &R, const cv::Point3f &C);

void splatProjection4param2(cv::Point2f &imagePoint, cv::Point3f &color,
                            const cv::Point2f &alpha, const cv::Point2f &beta,
                            const cv::Point3f& paramS, const cv::Point3f& paramT, const cv::Point3f& param0,
                            const cv::Mat &K, const cv::Mat &R, const cv::Point3f &C);

void splatProjection6param2(cv::Point2f &imagePoint, cv::Point3f &color,
                            const cv::Point2f &alphau, const cv::Point2f &alphav, const cv::Point2f &beta,
                            const cv::Point3f& paramS, const cv::Point3f& paramT, const cv::Point3f& param0,
                            const cv::Mat &K, const cv::Mat &R, const cv::Point3f &C);

// compute sample points and eigenvalues, necessary for the residual
void computeResidualParameters(int nbSamples, const std::vector<cv::Point2f> &flow, const std::vector<cv::Mat> &K_inv, const std::vector<cv::Mat> &R_transp, const std::vector<cv::Point3f> &C,
                               std::vector<float>& eigenVectors, std::vector<float>& eigenValues, std::vector<float>& samplePoint);

// estimate the color parameter for a linear model (a function of s)
void colorRegression(int nbSamples, const std::vector<cv::Point2f> &flow, const std::vector<cv::Point3f>& colorSampleSet,
                     const std::vector<cv::Mat> &K_inv, const std::vector<cv::Mat> &R_transp, const std::vector<cv::Point3f> &C,
                     std::vector<float> &x, float &finalCost, bool verbose);

// compute model parameters from flowed LF by curve fitting
void triangulationLF(int nbSamples, const std::vector<cv::Point2f> &flow,
                     const std::vector<cv::Mat> &K_inv, const std::vector<cv::Mat> &R_transp, const std::vector<cv::Point3f> &C,
                     std::vector<float> &x, float &finalCost, float &conditionNumber, bool verbose = false);

// compute 3D point coordinates from flowed LF by classic triangulation
void triangulationClassic(int nbSamples, const std::vector<cv::Point2f> &flow,
                          const std::vector<cv::Mat> &K, const std::vector<cv::Mat> &R, const std::vector<cv::Point3f> &t,
                          std::vector<float> &point3D, float &finalCost, bool verbose = false);

// compute the reprojection error for the computed 3D point
void reprojectionCompute(const std::vector<float> &point3D, int nbSamples, const std::vector<cv::Point2f> &flow,
                         const std::vector<cv::Mat> &K, const std::vector<cv::Mat> &R, const std::vector<cv::Point3f> &t,
                         float &error);

// check if the reprojection of the 3D point matches the TEST 3D samples
void reprojectionCheck(const std::vector<float> &point3D, int nbSamples,
                       const std::vector<cv::Mat> &K, const std::vector<cv::Mat> &R, const std::vector<cv::Point3f> &t);

void printEigen(const std::vector<float>& eigenVectors, const std::vector<float>& eigenValues, const std::vector<float>& samplePoint);

// perform DLT to initialize optimization
void DLT(uint nbSamples,
         const std::vector<float>& samplePoint,
         std::vector<float> &x,
         float& conditionNumber);

// perform inhomogeneous method to initialize optimization
void IHM(uint nbSamples,
         const std::vector<float>& samplePoint,
         std::vector<float> &x,
         float& conditionNumber);

// fit one set of sample with new LF model, polynome of degree 1
void optimize(int kNumObservations,
              const std::vector<float>& eigenVectors,
              const std::vector<float>& eigenValues,
              const std::vector<float>& samplePoint,
              std::vector<float> &x, float &finalCost, bool verbose = false);

void testLFTriangulation(int nbSamples,
                         const std::vector<cv::Mat> &K_inv, const std::vector<cv::Mat> &R_transp, const std::vector<cv::Point3f> &C,
                         std::vector<float> &point3D, float &finalCost);

// triangulate one set of sample with classic triangulation method
void triangulateSamplesClassic(int kNumObservations, const std::vector<float>& observation,
                               const std::vector<float>& eigenVectors,
                               const std::vector<float>& eigenValues,
                               const std::vector<float>& samplePoint,
                               std::vector<float> &point3D, float &finalCost, bool verbose = false);

void testClassicTriangulation(int nbSamples,
                              const std::vector<cv::Mat> &K, const std::vector<cv::Mat> &R, const std::vector<cv::Point3f> &t,
                              std::vector<float> &point3D, float &finalCost);

#endif /* #ifndef TRIANGULATION_H */
