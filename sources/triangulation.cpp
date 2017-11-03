#include <iostream>
#include <ceres/ceres.h>
#include "optical_flow/CvUtil.h"

#include "triangulation.h"

using namespace ceres;
using namespace cv;

static const float dist = 1000.0; // distance between the two planes

static std::vector<cv::Point2f> testFlow = {cv::Point2f(500, 200),
                                            cv::Point2f(534.76953, 217.84976),
                                            cv::Point2f(576.31268, 191.77184),
                                            cv::Point2f(590.68207, 200.87665),
                                            cv::Point2f(607.8374, 177.04419)};

void splatProjection3param(cv::Point2f &imagePoint, const Point3f &parameters,
                           const cv::Mat &K, const cv::Mat &R, const cv::Point3f &C) {

    Mat x = (Mat_<float>(3, 1) << 0, 0, 0);

    if(C.z > 0.0000001) { // the optical center of the target view is NOT located on the plane (s,t)

        // uv = A * st + B
        // Ap -> lambertian point
        // Ac -> pinhole camera
        // Bp -> lambertian point
        // Bc -> pinhole camera
        // matrix M = Ap - Ac
        // vector b = Bc - Bp

        // Ap = ap * I2 (lambertian point)
        // Ac = ac * I2, with ac = (c3 - dist)/c3 (behaves like lambertian)
        // M = m * I2, with m = ap - ac

        float m = parameters.x - (C.z - dist)/C.z;
        Mat M = (Mat_<float>(2, 2) << m, 0, 0, m);
        Mat M_inv = M.inv();
        float buc = dist*C.x/C.z;
        float bvc = dist*C.y/C.z;
        Mat b = (Mat_<float>(2, 1) << buc - parameters.y, bvc - parameters.z);
        Mat st = M_inv * b;

        // we project the point (s,t,0)

        Mat X = (Mat_<float>(3, 1) << st.at<float>(0, 0) - C.x, st.at<float>(1, 0) - C.y, - C.z);
        x = K * R * X;

    } else {

        // simplified rendering equations
        Mat st = (Mat_<float>(2, 1) << C.x, C.y);

        // we project the point (u,v,dist)

        Mat Ap = (Mat_<float>(2, 2) << parameters.x, 0, 0, parameters.x);
        Mat Bp = (Mat_<float>(2, 1) << parameters.y, parameters.z);
        Mat uv = Ap*st + Bp;

        Mat X = (Mat_<float>(3, 1) << uv.at<float>(0, 0) - C.x, uv.at<float>(1, 0) - C.y, dist - C.z);
        x = K * R * X;
    }

    imagePoint.x = x.at<float>(0, 0)/x.at<float>(2, 0);
    imagePoint.y = x.at<float>(1, 0)/x.at<float>(2, 0);
}

void splatProjection4param(cv::Point2f &imagePoint, const cv::Point2f &alpha, const cv::Point2f &beta,
                           const cv::Mat &K, const cv::Mat &R, const cv::Point3f &C) {

    Mat x = (Mat_<float>(3, 1) << 0, 0, 0);

    if(C.z > 0.0000001) { // the optical center of the target view is NOT located on the plane (s,t)

        // uv = A * st + B
        // Ap -> lambertian point
        // Ac -> pinhole camera
        // Bp -> lambertian point
        // Bc -> pinhole camera
        // matrix M = Ap - Ac
        // vector b = Bc - Bp

        // Ap = [au, 0 ; 0, av]
        // Ac = ac * I2, with ac = (c3 - dist)/c3 (behaves like lambertian)
        // M = [au - ac, 0 ; 0, av - ac]

        float ac = (C.z - dist)/C.z;
        Mat M = (Mat_<float>(2, 2) << alpha.x - ac, 0, 0, alpha.y - ac);
        Mat M_inv = M.inv();
        float buc = dist*C.x/C.z;
        float bvc = dist*C.y/C.z;
        Mat b = (Mat_<float>(2, 1) << buc - beta.x, bvc - beta.y);
        Mat st = M_inv * b;

        Mat X = (Mat_<float>(3, 1) << st.at<float>(0, 0) - C.x, st.at<float>(1, 0) - C.y, - C.z);
        x = K * R * X;

    } else {

        // simplified rendering equations
        Mat st = (Mat_<float>(2, 1) << C.x, C.y);

        // we project the point (u,v,dist)

        Mat Ap = (Mat_<float>(2, 2) << alpha.x, 0, 0, alpha.y);
        Mat Bp = (Mat_<float>(2, 1) << beta.x, beta.y);
        Mat uv = Ap*st + Bp;

        Mat X = (Mat_<float>(3, 1) << uv.at<float>(0, 0) - C.x, uv.at<float>(1, 0) - C.y, dist - C.z);
        x = K * R * X;
    }

    imagePoint.x = x.at<float>(0, 0)/x.at<float>(2, 0);
    imagePoint.y = x.at<float>(1, 0)/x.at<float>(2, 0);
}

void splatProjection6param(cv::Point2f &imagePoint, const cv::Point2f &alphau, const cv::Point2f &alphav, const cv::Point2f &beta,
                           const cv::Mat &K, const cv::Mat &R, const cv::Point3f &C) {

    Mat x = (Mat_<float>(3, 1) << 0, 0, 0);

    if(C.z > 0.0000001) { // the optical center of the target view is NOT located on the plane (s,t)

        // uv = A * st + B
        // Ap -> lambertian point
        // Ac -> pinhole camera
        // Bp -> lambertian point
        // Bc -> pinhole camera
        // matrix M = Ap - Ac
        // vector b = Bc - Bp

        // Ap = [au, 0 ; 0, av]
        // Ac = ac * I2, with ac = (c3 - 1)/c3 (behaves like lambertian)
        // M = [aus - ac, aut ; avs, avt - ac]

        float ac = (C.z - dist)/C.z;
        Mat M = (Mat_<float>(2, 2) << alphau.x - ac, alphau.y, alphav.x, alphav.y - ac);
        Mat M_inv = M.inv();
        float buc = dist*C.x/C.z;
        float bvc = dist*C.y/C.z;
        Mat b = (Mat_<float>(2, 1) << buc - beta.x, bvc - beta.y);
        Mat st = M_inv * b;

        Mat X = (Mat_<float>(3, 1) << st.at<float>(0, 0) - C.x, st.at<float>(1, 0) - C.y, - C.z);
        x = K * R * X;

    } else {

        // simplified rendering equations
        Mat st = (Mat_<float>(2, 1) << C.x, C.y);

        // we project the point (u,v,dist)

        Mat Ap = (Mat_<float>(2, 2) << alphau.x, alphau.y, alphav.x, alphav.y);
        Mat Bp = (Mat_<float>(2, 1) << beta.x, beta.y);
        Mat uv = Ap*st + Bp;

        Mat X = (Mat_<float>(3, 1) << uv.at<float>(0, 0) - C.x, uv.at<float>(1, 0) - C.y, dist - C.z);
        x = K * R * X;
    }

    imagePoint.x = x.at<float>(0, 0)/x.at<float>(2, 0);
    imagePoint.y = x.at<float>(1, 0)/x.at<float>(2, 0);
}

void splatProjection3param2(cv::Point2f &imagePoint, cv::Point3f &color,
                            const Point3f &parameters,
                            const cv::Point3f& paramS, const cv::Point3f& paramT, const cv::Point3f& param0,
                            const cv::Mat &K, const cv::Mat &R, const cv::Point3f &C) {

    Mat x = (Mat_<float>(3, 1) << 0, 0, 0);
    Mat st = (Mat_<float>(2, 1) << 0, 0);

    if(C.z > 0.0000001) { // the optical center of the target view is NOT located on the plane (s,t)

        float m = parameters.x - (C.z - dist)/C.z;
        Mat M = (Mat_<float>(2, 2) << m, 0, 0, m);
        Mat M_inv = M.inv();
        float buc = dist*C.x/C.z;
        float bvc = dist*C.y/C.z;
        Mat b = (Mat_<float>(2, 1) << buc - parameters.y, bvc - parameters.z);
        st = M_inv * b;

        // we project the point (s,t,0)

        Mat X = (Mat_<float>(3, 1) << st.at<float>(0, 0) - C.x, st.at<float>(1, 0) - C.y, - C.z);
        x = K * R * X;

    } else {

        // simplified rendering equations
        st = (Mat_<float>(2, 1) << C.x, C.y);

        // we project the point (u,v,dist)

        Mat Ap = (Mat_<float>(2, 2) << parameters.x, 0, 0, parameters.x);
        Mat Bp = (Mat_<float>(2, 1) << parameters.y, parameters.z);
        Mat uv = Ap*st + Bp;

        Mat X = (Mat_<float>(3, 1) << uv.at<float>(0, 0) - C.x, uv.at<float>(1, 0) - C.y, dist - C.z);
        x = K * R * X;
    }

    imagePoint.x = x.at<float>(0, 0)/x.at<float>(2, 0);
    imagePoint.y = x.at<float>(1, 0)/x.at<float>(2, 0);

    Mat A = (Mat_<float>(3, 2) << paramS.x, paramT.x, paramS.y, paramT.y, paramS.z, paramT.z);
    Mat d = (Mat_<float>(3, 1) << param0.x, param0.y, param0.z);
    Mat I = A*st + d;

    color.x = I.at<float>(0, 0);
    color.y = I.at<float>(1, 0);
    color.z = I.at<float>(2, 0);
}

void splatProjection4param2(cv::Point2f &imagePoint, cv::Point3f &color,
                            const cv::Point2f &alpha, const cv::Point2f &beta,
                            const cv::Point3f& paramS, const cv::Point3f& paramT, const cv::Point3f& param0,
                            const cv::Mat &K, const cv::Mat &R, const cv::Point3f &C) {

    Mat x = (Mat_<float>(3, 1) << 0, 0, 0);
    Mat st = (Mat_<float>(2, 1) << 0, 0);

    if(C.z > 0.0000001) { // the optical center of the target view is NOT located on the plane (s,t)

        float ac = (C.z - dist)/C.z;
        Mat M = (Mat_<float>(2, 2) << alpha.x - ac, 0, 0, alpha.y - ac);
        Mat M_inv = M.inv();
        float buc = dist*C.x/C.z;
        float bvc = dist*C.y/C.z;
        Mat b = (Mat_<float>(2, 1) << buc - beta.x, bvc - beta.y);
        st = M_inv * b;

        Mat X = (Mat_<float>(3, 1) << st.at<float>(0, 0) - C.x, st.at<float>(1, 0) - C.y, - C.z);
        x = K * R * X;

    } else {

        // simplified rendering equations
        st = (Mat_<float>(2, 1) << C.x, C.y);

        // we project the point (u,v,dist)

        Mat Ap = (Mat_<float>(2, 2) << alpha.x, 0, 0, alpha.y);
        Mat Bp = (Mat_<float>(2, 1) << beta.x, beta.y);
        Mat uv = Ap*st + Bp;

        Mat X = (Mat_<float>(3, 1) << uv.at<float>(0, 0) - C.x, uv.at<float>(1, 0) - C.y, dist - C.z);
        x = K * R * X;
    }

    imagePoint.x = x.at<float>(0, 0)/x.at<float>(2, 0);
    imagePoint.y = x.at<float>(1, 0)/x.at<float>(2, 0);

    Mat A = (Mat_<float>(3, 2) << paramS.x, paramT.x, paramS.y, paramT.y, paramS.z, paramT.z);
    Mat d = (Mat_<float>(3, 1) << param0.x, param0.y, param0.z);
    Mat I = A*st + d;

    color.x = I.at<float>(0, 0);
    color.y = I.at<float>(1, 0);
    color.z = I.at<float>(2, 0);
}

void splatProjection6param2(cv::Point2f &imagePoint, cv::Point3f &color,
                            const cv::Point2f &alphau, const cv::Point2f &alphav, const cv::Point2f &beta,
                            const cv::Point3f& paramS, const cv::Point3f& paramT, const cv::Point3f& param0,
                            const cv::Mat &K, const cv::Mat &R, const cv::Point3f &C) {

    Mat x = (Mat_<float>(3, 1) << 0, 0, 0);
    Mat st = (Mat_<float>(2, 1) << 0, 0);

    if(C.z > 0.0000001) { // the optical center of the target view is NOT located on the plane (s,t)

        float ac = (C.z - dist)/C.z;
        Mat M = (Mat_<float>(2, 2) << alphau.x - ac, alphau.y, alphav.x, alphav.y - ac);
        Mat M_inv = M.inv();
        float buc = dist*C.x/C.z;
        float bvc = dist*C.y/C.z;
        Mat b = (Mat_<float>(2, 1) << buc - beta.x, bvc - beta.y);
        st = M_inv * b;

        Mat X = (Mat_<float>(3, 1) << st.at<float>(0, 0) - C.x, st.at<float>(1, 0) - C.y, - C.z);
        x = K * R * X;

    } else {

        // simplified rendering equations
        st = (Mat_<float>(2, 1) << C.x, C.y);

        // we project the point (u,v,dist)

        Mat Ap = (Mat_<float>(2, 2) << alphau.x, alphau.y, alphav.x, alphav.y);
        Mat Bp = (Mat_<float>(2, 1) << beta.x, beta.y);
        Mat uv = Ap*st + Bp;

        Mat X = (Mat_<float>(3, 1) << uv.at<float>(0, 0) - C.x, uv.at<float>(1, 0) - C.y, dist - C.z);
        x = K * R * X;
    }

    imagePoint.x = x.at<float>(0, 0)/x.at<float>(2, 0);
    imagePoint.y = x.at<float>(1, 0)/x.at<float>(2, 0);

    Mat A = (Mat_<float>(3, 2) << paramS.x, paramT.x, paramS.y, paramT.y, paramS.z, paramT.z);
    Mat d = (Mat_<float>(3, 1) << param0.x, param0.y, param0.z);
    Mat I = A*st + d;

    color.x = I.at<float>(0, 0);
    color.y = I.at<float>(1, 0);
    color.z = I.at<float>(2, 0);
}

void computeResidualParameters(int nbSamples, const std::vector<cv::Point2f> &flow, const std::vector<cv::Mat> &K_inv, const std::vector<cv::Mat> &R_transp, const std::vector<cv::Point3f> &C,
                               std::vector<float>& eigenVectors, std::vector<float>& eigenValues, std::vector<float>& samplePoint) {

    for(int k = 0 ; k < nbSamples ; ++k) {

        const cv::Mat K_invk = K_inv[k];
        const cv::Mat R_transpk = R_transp[k];
        const cv::Point3f Ck = C[k];

        // lightfield parametrization
        Mat ray = R_transpk * K_invk * Mat(cv::Point3f(flow[k].x, flow[k].y, 1.0));
        Mat a = (Mat_<float>(2, 1) << ray.at<float>(0, 0)/ray.at<float>(2, 0),
                 ray.at<float>(1, 0)/ray.at<float>(2, 0));

        samplePoint[4*k+0] = Ck.x + a.at<float>(0, 0)*(dist - Ck.z);
        samplePoint[4*k+1] = Ck.y + a.at<float>(1, 0)*(dist - Ck.z);
        samplePoint[4*k+2] = Ck.x + a.at<float>(0, 0)*(-Ck.z);
        samplePoint[4*k+3] = Ck.y + a.at<float>(1, 0)*(-Ck.z);

        // compute the Jacobian
        Mat temp1 = (Mat_<float>(3, 2) << 1, 0, 0, 1, 0, 0);
        Mat temp2 = (Mat_<float>(2, 3) << 1, 0, -a.at<float>(0, 0), 0, 1, -a.at<float>(1, 0));
        temp1 = R_transpk * K_invk * temp1;
        Mat Ja = temp2*temp1 / ray.at<float>(2, 0);
        Mat Ja_transp = (Mat_<float>(2,2) << 0, 0, 0, 0);
        transpose(Ja, Ja_transp);

        // compute the covariance matrix
        Mat sigmaxx = (Mat_<float>(2, 2) << 1, 0, 0, 1);
        Mat S = Ja * sigmaxx;
        S = S * Ja_transp;

        Mat sigmaSS = (-Ck.z)*(-Ck.z)*S;
        Mat sigmaUU = (dist - Ck.z)*(dist - Ck.z)*S;
        Mat sigmaUS = (-Ck.z)*(dist - Ck.z)*S;
        Mat sigmaSU = (-Ck.z)*(dist - Ck.z)*S;

        Mat H1 = (Mat_<float>(2, 2) << 0, 0, 0, 0);
        Mat H2 = (Mat_<float>(2, 2) << 0, 0, 0, 0);
        hconcat(sigmaUU, sigmaUS, H1);
        hconcat(sigmaSU, sigmaSS, H2);

        Mat sigma = (Mat_<float>(4, 4) << 0, 0, 0, 0, 0, 0, 0, 0);
        vconcat(H1, H2, sigma);

        SVD svd(sigma);

        eigenValues[2*k+0] = svd.w.at<float>(0, 0);
        eigenValues[2*k+1] = svd.w.at<float>(1, 0);

        eigenVectors[8*k+0] = svd.u.at<float>(0, 0);
        eigenVectors[8*k+1] = svd.u.at<float>(1, 0);
        eigenVectors[8*k+2] = svd.u.at<float>(2, 0);
        eigenVectors[8*k+3] = svd.u.at<float>(3, 0);

        eigenVectors[8*k+4] = svd.u.at<float>(0, 1);
        eigenVectors[8*k+5] = svd.u.at<float>(1, 1);
        eigenVectors[8*k+6] = svd.u.at<float>(2, 1);
        eigenVectors[8*k+7] = svd.u.at<float>(3, 1);
    }
}


// Polynomial degree 1, 9 parameters x = [rs, rt, r0 ; gs, gt, g0 ; bs, bt, b0]
struct residualIntensity {

    residualIntensity(const std::vector<float>& sK, const std::vector<float>& colorK)
        : _sK(sK), _colorK(colorK) {}

    template <typename T>
    bool operator()(const T* const x, T* residual) const {

        const T r = T(_colorK[0]);
        const T g = T(_colorK[1]);
        const T b = T(_colorK[2]);

        const T rs = T(x[0]);
        const T rt = T(x[1]);
        const T r0 = T(x[2]);
        const T gs = T(x[3]);
        const T gt = T(x[4]);
        const T g0 = T(x[5]);
        const T bs = T(x[6]);
        const T bt = T(x[7]);
        const T b0 = T(x[8]);

        const T s = T(_sK[0]);
        const T t = T(_sK[1]);

        residual[0] = r - rs*s - rt*t - r0;
        residual[1] = g - gs*s - gt*t - g0;
        residual[2] = b - bs*s - bt*t - b0;

        return true;
    }

private:
    // Observations for a sample.
    const std::vector<float> _sK;
    const std::vector<float> _colorK;
};

void colorRegression(int nbSamples, const std::vector<cv::Point2f> &flow, const std::vector<cv::Point3f>& colorSampleSet,
                     const std::vector<cv::Mat> &K_inv, const std::vector<cv::Mat> &R_transp, const std::vector<cv::Point3f> &C,
                     std::vector<float> &x, float &finalCost, bool verbose) {

    // residual parameters
    std::vector<float> S(nbSamples*2);

    // Build the problem.
    Problem problem;
    const uint nbParams = x.size();

    // init model parameters
    const double initialValue = 0.0;
    std::vector<double> xd(nbParams);
    for(uint i = 0 ; i < xd.size() ; ++i) {

        xd[i] = initialValue;
    }

    if(verbose) {
        std::cout << "COLOR REGRESSION, POLYNOME OF DEGREE 1, " << nbParams << " PARAMETERS" << std::endl;
    }

    for (int k = 0; k < nbSamples; ++k) {

        const cv::Mat K_invk = K_inv[k];
        const cv::Mat R_transpk = R_transp[k];
        const cv::Point3f Ck = C[k];

        // lightfield parametrization
        Mat ray = R_transpk * K_invk * Mat(cv::Point3f(flow[k].x, flow[k].y, 1.0));
        Mat a = (Mat_<float>(2, 1) << ray.at<float>(0, 0)/ray.at<float>(2, 0),
                 ray.at<float>(1, 0)/ray.at<float>(2, 0));

        S[2*k+0] = Ck.x + a.at<float>(0, 0)*(-Ck.z); // s
        S[2*k+1] = Ck.y + a.at<float>(1, 0)*(-Ck.z); // t
        assert(S[2*k+0] == S[2*k+0]);
        assert(S[2*k+1] == S[2*k+1]);
        assert(colorSampleSet[k].x == colorSampleSet[k].x);
        assert(colorSampleSet[k].y == colorSampleSet[k].y);
        assert(colorSampleSet[k].z == colorSampleSet[k].z);

        std::vector<float> sK = {S[2*k+0], S[2*k+1]};
        std::vector<float> colorK = {colorSampleSet[k].x, colorSampleSet[k].y, colorSampleSet[k].z};

        CostFunction* costFunction;

        costFunction =
                new AutoDiffCostFunction<residualIntensity, 3, 9>(
                    new residualIntensity(sK, colorK));
        problem.AddResidualBlock(costFunction, NULL, xd.data());
    }

    // Run the solver!
    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = verbose;
    options.max_num_iterations = 200;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    finalCost = summary.final_cost;

    for(uint i = 0 ; i < xd.size() ; ++i) {

        x[i] = (float)xd[i];
    }

    if(verbose) {
        std::cout << summary.FullReport() << std::endl;
        for(uint i = 0 ; i < nbParams ; ++i) {
            std::cout << "parameter " << i << ": " << initialValue << " -> " << x[i] << std::endl;
        }
    }
}

void triangulationLF(int nbSamples, const std::vector<cv::Point2f> &flow,
                     const std::vector<cv::Mat> &K_inv, const std::vector<cv::Mat> &R_transp, const std::vector<cv::Point3f> &C,
                     std::vector<float> &x, float &finalCost, float &conditionNumber, bool verbose) {

    // residual parameters
    std::vector<float> eigenVectors(nbSamples*8), eigenValues(nbSamples*2), samplePoint(nbSamples*4);

    computeResidualParameters(nbSamples, flow, K_inv, R_transp, C,
                              eigenVectors, eigenValues, samplePoint);

//    // perform DLT to initialize optimization
//    DLT(nbSamples, samplePoint, x, conditionNumber);

    // perform inhomogeneous method to initialize optimization
    IHM(nbSamples, samplePoint, x, conditionNumber);

    // main optimization (non-linear least square)
    optimize(nbSamples, eigenVectors, eigenValues, samplePoint, x, finalCost, verbose);
}

void triangulationClassic(int nbSamples, const std::vector<cv::Point2f> &flow,
                          const std::vector<cv::Mat> &K, const std::vector<cv::Mat> &R, const std::vector<cv::Point3f> &t,
                          std::vector<float> &point3D, float &finalCost, bool verbose) {

    std::vector<float> observation(2*nbSamples);

    std::vector<float> Kvec(nbSamples*9);
    std::vector<float> Rvec(nbSamples*9);
    std::vector<float> tvec(nbSamples*3);

    for (int k = 0; k < nbSamples; ++k) {

        observation[2*k+0] = flow[k].x;
        observation[2*k+1] = flow[k].y;

        Kvec[9*k+0] = K[k].at<float>(0, 0); Kvec[9*k+1] = K[k].at<float>(0, 1); Kvec[9*k+2] = K[k].at<float>(0, 2);
        Kvec[9*k+3] = K[k].at<float>(1, 0); Kvec[9*k+4] = K[k].at<float>(1, 1); Kvec[9*k+5] = K[k].at<float>(1, 2);
        Kvec[9*k+6] = K[k].at<float>(2, 0); Kvec[9*k+7] = K[k].at<float>(2, 1); Kvec[9*k+8] = K[k].at<float>(2, 2);

        Rvec[9*k+0] = R[k].at<float>(0, 0); Rvec[9*k+1] = R[k].at<float>(0, 1); Rvec[9*k+2] = R[k].at<float>(0, 2);
        Rvec[9*k+3] = R[k].at<float>(1, 0); Rvec[9*k+4] = R[k].at<float>(1, 1); Rvec[9*k+5] = R[k].at<float>(1, 2);
        Rvec[9*k+6] = R[k].at<float>(2, 0); Rvec[9*k+7] = R[k].at<float>(2, 1); Rvec[9*k+8] = R[k].at<float>(2, 2);

        tvec[3*k+0] = t[k].x; tvec[3*k+1] = t[k].y; tvec[3*k+2] = t[k].z;

        if(false) { // verbose

            std::cout << "Kvec: (" << Kvec[9*k+0] << ", " << Kvec[9*k+1] << ", " << Kvec[9*k+2] << ", "
                      << Kvec[9*k+3] << ", " << Kvec[9*k+4] << ", " << Kvec[9*k+5] << ", "
                      << Kvec[9*k+6] << ", " << Kvec[9*k+7] << ", " << Kvec[9*k+8] << ")" << std::endl;

            std::cout << "Rvec: (" << Rvec[9*k+0] << ", " << Rvec[9*k+1] << ", " << Rvec[9*k+2] << ", "
                      << Rvec[9*k+3] << ", " << Rvec[9*k+4] << ", " << Rvec[9*k+5] << ", "
                      << Rvec[9*k+6] << ", " << Rvec[9*k+7] << ", " << Rvec[9*k+8] << ")" << std::endl;

            std::cout << "tvec: (" << tvec[3*k+0] << ", " << tvec[3*k+1] << ", " << tvec[3*k+2] << ")" << std::endl;
        }
    }

    triangulateSamplesClassic(nbSamples, observation, Kvec, Rvec, tvec, point3D, finalCost, verbose);
}

void reprojectionCompute(const std::vector<float> &point3D, int nbSamples, const std::vector<cv::Point2f> &flow,
                         const std::vector<cv::Mat> &K, const std::vector<cv::Mat> &R, const std::vector<cv::Point3f> &t,
                         float &error) {

    Mat X = (Mat_<float>(3,1) << point3D[0], point3D[1], point3D[2]);

    error = 0.0;

    for(int k = 0 ; k < nbSamples ; ++k) {

        const std::vector<float> imagePointObserved = {flow[k].x,flow[k].y};
        std::vector<float> imagePointEstimates = {0.0, 0.0};

        Mat x = R[k]*X + Mat(t[k]);
        x = K[k]*x;
        imagePointEstimates[0] = x.at<float>(0, 0)/x.at<float>(2, 0);
        imagePointEstimates[1] = x.at<float>(1, 0)/x.at<float>(2, 0);

        error += (imagePointEstimates[0] - imagePointObserved[0])*(imagePointEstimates[0] - imagePointObserved[0]) +
                (imagePointEstimates[1] - imagePointObserved[1])*(imagePointEstimates[1] - imagePointObserved[1]);
    }
}

void reprojectionCheck(const std::vector<float> &point3D, int nbSamples,
                       const std::vector<cv::Mat> &K, const std::vector<cv::Mat> &R, const std::vector<cv::Point3f> &t) {

    Mat X = (Mat_<float>(3,1) << point3D[0], point3D[1], point3D[2]);

    for(int k = 0 ; k < nbSamples ; ++k) {

        const std::vector<float> imagePointObserved = {testFlow[k].x,testFlow[k].y};
        std::vector<float> imagePointEstimates = {0.0, 0.0};

        Mat x = R[k]*X + Mat(t[k]);
        x = K[k]*x;
        imagePointEstimates[0] = x.at<float>(0, 0)/x.at<float>(2, 0);
        imagePointEstimates[1] = x.at<float>(1, 0)/x.at<float>(2, 0);

        float error = (imagePointEstimates[0] - imagePointObserved[0])*(imagePointEstimates[0] - imagePointObserved[0]) +
                (imagePointEstimates[1] - imagePointObserved[1])*(imagePointEstimates[1] - imagePointObserved[1]);

        std::cout << "Reprojection error of sample " << k << ": " << error << std::endl;
    }
}

// for each block residual:
// residual is scalar
// there are 3 parameters (observation)
// - eigen vectors (2*4 elements), TODO: reduce dimension
// - eigen values (2 elements)
// - sample points (1*4 elements)

// all 3 parameters (alpha, beta and gamma) are scalar

/*
 * there are 2 eigenvectors, e1 and e2
 * their coordinates are stored as [e1u, e1v, e1s, e1t, e2u, e2v, e2s, e2t]
 * there are 2 eigenvalues, l1 and l2, stored as [l1, l2]
 * there are 4 sample point elements, [pu, pv, ps, pt]
 * */

template <typename T>
float testResidual(const T* const alpha, const T* const beta, const T* const gamma, T* residual,
                   float _eigenVectors[], float _eigenValues[], float _samplePoint[]) {

    const T e1u = T(_eigenVectors[0]);
    const T e1v = T(_eigenVectors[1]);
    const T e1s = T(_eigenVectors[2]);
    const T e1t = T(_eigenVectors[3]);

    const T e2u = T(_eigenVectors[4]);
    const T e2v = T(_eigenVectors[5]);
    const T e2s = T(_eigenVectors[6]);
    const T e2t = T(_eigenVectors[7]);

    const T l1 = T(_eigenValues[0]);
    const T l2 = T(_eigenValues[1]);

    const T pu = T(_samplePoint[0]);
    const T pv = T(_samplePoint[1]);
    const T ps = T(_samplePoint[2]);
    const T pt = T(_samplePoint[3]);

    const T D = (e1u - *alpha*e1s)*(e2v - *alpha*e2t) - (e2u - *alpha*e2s)*(e1v - *alpha*e1t);
    const T mu1 = ((e2v - *alpha*e2t)*(*alpha*ps + *beta - pu) - (e2u - *alpha*e2s)*(*alpha*pt + *gamma - pv))/D;
    const T mu2 = ((e1u - *alpha*e1s)*(*alpha*pt + *gamma - pv) - (e1v - *alpha*e1t)*(*alpha*ps + *beta - pu))/D;
    *residual = mu1*mu1/l1 + mu2*mu2/l2;

    return true;
}

// Lambertian model, polynomial degree 1, 3 parameters x = (a, bu, bv)
struct residual3param {

    residual3param(const std::vector<float>& eigenVectors, const std::vector<float>& eigenValues, const std::vector<float>& samplePoint)
        : _eigenVectors(eigenVectors), _eigenValues(eigenValues), _samplePoint(samplePoint) {}

    template <typename T>
    bool operator()(const T* const x, T* residual) const {

        const T e1u = T(_eigenVectors[0]);
        const T e1v = T(_eigenVectors[1]);
        const T e1s = T(_eigenVectors[2]);
        const T e1t = T(_eigenVectors[3]);

        const T e2u = T(_eigenVectors[4]);
        const T e2v = T(_eigenVectors[5]);
        const T e2s = T(_eigenVectors[6]);
        const T e2t = T(_eigenVectors[7]);

        const T l1 = T(ceres::sqrt(_eigenValues[0]));
        const T l2 = T(ceres::sqrt(_eigenValues[1]));

        const T pu = T(_samplePoint[0]);
        const T pv = T(_samplePoint[1]);
        const T ps = T(_samplePoint[2]);
        const T pt = T(_samplePoint[3]);

        const T D = (e1u - x[0]*e1s)*(e2v - x[0]*e2t) - (e2u - x[0]*e2s)*(e1v - x[0]*e1t);
        const T mu1 = ((e2v - x[0]*e2t)*(x[0]*ps + x[1] - pu) - (e2u - x[0]*e2s)*(x[0]*pt + x[2] - pv))/D;
        const T mu2 = ((e1u - x[0]*e1s)*(x[0]*pt + x[2] - pv) - (e1v - x[0]*e1t)*(x[0]*ps + x[1] - pu))/D;
        residual[0] = mu1/l1;
        residual[1] = mu2/l2;
        // ||r||^2 = mu1^2/l1 + mu2^2/l2;
        // ceres minimizes 1/2*||r||^2

        return true;
    }

private:
    // Observations for a sample.
    const std::vector<float> _eigenVectors;
    const std::vector<float> _eigenValues;
    const std::vector<float> _samplePoint;
};

// Model 2 polynomial degree 1, 4 parameters x = (au, av, bu, bv)
struct residual4param {

    residual4param(const std::vector<float>& eigenVectors, const std::vector<float>& eigenValues, const std::vector<float>& samplePoint)
        : _eigenVectors(eigenVectors), _eigenValues(eigenValues), _samplePoint(samplePoint) {}

    template <typename T>
    bool operator()(const T* const x, T* residual) const {

        const T e1u = T(_eigenVectors[0]);
        const T e1v = T(_eigenVectors[1]);
        const T e1s = T(_eigenVectors[2]);
        const T e1t = T(_eigenVectors[3]);

        const T e2u = T(_eigenVectors[4]);
        const T e2v = T(_eigenVectors[5]);
        const T e2s = T(_eigenVectors[6]);
        const T e2t = T(_eigenVectors[7]);

        const T l1 = T(ceres::sqrt(_eigenValues[0]));
        const T l2 = T(ceres::sqrt(_eigenValues[1]));

        const T pu = T(_samplePoint[0]);
        const T pv = T(_samplePoint[1]);
        const T ps = T(_samplePoint[2]);
        const T pt = T(_samplePoint[3]);

        const T D = (e1u - x[0]*e1s)*(e2v - x[1]*e2t) - (e2u - x[0]*e2s)*(e1v - x[1]*e1t);
        const T mu1 = ((e2v - x[1]*e2t)*(x[0]*ps + x[2] - pu) - (e2u - x[0]*e2s)*(x[1]*pt + x[3] - pv))/D;
        const T mu2 = ((e1u - x[0]*e1s)*(x[1]*pt + x[3] - pv) - (e1v - x[1]*e1t)*(x[0]*ps + x[2] - pu))/D;
        residual[0] = mu1/l1;
        residual[1] = mu2/l2;
        // ||r||^2 = mu1^2/l1 + mu2^2/l2;
        // ceres minimizes 1/2*||r||^2

        return true;
    }

private:
    // Observations for a sample.
    const std::vector<float> _eigenVectors;
    const std::vector<float> _eigenValues;
    const std::vector<float> _samplePoint;
};

// Model 3 polynomial degree 1, 6 parameters x = (aus, aut, avs, avt, bu, bv)
struct residual6param {

    residual6param(const std::vector<float>& eigenVectors, const std::vector<float>& eigenValues, const std::vector<float>& samplePoint)
        : _eigenVectors(eigenVectors), _eigenValues(eigenValues), _samplePoint(samplePoint) {}

    template <typename T>
    bool operator()(const T* const x, T* residual) const {

        const T e1u = T(_eigenVectors[0]);
        const T e1v = T(_eigenVectors[1]);
        const T e1s = T(_eigenVectors[2]);
        const T e1t = T(_eigenVectors[3]);

        const T e2u = T(_eigenVectors[4]);
        const T e2v = T(_eigenVectors[5]);
        const T e2s = T(_eigenVectors[6]);
        const T e2t = T(_eigenVectors[7]);

        const T l1 = T(ceres::sqrt(_eigenValues[0]));
        const T l2 = T(ceres::sqrt(_eigenValues[1]));

        const T pu = T(_samplePoint[0]);
        const T pv = T(_samplePoint[1]);
        const T ps = T(_samplePoint[2]);
        const T pt = T(_samplePoint[3]);

        const T aus = x[0];
        const T aut = x[1];
        const T avs = x[2];
        const T avt = x[3];
        const T bu = x[4];
        const T bv = x[5];

        const T D = (e1u - aus*e1s - aut*e1t)*(e2v - avs*e2s - avt*e2t) - (e2u - aus*e2s - aut*e2t)*(e1v - avs*e1s - avt*e1t);
        const T mu1 = ((e2v - avs*e2s - avt*e2t)*(aus*ps + aut*pt + bu - pu) - (e2u - aus*e2s - aut*e2t)*(avs*ps + avt*pt + bv - pv))/D;
        const T mu2 = ((e1u - aus*e1s - aut*e1t)*(avs*ps + avt*pt + bv - pv) - (e1v - avs*e1s - avt*e1t)*(aus*ps + aut*pt + bu - pu))/D;
        residual[0] = mu1/l1;
        residual[1] = mu2/l2;
        // ||r||^2 = mu1^2/l1 + mu2^2/l2;
        // ceres minimizes 1/2*||r||^2

        return true;
    }

private:
    // Observations for a sample.
    const std::vector<float> _eigenVectors;
    const std::vector<float> _eigenValues;
    const std::vector<float> _samplePoint;
};

void printEigen(const std::vector<float>& eigenVectors,
                const std::vector<float>& eigenValues,
                const std::vector<float>& samplePoint) {

    std::cout << "eigenVectors: (" << eigenVectors[0] << ", "
              << eigenVectors[1] << ", "
              << eigenVectors[2] << ", "
              << eigenVectors[3] << ")   ("
              << eigenVectors[4] << ", "
              << eigenVectors[5] << ", "
              << eigenVectors[6] << ", "
              << eigenVectors[7] << ") " << std::endl;

    std::cout << "eigenValues: (" << eigenValues[0] << ", "
              << eigenValues[1] << ") " << std::endl;

    std::cout << "samplePoint: (" << samplePoint[0] << ", "
              << samplePoint[1] << ", "
              << samplePoint[2] << ", "
              << samplePoint[3] << ")" << std::endl;
}

void DLT(uint nbSamples,
         const std::vector<float>& samplePoint,
         std::vector<float> &x,
         float& conditionNumber) {

    const uint nbParams = x.size();

    switch(nbParams) {

    case 3:
    {
        cv::Mat M4 = cv::Mat::zeros(2*nbSamples, 4, CV_32FC1); // 3g + 1 (homogeneous)

        // fill system matrix
        for(uint k(0) ; k < nbSamples ; ++k) {

            M4.at<float>(2*k + 0, 0) = samplePoint[4*k+2]; M4.at<float>(2*k + 0, 1) = 1; M4.at<float>(2*k + 0, 2) = 0; M4.at<float>(2*k + 0, 3) = -samplePoint[4*k+0];
            M4.at<float>(2*k + 1, 0) = samplePoint[4*k+3]; M4.at<float>(2*k + 1, 1) = 0; M4.at<float>(2*k + 1, 2) = 1; M4.at<float>(2*k + 1, 3) = -samplePoint[4*k+1];
        }

        cv::SVD svd4(M4);

        //        assert(svd4.vt.at<float>(svd4.vt.rows-1, 3) > 0.00001);
        x[0] = svd4.vt.at<float>(svd4.vt.rows-1, 0)/svd4.vt.at<float>(svd4.vt.rows-1, 3);
        x[1] = svd4.vt.at<float>(svd4.vt.rows-1, 1)/svd4.vt.at<float>(svd4.vt.rows-1, 3);
        x[2] = svd4.vt.at<float>(svd4.vt.rows-1, 2)/svd4.vt.at<float>(svd4.vt.rows-1, 3);

        conditionNumber = svd4.w.at<float>(0, 0)/svd4.w.at<float>(svd4.w.rows-1, 0);
    }
        break;

    case 4:
    {
        cv::Mat M5 = cv::Mat::zeros(2*nbSamples, 5, CV_32FC1); // 4g + 1 (homogeneous)

        // fill system matrix
        for(uint k(0) ; k < nbSamples ; ++k) {

            M5.at<float>(2*k + 0, 0) = samplePoint[4*k+2]; M5.at<float>(2*k + 0, 1) = 0; M5.at<float>(2*k + 0, 2) = 1; M5.at<float>(2*k + 0, 3) = 0; M5.at<float>(2*k + 0, 4) = -samplePoint[4*k+0];
            M5.at<float>(2*k + 1, 0) = 0; M5.at<float>(2*k + 1, 1) = samplePoint[4*k+3]; M5.at<float>(2*k + 1, 2) = 0; M5.at<float>(2*k + 1, 3) = 1; M5.at<float>(2*k + 1, 4) = -samplePoint[4*k+1];
        }

        cv::SVD svd5(M5);

        //        assert(svd5.vt.at<float>(svd5.vt.rows-1, 4) > 0.00001);
        x[0] = svd5.vt.at<float>(svd5.vt.rows-1, 0)/svd5.vt.at<float>(svd5.vt.rows-1, 4);
        x[1] = svd5.vt.at<float>(svd5.vt.rows-1, 1)/svd5.vt.at<float>(svd5.vt.rows-1, 4);
        x[2] = svd5.vt.at<float>(svd5.vt.rows-1, 2)/svd5.vt.at<float>(svd5.vt.rows-1, 4);
        x[3] = svd5.vt.at<float>(svd5.vt.rows-1, 3)/svd5.vt.at<float>(svd5.vt.rows-1, 4);

        conditionNumber = svd5.w.at<float>(0, 0)/svd5.w.at<float>(svd5.w.rows-1, 0);
    }
        break;

    case 6:
    {
        cv::Mat M7 = cv::Mat::zeros(2*nbSamples, 7, CV_32FC1); // 6g + 1 (homogeneous)

        // fill system matrix
        for(uint k(0) ; k < nbSamples ; ++k) {

            M7.at<float>(2*k + 0, 0) = samplePoint[4*k+2]; M7.at<float>(2*k + 0, 1) = samplePoint[4*k+3]; M7.at<float>(2*k + 0, 2) = 0; M7.at<float>(2*k + 0, 3) = 0; M7.at<float>(2*k + 0, 4) = 1; M7.at<float>(2*k + 0, 5) = 0; M7.at<float>(2*k + 0, 6) = -samplePoint[4*k+0];
            M7.at<float>(2*k + 1, 0) = 0; M7.at<float>(2*k + 1, 1) = 0; M7.at<float>(2*k + 1, 2) = samplePoint[4*k+2]; M7.at<float>(2*k + 1, 3) = samplePoint[4*k+3]; M7.at<float>(2*k + 1, 4) = 0; M7.at<float>(2*k + 1, 5) = 1; M7.at<float>(2*k + 1, 6) = -samplePoint[4*k+1];
        }

        cv::SVD svd7(M7);

        //        assert(svd7.vt.at<float>(svd7.vt.rows-1, 6) > 0.00001);
        x[0] = svd7.vt.at<float>(svd7.vt.rows-1, 0)/svd7.vt.at<float>(svd7.vt.rows-1, 6);
        x[1] = svd7.vt.at<float>(svd7.vt.rows-1, 1)/svd7.vt.at<float>(svd7.vt.rows-1, 6);
        x[2] = svd7.vt.at<float>(svd7.vt.rows-1, 2)/svd7.vt.at<float>(svd7.vt.rows-1, 6);
        x[3] = svd7.vt.at<float>(svd7.vt.rows-1, 3)/svd7.vt.at<float>(svd7.vt.rows-1, 6);
        x[4] = svd7.vt.at<float>(svd7.vt.rows-1, 4)/svd7.vt.at<float>(svd7.vt.rows-1, 6);
        x[5] = svd7.vt.at<float>(svd7.vt.rows-1, 5)/svd7.vt.at<float>(svd7.vt.rows-1, 6);

        conditionNumber = svd7.w.at<float>(0, 0)/svd7.w.at<float>(svd7.w.rows-1, 0);
    }
        break;

    default:
        assert(false);
        break;
    }
}

void IHM(uint nbSamples,
         const std::vector<float>& samplePoint,
         std::vector<float> &x,
         float& conditionNumber) {

    const uint nbParams = x.size();

    switch(nbParams) {

    case 3:
    {
        cv::Mat M3 = cv::Mat::zeros(2*nbSamples, 3, CV_32FC1); // 3g (inhomogeneous)
        cv::Mat b3 = cv::Mat::zeros(2*nbSamples, 1, CV_32FC1);

        // fill system matrix
        for(uint k(0) ; k < nbSamples ; ++k) {

            M3.at<float>(2*k + 0, 0) = samplePoint[4*k+2]; M3.at<float>(2*k + 0, 1) = 1; M3.at<float>(2*k + 0, 2) = 0;
            M3.at<float>(2*k + 1, 0) = samplePoint[4*k+3]; M3.at<float>(2*k + 1, 1) = 0; M3.at<float>(2*k + 1, 2) = 1;
            b3.at<float>(2*k + 0, 0) = samplePoint[4*k+0];
            b3.at<float>(2*k + 1, 0) = samplePoint[4*k+1];
        }

        cv::Mat M3t = cv::Mat::zeros(2*nbSamples, 3, CV_32FC1);
        cv::transpose(M3, M3t);
        cv::Mat M3tM3 = M3t * M3;
        cv::Mat M3tM3_inv = cv::Mat::zeros(2*nbSamples, 3, CV_32FC1);
        cv::invert(M3tM3, M3tM3_inv);
        cv::Mat res3 = M3tM3_inv * (M3t * b3);
        x[0] = res3.at<float>(0, 0);
        x[1] = res3.at<float>(1, 0);
        x[2] = res3.at<float>(2, 0);

        cv::SVD svd3(M3);
        conditionNumber = svd3.w.at<float>(0, 0)/svd3.w.at<float>(svd3.w.rows-1, 0);
    }
        break;

    case 4:
    {
        cv::Mat M4 = cv::Mat::zeros(2*nbSamples, 4, CV_32FC1); // 4g (inhomogeneous)
        cv::Mat b4 = cv::Mat::zeros(2*nbSamples, 1, CV_32FC1);

        // fill system matrix
        for(uint k(0) ; k < nbSamples ; ++k) {

            M4.at<float>(2*k + 0, 0) = samplePoint[4*k+2]; M4.at<float>(2*k + 0, 1) = 0; M4.at<float>(2*k + 0, 2) = 1; M4.at<float>(2*k + 0, 3) = 0;
            M4.at<float>(2*k + 1, 0) = 0; M4.at<float>(2*k + 1, 1) = samplePoint[4*k+3]; M4.at<float>(2*k + 1, 2) = 0; M4.at<float>(2*k + 1, 3) = 1;
            b4.at<float>(2*k + 0, 0) = samplePoint[4*k+0];
            b4.at<float>(2*k + 1, 0) = samplePoint[4*k+1];
        }

        cv::Mat M4t = cv::Mat::zeros(2*nbSamples, 4, CV_32FC1);
        cv::transpose(M4, M4t);
        cv::Mat M4tM4 = M4t * M4;
        cv::Mat M4tM4_inv = cv::Mat::zeros(2*nbSamples, 4, CV_32FC1);
        cv::invert(M4tM4, M4tM4_inv);
        cv::Mat res4 = M4tM4_inv * (M4t * b4);
        x[0] = res4.at<float>(0, 0);
        x[1] = res4.at<float>(1, 0);
        x[2] = res4.at<float>(2, 0);
        x[3] = res4.at<float>(3, 0);

        cv::SVD svd4(M4);
        conditionNumber = svd4.w.at<float>(0, 0)/svd4.w.at<float>(svd4.w.rows-1, 0);
    }
        break;

    case 6:
    {
        cv::Mat M6 = cv::Mat::zeros(2*nbSamples, 6, CV_32FC1); // 6g (inhomogeneous)
        cv::Mat b6 = cv::Mat::zeros(2*nbSamples, 1, CV_32FC1);

        // fill system matrix
        for(uint k(0) ; k < nbSamples ; ++k) {

            M6.at<float>(2*k + 0, 0) = samplePoint[4*k+2]; M6.at<float>(2*k + 0, 1) = samplePoint[4*k+3]; M6.at<float>(2*k + 0, 2) = 0; M6.at<float>(2*k + 0, 3) = 0; M6.at<float>(2*k + 0, 4) = 1; M6.at<float>(2*k + 0, 5) = 0;
            M6.at<float>(2*k + 1, 0) = 0; M6.at<float>(2*k + 1, 1) = 0; M6.at<float>(2*k + 1, 2) = samplePoint[4*k+2]; M6.at<float>(2*k + 1, 3) = samplePoint[4*k+3]; M6.at<float>(2*k + 1, 4) = 0; M6.at<float>(2*k + 1, 5) = 1;
            b6.at<float>(2*k + 0, 0) = samplePoint[4*k+0];
            b6.at<float>(2*k + 1, 0) = samplePoint[4*k+1];
        }

        cv::Mat M6t = cv::Mat::zeros(2*nbSamples, 6, CV_32FC1);
        cv::transpose(M6, M6t);
        cv::Mat M6tM6 = M6t * M6;
        cv::Mat M6tM6_inv = cv::Mat::zeros(2*nbSamples, 6, CV_32FC1);
        cv::invert(M6tM6, M6tM6_inv);
        cv::Mat res6 = M6tM6_inv * (M6t * b6);
        x[0] = res6.at<float>(0, 0);
        x[1] = res6.at<float>(1, 0);
        x[2] = res6.at<float>(2, 0);
        x[3] = res6.at<float>(3, 0);
        x[4] = res6.at<float>(4, 0);
        x[5] = res6.at<float>(5, 0);

        cv::SVD svd6(M6);
        conditionNumber = svd6.w.at<float>(0, 0)/svd6.w.at<float>(svd6.w.rows-1, 0);
    }
        break;

    default:
        assert(false);
        break;
    }
}

void optimize(int kNumObservations,
              const std::vector<float>& eigenVectors,
              const std::vector<float>& eigenValues,
              const std::vector<float>& samplePoint,
              std::vector<float> &x, float &finalCost, bool verbose) {

    // Build the problem.
    Problem problem;
    const uint nbParams = x.size();

    // init model parameters
    std::vector<double> xd(nbParams);
    for(uint i = 0 ; i < xd.size() ; ++i) {

        xd[i] = (double) x[i];
    }

    if(verbose) {
        std::cout << "NEW LF TRIANGULATION, POLYNOME OF DEGREE 1, " << nbParams << " PARAMETERS" << std::endl;
    }

    for (int k = 0; k < kNumObservations; ++k) {

        std::vector<float> eigenVectorsk = {eigenVectors[8*k+0], eigenVectors[8*k+1], eigenVectors[8*k+2], eigenVectors[8*k+3],
                                            eigenVectors[8*k+4], eigenVectors[8*k+5], eigenVectors[8*k+6], eigenVectors[8*k+7]};

        std::vector<float> eigenValuesk = {eigenValues[2*k+0], eigenValues[2*k+1]};

        std::vector<float> samplePointk = {samplePoint[4*k+0], samplePoint[4*k+1], samplePoint[4*k+2], samplePoint[4*k+3]};

        CostFunction* costFunction;
        switch(nbParams) {

        case 3:
            costFunction =
                    new AutoDiffCostFunction<residual3param, 2, 3>(
                        new residual3param(eigenVectorsk, eigenValuesk, samplePointk));
            problem.AddResidualBlock(costFunction, NULL, xd.data());
            break;

        case 4:
            costFunction =
                    new AutoDiffCostFunction<residual4param, 2, 4>(
                        new residual4param(eigenVectorsk, eigenValuesk, samplePointk));
            problem.AddResidualBlock(costFunction, NULL, xd.data());
            break;

        case 6:
            costFunction =
                    new AutoDiffCostFunction<residual6param, 2, 6>(
                        new residual6param(eigenVectorsk, eigenValuesk, samplePointk));
            problem.AddResidualBlock(costFunction, NULL, xd.data());
            break;

        default:
            assert(false);
            break;
        }
    }

    // Run the solver!
    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = verbose;
    options.max_num_iterations = 200;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    finalCost = summary.final_cost;

    if(verbose) {
        std::cout << summary.FullReport() << std::endl;
        for(uint i = 0 ; i < nbParams ; ++i) {
            std::cout << "parameter " << i << ": " << x[i] << " -> " << xd[i] << std::endl;
        }
    }

    for(uint i = 0 ; i < xd.size() ; ++i) {

        x[i] = (float)xd[i];
    }
}

void testLFTriangulation(int nbSamples,
                         const std::vector<cv::Mat> &K_inv, const std::vector<cv::Mat> &R_transp, const std::vector<cv::Point3f> &C,
                         std::vector<float> &point3D, float &finalCost) {

    // residual parameters
    std::vector<float> eigenVectors(nbSamples*8), eigenValues(nbSamples*2), samplePoint(nbSamples*4);

    std::cout << "EIGEN VECTORS AND VALUES AND SAMPLE POINTS IN TEST_TRIANGULATION" << std::endl;

    for(int k = 0 ; k < nbSamples ; ++k) {

        const cv::Mat K_invk = K_inv[k];
        const cv::Mat R_transpk = R_transp[k];
        const cv::Point3f Ck = C[k];

        // lightfield parametrization
        Mat ray = R_transpk * K_invk * Mat(cv::Point3f(testFlow[k].x, testFlow[k].y, 1.0));
        Mat a = (Mat_<float>(2, 1) << ray.at<float>(0, 0)/ray.at<float>(2, 0),
                 ray.at<float>(1, 0)/ray.at<float>(2, 0));

        std::cout << "Ck.x: " << Ck.x << "  Ck.y: " << Ck.y << "  Ck.z: " << Ck.z << std::endl;

        samplePoint[4*k+0] = Ck.x + a.at<float>(0, 0)*(1 - Ck.z);
        samplePoint[4*k+1] = Ck.y + a.at<float>(1, 0)*(1 - Ck.z);
        samplePoint[4*k+2] = Ck.x + a.at<float>(0, 0)*(-Ck.z);
        samplePoint[4*k+3] = Ck.y + a.at<float>(1, 0)*(-Ck.z);

        // compute the Jacobian
        Mat temp1 = (Mat_<float>(3, 2) << 1, 0, 0, 1, 0, 0);
        Mat temp2 = (Mat_<float>(2, 3) << 1, 0, -a.at<float>(0, 0), 0, 1, -a.at<float>(1, 0));
        temp1 = R_transpk * K_invk * temp1;
        Mat Ja = temp2*temp1 / ray.at<float>(2, 0);
        Mat Ja_transp = (Mat_<float>(2,2) << 0, 0, 0, 0);
        transpose(Ja, Ja_transp);

        // compute the covariance matrix
        Mat sigmaxx = (Mat_<float>(2, 2) << 1, 0, 0, 1);
        Mat S = Ja * sigmaxx;
        S = S * Ja_transp;

        Mat sigmaSS = (-Ck.z)*(-Ck.z)*S;
        Mat sigmaUU = (1 - Ck.z)*(1 - Ck.z)*S;
        Mat sigmaUS = (-Ck.z)*(1 - Ck.z)*S;
        Mat sigmaSU = (-Ck.z)*(1 - Ck.z)*S;

        Mat H1 = (Mat_<float>(2, 2) << 0, 0, 0, 0);
        Mat H2 = (Mat_<float>(2, 2) << 0, 0, 0, 0);
        hconcat(sigmaUU, sigmaUS, H1);
        hconcat(sigmaSU, sigmaSS, H2);

        Mat sigma = (Mat_<float>(4, 4) << 0, 0, 0, 0, 0, 0, 0, 0);
        vconcat(H1, H2, sigma);

        SVD svd(sigma);

        eigenValues[2*k+0] = svd.w.at<float>(0, 0);
        eigenValues[2*k+1] = svd.w.at<float>(1, 0);

        eigenVectors[8*k+0] = svd.u.at<float>(0, 0);
        eigenVectors[8*k+1] = svd.u.at<float>(1, 0);
        eigenVectors[8*k+2] = svd.u.at<float>(2, 0);
        eigenVectors[8*k+3] = svd.u.at<float>(3, 0);

        eigenVectors[8*k+4] = svd.u.at<float>(0, 1);
        eigenVectors[8*k+5] = svd.u.at<float>(1, 1);
        eigenVectors[8*k+6] = svd.u.at<float>(2, 1);
        eigenVectors[8*k+7] = svd.u.at<float>(3, 1);

        printEigen(eigenVectors, eigenValues, samplePoint);
    }

    // test cost function (evaluation)
    //    for (int k = 0; k < nbSamples; ++k) {

    //        float alpha(0.7), beta(0.0), gamma(0.0), residual(0.0);
    //        float eigenVectorsk[8] = {eigenVectors[k+0], eigenVectors[k+1], eigenVectors[k+2], eigenVectors[k+3],
    //                                   eigenVectors[k+4], eigenVectors[k+5], eigenVectors[k+6], eigenVectors[k+7]};

    //        float eigenValuesk[2] = {eigenValues[k+0], eigenValues[k+1]};

    //        float samplePointk[4] = {samplePoint[k+0], samplePoint[k+1], samplePoint[k+2], samplePoint[k+3]};

    //        testResidual<float>(&alpha, &beta, &gamma, &residual,
    //                             eigenVectorsk, eigenValuesk, samplePointk);

    //        std::cout << "Block nb " << k << " alpha: " << alpha << " beta: " << beta << " gamma: " << gamma << " residual: " << residual << std::endl;
    //    }

    std::vector<float> x(3);
    optimize(nbSamples, eigenVectors, eigenValues, samplePoint, x, finalCost);

    // compute 3D point
    point3D[0] = x[1]/(1-x[0]);
    point3D[1] = x[2]/(1-x[0]);
    point3D[2] = 1.0/(1-x[0]);
}

struct SnavelyReprojectionError {

    SnavelyReprojectionError(const std::vector<float>& observation,
                             const std::vector<float>& K,
                             const std::vector<float>& R,
                             const std::vector<float>& t)
        : _observation(observation), _K(K), _R(R), _t(t) {}

    template <typename T>
    bool operator()(const T* const point, T* residuals) const {

        const T temp1x = T(_R[0])*point[0] + T(_R[1])*point[1] + T(_R[2])*point[2] + T(_t[0]);
        const T temp1y = T(_R[3])*point[0] + T(_R[4])*point[1] + T(_R[5])*point[2] + T(_t[1]);
        const T temp1z = T(_R[6])*point[0] + T(_R[7])*point[1] + T(_R[8])*point[2] + T(_t[2]);

        const T temp2x = T(_K[0])*temp1x + T(_K[1])*temp1y + T(_K[2])*temp1z;
        const T temp2y = T(_K[3])*temp1x + T(_K[4])*temp1y + T(_K[5])*temp1z;
        const T temp2z = T(_K[6])*temp1x + T(_K[7])*temp1y + T(_K[8])*temp1z;

        const T predicted_x = temp2x / temp2z;
        const T predicted_y = temp2y / temp2z;

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - T(_observation[0]);
        residuals[1] = predicted_y - T(_observation[1]);

        return true;
    }

    const std::vector<float> _observation;
    const std::vector<float> _K;
    const std::vector<float> _R;
    const std::vector<float> _t;
};

void triangulateSamplesClassic(int kNumObservations, const std::vector<float>& observation,
                               const std::vector<float>& K,
                               const std::vector<float>& R,
                               const std::vector<float>& t,
                               std::vector<float> &point3D, float &finalCost, bool verbose) {

    // Build the problem.
    Problem problem;

    const double initX = 1.0;
    const double initY = 1.0;
    const double initZ = 1.0;

    // 3 parameters of a Lambertian point
    // initialization
    std::vector<double> point3Dd(3);
    point3Dd[0] = initX;
    point3Dd[1] = initY;
    point3Dd[2] = initZ;

    if(verbose) {
        std::cout << "CLASSIC TRIANGULATION" << std::endl;
    }

    for (int k = 0; k < kNumObservations; ++k) {

        std::vector<float> observationk = {observation[2*k+0], observation[2*k+1]};
        std::vector<float> Kk = {K[9*k+0], K[9*k+1], K[9*k+2],
                                 K[9*k+3], K[9*k+4], K[9*k+5],
                                 K[9*k+6], K[9*k+7], K[9*k+8]};
        std::vector<float> Rk = {R[9*k+0], R[9*k+1], R[9*k+2],
                                 R[9*k+3], R[9*k+4], R[9*k+5],
                                 R[9*k+6], R[9*k+7], R[9*k+8]};
        std::vector<float> tk = {t[3*k+0], t[3*k+1], t[3*k+2]};

        CostFunction* costFunction =
                new AutoDiffCostFunction<SnavelyReprojectionError, 2, 3>(
                    new SnavelyReprojectionError(observationk, Kk, Rk, tk));
        problem.AddResidualBlock(costFunction, NULL, point3Dd.data());
    }

    // Run the solver!
    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = verbose;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    finalCost = summary.final_cost;

    point3D[0] = (float)point3Dd[0];
    point3D[1] = (float)point3Dd[1];
    point3D[2] = (float)point3Dd[2];

    if(verbose) {
        std::cout << summary.BriefReport() << std::endl;
        std::cout << "X : " << initX << " -> " << point3D[0] << std::endl;
        std::cout << "Y : " << initY << " -> " << point3D[1] << std::endl;
        std::cout << "Z : " << initZ << " -> " << point3D[2] << std::endl;
    }
}

void testClassicTriangulation(int nbSamples,
                              const std::vector<cv::Mat> &K, const std::vector<cv::Mat> &R, const std::vector<cv::Point3f> &t,
                              std::vector<float> &point3D, float &finalCost) {

    const std::vector<float> observation = {testFlow[0].x, testFlow[0].y,
                                            testFlow[1].x, testFlow[1].y,
                                            testFlow[2].x, testFlow[2].y,
                                            testFlow[3].x, testFlow[3].y,
                                            testFlow[4].x, testFlow[4].y};

    std::vector<float> Kvec(nbSamples*9);
    std::vector<float> Rvec(nbSamples*9);
    std::vector<float> tvec(nbSamples*3);

    for (int k = 0; k < nbSamples; ++k) {

        Kvec[9*k+0] = K[k].at<float>(0, 0); Kvec[9*k+1] = K[k].at<float>(0, 1); Kvec[9*k+2] = K[k].at<float>(0, 2);
        Kvec[9*k+3] = K[k].at<float>(1, 0); Kvec[9*k+4] = K[k].at<float>(1, 1); Kvec[9*k+5] = K[k].at<float>(1, 2);
        Kvec[9*k+6] = K[k].at<float>(2, 0); Kvec[9*k+7] = K[k].at<float>(2, 1); Kvec[9*k+8] = K[k].at<float>(2, 2);

        Rvec[9*k+0] = R[k].at<float>(0, 0); Rvec[9*k+1] = R[k].at<float>(0, 1); Rvec[9*k+2] = R[k].at<float>(0, 2);
        Rvec[9*k+3] = R[k].at<float>(1, 0); Rvec[9*k+4] = R[k].at<float>(1, 1); Rvec[9*k+5] = R[k].at<float>(1, 2);
        Rvec[9*k+6] = R[k].at<float>(2, 0); Rvec[9*k+7] = R[k].at<float>(2, 1); Rvec[9*k+8] = R[k].at<float>(2, 2);

        tvec[3*k+0] = t[k].x; tvec[3*k+1] = t[k].y; tvec[3*k+2] = t[k].z;

        std::cout << "Kvec: (" << Kvec[9*k+0] << ", " << Kvec[9*k+1] << ", " << Kvec[9*k+2] << ", "
                  << Kvec[9*k+3] << ", " << Kvec[9*k+4] << ", " << Kvec[9*k+5] << ", "
                  << Kvec[9*k+6] << ", " << Kvec[9*k+7] << ", " << Kvec[9*k+8] << ")" << std::endl;

        std::cout << "Rvec: (" << Rvec[9*k+0] << ", " << Rvec[9*k+1] << ", " << Rvec[9*k+2] << ", "
                  << Rvec[9*k+3] << ", " << Rvec[9*k+4] << ", " << Rvec[9*k+5] << ", "
                  << Rvec[9*k+6] << ", " << Rvec[9*k+7] << ", " << Rvec[9*k+8] << ")" << std::endl;

        std::cout << "tvec: (" << tvec[3*k+0] << ", " << tvec[3*k+1] << ", " << tvec[3*k+2] << ")" << std::endl;
    }

    triangulateSamplesClassic(nbSamples, observation, Kvec, Rvec, tvec, point3D, finalCost);
}




