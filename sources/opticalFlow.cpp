#include <iomanip>

#include "opticalFlow.h"

#include "optical_flow/CvUtil.h"
#include "optical_flow/NovelView.h"
#include "optical_flow/OpticalFlowVisualization.h"
#include "optical_flow/SystemUtil.h"
#include "optical_flow/VrCamException.h"

using namespace std;
using namespace cv;
using namespace surround360;
using namespace surround360::util;
using namespace surround360::optical_flow;

// flowLtoRvec is an std::vector
// flowLtoR is a Mat opencv basic structure (matrix)

void prepareOpticalFlow(string leftImageName,
                        string rightImageName,
                        std::vector<Point2f>* flowLtoRvec,
                        NovelViewGenerator* novelViewGen)
{
    // reading images
    Mat colorImageL = imreadExceptionOnFail(leftImageName, -1); // -1 = load RGBA
    Mat colorImageR = imreadExceptionOnFail(rightImageName, -1);

    // we need alpha channels for flow. if they are missing, convert
    if (colorImageL.type() == CV_8UC3)
    {
        cvtColor(colorImageL, colorImageL, CV_BGR2BGRA);
    }
    if (colorImageR.type() == CV_8UC3)
    {
        cvtColor(colorImageR, colorImageR, CV_BGR2BGRA);
    }

    cout << "constructing novel view generator" << endl;

    cout << "calling prepare" << endl;
    double prepareStartTime = getCurrTimeSec();
    const Mat prevFlowLtoR = Mat();
    const Mat prevFlowRtoL = Mat();
    const Mat prevColorImageL = Mat();
    const Mat prevColorImageR = Mat();

    novelViewGen->prepare(
                colorImageL,
                colorImageR,
                prevFlowLtoR,
                prevFlowRtoL,
                prevColorImageL,
                prevColorImageR);

    double prepareEndTime = getCurrTimeSec();
    cout << "RUNTIME (sec) = " << (prepareEndTime - prepareStartTime) << endl;

    const Mat flowLtoR = novelViewGen->getFlowLtoR();

    for (int y = 0; y < flowLtoR.rows; ++y)
    {
        for (int x = 0; x < flowLtoR.cols; ++x)
        {
            ((*flowLtoRvec)[y*flowLtoR.cols + x]) = flowLtoR.at<Point2f>(y, x);
        }
    }
}

void computeOpticalFlow(string leftImageName,
                        string rightImageName,
                        string flowAlg,
                        std::vector<Point2f>* flowLtoRvec,
                        std::vector<Point2f>* flowRtoLvec)
{
    // reading images
    Mat colorImageL = imreadExceptionOnFail(leftImageName, -1); // -1 = load RGBA
    Mat colorImageR = imreadExceptionOnFail(rightImageName, -1);

    // we need alpha channels for flow. if they are missing, convert
    if (colorImageL.type() == CV_8UC3)
    {
        cvtColor(colorImageL, colorImageL, CV_BGR2BGRA);
    }
    if (colorImageR.type() == CV_8UC3)
    {
        cvtColor(colorImageR, colorImageR, CV_BGR2BGRA);
    }

    cout << "constructing novel view generator" << endl;
    NovelViewGenerator* novelViewGen = new NovelViewGeneratorAsymmetricFlow(flowAlg);

    cout << "calling prepare" << endl;
    double prepareStartTime = getCurrTimeSec();
    const Mat prevFlowLtoR = Mat();
    const Mat prevFlowRtoL = Mat();
    const Mat prevColorImageL = Mat();
    const Mat prevColorImageR = Mat();

    novelViewGen->prepare(
                colorImageL,
                colorImageR,
                prevFlowLtoR,
                prevFlowRtoL,
                prevColorImageL,
                prevColorImageR);

    double prepareEndTime = getCurrTimeSec();
    cout << "RUNTIME (sec) = " << (prepareEndTime - prepareStartTime) << endl;

    const Mat flowLtoR = novelViewGen->getFlowLtoR();
    const Mat flowRtoL = novelViewGen->getFlowRtoL();

    for (int y = 0; y < flowLtoR.rows; ++y)
    {
        for (int x = 0; x < flowLtoR.cols; ++x)
        {
            (*flowLtoRvec)[y*flowLtoR.cols + x] = flowLtoR.at<Point2f>(y, x);
            (*flowRtoLvec)[y*flowLtoR.cols + x] = flowRtoL.at<Point2f>(y, x);
        }
    }
}

void testOpticalFlow(string leftImageName, string rightImageName, string outdir)
{
    // reading images
    Mat colorImageL = imreadExceptionOnFail(leftImageName, -1); // -1 = load RGBA
    Mat colorImageR = imreadExceptionOnFail(rightImageName, -1);

    // we need alpha channels for flow. if they are missing, convert
    if (colorImageL.type() == CV_8UC3)
    {
        cvtColor(colorImageL, colorImageL, CV_BGR2BGRA);
    }
    if (colorImageR.type() == CV_8UC3)
    {
        cvtColor(colorImageR, colorImageR, CV_BGR2BGRA);
    }

    const int repetitions = 1; // number of times to repeat the flow calculation
    const std::string flowAlg = "pixflow_low"; // optical flow algorithm to use
    const int numIntermediateViews = 0; // number of views to make
    const bool saveAsymetricNovelViews = false; // if true, we will save the non-merged novel views that are obtained by warping the left/right images, in addition to the combined novel view

    for (int rep = 0; rep < repetitions; ++rep)
    {
        cout << "---- repetition " << rep << endl;

        cout << "constructing novel view generator" << endl;
        NovelViewGenerator* novelViewGen = new NovelViewGeneratorAsymmetricFlow(flowAlg);

        cout << "calling prepare" << endl;
        double prepareStartTime = getCurrTimeSec();
        const Mat prevFlowLtoR = Mat();
        const Mat prevFlowRtoL = Mat();
        const Mat prevColorImageL = Mat();
        const Mat prevColorImageR = Mat();

        novelViewGen->prepare(
                    colorImageL,
                    colorImageR,
                    prevFlowLtoR,
                    prevFlowRtoL,
                    prevColorImageL,
                    prevColorImageR);
        double prepareEndTime = getCurrTimeSec();
        cout << "RUNTIME (sec) = " << (prepareEndTime - prepareStartTime) << endl;

        const bool visualize = true;

        if(visualize)
        {
            cout << "building visualizations" << endl;

            Mat flowVisLtoR                = visualizeFlowAsGreyDisparity(novelViewGen->getFlowLtoR());
            Mat flowVisRtoL                = visualizeFlowAsGreyDisparity(novelViewGen->getFlowRtoL());
            Mat flowVisLtoRColorWheel      = visualizeFlowColorWheel(novelViewGen->getFlowLtoR());
            Mat flowVisRtoLColorWheel      = visualizeFlowColorWheel(novelViewGen->getFlowRtoL());
            Mat flowVisLtoRColorWithLines  = visualizeFlowAsVectorField(novelViewGen->getFlowLtoR(), colorImageL);
            Mat flowVisRtoLColorWithLines  = visualizeFlowAsVectorField(novelViewGen->getFlowRtoL(), colorImageR);

            cvtColor(flowVisRtoL,                 flowVisRtoL,                CV_GRAY2BGRA);
            cvtColor(flowVisLtoR,                 flowVisLtoR,                CV_GRAY2BGRA);
            cvtColor(flowVisLtoRColorWheel,       flowVisLtoRColorWheel,      CV_BGR2BGRA);
            cvtColor(flowVisRtoLColorWheel,       flowVisRtoLColorWheel,      CV_BGR2BGRA);

            Mat horizontalVisLtoR = stackHorizontal(vector<Mat>({flowVisLtoR, flowVisLtoRColorWheel, flowVisLtoRColorWithLines}));
            Mat horizontalVisRtoL = stackHorizontal(vector<Mat>({flowVisRtoL, flowVisRtoLColorWheel, flowVisRtoLColorWithLines}));

            imwriteExceptionOnFail(outdir + "/LtoR_" + flowAlg + ".png", horizontalVisLtoR);
            imwriteExceptionOnFail(outdir + "/RtoL_" + flowAlg + ".png", horizontalVisRtoL);

            // system(string("rm " + outdir + "/novel*").c_str());

            for (int v = 0; v < numIntermediateViews; ++v)
            {
                const double shiftFromLeft = double(v) / double(numIntermediateViews - 1);

                Mat novelViewMerged = Mat(); // init here so we don't crash if nothing
                Mat novelViewFromL = Mat();  // is written
                Mat novelViewFromR = Mat();
                novelViewGen->generateNovelView(shiftFromLeft, novelViewMerged, novelViewFromL, novelViewFromR);

                stringstream ss;
                ss << std::setfill('0') << std::setw(6) << v;
                const string frameIdxPadded = ss.str();

                imwriteExceptionOnFail(outdir + "/novel_view_" + flowAlg + "_" + frameIdxPadded + ".png", novelViewMerged);

                if (saveAsymetricNovelViews)
                {
                    imwriteExceptionOnFail(outdir + "/novelFromL_" + frameIdxPadded + ".png", novelViewFromL);
                    imwriteExceptionOnFail( outdir + "/novelFromR_" + frameIdxPadded + ".png", novelViewFromR);
                }
            }
        }

        delete novelViewGen;
    }
}

void displayMatrix(const Mat &M)
{
    for(int i = 0 ; i < M.rows ; ++i)
    {
        for(int j = 0 ; j < M.cols ; ++j)
        {
            std::cout << "( " << i << ", " << j << " ): " << M.at<float>(i, j) << std::endl;
        }
    }
}

void displayMatrix2(const Mat &M)
{
    for(MatConstIterator_<float> it = M.begin<float>() ; it != M.end<float>() ; ++it)
    {
        std::cout << *it << std::endl;
    }
}

void computeCovariance(std::vector<Point2f> &imageData, std::vector<Point3f> &centerData, Point3f targetCenter,
                       float *trace, float *determinant, float *eigenRatio, Point2f *eigenValues, Point3f *eigenVector,
                       Point2f *targetImage)
{
    const uint nbSamples = imageData.size();
    const uint minNbSamples = 0;

    if(nbSamples > minNbSamples)
    {
        Mat X = Mat::zeros(nbSamples, 2, CV_32FC1);
        Mat C = Mat::zeros(nbSamples, 3, CV_32FC1);

        Point2f meanX(0.0f, 0.0f);
        for(uint k = 0 ; k < nbSamples ; ++k)
        {
            meanX += imageData[k];
        }
        meanX /= (float)nbSamples;

        for(uint k = 0 ; k < nbSamples ; ++k)
        {
            Point2f point = imageData[k] - meanX;

            X.at<float>(k, 0) = point.x;
            X.at<float>(k, 1) = point.y;

            // std::cout << point << "   ";
        }

        Point3f meanC(0.0f, 0.0f, 0.0f);
        for(uint k = 0 ; k < nbSamples ; ++k)
        {
            meanC += centerData[k];
        }
        meanC /= (float)nbSamples;

        for(uint k = 0 ; k < nbSamples ; ++k)
        {
            Point3f point = centerData[k] - meanC;

            C.at<float>(k, 0) = point.x;
            C.at<float>(k, 1) = point.y;
            C.at<float>(k, 2) = point.z;

            // std::cout << point << "   ";
        }

        Mat X_t = X.t();
        Mat sigmaXX = X_t * X;
        sigmaXX = sigmaXX/((float)nbSamples - 1.0);

        Mat C_t = C.t();
        Mat sigmaCC = C_t * C;
        sigmaCC = sigmaCC/((float)nbSamples - 1.0);

        Mat sigmaCX = C_t * X;
        sigmaCX = sigmaCX/((float)nbSamples - 1.0);

        Mat sigmaXC = X_t * C;
        sigmaXC = sigmaXC/((float)nbSamples - 1.0);

        Mat H1 = Mat::zeros(3, 5, CV_32FC1);
        Mat H2 = Mat::zeros(2, 5, CV_32FC1);
        hconcat(sigmaCC, sigmaCX, H1);
        hconcat(sigmaXC, sigmaXX, H2);

        Mat sigma = Mat::zeros(5, 5, CV_32FC1);
        vconcat(H1, H2, sigma);

        SVD svd(sigma);

        *trace = (float)cv::trace(sigma)[0];
        *determinant = (float)cv::determinant(sigma);
        *eigenRatio = (float)(svd.w.at<float>(0, 0)/svd.w.at<float>(1, 0));
        eigenValues->x = svd.w.at<float>(0, 0);
        eigenValues->y = svd.w.at<float>(1, 0);
        eigenVector->x = svd.u.at<float>(0, 0);
        eigenVector->y = svd.u.at<float>(1, 0);
        eigenVector->z = svd.u.at<float>(2, 0);

        Mat sigmaCC_inv = sigmaCC.inv();
        Mat deltaC(targetCenter - meanC);
        Mat res = Mat(meanX) + sigmaXC * sigmaCC_inv * Mat(deltaC);
        *targetImage = Point2f(res.at<float>(0, 0), res.at<float>(1, 0));

        const bool verbose = false;
        if(verbose) {

            std::cout << "sigmaXX: " << sigmaXX << std::endl;
            std::cout << "sigmaCC: " << sigmaCC << std::endl;
            std::cout << "sigmaCX: " << sigmaCX << std::endl;
            std::cout << "sigmaXC: " << sigmaXC << std::endl;

            std::cout << "sigma: " << sigma << std::endl;

            std::cout << "vt: " << svd.vt << std::endl;
            std::cout << "u: " << svd.u << std::endl;
            std::cout << "w: " << svd.w << std::endl;
        }
    }
    else
    {
        *trace = 0.0f;
        *determinant = 0.0f;
        *eigenRatio = 0.0f;
        eigenValues->x = eigenValues->y = 0.0f;
        eigenVector->x = eigenVector->y = eigenVector->z = 0.0f;
        targetImage->x = targetImage->y = 0.0f;
    }
}

OpticalFlow::OpticalFlow( uint nbImages ) : _nbImages(nbImages) {

    // setup paralllel optical flow
    _startOpticalFlowTime = getCurrTimeSec();
    _novelViewGenerators.resize(nbImages - 1); // n images, n-1 flows
    //    for (int leftIdx = 0; leftIdx < nbImages - 1; ++leftIdx) {
    //        pushThread(leftIdx);
    //    }

    _opticalFlowRuntime = getCurrTimeSec() - _startOpticalFlowTime;
}

OpticalFlow::~OpticalFlow() {

    for (uint flowIdx = 0 ; flowIdx < _nbImages - 1 ; ++flowIdx) {
        if(_novelViewGenerators[flowIdx] != 0) {
            delete _novelViewGenerators[flowIdx];
            _novelViewGenerators[flowIdx] = 0;
        }
    }
}

void OpticalFlow::pushThread(uint flowIdx,
                             string leftImageName,
                             string rightImageName,
                             string flowAlg,
                             std::vector<Point2f>& flowLtoRvec) {

    _novelViewGenerators[flowIdx] = new NovelViewGeneratorAsymmetricFlow(flowAlg);
    _threads.push_back(std::thread(prepareOpticalFlow,
                                   leftImageName,
                                   rightImageName,
                                   &flowLtoRvec,
                                   _novelViewGenerators[flowIdx]));
}

void OpticalFlow::join() {

    for (std::thread& t : _threads) { t.join(); }
}


