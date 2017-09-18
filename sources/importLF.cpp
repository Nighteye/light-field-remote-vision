#include "lfScene.h"
#include <cocolib/cocolib/common/debug.h>
#include <iostream>
#include <vector>
#include <sys/stat.h>

#define cimg_display 0
#define cimg_use_tiff
#define cimg_use_png
#include "CImg.h"

void loadPFM(std::vector<float>& output,
             int width, int height,
             std::string name,
             bool flip)
{
    cimg_library::CImg<float> image(name.c_str());

    image.resize(width, height, 1, 1);

    for(uint y = 0 ; y < (uint)height ; ++y)
    {
        for(uint x = 0 ; x < (uint)width ; ++x)
        {
            const uint i = y*width + x;
            uint i_flip = 0;
            if(flip) {
                i_flip = (height - 1 - y)*width + x;
            } else {
                i_flip = i;
            }
            output[i] = *(((float*)(&image(0,0,0,0))) + i_flip);
        }
    }
}

void loadPFM(std::vector<cv::Point2f>& output,
             int width, int height,
             std::string name,
             bool flip)
{
    cimg_library::CImg<float> image(name.c_str());

    image.resize(width, height, 1, 2);

    for(uint y = 0 ; y < (uint)height ; ++y)
    {
        for(uint x = 0 ; x < (uint)width ; ++x)
        {
            const uint i = y*width + x;
            uint i_flip = 0;
            if(flip) {
                i_flip = (height - 1 - y)*width + x;
            } else {
                i_flip = i;
            }
            output[i].x = *(((float*)(&image(0,0,0,0))) + i_flip);
            output[i].y = *(((float*)(&image(0,0,0,1))) + i_flip);
        }
    }
}

void loadPFM(std::vector<cv::Point3f>& output,
             int width, int height,
             std::string name,
             bool flip)
{
    cimg_library::CImg<float> image(name.c_str());

    image.resize(width, height, 1, 3);

    for(uint y = 0 ; y < (uint)height ; ++y)
    {
        for(uint x = 0 ; x < (uint)width ; ++x)
        {
            const uint i = y*width + x;
            uint i_flip = 0;
            if(flip) {
                i_flip = (height - 1 - y)*width + x;
            } else {
                i_flip = i;
            }
            output[i].x = *(((float*)(&image(0,0,0,0))) + i_flip);
            output[i].y = *(((float*)(&image(0,0,0,1))) + i_flip);
            output[i].z = *(((float*)(&image(0,0,0,2))) + i_flip);
        }
    }
}

void loadPFM(std::vector<bool>& output,
             int width, int height,
             std::string name,
             bool flip)
{
    cimg_library::CImg<float> image(name.c_str());

    image.resize(width, height, 1, 1);

    for(uint y = 0 ; y < (uint)height ; ++y)
    {
        for(uint x = 0 ; x < (uint)width ; ++x)
        {
            const uint i = y*width + x;
            uint i_flip = 0;
            const float epsilon = 0.00001;
            if(flip) {
                i_flip = (height - 1 - y)*width + x;
            } else {
                i_flip = i;
            }
            if(*(((float*)(&image(0,0,0,0))) + i_flip) < epsilon) {
                output[i] = false;
            } else {
                output[i] = true;
            }
        }
    }
}

void savePFM(std::vector<std::vector<float> >& input,
             int width, int height,
             std::string name,
             bool flip)
{
    cimg_library::CImg<float> image;

    image.resize(width, height, 1, input.size());

    assert(input.size() > 0);
    assert(input.size() <= 3);

    for(uint channel = 0 ; channel < input.size() ; ++channel)
    {
        memcpy(&image(0,0,0,channel), input[channel].data(), width*height*sizeof(float));
    }

    if(flip)
    {
        image.mirror('y');
    }
    try
    {
        image.save(name.c_str());
    }
    catch (cimg_library::CImgIOException)
    {
        printf("Exception COUGHT: file not saved\n");
    }
}

// bool* arr;

// bool (*arrt)[] = arr
void savePFM(std::vector<bool>& input,
             int width, int height,
             std::string name,
             bool flip)
{
    cimg_library::CImg<float> image;

    image.resize(width, height, 1, 1);

    for(size_t i = 0 ; i < input.size() ; ++i)
    {
        float value = (float)input[i];
        *(((float*)(&image(0,0,0,0))) + i) = value;
    }

    if(flip)
    {
        image.mirror('y');
    }
    try
    {
        image.save(name.c_str());
    }
    catch (cimg_library::CImgIOException)
    {
        printf("Exception COUGHT: file not saved\n");
    }
}

void savePFM(std::vector<float>& input,
             int width, int height,
             std::string name,
             bool flip)
{
    cimg_library::CImg<float> image;

    image.resize(width, height, 1, 1);

    for(size_t i = 0 ; i < input.size() ; ++i)
    {
        float value = (float)input[i];
        *(((float*)(&image(0,0,0,0))) + i) = value;
    }

    if(flip)
    {
        image.mirror('y');
    }
    try
    {
        image.save(name.c_str());
    }
    catch (cimg_library::CImgIOException)
    {
        printf("Exception COUGHT: file not saved\n");
    }
}

void savePFM(std::vector<cv::Point2f>& input,
             int width, int height,
             std::string name,
             bool flip)
{
    cimg_library::CImg<float> image;

    image.resize(width, height, 1, 2);

    for(size_t i = 0 ; i < input.size() ; ++i)
    {
        float x = (float)input[i].x;
        *(((float*)(&image(0,0,0,0))) + i) = x;
        float y = (float)input[i].y;
        *(((float*)(&image(0,0,0,1))) + i) = y;
    }

    if(flip)
    {
        image.mirror('y');
    }
    try
    {
        image.save(name.c_str());
    }
    catch (cimg_library::CImgIOException)
    {
        printf("Exception COUGHT: file not saved\n");
    }
}

void savePFM(std::vector<cv::Point3f>& input,
             int width, int height,
             std::string name,
             bool flip)
{
    cimg_library::CImg<float> image;

    image.resize(width, height, 1, 3);

    for(size_t i = 0 ; i < input.size() ; ++i)
    {
        float x = (float)input[i].x;
        *(((float*)(&image(0,0,0,0))) + i) = x;
        float y = (float)input[i].y;
        *(((float*)(&image(0,0,0,1))) + i) = y;
        float z = (float)input[i].z;
        *(((float*)(&image(0,0,0,2))) + i) = z;
    }

    if(flip)
    {
        image.mirror('y');
    }
    try
    {
        image.save(name.c_str());
    }
    catch (cimg_library::CImgIOException)
    {
        printf("Exception COUGHT: file not saved\n");
    }
}

void savePNG(std::vector<cv::Point3f>& input,
             int width, int height,
             std::string name,
             bool flip) {

    cimg_library::CImg<unsigned char> image;

    image.resize(width, height, 1, 3);

    for(size_t i = 0 ; i < input.size() ; ++i) {

        uchar x = (uchar)(input[i].x * 255.0f);
        *(((uchar*)(&image(0,0,0,0))) + i) = x;
        uchar y = (uchar)(input[i].y * 255.0f);
        *(((uchar*)(&image(0,0,0,1))) + i) = y;
        uchar z = (uchar)(input[i].z * 255.0f);
        *(((uchar*)(&image(0,0,0,2))) + i) = z;
    }

    if(flip) {
        image.mirror('y');
    }
    try {
        image.save(name.c_str());
    } catch (cimg_library::CImgIOException) {
        printf("Exception COUGHT: file not saved\n");
    }
}

// Import camera image from PNG file
bool InputCam::importTexture( const char* imageName, bool flip ) {

    cimg_library::CImg<float> image(imageName);

    image.resize(_W, _H, 1, 3);

    for(uint y = 0 ; y < (uint)_H ; ++y)
    {
        for(uint x = 0 ; x < (uint)_W ; ++x)
        {
            const uint i = y*_W + x;
            uint i_flip = 0;
            if(flip) {
                i_flip = (_H - 1 - y)*_W + x;
            } else {
                i_flip = i;
            }
            _texture[i].x = *(((float*)(&image(0,0,0,0))) + i_flip) / 255.0f;
            _texture[i].y = *(((float*)(&image(0,0,0,1))) + i_flip) / 255.0f;
            _texture[i].z = *(((float*)(&image(0,0,0,2))) + i_flip) / 255.0f;
        }
    }

    return true;
}

// Import camera parameters and load vbos
// Read the parameters from Stanford images files
bool InputCam::importCamParametersStanford(double centerX, double centerY) {

    float focal_length = 1.0;
    float pixel_aspect = 1.0;
    float principal_point[] = {0.5, 0.5};
    glm::mat3 R(1.0);
    glm::vec3 t(0.0);
    glm::mat3 K(0.0);

    // TODO: use centerX and centerY to determine the centers of hte cameras

    // focal_length = f1 in pixels divided by larger side
    // pixel_aspect = pixel width divided by pixel height
    // principal_point is also normalized and independent of the image size
    if( _W >= _H ) {
        K[0][0] = _W * focal_length;
    } else {
        K[0][0] = _H * focal_length;
    }
    K[1][1] = K[0][0] / pixel_aspect;
    K[2][2] = 1.0;
    K[2][0] = _W * principal_point[0];
    K[2][1] = _H * principal_point[1];

    // t = -RC
    t[0] = - centerX;
    t[1] = - centerY;
    t[2] = 0;

    _pinholeCamera = PinholeCamera( K, R, t, _W, _H );

    return true;
}

// Import camera parameters and load vbos
// Read the camera matrices from INI file (MVE format)
bool InputCam::importCamParameters( char *cameraName ) {

    std::ifstream in( cameraName, std::ifstream::in );
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

    if( !strcmp( "[view]", tmp.c_str() ) || count >= nbMaxWordsHeader) {

        // No camera parameter has been estimated by sfm
        return false;

    } else {

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
        if( _W >= _H ) {
            K[0][0] = _W * focal_length;
        } else {
            K[0][0] = _H * focal_length;
        }
        K[1][1] = K[0][0] / pixel_aspect;
        K[2][2] = 1.0;
        K[2][0] = _W * principal_point[0];
        K[2][1] = _H * principal_point[1];

        _pinholeCamera = PinholeCamera( K, R, t, _W, _H );

        // _pinholeCamera.display();

        return true;
    }
}

// Import camera parameters (translation) and set fixed focal length (hack)
// Read the camera translation vectors from XML file (TOLF format)
bool InputCam::importCamTranslations( char *cameraName, uint viewIndex ) {

    std::string tmp;

    const uint referenceViewIndex = 12; // HACK, TODO: variable for reference view number

    glm::vec3 r(0.0);
    glm::mat3 R(1.0);
    glm::vec3 t(0.0);
    glm::mat3 K(0.0);
    uint nbMaxWordsHeader = 100;

    std::ifstream in( cameraName, std::ifstream::in );
    assert( in.is_open() );
    assert( in );

    uint count = 0; // for safety
    while( strcmp( "<data>", tmp.c_str() ) && strcmp( "[view]", tmp.c_str() ) && count < nbMaxWordsHeader ) {

        in >> tmp;
        ++count;
    }

    assert( strcmp( "</data>", tmp.c_str() ) && count < nbMaxWordsHeader);

    for(uint i = 0 ; i < referenceViewIndex ; ++i) {

        for(uint j = 0 ; j < 6 ; ++j) {
            in >> tmp;
        }
    }

    // back to the beginning of the file
    in.clear();
    in.seekg(0, std::ios::beg);

    count = 0; // for safety
    while( strcmp( "<data>", tmp.c_str() ) && strcmp( "[view]", tmp.c_str() ) && count < nbMaxWordsHeader ) {

        in >> tmp;
        ++count;
    }

    if( !strcmp( "</data>", tmp.c_str() ) || count >= nbMaxWordsHeader) {

        // No camera parameter has been estimated by sfm
        return false;

    } else {

        for(uint i = 0 ; i < viewIndex ; ++i) {

            for(uint j = 0 ; j < 6 ; ++j) {
                in >> tmp;
            }
        }

        in >> r[0] >> r[1] >> r[2] >> t[0] >> t[1] >> t[2];
        in.close();

        float theta = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
        if(theta > 0.0000001) {

            r /= theta;
            R[0][0] = cos(theta) + (1 - cos(theta))*r[0]*r[0];
            R[1][0] = (1 - cos(theta))*r[0]*r[1] - sin(theta)*r[2];
            R[2][0] = (1 - cos(theta))*r[0]*r[2] + sin(theta)*r[1];
            R[0][1] = (1 - cos(theta))*r[1]*r[0] + sin(theta)*r[2];
            R[1][1] = cos(theta) + (1 - cos(theta))*r[1]*r[1];
            R[2][1] = (1 - cos(theta))*r[1]*r[2] - sin(theta)*r[0];
            R[0][2] = (1 - cos(theta))*r[2]*r[0] - sin(theta)*r[1];
            R[1][2] = (1 - cos(theta))*r[2]*r[1] + sin(theta)*r[0];
            R[2][2] = cos(theta) + (1 - cos(theta))*r[2]*r[2];

        } else {

            R = glm::mat3(1.0);
        }

        // update the camera paramters with respect to the reference view
        K[0][0] = 905.44060602527395;
        K[1][1] = 907.68102844590101;
        K[2][2] = 1.0;
        K[2][0] = 308.29227330174598;
        K[2][1] = 252.10539485267773;
        t = glm::transpose(R)*t;
        //        t.z += 10; // HACK
        R = glm::mat3(1.0); // the rotation is the rotation of the reference view, so the identity
        // the center of projection remains the same

        _pinholeCamera = PinholeCamera( K, R, t, _W, _H );

        //        _pinholeCamera.display();

        return true;
    }
}

bool LFScene::checkExistence(const std::string& name) {

    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}

bool LFScene::checkExistenceAllViews(const std::string& name) {

    bool exists = true;

    for(uint viewIndex = 0 ; viewIndex < _nbCameras ; ++viewIndex) {

        if(viewIndex == _nbCameras/2) {
            continue;
        }

        char temp[500];
        std::fill_n(temp, 500, 0.0);
        sprintf( temp, name.c_str(), viewIndex );
        struct stat buffer;
        exists &= (stat (temp, &buffer) == 0);
    }

    return exists;
}

// load standford lightfield dataset (image and camera matrices)
// camera matrices are obainted thank to openMVG calibration
// MVE FORMAT (convert rows to rows and column)
void LFScene::importStanfordMVEViews() {

    assert(!_mveName.empty());

    std::cout << "Import cameras and images" << std::endl;

    uint viewIndex = 0;

    for ( int t = _tMin ; t <= _tMax ; ++t ) {

        for ( int s = _sMin ; s <= _sMax ; ++s ) {

            char imageNameChar[500];
            char cameraNameChar[500];
            int mveIndex = t*17 + s; // HACK, TODO: stanford LF range as parameter
            sprintf( imageNameChar, _imageName.c_str(), mveIndex );
            sprintf( cameraNameChar, _cameraName.c_str(), mveIndex );

            std::cout << "Import camera " << cameraNameChar << " and image " << imageNameChar << " ..." << std::endl;

            InputCam *v_k = new InputCam( _camWidth, _camHeight, _outdir );

            // Import camera parameters and load vbos
            if( v_k->importTexture(imageNameChar) ) {

                // Import camera parameters and load vbos
                if( v_k->importCamParameters(cameraNameChar) ) {

                    _vCam.push_back( v_k );

                    // import the view to synthetize also, since we remove it only when running cocolib
                    if ( s == _sRmv ) {

                        _renderIndex = viewIndex;
                    }
                    ++viewIndex;

                } else {

                    assert( s != _sRmv && t != _tRmv );
                    std::cout << "Error while loading camera parameters of view " << s << std::endl;
                    delete v_k;
                    v_k = 0;
                }

            } else {

                assert( s != _sRmv && t != _tRmv );
                std::cout << "Error while loading input image of view " << s << ": file doesn't exist" << std::endl;
                delete v_k;
                v_k = 0;
            }
        }
    }

    assert(viewIndex == _vCam.size());
    _nbCameras = viewIndex;

    std::cout << "done!" << std::endl;
}

// load standford lightfield dataset (image and camera matrices)
// camera matrices are obainted thank to openMVG calibration
// MVE FORMAT (convert rows to rows and column)
// custom camera configuration
void LFScene::importCustomMVEViews() {

    assert(!_mveName.empty());

    // custom configuration // HACK
    std::vector<int> sIndices = {_sMin, _sMin + 2, _sMin + 4, _sMin + 6, _sMin + 8,
                                 _sMin, _sMin + 2, _sMin + 4, _sMin + 6, _sMin + 8,
                                 _sMin, _sMin + 2, _sMin + 4, _sMin + 6, _sMin + 8,
                                 _sMin, _sMin + 2, _sMin + 4, _sMin + 6, _sMin + 8,
                                 _sMin, _sMin + 2, _sMin + 4, _sMin + 6, _sMin + 8};

    std::vector<int> tIndices = {_tMin, _tMin, _tMin, _tMin, _tMin,
                                 _tMin + 2, _tMin + 2, _tMin + 2, _tMin + 2, _tMin + 2,
                                 _tMin + 4, _tMin + 4, _tMin + 4, _tMin + 4, _tMin + 4,
                                 _tMin + 6, _tMin + 6, _tMin + 6, _tMin + 6, _tMin + 6,
                                 _tMin + 8, _tMin + 8, _tMin + 8, _tMin + 8, _tMin + 8};

    assert(_sMax == _sMin + 8);
    assert(_tMax == _tMin + 8);
    _nbCameras = 25;
    _centralIndex = _nbCameras/2;

    std::cout << "Import cameras and images" << std::endl;

    for ( uint viewIndex = 0 ; viewIndex < _nbCameras ; ++viewIndex ) {

        char imageNameChar[500];
        char cameraNameChar[500];
        int s = sIndices[viewIndex];
        int t = tIndices[viewIndex];
        int mveIndex = t*17 + s; // HACK
        sprintf( imageNameChar, _imageName.c_str(), mveIndex );
        sprintf( cameraNameChar, _cameraName.c_str(), mveIndex );

        std::cout << "Import camera " << cameraNameChar << " and image " << imageNameChar << " ..." << std::endl;

        InputCam *v_k = new InputCam( _camWidth, _camHeight, _outdir );

        // Import camera parameters and load vbos
        if( v_k->importTexture(imageNameChar) ) {

            // Import camera parameters and load vbos
            if( v_k->importCamParameters(cameraNameChar) ) {

                _vCam.push_back( v_k );

            } else {

                assert( s != _sRmv && t != _tRmv );
                std::cout << "Error while loading camera parameters of view " << mveIndex << std::endl;
                delete v_k;
                v_k = 0;
            }

        } else {

            assert( s != _sRmv && t != _tRmv );
            std::cout << "Error while loading input image of view " << mveIndex << ": file doesn't exist" << std::endl;
            delete v_k;
            v_k = 0;
        }
    }

    assert(_nbCameras == _vCam.size());

    std::cout << "done!" << std::endl;
}

// load TOLF dataset (image and camera matrices)
// camera matrices are obainted thank to their own calibration
// TOLF FORMAT
// custom camera configuration
void LFScene::importCustomTOLFViews() {

    assert(!_imageName.empty());
    assert(!_cameraName.empty());

    _nbCameras = 25;
    _centralIndex = _nbCameras/2;

    std::cout << "Import cameras and images" << std::endl;

    for ( uint viewIndex = 0 ; viewIndex < _nbCameras ; ++viewIndex ) {

        char imageNameChar[500];
        char cameraNameChar[500];
        sprintf( imageNameChar, _imageName.c_str(), viewIndex );
        sprintf( cameraNameChar, _cameraName.c_str() ); // should be the same whatever the view index

        std::cout << "Import camera " << cameraNameChar << " and image " << imageNameChar << " ..." << std::endl;

        InputCam *v_k = new InputCam( _camWidth, _camHeight, _outdir );

        // Import camera parameters and load vbos
        if( v_k->importTexture(imageNameChar) ) {

            // Import camera parameters and load vbos
            if( v_k->importCamTranslations(cameraNameChar, viewIndex) ) {

                _vCam.push_back( v_k );

            } else {

                assert( viewIndex != _centralIndex );
                std::cout << "Error while loading camera parameters of view " << viewIndex << std::endl;
                delete v_k;
                v_k = 0;
            }

        } else {

            assert( viewIndex != _centralIndex );
            std::cout << "Error while loading input image of view " << viewIndex << ": file doesn't exist" << std::endl;
            delete v_k;
            v_k = 0;
        }
    }

    assert(_nbCameras == _vCam.size());

    std::cout << "done!" << std::endl;
}

// for each view import source images and camera parameters (blender datasets for example)
// BLENDER FORMAT (rows only)
void LFScene::importViewsNoDepth() {

    std::cout << "Import cameras and images" << std::endl;

    uint viewIndex = 0;
    for ( int s = _sMin ; s <= _sMax ; ++s ) {

        char imageNameChar[500];
        char cameraNameChar[500];

        sprintf( imageNameChar, _imageName.c_str(), s );
        sprintf( cameraNameChar, _cameraName.c_str(), s );

        // std::cout << "Import camera " << cameraNameChar << " and image " << imageNameChar << " ..." << std::endl;

        InputCam *v_k = new InputCam( _camWidth, _camHeight, _outdir );

        // Import camera parameters and load vbos
        if( v_k->importTexture(imageNameChar) ) {

            // Import camera parameters and load vbos
            if( v_k->importCamParameters(cameraNameChar) ) {

                _vCam.push_back( v_k );

                // import the view to synthetize also, since we remove it only when running cocolib
                if ( s == _sRmv ) {

                    _renderIndex = viewIndex;
                }
                ++viewIndex;

            } else {

                assert( s != _sRmv );
                // std::cout << "Error while loading camera parameters of view " << s << std::endl;
                delete v_k;
                v_k = 0;
            }

        } else {

            assert( s != _sRmv );
            // std::cout << "Error while loading input image of view " << s << ": file doesn't exist" << std::endl;
            delete v_k;
            v_k = 0;
        }
    }

    assert(viewIndex == _vCam.size());
    _nbCameras = viewIndex;

    std::cout << "done!" << std::endl;
}

