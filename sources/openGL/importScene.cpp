#include "../ply_io.h"
#include "view.h"
#include "mesh.h"
#include "scene.h"
#include "frameBuffer.h"

#include <cocolib/cocolib/common/debug.h>
#include <iostream>
#include <vector>

using namespace coco;

void Scene::importMesh( std::string mveName ) {

    checkGLErrors();

    std::string meshName = mveName + "/fssMesh-clean.ply";

    std::vector< float > vec_points;
    std::vector< float > vec_normals;
    std::vector< unsigned int > vec_triangles;

    std::vector< GLfloat > vertexArray;
    std::vector< GLfloat > normalArray;
    std::vector<GLuint> triangleArray;

    if(read_ply( meshName.c_str(), vec_points, vec_normals, vec_triangles )) {

        std::cout << "Import mesh " << meshName << " ..." << std::endl;

        vertexArray.resize(vec_points.size());
        normalArray.resize(vec_points.size());
        for ( uint i = 0 ; i < vertexArray.size() ; ++i ) {

            vertexArray[i] = (GLfloat)vec_points[i];
            normalArray[i] = (GLfloat)vec_normals[i];
        }

        triangleArray.resize(vec_triangles.size());
        for ( uint i = 0 ; i < triangleArray.size() ; ++i ) {

            triangleArray[i] = (GLuint)vec_triangles[i];
        }

        Mesh *mesh = new Mesh( vertexArray, normalArray, triangleArray );

        assert( mesh->isMeshOK() );
        _meshes.push_back( mesh );
        assert( _meshes[0]->isMeshOK() );

        std::cout << "done!" << std::endl;

    } else {

        std::cout << "No mesh found in " << meshName << std::endl;
    }

    checkGLErrors();
}

void Scene::importPointCloud( std::string mveName ) {

    checkGLErrors();

    std::string pointCloudName = mveName + "/pointCloud.ply";

    std::vector< float > vec_points;
    std::vector< float > vec_normals;
    std::vector< unsigned int > vec_triangles;

    std::vector< GLfloat > vertexArray;
    std::vector< GLfloat > normalArray;
    std::vector<GLuint> triangleArray;

    if( read_ply( pointCloudName.c_str(), vec_points, vec_normals, vec_triangles )) {

        std::cout << "Import point cloud " << pointCloudName << " ..." << std::endl;

        vertexArray.resize(vec_points.size());
        normalArray.resize(vec_points.size());
        triangleArray.resize(0);
        for ( uint i = 0 ; i < vertexArray.size() ; ++i ) {

            vertexArray[i] = (GLfloat)vec_points[i];
            normalArray[i] = (GLfloat)vec_normals[i];
        }
        Mesh *pointCloud = new Mesh( vertexArray, normalArray, triangleArray );

        assert( pointCloud->isMeshOK() );
        _meshes.push_back( pointCloud );
        assert( _meshes[1]->isMeshOK() );

        std::cout << "done!" << std::endl;

    } else {

        std::cout << "No point cloud found in " << pointCloudName << std::endl;
    }

    checkGLErrors();
}

// for each view import all necessary data to compute cocolib input parameters (warps and their derivatives)
void Scene::importViews( std::string mveName, uint scale_min, uint scale_max ) {

    checkGLErrors();

    const bool verbose = false;
    float count = 0;

    std::cout << "Import cameras and pictures from " << mveName + "/views" + "/view_%04i.mve/ ..." << std::endl;

    uint viewIndex = 0;
    for ( int s = _sMin ; s <= _sMax ; ++s ) {

        std::string cameraName = mveName + "/views" + "/view_%04i.mve" + "/meta.ini";
        std::string imageName = mveName + "/views" + "/view_%04i.mve" + "/undistorted.png";
        std::string maskName = "/local/home/gnietol/Pictures/mask060_%02i.png"; // HACK
        std::string depthName = mveName + "/views" + "/view_%04i.mve" + "/depth-L%01i.mvei";
        std::string normalName = mveName + "/views" + "/view_%04i.mve" + "/dz-L%01i.mvei";

        char cameraNameChar[500];
        char imageNameChar[500];
        char maskNameChar[500];
        char depthNameChar[500];
        char normalNameChar[500];

        sprintf( cameraNameChar, cameraName.c_str(), s );
        sprintf( imageNameChar, imageName.c_str(), s );
        sprintf( maskNameChar, maskName.c_str(), s );

        // std::cout << "Import camera " << cameraNameChar << " and image " << imageNameChar << " ..." << std::endl;

        InputView *v_k = new InputView( _camWidth, _camHeight, _outdir, _pyramidHeight );

        // Import camera parameters and load vbos
        if( v_k->importTexture(imageNameChar) ) {

            assert( v_k->importMask(maskNameChar, count) );

            // Import camera parameters and load vbos
            if( v_k->importCamParameters(cameraNameChar) ) {

                v_k->initDepthMap();

                Texture* depthMapTmp(0);

                for ( uint scale = scale_min ; scale <= scale_max ; ++scale ) {

                    sprintf( depthNameChar, depthName.c_str(), s, scale );
                    sprintf( normalNameChar, normalName.c_str(), s, scale );

                    // Import depth and normal map from mvei and append them as color buffer to Map object
                    depthMapTmp = v_k->setDepthAndNormals(depthNameChar, normalNameChar);

                    if( depthMapTmp != 0 ) {

                        v_k->addDepthScale(depthMapTmp, _FBO, scale, _addDepthScaleShader);

                    } else { // if at least one of the depth maps is not loaded, we break the loop

                        assert( s != _sRmv );
                        std::cout << "Error while loading normal and depth map of view " << s << " , scale " << scale << std::endl;
                        delete v_k;
                        v_k = 0;
                        break;
                    }
                }

                if(verbose && !_outdir.empty()) {

                    char outDepthNameChar[500];
                    const std::string outDepthName = _outdir + "/depthMap_%02i.pfm";
                    sprintf( outDepthNameChar, outDepthName.c_str(), s );
                    std::cout << "Export depth map " << s << " in " << outDepthNameChar << std::endl;
                    v_k->saveDepthMap( std::string(outDepthNameChar) );
                    depthMapTmp->saveRGBAFloatTexture( _camWidth, _camHeight, 3, std::string(outDepthNameChar), false );
                }

                if( depthMapTmp != 0 ) {

                    _vCam.push_back( v_k );

                    // import the view to synthetize also, since we remove it only when running cocolib
                    if ( s == _sRmv ) {

                        _uCam = new TargetView( _camWidth, _camHeight, v_k->getPinholeCamera(), _outdir, _pyramidHeight );
                        // empty depth map
                        _uCam->initDepthMap();
                        // clear depth (set first channel to invalid value)
                        _FBO->clearTexture( _uCam->getDepthMap(), INVALID_DEPTH );

                        _renderIndex = viewIndex;
                    }
                    ++viewIndex;

                } else {

                    delete v_k;
                    v_k = 0;
                }

                delete depthMapTmp;
                depthMapTmp = 0;

            } else {

                assert( s != _sRmv );
                // std::cout << "Error while loading camera parameters of view " << s << std::endl;
                delete v_k;
                v_k = 0;
            }

        } else {

            checkGLErrors();
            assert( s != _sRmv );
            // std::cout << "Error while loading input image of view " << s << ": file doesn't exist" << std::endl;
            checkGLErrors();
            delete v_k;
            v_k = 0;
            checkGLErrors();
        }
    }

    count /= (_sMax - _sMin + 1);
    std::cout << "Average ratio of occluded areas: " << count << std::endl;

    assert(viewIndex == _vCam.size());
    _nbCameras = viewIndex;

    // in the case we don't want to generate an input view
    if(_uCam == 0) {

        PinholeCamera pinholeCamera;
        _uCam = new TargetView( _camWidth, _camHeight, pinholeCamera, _outdir, _pyramidHeight );
        // empty depth map
        _uCam->initDepthMap();
        // clear depth (set first channel to invalid value)
        _FBO->clearTexture( _uCam->getDepthMap(), INVALID_DEPTH );
    }

    std::cout << "done!" << std::endl;

    checkGLErrors();
}

// load standford lightfield dataset (images only)
void Scene::importStanfordViews( std::string imageName ) {

    checkGLErrors();

    // imported from cocolib
    // -------------------------------------------------------
    // loop over all directory entries
    std::string lf_name, lf_dir;
    breakupFileName( imageName, lf_dir, lf_name );
    coco::Directory::cd( lf_dir );
    TRACE( "  searching for images in subdirectory " << lf_dir << ", format *.png" << std::endl );
    std::vector<std::string> files = coco::Directory::files( "*.png" );
    assert( files.size() > 0 );
    coco::Directory::cd( coco::Directory::base() ); // reset work directory

    int S(_sMax - _sMin + 1), T(_tMax - _tMin + 1);

    // Read views
    TRACE( "  [" );
    for(size_t i = 0 ; i < files.size() ; i++) {

        if ( (i%S)==0 ) TRACE( "." );
        TRACE5( "   reading image " << files[i] );

        // HACK: ASSUMES FORMAT "out_%02i_%02i"
        if ( lf_name != "out_%02i_%02i.png" ) {
            ERROR( "Unsupported format string." << std::endl );
            assert( false );
            continue;
        }

        // stanford format: first two entries are s and t
        const char *buf = files[i].c_str();
        int s = (buf[7]-'0') * 10 + (buf[8]-'0');
        int t = (buf[4]-'0') * 10 + (buf[5]-'0');
        int viewIndex = S*(t - _tMin) + (s - _sMin);
        TRACE5( "  coords " << s << " " << t << std::endl );
        // Check if view ok.
        if ( s < _sMin ||  _sMax < s ) {
            // ERROR( "  index s " << s << " out of range." << std::endl );
            continue;
        }
        if ( t < _tMin ||  _tMax < t ) {
            // ERROR( "  index t " << t << " out of range." << std::endl );
            continue;
        }

        InputView *v_k = new InputView( _camWidth, _camHeight, _outdir, _pyramidHeight );

        // views are not loaded in the right order, so don't pushback
        _vCam.resize(S*T);

        if( v_k->importTexture((lf_dir + files[i]).c_str()) ) {

            v_k->importCamParametersStanford(-(t - _tMin - T/2), (s - _sMin - S/2));

            v_k->initDepthMap();

            Texture* depthMapTmp(0);

            uint scale = 0;

            depthMapTmp = v_k->initDepthAndNormals();

            v_k->addDepthScale(depthMapTmp, _FBO, scale, _addDepthScaleShader);

            _vCam[viewIndex] = v_k;

            // import the view to synthetize also, since we remove it only when running cocolib
            if ( s == _sRmv && t == _tRmv) {

                _renderIndex = viewIndex;
            }

        } else {

            assert(s != _sRmv && t != _tRmv);
            delete v_k;
            v_k = 0;
        }
    }

    TRACE( "] done." << std::endl );
    TRACE( "switched back to dir " << coco::Directory::current() << std::endl );
    // -------------------------------------------------------

    _nbCameras = S*T;

    std::cout << "done!" << std::endl;

    checkGLErrors();
}

// load standford lightfield dataset (image and camera matrices)
// camera matrices are obainted thank to openMVG calibration
// STANFORD FORMAT (rows and columns)
void Scene::importStanfordOriginalViews( std::string cameraName, std::string imageName ) {

    std::cout << "Import cameras and images" << std::endl;

    uint viewIndex = 0;

    for ( int t = _tMin ; t <= _tMax ; ++t ) {

        for ( int s = _sMin ; s <= _sMax ; ++s ) {

            char cameraNameChar[500];
            char imageNameChar[500];

            sprintf( cameraNameChar, cameraName.c_str(), t, s );
            sprintf( imageNameChar, imageName.c_str(), t, s );

            std::cout << "Import camera " << cameraNameChar << " and image " << imageNameChar << " ..." << std::endl;

            InputView *v_k = new InputView( _camWidth, _camHeight, _outdir, _pyramidHeight );

            // Import camera parameters and load vbos
            if( v_k->importTexture(imageNameChar) ) {

                // Import camera parameters and load vbos
                if( v_k->importCamParameters(cameraNameChar) ) {

                    _vCam.push_back( v_k );

                    // import the view to synthetize also, since we remove it only when running cocolib
                    if ( s == _sRmv ) {

                        _uCam = new TargetView( _camWidth, _camHeight, v_k->getPinholeCamera(), _outdir, _pyramidHeight );
                        // empty depth map
                        _uCam->initDepthMap();
                        // clear depth (set first channel to invalid value)
                        _FBO->clearTexture( _uCam->getDepthMap(), INVALID_DEPTH );

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

    // in the case we don't want to generate an input view
    if(_uCam == 0) {

        PinholeCamera pinholeCamera;
        _uCam = new TargetView( _camWidth, _camHeight, pinholeCamera, _outdir, _pyramidHeight );
        // empty depth map
        _uCam->initDepthMap();
        // clear depth (set first channel to invalid value)
        _FBO->clearTexture( _uCam->getDepthMap(), INVALID_DEPTH );
    }

    std::cout << "done!" << std::endl;
}

