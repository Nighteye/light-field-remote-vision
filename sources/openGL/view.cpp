#include "view.h"
#include "frameBuffer.h"
#include "pyramid.h"

#include "glm/ext.hpp"

#include <fstream>
#include <string>

InputView::InputView( uint W, uint H, std::string outdir, int pyramidHeight ) :

    _pyramid(0),
    _W(W), _H(H), _outdir(outdir), _pyramidHeight(pyramidHeight),
    _cameraWireShader( "sources/openGL/shaders/triangleMesh.vert", "sources/openGL/shaders/triangleMesh.frag" ),
    _cameraTextureShader( "sources/openGL/shaders/texture.vert", "sources/openGL/shaders/texture.frag" ),
    _cameraFrustumVAO(0), _cameraFrustumVBO(0), _cameraFrustumEBO(0),
    _cameraCentersVAO(0), _cameraCentersVBO(0),
    _cameraImagetteVAO(0), _imagetteVertVBO(0), _imagetteTexCoordVBO(0), _imagetteEBO(0),
    _depthMap(0),
    _texture(0),
    _mask(0) {

    _pyramid = new Pyramid(W, H);

    checkGLErrors();

    // set up the array of vertices of wired camera mesh

    checkGLErrors();

    initCameraWireShader( &_cameraWireShader );
    initCameraTextureShader( &_cameraTextureShader );

    checkGLErrors();
}

InputView::~InputView( ) {

    checkGLErrors();

    glDeleteVertexArrays(1, &_cameraFrustumVAO);
    glDeleteBuffers(1, &_cameraFrustumVBO);
    glDeleteBuffers(1, &_cameraFrustumEBO);

    glDeleteVertexArrays(1, &_cameraCentersVAO);
    glDeleteBuffers(1, &_cameraCentersVBO);

    glDeleteVertexArrays(1, &_cameraImagetteVAO);
    glDeleteBuffers(1, &_imagetteVertVBO);
    glDeleteBuffers(1, &_imagetteTexCoordVBO);
    glDeleteBuffers(1, &_imagetteEBO);

    if(_texture != 0) {
        delete _texture;
        _texture = 0;
    }

    if(_mask != 0) {
        delete _mask;
        _mask = 0;
    }

    if(_depthMap != 0) {
        delete _depthMap;
        _depthMap = 0;
    }

    if(_pyramid != 0) {
        delete _pyramid;
        _pyramid = 0;
    }

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

    for(uint i = 0 ; i < _depthPyramidTex.size() ; ++i) {
        if(_depthPyramidTex[i] != 0) {
            delete _depthPyramidTex[i];
            _depthPyramidTex[i] = 0;
        }
    }
    for(uint i = 0 ; i < _depthPyramidArray.size() ; ++i) {
        if(_depthPyramidArray[i] != 0) {
            delete[] _depthPyramidArray[i];
            _depthPyramidArray[i] = 0;
        }
    }

    checkGLErrors();
}

// Import camera image from PNG file
bool InputView::importTexture( const char* imageName ) {

    std::ifstream in( imageName, std::ifstream::in );

    if(in.is_open()) {

        _texture = new Texture(imageName);
        _texture->load();
        return true;

    } else {

        return false;
    }
}

// Import mask from PNG file (camouflage)
bool InputView::importMask( const char* maskName, float &count ) {

    std::ifstream in( maskName, std::ifstream::in );

    if(in.is_open()) {

        _mask = new Texture(maskName);
        _mask->load();

        float imCount = 0;
        if(true) {

            const uint nbChannels = 3;
            std::vector< std::vector<float> > mask(_W*_H);

            getMask(mask, nbChannels);

            for(uint i = 0 ; i < mask.size() ; ++i) {
                for(uint c = 0 ; c < mask[i].size() ; ++c) {
//                    std::cout << mask[i][c] << "  ";
                    if(mask[i][c] > 0.5) {
                        imCount += 1.0;
                    }
                }
//                std::cout << std::endl;
            }

            imCount /= nbChannels*_W*_H;
            // std::cout << "Ratio of occluded areas in source image " << maskName << ": " << imCount << std::endl;
            count += imCount;
        }

        return true;

    } else {

        return false;
    }
}

// Import camera parameters and load vbos
// Read the parameters from Stanford images files
bool InputView::importCamParametersStanford(double centerX, double centerY) {

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

    // load vbos
    load();

    return true;
}

// Import camera parameters and load vbos
// Read the camera matrices from INI file (MVE format)
bool InputView::importCamParameters( char *cameraName ) {

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

        // load vbos
        load();

        return true;
    }
}

// Read an MVEI file (MVE format)
bool InputView::importMVEIFile( char *fileName, std::vector< std::vector<float> > &data, uint &W, uint &H ) {

    if(! std::ifstream( fileName )) {

        std::cout << "File " << std::string(fileName) << " doesn't exist!" << std::endl;
        return false;

    } else {

        // std::cout << "Import MVE file " << std::string(fileName) << std::endl;

        std::ifstream in(fileName, std::ios::binary);

        assert( in.is_open() );
        assert( in );

        // header
        char buff[1000];
        in.getline(buff, 1000);

        uint channels, type;
        in.read(reinterpret_cast<char*>(&W), sizeof(W));
        in.read(reinterpret_cast<char*>(&H), sizeof(H));
        in.read(reinterpret_cast<char*>(&channels), sizeof(channels));
        in.read(reinterpret_cast<char*>(&type), sizeof(type));


        // std::cout << "W: " << W << " H: " << H << " channels: " << channels << " type: " << type << std::endl;

        data.resize(channels);
        for(uint i = 0 ; i < channels ; ++i) {
            data[i].resize(W*H);
        }

        for(uint i = 0 ; i < H ; ++i) {
            for(uint j = 0 ; j < W ; ++j) {
                for(uint k = 0 ; k < channels ; ++k) {

                    assert(in.read(reinterpret_cast<char*>(&data[k][i*W+j]), sizeof(float)));
                }
            }
        }

        in.close();

        return true;
    }
}

void InputView::load( ) {

    checkGLErrors();

    // Frustum, wired camera triangulation

    // The vertices containing the cameras frustum (4 for the imagette plus the optical center)
    const uint nbFrustumVertices = 5 * 3;
    GLfloat frustumVertices[nbFrustumVertices];

    // the indices of the triangles
    const uint nbFrustumIndices =  4 * 3;
    GLushort frustumIndices[nbFrustumIndices] = { 0,1,2, 0,2,3, 0,3,4, 0,4,1 };

    const uint nbImagetteVertices = 4 * 3;
    GLfloat imagetteVertices[nbImagetteVertices];

    // 4 Vertices * 2 texture coordinates
    const uint nbImagetteTexCoord = 4 * 2;
    GLfloat imagetteTexCoord[nbImagetteTexCoord] = { 0.,0., (GLfloat)_W,0., (GLfloat)_W,(GLfloat)_H, 0.,(GLfloat)_H };

    // the indices of the imagettes
    const uint nbImagetteIndices = 4;
    GLushort imagetteIndices[nbImagetteIndices] = { 0, 1, 3, 2 };

    // compute image corners coordinated with normalized focal (f=normalized_focal)
    float normalized_focal = 0.5f;

    float focal = _pinholeCamera._K[0][0];
    // use principal point to adjust image center
    glm::vec2 pp(_pinholeCamera._K[2][0], _pinholeCamera._K[2][1]);

    // the corners of the image in the frustum
    glm::mat4x3 c;
    c[0] = glm::vec3(    -pp[0]/focal * normalized_focal, (-pp[1]+_H)/focal * normalized_focal, normalized_focal);
    c[1] = glm::vec3((-pp[0]+_W)/focal * normalized_focal, (-pp[1]+_H)/focal * normalized_focal, normalized_focal);
    c[2] = glm::vec3((-pp[0]+_W)/focal * normalized_focal,     -pp[1]/focal * normalized_focal, normalized_focal);
    c[3] = glm::vec3(    -pp[0]/focal * normalized_focal,     -pp[1]/focal * normalized_focal, normalized_focal);

    // Convert to World coordinates
    for ( int i = 0 ; i < 4 ; ++i ) {
        c[i] = glm::transpose(_pinholeCamera._R) * c[i] + _pinholeCamera._C;
    }

    //    std::cout << glm::to_string(c) << std::endl;

    // add to structure
    // first point is optical center
    for( int j = 0 ; j < 3 ; ++j ) {
        frustumVertices[j] = _pinholeCamera._C[j];
    }

    for ( int i = 0 ; i < 4 ; ++i ) {
        for( int j = 0 ; j < 3 ; ++j ) {
            // Vertices vector has groups of 5 Vertices. The first is the optical center, then the frustum Vertices
            frustumVertices[3 + i*3 + j] = c[i][j];
            // imagette Vertices vector has only the 4 frustum Vertices
            imagetteVertices[i*3 + j] = c[i][j];
        }
    }

    // Camera centers, optical center and image center
    glm::vec3 opticalAxis = glm::transpose(_pinholeCamera._R) * glm::vec3( 0.0f, 0.0f, 1.0f );
    glm::vec3 imageCenter = _pinholeCamera._C + normalized_focal * opticalAxis;

    int nbCenters = 2*3;
    GLfloat centers[nbCenters] = { _pinholeCamera._C[0], _pinholeCamera._C[1], _pinholeCamera._C[2],
                                   imageCenter[0], imageCenter[1], imageCenter[2] };

    // --------------- generate camera frustum VAO/VBO/EBO -------------------------------

    if(glIsVertexArray(_cameraFrustumVAO) == GL_TRUE) {
        glDeleteVertexArrays(1, &_cameraFrustumVAO);
    }
    if(glIsBuffer(_cameraFrustumVBO) == GL_TRUE) {
        glDeleteBuffers(1, &_cameraFrustumVBO);
    }
    if(glIsBuffer(_cameraFrustumEBO) == GL_TRUE) {
        glDeleteBuffers(1, &_cameraFrustumEBO);
    }
    glGenVertexArrays(1, &_cameraFrustumVAO);
    glGenBuffers(1, &_cameraFrustumVBO);
    glGenBuffers(1, &_cameraFrustumEBO);

    glBindVertexArray(_cameraFrustumVAO);

    glBindBuffer(GL_ARRAY_BUFFER, _cameraFrustumVBO);
    glBufferData(GL_ARRAY_BUFFER, nbFrustumVertices * sizeof(GLfloat), frustumVertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _cameraFrustumEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, nbFrustumIndices * sizeof(GLushort), frustumIndices, GL_STATIC_DRAW);

    // Vertex Positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);

    // --------------- generate camera frustum VAO/VBO/EBO -------------------------------

    if(glIsVertexArray(_cameraCentersVAO) == GL_TRUE) {
        glDeleteVertexArrays(1, &_cameraCentersVAO);
    }
    if(glIsBuffer(_cameraCentersVBO) == GL_TRUE) {
        glDeleteBuffers(1, &_cameraCentersVBO);
    }
    glGenVertexArrays(1, &_cameraCentersVAO);
    glGenBuffers(1, &_cameraCentersVBO);

    glBindVertexArray(_cameraCentersVAO);

    glBindBuffer(GL_ARRAY_BUFFER, _cameraCentersVBO);
    glBufferData(GL_ARRAY_BUFFER, nbCenters * sizeof(GLfloat), centers, GL_STATIC_DRAW);

    // Vertex Positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);

    // --------------- generate imagette points and quads (triangle strip) ---------------

    if(glIsVertexArray(_cameraImagetteVAO) == GL_TRUE) {
        glDeleteVertexArrays(1, &_cameraImagetteVAO);
    }
    if(glIsBuffer(_imagetteVertVBO) == GL_TRUE) {
        glDeleteBuffers(1, &_imagetteVertVBO);
    }
    if(glIsBuffer(_imagetteTexCoordVBO) == GL_TRUE) {
        glDeleteBuffers(1, &_imagetteTexCoordVBO);
    }
    if(glIsBuffer(_imagetteEBO) == GL_TRUE) {
        glDeleteBuffers(1, &_imagetteEBO);
    }
    glGenVertexArrays(1, &_cameraImagetteVAO);
    glGenBuffers(1, &_imagetteVertVBO);
    glGenBuffers(1, &_imagetteTexCoordVBO);
    glGenBuffers(1, &_imagetteEBO);

    glBindVertexArray(_cameraImagetteVAO);

    glBindBuffer(GL_ARRAY_BUFFER, _imagetteVertVBO);
    glBufferData(GL_ARRAY_BUFFER, nbImagetteVertices * sizeof(GLfloat), imagetteVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, _imagetteTexCoordVBO);
    glBufferData(GL_ARRAY_BUFFER, nbImagetteTexCoord * sizeof(GLfloat), imagetteTexCoord, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _imagetteEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, nbImagetteIndices * sizeof(GLushort), imagetteIndices, GL_STATIC_DRAW);

    glBindVertexArray(0);

    checkGLErrors();
}

void InputView::display( const glm::mat4 &renderMatrix ) {

    checkGLErrors();

    glUseProgram( _cameraWireShader.getProgramID() );

    // --------------- Draw wired triangulation (frustum) ------------------------

    glEnable( GL_LINE_SMOOTH );
    glEnable( GL_POLYGON_SMOOTH );
    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
    glEnable( GL_DEPTH_TEST );
    glCullFace( GL_BACK );

    glBindVertexArray(_cameraFrustumVAO);

    // send render matrix
    _cameraWireShader.setUniformMat4("modelviewProjection", renderMatrix);

    GLfloat redColor[] = {1.,0.,0.};
    _cameraWireShader.setUniform3fv("in_color", redColor);

    // number of vertex making the triangles : N triangles = 3* vertex
    int nbFrustumIndices = 4*3;

    glDrawElements( GL_TRIANGLES, nbFrustumIndices, GL_UNSIGNED_SHORT, 0 );

    glBindVertexArray(0);

    glDisable( GL_DEPTH_TEST );
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
    glDisable( GL_POLYGON_SMOOTH );
    glDisable( GL_LINE_SMOOTH );

    // --------------- Draw camera optical center and image center ---------------

    glPointSize(3);

    glBindVertexArray(_cameraCentersVAO);

    // optical center
    GLfloat greenColor[] = {0.,1.,0.};
    _cameraWireShader.setUniform3fv("in_color", greenColor);
    glDrawArrays( GL_POINTS, 0, 1 );

    // image center
    GLfloat whiteColor[] = {1.,1.,1.};
    _cameraWireShader.setUniform3fv("in_color", whiteColor);
    glDrawArrays( GL_POINTS, 1, 1 );

    glBindVertexArray(0);

    glUseProgram(0);

    // --------------- Draw imagette ---------------------------------------------

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram( _cameraTextureShader.getProgramID() );

    glBindVertexArray( _cameraImagetteVAO );

    _cameraTextureShader.setUniformMat4( "modelviewProjection", renderMatrix );
    _cameraTextureShader.setUniformi( "myTexture", 0 );
    _cameraTextureShader.setUniformf( "alpha", 0.6 );
    _cameraTextureShader.setUniformi( "H", _H );

    glBindTexture(GL_TEXTURE_RECTANGLE, _texture->getID());

    glDrawElements( GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_SHORT, 0 );

    glBindTexture(GL_TEXTURE_RECTANGLE, 0);

    glBindVertexArray(0);

    glUseProgram(0);

    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);

    checkGLErrors();
}

void InputView::initCameraWireShader( Shader *shader ) {

    checkGLErrors();

    assert( shader->add() );

    // Bind shader inputs
    shader->bindAttribLocation( 0, "in_position" );

    assert( shader->link() );

    checkGLErrors();
}

void InputView::initCameraTextureShader( Shader *shader ) {

    assert( shader->add() );

    // Bind shader inputs
    shader->bindAttribLocation( 0, "in_Vertex" );
    shader->bindAttribLocation( 1, "in_TexCoord0" );

    assert( shader->link() );
}

void InputView::setTexMappingUniform( Shader *triangleMeshShader ) {

    triangleMeshShader->setUniformi( "myTexture", 0 );
    triangleMeshShader->setUniformf( "vi_width", _W );
    triangleMeshShader->setUniformf( "vi_height", _H );
    triangleMeshShader->setUniformMat4x3( "vi_P", _pinholeCamera._P );
    triangleMeshShader->setUniformMat3( "vi_R", _pinholeCamera._R );
    triangleMeshShader->setUniform3fv( "vi_C", _pinholeCamera._C );
}

Texture* InputView::setDepthAndNormals(char *depthName, char *normalName) {

    std::vector< std::vector<float> > depthMap, normalMap;
    uint depthMapW(0), depthMapH(0), normalMapW(0), normalMapH(0);

    bool ok;
    ok = importMVEIFile(depthName, depthMap, depthMapW, depthMapH);
    ok = ok && importMVEIFile(normalName, normalMap, normalMapW, normalMapH);

    assert(depthMapW == normalMapW && depthMapH == normalMapH);

    std::vector< std::vector<float> > combinedDepthMap;

    combinedDepthMap.push_back(depthMap[0]);
    combinedDepthMap.push_back(normalMap[0]);
    combinedDepthMap.push_back(normalMap[1]);

    if(ok) {

        Texture *combinedDepthMapTex = new Texture(0, _W, _H, GL_RGB, GL_FLOAT, GL_RGB32F, true);

        combinedDepthMapTex->loadFromData(combinedDepthMap, _W/depthMapW, (float)INVALID_DEPTH, false);

        return combinedDepthMapTex;
    } else {
        return 0;
    }
}

// when no depth data is available (refocussing, no preliminary reconstruction), standard initialization
Texture* InputView::initDepthAndNormals() {

    const uint mapSize = _W*_H;
    std::vector< std::vector<float> > depthMap(1), normalMap(2);
    depthMap[0].resize(mapSize);
    normalMap[0].resize(mapSize);
    normalMap[1].resize(mapSize);

    for(uint i = 0 ; i < mapSize ; ++i) {
        depthMap[0][i] = 0.0;
        normalMap[0][i] = 0.0;
        normalMap[1][i] = 0.0;
    }

    std::vector< std::vector<float> > combinedDepthMap;

    combinedDepthMap.push_back(depthMap[0]);
    combinedDepthMap.push_back(normalMap[0]);
    combinedDepthMap.push_back(normalMap[1]);

    Texture *combinedDepthMapTex = new Texture(0, _W, _H, GL_RGB, GL_FLOAT, GL_RGB32F, true);
    combinedDepthMapTex->loadFromData(combinedDepthMap, 1, (float)INVALID_DEPTH, false);

    return combinedDepthMapTex;
}

void InputView::exportWarp(char* lfName, char* tauName, char* dpartName,
                           TargetView *uCam, char* uWarpedName, Texture* textureToWarp, FrameBuffer* FBO,
                           Shader* tauWarpShader, Shader* tauPartialShader, Shader* warpVkShader) {

    // add a 3-channel color buffer for the warps (and deformation weights)

    Texture* _tauWarp = new Texture(0, _W, _H, GL_RGB, GL_FLOAT, GL_RGB32F, true);
    Texture* _tauPartial = new Texture(0, _W, _H, GL_RGB, GL_FLOAT, GL_RGB32F, true);
    Texture* _uWarped = new Texture(0, _W, _H, GL_RGB, GL_FLOAT, GL_RGB32F, true);
    _tauWarp->loadEmptyTexture();
    _tauPartial->loadEmptyTexture();
    _uWarped->loadEmptyTexture();

    PinholeCamera uPinholeCam = uCam->getPinholeCamera();

    FBO->computeTauWarps( uCam->getDepthMap(), _depthMap, _tauWarp,
                          uPinholeCam._K, uPinholeCam._R, uPinholeCam._t,
                          _pinholeCamera._K, _pinholeCamera._R, _pinholeCamera._t,
                          tauWarpShader );

    FBO->computeTauPartial( _depthMap, _tauPartial,
                            uPinholeCam._K, uPinholeCam._R, uPinholeCam._t,
                            _pinholeCamera._K, _pinholeCamera._R, _pinholeCamera._t,
                            tauPartialShader );

    if(lfName != 0) {
        //std::cout << "Save view in " << lfName << std::endl;
        _texture->saveRGBAIntTexture(_W, _H, 3, std::string(lfName), false);
    } else {
        //std::cout << "View path is empty" << std::endl;
    }

    if(uWarpedName != 0) {
        FBO->warpVk( _tauWarp, textureToWarp, _uWarped,
                     warpVkShader );
        // std::cout << "Save warped target view in " << uWarpedName << std::endl;
        _uWarped->saveRGBAFloatTexture( _W, _H, 3, uWarpedName, false );
    } else {
        //std::cout << "Don't save the warped target view" << std::endl;
    }

    if(dpartName != 0) {
        //std::cout << "Save partial tau map in " << dpartName << std::endl;
        _tauPartial->saveRGBAFloatTexture( _W, _H, 3, dpartName, false );
    } else {
        //std::cout << "Partial tau path is empty" << std::endl;
    }

    if(tauName != 0) {
        //std::cout << "Save tau warp in " << tauName << std::endl;
        _tauWarp->saveRGBAFloatTexture( _W, _H, 3, tauName, false );
    } else {
        //std::cout << "Tau warp path is empty" << std::endl;
    }

    delete _tauWarp;
    delete _tauPartial;
    delete _uWarped;
}

void InputView::filterDepthmap(FrameBuffer* FBO, Shader* fromRadial2OrthoShader, char *filteredDepthNameChar) {

    FBO->fromRadial2Ortho( _depthMap, _pinholeCamera._K, fromRadial2OrthoShader );
    // bilateralFiltering( _texture->getID(), *tempIndex, *_depthMapIndex );

    if(filteredDepthNameChar != 0) {

        std::cout << "Save filtered depth map in " << filteredDepthNameChar << std::endl;
        saveDepthMap(std::string(filteredDepthNameChar));
    }
}

// Multi-scale depth map accumulation
void InputView::addDepthScale(Texture* depthMapToAdd, FrameBuffer* FBO, uint scale, Shader* addDepthScaleShader) {

    // clear depth (set first channel to invalid value)
    FBO->clearTexture( _depthMap, INVALID_DEPTH );

    FBO->addDepthScale(depthMapToAdd, _depthMap, scale, addDepthScaleShader);
}

//void InputView::createGaussianPyramid(uint k) {

//    if(_gaussianPyramid != 0) {

//        delete _gaussianPyramid;
//    }

//    _gaussianPyramid = new Pyramid(_W, _H);

//    _gaussianPyramid->addInitialImage(*_texture);

//    const bool verbose = false;

//    std::string gaussianName = _outdir + "/gaussianScale%02i_%02i.pfm";
//    std::string laplacianName = _outdir + "/laplacianScale%02i_%02i.pfm";
//    char tmpNameChar[500];

//    if(verbose) {
//        sprintf( tmpNameChar, gaussianName.c_str(), 0, k );
//        std::cout << "Save original image in " << tmpNameChar << std::endl;
//        _gaussianPyramid->saveColorBuffer(tmpNameChar, 0);
//    }

//    // for every scale. don't go to max scale
//    uint s = 0;
//    while(_W / (uint)pow(2.0, (double)s+4) > 0 && _H / (uint)pow(2.0, (double)s+4) > 0) {

//        // add the scale s+1
//        _gaussianPyramid->addScale();
//        // reduce the scale s to s+1
//        _gaussianPyramid->reduce(s);

//        if(verbose) {
//            sprintf( tmpNameChar, gaussianName.c_str(), s+1, k );
//            std::cout << "Save scale " << s+1 << " image in " << tmpNameChar << std::endl;
//            _gaussianPyramid->saveColorBuffer(tmpNameChar, s+1);
//        }

//        _gaussianPyramid->computeLaplacian(s);

//        if(verbose) {
//            sprintf( tmpNameChar, laplacianName.c_str(), s, k );
//            std::cout << "Save Laplacian scale " << s << " image in " << tmpNameChar << std::endl;
//            _gaussianPyramid->saveColorBuffer(tmpNameChar, s);
//        }

//        ++s;
//    }
//}

void InputView::createGaussianScaleSpace(uint k) {

    const uint nbChannels = 3;
    assert(_texture->getInternalFormat() == GL_RGB);
    const bool verbose = false;

    unsigned char *charBuffer = new unsigned char[nbChannels*_W*_H];
    float *floatBuffer = new float[nbChannels*_W*_H];

    glBindTexture(GL_TEXTURE_RECTANGLE, _texture->getID());
    glGetTexImage( GL_TEXTURE_RECTANGLE, 0, _texture->getFormat(), _texture->getType(), charBuffer );
    glBindTexture(GL_TEXTURE_RECTANGLE, 0);

    for(uint i = 0 ; i < nbChannels*_W*_H ; ++i) {

        floatBuffer[i] = (float)charBuffer[i] / 255.0f;
    }
    delete[] charBuffer;
    _pyramid->_gaussianPyramidArray.push_back(floatBuffer);

    std::string gaussianName = _outdir + "/gaussian2Scale%02i_%02i.pfm";
    std::string laplacianName = _outdir + "/laplacian2Scale%02i_%02i.pfm";
    char tmpNameChar[500];

    Texture* texture = new Texture(0, _W, _H, GL_RGB, GL_FLOAT, GL_RGB32F, false);

    texture->loadFromData(_pyramid->_gaussianPyramidArray[0]);
    _pyramid->_gaussianPyramidTex.push_back(texture);

    if(verbose) {
        sprintf( tmpNameChar, gaussianName.c_str(), 0, k );
        std::cout << "Save original image in " << tmpNameChar << std::endl;
        _pyramid->_gaussianPyramidTex[0]->saveRGBAFloatTexture(_W, _H, nbChannels, std::string(tmpNameChar), false);
    }

    // for every scale. don't go to max scale
    // uint s = 0;
    // while(_W / (uint)pow(2.0, (double)s) > 0 && _H / (uint)pow(2.0, (double)s) > 0) {
    for(uint s = 1 ; s <= (uint)_pyramidHeight ; ++s) {

        float *gaussianArray = new float[nbChannels*_W*_H];
        Texture* gaussianTex = new Texture(0, _W, _H, GL_RGB, GL_FLOAT, GL_RGB32F, false);

        memset(gaussianArray, 0, nbChannels*_W*_H*sizeof(float));
        _pyramid->_gaussianPyramidArray.push_back(gaussianArray);

        _pyramid->oddHDC(_W, _H, nbChannels, s);

        gaussianTex->loadFromData(_pyramid->_gaussianPyramidArray[s]);
        _pyramid->_gaussianPyramidTex.push_back(gaussianTex);

        if(verbose) {
            sprintf( tmpNameChar, gaussianName.c_str(), s, k );
            std::cout << "Save scale " << s << " image in " << tmpNameChar << std::endl;
            _pyramid->_gaussianPyramidTex[s]->saveRGBAFloatTexture(_W, _H, nbChannels, std::string(tmpNameChar), false);
        }

        float *laplacianArray = new float[nbChannels*_W*_H]; // scale (s-1)
        Texture* laplacianTex = new Texture(0, _W, _H, GL_RGB, GL_FLOAT, GL_RGB32F, false);

        memset(laplacianArray, 0, nbChannels*_W*_H*sizeof(float));
        _pyramid->_laplacianPyramidArray.push_back(laplacianArray);

        _pyramid->dog(_W, _H, nbChannels, s-1);

        laplacianTex->loadFromData(_pyramid->_laplacianPyramidArray[s-1]);
        _pyramid->_laplacianPyramidTex.push_back(laplacianTex);

        if(verbose) {
            sprintf( tmpNameChar, laplacianName.c_str(), s-1, k );
            std::cout << "Save scale " << s-1 << " Laplacian in " << tmpNameChar << std::endl;
            _pyramid->_laplacianPyramidTex[s-1]->saveRGBAFloatTexture(_W, _H, nbChannels, std::string(tmpNameChar), false);
        }
    }
}

void InputView::createGaussianPyramidCPU(uint k) {

    uint nbChannels = 0;
    const bool verbose = false;

    // check internal format
    if(_texture->getInternalFormat() == GL_RGB) {

        nbChannels = 3;

    } else if(_texture->getInternalFormat() == GL_RGBA) {

        nbChannels = 4;

    } else {

        assert(false);
    }

    unsigned char *charBuffer = new unsigned char[nbChannels*_W*_H];
    float *floatBuffer = new float[nbChannels*_W*_H];

    glBindTexture(GL_TEXTURE_RECTANGLE, _texture->getID());
    glGetTexImage( GL_TEXTURE_RECTANGLE, 0, _texture->getFormat(), _texture->getType(), charBuffer );
    glBindTexture(GL_TEXTURE_RECTANGLE, 0);

    for(uint i = 0 ; i < nbChannels*_W*_H ; ++i) {

        floatBuffer[i] = (float)charBuffer[i] / 255.0f;
    }
    delete[] charBuffer;
    _gaussianPyramidArray.push_back(floatBuffer);

    std::string gaussianName = _outdir + "/gaussianScale%02i_%02i.pfm";
    std::string laplacianName = _outdir + "/laplacianScale%02i_%02i.pfm";
    char tmpNameChar[500];

    Texture* texture(0);

    if(nbChannels == 3) {
        texture = new Texture(0, _W, _H, GL_RGB, GL_FLOAT, GL_RGB32F, false);
    } else {
        texture = new Texture(0, _W, _H, GL_RGBA, GL_FLOAT, GL_RGBA32F, false);
    }

    texture->loadFromData(_gaussianPyramidArray[0]);
    _gaussianPyramidTex.push_back(texture);

    if(verbose) {
        sprintf( tmpNameChar, gaussianName.c_str(), 0, k );
        std::cout << "Save original image in " << tmpNameChar << std::endl;
        _gaussianPyramidTex[0]->saveRGBAFloatTexture(_W, _H, nbChannels, std::string(tmpNameChar), false);
    }

    // for every scale. don't go to max scale
    // uint s = 0;
    // while(_W / (uint)pow(2.0, (double)s) > 0 && _H / (uint)pow(2.0, (double)s) > 0) {
    for(uint s = 1 ; s <= (uint)_pyramidHeight ; ++s) {

        // ++s;

        uint W = _W / (uint)pow(2.0, (double)(s-1));
        uint H = _H / (uint)pow(2.0, (double)(s-1));
        uint w = _W / (uint)pow(2.0, (double)s);
        uint h = _H / (uint)pow(2.0, (double)s);

        float *gaussianArray = new float[nbChannels*w*h];
        Texture* gaussianTex(0);
        if(nbChannels == 3) {
            gaussianTex = new Texture(0, w, h, GL_RGB, GL_FLOAT, GL_RGB32F, false);
        } else {
            gaussianTex = new Texture(0, w, h, GL_RGBA, GL_FLOAT, GL_RGBA32F, false);
        }

        memset(gaussianArray, 0, nbChannels*w*h*sizeof(float));

        reduceGaussian(W, H, w, h, nbChannels, _gaussianPyramidArray[s-1], gaussianArray);

        _gaussianPyramidArray.push_back(gaussianArray);
        gaussianTex->loadFromData(_gaussianPyramidArray[s]);
        _gaussianPyramidTex.push_back(gaussianTex);

        if(verbose) {
            sprintf( tmpNameChar, gaussianName.c_str(), s, k );
            std::cout << "Save scale " << s << " image in " << tmpNameChar << std::endl;
            _gaussianPyramidTex[s]->saveRGBAFloatTexture(w, h, nbChannels, std::string(tmpNameChar), false);
        }

        float *laplacianArray = new float[nbChannels*W*H]; // scale (s-1)
        Texture* laplacianTex(0);
        if(nbChannels == 3) {
            laplacianTex = new Texture(0, W, H, GL_RGB, GL_FLOAT, GL_RGB32F, false);
        } else {
            laplacianTex = new Texture(0, W, H, GL_RGBA, GL_FLOAT, GL_RGBA32F, false);
        }

        memset(laplacianArray, 0, nbChannels*W*H*sizeof(float));
        expandGaussian(W, H, w, h, nbChannels, _gaussianPyramidArray[s], laplacianArray);
        if(verbose) {
            laplacianTex->loadFromData(laplacianArray);
            std::string expandedName = _outdir + "/expandedScale%02i_%02i.pfm";
            sprintf( tmpNameChar, expandedName.c_str(), s-1, k );
            std::cout << "Save scale " << s-1 << " expanded image in " << tmpNameChar << std::endl;
            laplacianTex->saveRGBAFloatTexture(W, H, nbChannels, std::string(tmpNameChar), false);
        }
        computeLaplacian(W, H, nbChannels, _gaussianPyramidArray[s-1], laplacianArray);

        _laplacianPyramidArray.push_back(laplacianArray);
        laplacianTex->loadFromData(_laplacianPyramidArray[s-1]);
        _laplacianPyramidTex.push_back(laplacianTex);

        if(verbose) {
            sprintf( tmpNameChar, laplacianName.c_str(), s-1, k );
            std::cout << "Save scale " << s-1 << " Laplacian in " << tmpNameChar << std::endl;
            _laplacianPyramidTex[s-1]->saveRGBAFloatTexture(W, H, nbChannels, std::string(tmpNameChar), false);
        }
    }
}

void InputView::createDepthPyramidCPU(uint k, int depthPyramidHeight) {

    const uint nbChannels = 3;
    const bool verbose = false;

    assert(_depthMap->getInternalFormat() == GL_RGB32F);

    float *floatBuffer = new float[nbChannels*_W*_H];

    glBindTexture(GL_TEXTURE_RECTANGLE, _depthMap->getID());
    glGetTexImage( GL_TEXTURE_RECTANGLE, 0, _depthMap->getFormat(), _depthMap->getType(), floatBuffer );
    glBindTexture(GL_TEXTURE_RECTANGLE, 0);

    _depthPyramidArray.push_back(floatBuffer);

    std::string gaussianName = _outdir + "/depthScale%02i_%02i.pfm";
    char tmpNameChar[500];

    Texture* initialScale(0);

    initialScale = new Texture(0, _W, _H, GL_RGB, GL_FLOAT, GL_RGB32F, false);

    initialScale->loadFromData(_depthPyramidArray[0]);
    _depthPyramidTex.push_back(initialScale);

    if(verbose) {
        sprintf( tmpNameChar, gaussianName.c_str(), 0, k );
        std::cout << "Save original depth map in " << tmpNameChar << std::endl;
        _depthPyramidTex[0]->saveRGBAFloatTexture(_W, _H, nbChannels, std::string(tmpNameChar), false);
    }

    // for every scale. don't go to max scale
    // uint s = 0;
    // while(_W / (uint)pow(2.0, (double)s) > 0 && _H / (uint)pow(2.0, (double)s) > 0) {
    for(uint s = 1 ; s <= (uint)depthPyramidHeight ; ++s) {

        // ++s;

        uint W = _W / (uint)pow(2.0, (double)(s-1));
        uint H = _H / (uint)pow(2.0, (double)(s-1));
        uint w = _W / (uint)pow(2.0, (double)s);
        uint h = _H / (uint)pow(2.0, (double)s);

        float *depthArray = new float[nbChannels*w*h];
        Texture* gaussianTex(0);
        if(nbChannels == 3) {
            gaussianTex = new Texture(0, w, h, GL_RGB, GL_FLOAT, GL_RGB32F, false);
        } else {
            gaussianTex = new Texture(0, w, h, GL_RGBA, GL_FLOAT, GL_RGBA32F, false);
        }

        memset(depthArray, 0, nbChannels*w*h*sizeof(float));

        oddHDCDepthReduce(W, H, w, h, nbChannels, _depthPyramidArray[s-1], depthArray);

        _depthPyramidArray.push_back(depthArray);
        gaussianTex->loadFromData(depthArray);
        _depthPyramidTex.push_back(gaussianTex);

        if(verbose) {
            sprintf( tmpNameChar, gaussianName.c_str(), s, k );
            std::cout << "Save scale " << s << " depth map in " << tmpNameChar << std::endl;
            _depthPyramidTex[s]->saveRGBAFloatTexture(w, h, nbChannels, std::string(tmpNameChar), false);
        }
    }

    for(uint i = 0 ; i < _depthPyramidArray.size() ; ++i) {
        if(_depthPyramidArray[i] != 0) {
            delete[] _depthPyramidArray[i];
            _depthPyramidArray[i] = 0;
        }
    }
}

// reduce depthmap according to the scale parameter
void InputView::createDepthScaleSpaceCPU(uint k, int depthScale) {

    const uint nbChannels = 3;
    const bool verbose = false;

    assert(_depthMap->getInternalFormat() == GL_RGB32F);

    // declare dimensions
    uint hiW(_W), hiH(_H), loW(_W), loH(_H);
    // init temp buffer
    float *tempDepth1 = new float[nbChannels*_W*_H];

    glBindTexture(GL_TEXTURE_RECTANGLE, _depthMap->getID());
    glGetTexImage( GL_TEXTURE_RECTANGLE, 0, _depthMap->getFormat(), _depthMap->getType(), tempDepth1 );
    glBindTexture(GL_TEXTURE_RECTANGLE, 0);

    std::string reducedName = _outdir + "/reducedDepthScale%02i_%02i.pfm";
    std::string expandedName = _outdir + "/expandedDepthScale%02i_%02i.pfm";
    char tmpNameChar[500];

    if(verbose) {
        sprintf( tmpNameChar, reducedName.c_str(), 0, k );
        std::cout << "Save original depth map in " << tmpNameChar << std::endl;
        _depthMap->saveRGBAFloatTexture(_W, _H, nbChannels, std::string(tmpNameChar), false);
    }

    // for every scale until the scale we want
    for(uint s = 1 ; s <= (uint)depthScale ; ++s) {

        hiW = _W / (uint)pow(2.0, (double)(s-1));
        hiH = _H / (uint)pow(2.0, (double)(s-1));
        loW = _W / (uint)pow(2.0, (double)s);
        loH = _H / (uint)pow(2.0, (double)s);

        float *tempDepth2 = new float[nbChannels*loW*loH];
        memset(tempDepth2, 0, nbChannels*loW*loH*sizeof(float));

        // reduce depth
        // oddHDCDepth(_W, _H, nbChannels, s, tempDepth1, tempDepth2);
        oddHDCDepthReduce(hiW, hiH, loW, loH, nbChannels, tempDepth1, tempDepth2);
        // reduceDepth(hiW, hiH, loW, loH, nbChannels, tempDepth1, tempDepth2);
        // oddGaussianDepth(_W, _H, nbChannels, s, tempDepth1, tempDepth2);

        if(verbose) {
            Texture* depthTex(0);
            if(nbChannels == 3) {
                depthTex = new Texture(0, loW, loH, GL_RGB, GL_FLOAT, GL_RGB32F, false);
            } else {
                depthTex = new Texture(0, loW, loH, GL_RGBA, GL_FLOAT, GL_RGBA32F, false);
            }
            depthTex->loadFromData(tempDepth2);
            sprintf( tmpNameChar, reducedName.c_str(), s, k );
            std::cout << "Save reduced scale " << s << " depth map in " << tmpNameChar << std::endl;
            depthTex->saveRGBAFloatTexture(loW, loH, nbChannels, std::string(tmpNameChar), false);
            delete depthTex;
        }

        delete[] tempDepth1;
        tempDepth1 = new float[nbChannels*loW*loH];
        memcpy(tempDepth1, tempDepth2, nbChannels*loW*loH*sizeof(float)); // cpy tempDepth2 to tempDepth1
        delete[] tempDepth2;
    }

    assert(depthScale > 0);
    // scale ascent, until the original scale
    for(uint s = depthScale ; s > 0 ; --s) {

        hiW = _W / (uint)pow(2.0, (double)(s-1));
        hiH = _H / (uint)pow(2.0, (double)(s-1));
        loW = _W / (uint)pow(2.0, (double)s);
        loH = _H / (uint)pow(2.0, (double)s);

        float *tempDepth2 = new float[nbChannels*hiW*hiH];
        memset(tempDepth2, 0, nbChannels*hiW*hiH*sizeof(float));

        // expand depth
        oddHDCDepthExpand(loW, loH, hiW, hiH, nbChannels, tempDepth1, tempDepth2);

        if(verbose) {
            Texture* depthTex(0);
            if(nbChannels == 3) {
                depthTex = new Texture(0, hiW, hiH, GL_RGB, GL_FLOAT, GL_RGB32F, false);
            } else {
                depthTex = new Texture(0, hiW, hiH, GL_RGBA, GL_FLOAT, GL_RGBA32F, false);
            }
            depthTex->loadFromData(tempDepth2);
            sprintf( tmpNameChar, expandedName.c_str(), s-1, k );
            std::cout << "Save expanded scale " << s-1 << " depth map in " << tmpNameChar << std::endl;
            depthTex->saveRGBAFloatTexture(hiW, hiH, nbChannels, std::string(tmpNameChar), false);
            delete depthTex;
        }

        delete[] tempDepth1;
        tempDepth1 = new float[nbChannels*hiW*hiH];
        memcpy(tempDepth1, tempDepth2, nbChannels*hiW*hiH*sizeof(float)); // cpy tempDepth2 to tempDepth1
        delete[] tempDepth2;
    }

    _depthMap->loadFromData(tempDepth1);

    delete[] tempDepth1;
}

void InputView::createTestInputView( float fov ) {

    float focal_length = 1.0 / (2*tan(fov*3.14/360));
    float pixel_aspect = 1.0;
    float principal_point[] = {0.5, 0.5};
    glm::mat3 R(1.0);
    glm::vec3 t(0.0);
    glm::mat3 K(0.0);

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

    t[0] = 0.0;
    t[1] = 0.0;
    t[2] = 5.0;

    _pinholeCamera = PinholeCamera( K, R, t, _W, _H );

    initViewTexture();
}

void InputView::drawTexture( FrameBuffer* FBO, Shader* textureShader ) {

    FBO->drawTexture( _texture, textureShader );
}

void InputView::saveDepthMap( const std::string &texName ) {

    _depthMap->saveRGBAFloatTexture( _W, _H, 3, texName, false );
}

// SETTERS

void InputView::initDepthMap() {

    _depthMap = new Texture(0, _W, _H, GL_RGB, GL_FLOAT, GL_RGB32F, true);
    _depthMap->loadEmptyTexture();
}

void InputView::initViewTexture() {

    _texture = new Texture(0, _W, _H, GL_RGB, GL_UNSIGNED_BYTE, GL_RGB, true);
    _texture->loadEmptyTexture();
}

void InputView::setPinholeCamera(const PinholeCamera &pinholeCamera) {

    _pinholeCamera = pinholeCamera;
}

// GETTERS

GLuint InputView::getTextureID() const {

    return _texture->getID();
}

PinholeCamera InputView::getPinholeCamera() const {

    return _pinholeCamera;
}

Texture* InputView::getTexture() const {

    return _texture;
}

std::vector<float>* InputView::getTextureVec() const {

    uint nbChannels = 3;
    unsigned char *charBuffer = new unsigned char[nbChannels*_W*_H];
    std::vector<float>* floatBuffer = new std::vector<float>(nbChannels*_W*_H);

    glBindTexture(GL_TEXTURE_RECTANGLE, _texture->getID());
    glGetTexImage(GL_TEXTURE_RECTANGLE, 0, _texture->getFormat(), _texture->getType(), charBuffer);
    glBindTexture(GL_TEXTURE_RECTANGLE, 0);

    for(uint i = 0 ; i < nbChannels*_W*_H ; ++i) {

        (*floatBuffer)[i] = (float)charBuffer[i] / 255.0f;
    }
    delete[] charBuffer;

    return floatBuffer;
}

void InputView::getTextureRGB(std::vector<cv::Point3f>& view) const {

    uint nbChannels = 3;
    unsigned char *charBuffer = new unsigned char[nbChannels*_W*_H];

    glBindTexture(GL_TEXTURE_RECTANGLE, _texture->getID());
    glGetTexImage(GL_TEXTURE_RECTANGLE, 0, _texture->getFormat(), _texture->getType(), charBuffer);
    glBindTexture(GL_TEXTURE_RECTANGLE, 0);

    for(uint i = 0 ; i < _W*_H ; ++i) {

        view[i].x = (float)charBuffer[i*nbChannels + 0] / 255.0f;
        view[i].y = (float)charBuffer[i*nbChannels + 1] / 255.0f;
        view[i].z = (float)charBuffer[i*nbChannels + 2] / 255.0f;
    }

    delete[] charBuffer;
}

void InputView::getMask(std::vector< std::vector<float> >& view, uint nbChannels) const {

    unsigned char *charBuffer = new unsigned char[nbChannels*_W*_H];

    glBindTexture(GL_TEXTURE_RECTANGLE, _mask->getID());
    glGetTexImage(GL_TEXTURE_RECTANGLE, 0, _mask->getFormat(), _mask->getType(), charBuffer);
    glBindTexture(GL_TEXTURE_RECTANGLE, 0);

    assert(_mask->getFormat() == GL_RGB);
    assert(_mask->getType() == GL_UNSIGNED_BYTE);

    for(uint i = 0 ; i < _W*_H ; ++i) {

        view[i].resize(nbChannels);
        for(uint c = 0 ; c < nbChannels ; ++c) {

            view[i][c] = (float)charBuffer[i*nbChannels + c] / 255.0f;
        }
    }

    delete[] charBuffer;
}

Texture* InputView::getDepthMap() const {

    return _depthMap;
}

Texture* InputView::getMask() const {

    return _mask;
}

TargetView::TargetView( uint W, uint H, const PinholeCamera &pinholeCamera, std::string outdir, int pyramidHeight ) :

    _pyramid(0),
    _depthMap(0),
    _W(W), _H(H),
    _outdir(outdir),
    _pyramidHeight(pyramidHeight),
    _pinholeCamera(pinholeCamera) {

    const uint nbChannels = 4; // RGB + weights

    _pyramid = new Pyramid(W, H);

    checkGLErrors();

    // -------------------------------- CREATE PYRAMID TEXTURES -------------------------------- //

    if(_pyramidHeight >= 0) {

        _pyramid->_laplacianPyramidTex.resize(_pyramidHeight);
        _pyramid->_laplacianPyramidArray.resize(_pyramidHeight);
        _pyramid->_gaussianPyramidArray.resize(_pyramidHeight+1);
        _pyramid->_gaussianPyramidTex.resize(_pyramidHeight+1);

        // Laplacian Pyramid
        for (uint scale = 0 ; scale < (uint)_pyramidHeight ; ++scale) {

            const int outR = (int)pow((double)2.0, (double)(scale));
            const int outW = W/outR;
            const int outH = H/outR;

            _pyramid->_laplacianPyramidArray[scale] = new float[nbChannels*outW*outH];
            _pyramid->_laplacianPyramidTex[scale] = new Texture(0, outW, outH, GL_RGBA, GL_FLOAT, GL_RGBA32F, false);
        }

        // Gaussian pyramid
        for (uint scale = 0 ; scale <= (uint)_pyramidHeight ; ++scale) {

            const int outR = (int)pow((double)2.0, (double)(scale));
            const int outW = W/outR;
            const int outH = H/outR;

            _pyramid->_gaussianPyramidArray[scale] = new float[nbChannels*outW*outH];
            _pyramid->_gaussianPyramidTex[scale] = new Texture(0, outW, outH, GL_RGBA, GL_FLOAT, GL_RGBA32F, false);
        }
    }

    initPyramid();

    //    if(_pyramidHeight >= 0) {

    //        _laplacianPyramidTex.resize(_pyramidHeight);
    //        _laplacianPyramidArray.resize(_pyramidHeight);
    //        _gaussianPyramidArray.resize(_pyramidHeight+1);
    //        _gaussianPyramidTex.resize(_pyramidHeight+1);

    //        // Laplacian Pyramid
    //        for (uint scale = 0 ; scale < (uint)_pyramidHeight ; ++scale) {

    //            uint w = W / (uint)pow(2.0, (double)scale);
    //            uint h = H / (uint)pow(2.0, (double)scale);

    //            _laplacianPyramidArray[scale] = new float[nbChannels*w*h];
    //            memset(_laplacianPyramidArray[scale], 0, nbChannels*w*h*sizeof(float));
    //            _laplacianPyramidTex[scale] = new Texture(0, w, h, GL_RGB, GL_FLOAT, GL_RGB32F, false);
    //        }

    //        // Gaussian pyramid
    //        for (uint scale = 0 ; scale <= (uint)_pyramidHeight ; ++scale) {

    //            uint w = W / (uint)pow(2.0, (double)scale);
    //            uint h = H / (uint)pow(2.0, (double)scale);

    //            _gaussianPyramidArray[scale] = new float[nbChannels*w*h];
    //            memset(_gaussianPyramidArray[scale], 0, nbChannels*w*h*sizeof(float));
    //            _gaussianPyramidTex[scale] = new Texture(0, w, h, GL_RGB, GL_FLOAT, GL_RGB32F, false);
    //        }
    //    }

    checkGLErrors();
}

TargetView::~TargetView( ) {

    checkGLErrors();

    if(_depthMap != 0) {
        delete _depthMap;
        _depthMap = 0;
    }

    if(_pyramid != 0) {
        delete _pyramid;
        _pyramid = 0;
    }

    for(uint i = 0 ; i < _gaussianPyramidTex.size() ; ++i) {

        if(_gaussianPyramidTex[i] != 0) {
            delete _gaussianPyramidTex[i];
        }
    }
    for(uint i = 0 ; i < _gaussianPyramidArray.size() ; ++i) {

        if(_gaussianPyramidArray[i] != 0) {
            delete[] _gaussianPyramidArray[i];
        }
    }

    for(uint i = 0 ; i < _laplacianPyramidTex.size() ; ++i) {
        if(_laplacianPyramidTex[i] != 0) {
            delete _laplacianPyramidTex[i];
        }
    }
    for(uint i = 0 ; i < _laplacianPyramidArray.size() ; ++i) {
        if(_laplacianPyramidArray[i] != 0) {
            delete[] _laplacianPyramidArray[i];
        }
    }

    checkGLErrors();
}

void TargetView::initPyramid() {

    const uint nbChannels = 4;

    // Laplacian Pyramid
    for (uint scale = 0 ; scale < (uint)_pyramidHeight ; ++scale) {

        const int outR = (int)pow((double)2.0, (double)(scale));
        const int outW = _W/outR;
        const int outH = _H/outR;

        memset(_pyramid->_laplacianPyramidArray[scale], 0, nbChannels*outW*outH*sizeof(float));
    }

    // Gaussian pyramid
    for (uint scale = 0 ; scale <= (uint)_pyramidHeight ; ++scale) {

        const int outR = (int)pow((double)2.0, (double)(scale));
        const int outW = _W/outR;
        const int outH = _H/outR;

        memset(_pyramid->_gaussianPyramidArray[scale], 0, nbChannels*outW*outH*sizeof(float));
    }
}

void TargetView::createDecomposeWarpedImage(uint inputScale, Texture* inputTex) {

    const uint nbChannels = 4;
    const bool verbose = false;

    Pyramid* pyramid = new Pyramid(_W, _H);

    float *floatBuffer = new float[nbChannels*_W*_H]; // from tex to array

    glBindTexture(GL_TEXTURE_RECTANGLE, inputTex->getID());
    glGetTexImage( GL_TEXTURE_RECTANGLE, 0, inputTex->getFormat(), inputTex->getType(), floatBuffer );
    glBindTexture(GL_TEXTURE_RECTANGLE, 0);

    pyramid->_gaussianPyramidArray.push_back(floatBuffer);

    std::string laplacianName = _outdir + "/decompositionScale%02i.tiff";
    char tmpNameChar[500];

    for(uint s = 1 ; s <= (uint)_pyramidHeight ; ++s) {

        const int inR = (int)pow((double)2.0, (double)(s-1));
        const int outR = (int)pow((double)2.0, (double)(s));
        const int W = _W/inR;
        const int H = _H/inR;
        const int w = _W/outR;
        const int h = _H/outR;

        float *gaussianArray = new float[nbChannels*w*h];

        memset(gaussianArray, 0, nbChannels*w*h*sizeof(float));
        pyramid->_gaussianPyramidArray.push_back(gaussianArray);

        pyramid->oddHDCreduced(_W, _H, nbChannels, s);

        float *laplacianArray = new float[nbChannels*W*H]; // scale (s-1)

        memset(laplacianArray, 0, nbChannels*W*H*sizeof(float));
        pyramid->_laplacianPyramidArray.push_back(laplacianArray);

        pyramid->gaussianToLaplacian(nbChannels, s-1);

        if(verbose) {
            Texture* laplacianTex = new Texture(0, W, H, GL_RGBA, GL_FLOAT, GL_RGBA32F, false);
            laplacianTex->loadFromData(pyramid->_laplacianPyramidArray[s-1]);
            pyramid->_laplacianPyramidTex.push_back(laplacianTex);
            sprintf( tmpNameChar, laplacianName.c_str(), s-1 );
            std::cout << "Save warped Laplacian scale " << inputScale << " decomposed at scale " << s-1 << " in " << tmpNameChar << std::endl;
            pyramid->_laplacianPyramidTex[s-1]->saveRGBAFloatTexture(W, H, nbChannels, std::string(tmpNameChar), false);
        }
    }

    const int radius1 = 1;
    const int radius2 = 1;

    uint scale1 = (int)inputScale<radius1?0:inputScale-radius1;
    uint scale2 = ((int)inputScale+radius2)<(_pyramidHeight-1)?inputScale+radius2:_pyramidHeight-1;
    for( uint s = scale1 ; s <= scale2 ; ++s ) {

        const int outR = (int)pow((double)2.0, (double)(s));
        const int w = _W/outR;
        const int h = _H/outR;

        assert(nbChannels == 4);
        _pyramid->addLevelContribution(w, h, nbChannels, s, pyramid->_laplacianPyramidArray[s]);
    }

    delete pyramid;
}

void TargetView::depthMapSplatting(InputView *vCam, FrameBuffer* FBO, Shader* depthSplattingShader) {

    // set fbo rendermatrix to current camera
    glm::mat4 renderMatrix;
    _pinholeCamera.setAsRenderCam( renderMatrix );

    PinholeCamera vPinholeCam = vCam->getPinholeCamera();

    FBO->splatDepth( vCam->getDepthMap(),
                     _pinholeCamera._K, _pinholeCamera._R, _pinholeCamera._t,
                     renderMatrix, vPinholeCam._K, vPinholeCam._R, vPinholeCam._t,
                     depthSplattingShader );
}

void TargetView::unscaledForwardWarping( InputView *vCam,
                                         FrameBuffer* splattingBuffer,
                                         bool visibilityPass,
                                         Texture* vkDepthMap,
                                         Texture* vkImage,
                                         Texture* vkMask,
                                         ShaderGeometry* imageSplattingShader,
                                         uint texIndex ) {

    // set fbo rendermatrix to current camera
    glm::mat4 renderMatrix;
    PinholeCamera uPinholeCam(_pinholeCamera);
    uPinholeCam.setAsRenderCam( renderMatrix );

    PinholeCamera vPinholeCam(vCam->getPinholeCamera());

    splattingBuffer->forwardWarping( visibilityPass,
                                     1,
                                     1,
                                     vkDepthMap,
                                     vkImage,
                                     vkMask,
                                     renderMatrix,
                                     vPinholeCam._K, vPinholeCam._R, vPinholeCam._t,
                                     uPinholeCam._R, uPinholeCam._t, // for refocussing
                                     imageSplattingShader,
                                     texIndex );
}

void TargetView::forwardWarping( InputView *vCam,
                                 FrameBuffer* splattingBuffer,
                                 bool visibilityPass,
                                 Texture* vkDepthMap,
                                 Texture* vkImage,
                                 Texture* vkMask,
                                 ShaderGeometry* imageSplattingShader,
                                 uint scale,
                                 bool originalDepth ) {

    // set fbo rendermatrix to current camera
    glm::mat4 renderMatrix;
    PinholeCamera uPinholeCam(_pinholeCamera);
    uPinholeCam.scaleCamera(scale);
    uPinholeCam.setAsRenderCam( renderMatrix );

    PinholeCamera vPinholeCam(vCam->getPinholeCamera());
    if(!originalDepth) {
        vPinholeCam.scaleCamera(scale);
    }

    uint ratioDepth; // ratio original dim / depth map or FBO dim
    uint ratioImage; // ratio depth map or FBO dim / input output tex dim
    if( originalDepth ) {
        ratioDepth = 1;
    } else {
        ratioDepth = (uint)pow(2.0, (double)scale);
    }

    if( originalDepth ) {
        ratioImage = (uint)pow(2.0, (double)scale);
    } else {
        ratioImage = 1;
    }

    splattingBuffer->forwardWarping( visibilityPass,
                                     ratioDepth,
                                     ratioImage,
                                     vkDepthMap,
                                     vkImage,
                                     vkMask,
                                     renderMatrix,
                                     vPinholeCam._K, vPinholeCam._R, vPinholeCam._t,
                                     uPinholeCam._R, uPinholeCam._t, // for refocussing
                                     imageSplattingShader );
}

// Normalised the splatted Laplacian
void TargetView::display() {

    // ------------------- LOAD VAO/VBO/EBO ------------------- //

    GLuint verticesVBO(0), texCoordVBO(0), VAO(0), EBO(0);
    Shader textureShader( "sources/openGL/shaders/2Dtexture.vert", "sources/openGL/shaders/2Dtexture.frag" );

    const int nbVertices = 8;
    const int nbIndices = 4;

    const GLfloat vertices[nbVertices] = {
        -1.0, -1.0,
        -1.0,  1.0,
        1.0, -1.0,
        1.0,  1.0
    };

    const GLfloat texCoord[nbVertices] = {
        0., 0.,
        0., (GLfloat)_H,
        (GLfloat)_W, 0.,
        (GLfloat)_W, (GLfloat)_H
    };

    const GLushort indices[nbIndices] = { 0, 1, 2, 3 };

    if( glIsVertexArray( VAO ) == GL_TRUE ) {
        glDeleteVertexArrays( 1, &VAO );
    }
    if(glIsBuffer( verticesVBO ) == GL_TRUE ) {
        glDeleteBuffers(1, &verticesVBO);
    }
    if(glIsBuffer( texCoordVBO ) == GL_TRUE ) {
        glDeleteBuffers(1, &texCoordVBO);
    }
    if(glIsBuffer ( EBO ) == GL_TRUE ) {
        glDeleteBuffers( 1, &EBO );
    }
    glGenVertexArrays( 1, &VAO );
    glGenBuffers( 1, &verticesVBO );
    glGenBuffers( 1, &texCoordVBO );
    glGenBuffers( 1, &EBO );

    glBindVertexArray( VAO );

    glBindBuffer( GL_ARRAY_BUFFER, verticesVBO );
    glBufferData( GL_ARRAY_BUFFER, nbVertices * sizeof(GLfloat), vertices, GL_STATIC_DRAW );
    glVertexAttribPointer( 0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0 );
    glEnableVertexAttribArray(0);

    glBindBuffer( GL_ARRAY_BUFFER, texCoordVBO );
    glBufferData( GL_ARRAY_BUFFER, nbVertices * sizeof(GLfloat), texCoord, GL_STATIC_DRAW );
    glVertexAttribPointer( 1, 2, GL_FLOAT, GL_FALSE, 0, (void*)0 );
    glEnableVertexAttribArray(1);

    glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, EBO );
    glBufferData( GL_ELEMENT_ARRAY_BUFFER, nbIndices * sizeof(GLushort), indices, GL_STATIC_DRAW );

    glBindVertexArray(0);

    checkGLErrors();

    // -------------------------------- INIT 2D SHADERS --------------------------- //

    checkGLErrors();

    assert( textureShader.add() );

    // Bind shader inputs
    textureShader.bindAttribLocation( 0, "in_pixel" );
    textureShader.bindAttribLocation( 1, "in_textureCoord" );

    assert( textureShader.link() );

    checkGLErrors();

    // -------------------------------- DRAW --------------------------- //

    // clear window

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glViewport( 0, 0, _W, _H );

    glUseProgram( textureShader.getProgramID() );

    glBindVertexArray( VAO ); // we draw an image-sized rectangle

    textureShader.setUniformi( "myTexture", 0 );
    textureShader.setUniformi( "H", _H );
    checkGLErrors();
    // input texture is blended temp tex
    glActiveTexture(GL_TEXTURE0);
    // TODO remove if scaled resolution pyramid
    glBindTexture( GL_TEXTURE_RECTANGLE, _pyramid->_gaussianPyramidTex[0]->getID() );
    //    glBindTexture( GL_TEXTURE_RECTANGLE, _gaussianPyramidTex[0]->getID() );

    glDrawElements( GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_SHORT, (void*)0 );

    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, 0 );

    glBindVertexArray(0);

    glUseProgram(0);

    checkGLErrors();

    // -------------------------------- DELETE --------------------------- //

    // vao
    glDeleteVertexArrays( 1, &VAO );
    glDeleteBuffers( 1, &verticesVBO );
    glDeleteBuffers( 1, &texCoordVBO );
    glDeleteBuffers( 1, &EBO );

    checkGLErrors();
}

void TargetView::move( const PinholeCamera &pinholeCamera1, const PinholeCamera &pinholeCamera2, uint count) {

    float step = 1000;
    float angle = 0.0003;

    _pinholeCamera._C = pinholeCamera1._C + (float)count * (pinholeCamera2._C - pinholeCamera1._C) / step;
    glm::mat3 Ry;
    Ry[0][0] = cos(count*angle); Ry[1][0] = 0; Ry[2][0] = sin(count*angle);
    Ry[0][1] = 0; Ry[1][1] = 1; Ry[2][1] = 0;
    Ry[0][2] = -sin(count*angle); Ry[1][2] = 0; Ry[2][2] = cos(count*angle);
    _pinholeCamera._R = Ry * pinholeCamera1._R;

    _pinholeCamera._t = -_pinholeCamera._R * _pinholeCamera._C;
    _pinholeCamera._P[0] = _pinholeCamera._R[0]; _pinholeCamera._P[1] = _pinholeCamera._R[1]; _pinholeCamera._P[2] = _pinholeCamera._R[2]; _pinholeCamera._P[3] = _pinholeCamera._t;
    _pinholeCamera._P = _pinholeCamera._K * _pinholeCamera._P;
}

void TargetView::createTestTargetView() {

    float focal_length = 1.0 / (2*tan(50*3.14/360));
    float pixel_aspect = 1.0;
    float principal_point[] = {0.5, 0.5};
    glm::mat3 R(1.0);
    glm::vec3 t(0.0);
    glm::mat3 K(0.0);

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

    t[0] = 0.0;
    t[1] = 0.0;
    t[2] = 5.0;

    _pinholeCamera = PinholeCamera( K, R, t, _W, _H );
}

void TargetView::saveDepthMap( const std::string &texName ) {

    _depthMap->saveRGBAFloatTexture( _W, _H, 3, texName, false );
}

// SETTERS

void TargetView::initDepthMap() {

    _depthMap = new Texture(0, _W, _H, GL_RGB, GL_FLOAT, GL_RGB32F, true);
    _depthMap->loadEmptyTexture();
}

// GETTERS

PinholeCamera TargetView::getPinholeCamera() const {

    return _pinholeCamera;
}

Texture* TargetView::getDepthMap() const {

    return _depthMap;
}

