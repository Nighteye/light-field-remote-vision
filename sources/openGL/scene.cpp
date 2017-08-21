#include "scene.h"
#include "cube.h"
#include "camera.h"
#include "../ply_io.h"
#include "mesh.h"
#include "view.h"
#include "frameBuffer.h"
#include "shader.h"
#include "pyramid.h"

#include <cocolib/cocolib/common/debug.h>
#include <iostream>
#include <GL/glew.h>
#include <vector>

using namespace coco;

#define checkGLErrors() {\
    GLenum error = glGetError(); \
    if(error != GL_NO_ERROR) { \
    std::cout << "GL_ERROR :" << __FILE__ << " "<< __LINE__ << " " << error << std::endl; \
    } \
    }

#define INVALID_POINT -1

Scene::Scene( std::string outdir,
              std::string winTitle,
              int winWidth, int winHeight,
              uint camWidth, uint camHeight,
              int sMin, int sMax, int sRmv,
              int tMin, int tMax, int tRmv,
              int pyramidHeight, float depthFocal ) :

    _outdir(outdir),
    _windowTitle(winTitle),
    _windowWidth(winWidth), _windowHeight(winHeight),
    _camWidth(camWidth), _camHeight(camHeight),
    _sMin(sMin), _sMax(sMax), _sRmv(sRmv),
    _tMin(tMin), _tMax(tMax), _tRmv(tRmv),
    _pyramidHeight(pyramidHeight), _depthFocal(depthFocal),
    _window(0), _OpenGLContext(0), _input(), _uCam(0), _FBO(0),

    _fromRadial2OrthoShader(0),
    _addDepthScaleShader(0),
    _tauWarpShader(0),
    _tauPartialShader(0),
    _warpVkShader(0),
    _normalizationShader(0),
    _addTexturesShader(0),
    _textureShader(0),
    _depthFromMeshShader(0),
    _depthSplattingShader(0),
    _forwardWarpingShader(0) {

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
}

Scene::~Scene() {

    for( uint i = 0 ; i < _meshes.size() ; ++i ) {
        delete _meshes[i];
        _meshes[i] = 0;
    }

    if(_uCam != 0) {
        delete _uCam;
    }
    _uCam = 0;

    for( uint i = 0 ; i < _nbCameras ; ++i ) {
        if(i < _vCam.size()){
            if(_vCam[i] != 0) {
                delete _vCam[i];
                _vCam[i] = 0;
            }
        }
    }

    if(_FBO != 0) {
        checkGLErrors();
        delete _FBO;
        _FBO = 0;
        checkGLErrors();
    }

    if(_fromRadial2OrthoShader != 0) {
        delete _fromRadial2OrthoShader;
        _fromRadial2OrthoShader = 0;
    }
    if(_addDepthScaleShader != 0) {
        delete _addDepthScaleShader;
        _addDepthScaleShader = 0;
    }
    if(_tauWarpShader != 0) {
        delete _tauWarpShader;
        _tauWarpShader = 0;
    }
    if(_tauPartialShader != 0) {
        delete _tauPartialShader;
        _tauPartialShader = 0;
    }
    if(_warpVkShader != 0) {
        delete _warpVkShader;
        _warpVkShader = 0;
    }
    if(_normalizationShader != 0) {
        delete _normalizationShader;
        _normalizationShader = 0;
    }
    if(_addTexturesShader != 0) {
        delete _addTexturesShader;
        _addTexturesShader = 0;
    }
    if(_textureShader != 0) {
        delete _textureShader;
        _textureShader = 0;
    }
    if(_depthFromMeshShader != 0) {
        delete _depthFromMeshShader;
        _depthFromMeshShader = 0;
    }

    if(_depthSplattingShader != 0) {
        delete _depthSplattingShader;
        _depthSplattingShader = 0;
    }
    if(_forwardWarpingShader != 0) {
        delete _forwardWarpingShader;
        _forwardWarpingShader = 0;
    }

    SDL_GL_DeleteContext(_OpenGLContext);
    SDL_DestroyWindow(_window);
    SDL_Quit();
}

bool Scene::initWindow() {

    // -------------------------- INITALIZING SDL ----------------------- //

    if(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_JOYSTICK) < 0) {

        std::cout << "Something went wrong with SDL initalization : " << SDL_GetError() << std::endl;
        SDL_Quit();

        return false;
    }

    // OpenGL version

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

    // Double Buffer

    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    // -------------------------- CREATING WINDOW ----------------------- //

    _window = SDL_CreateWindow(_windowTitle.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, _windowWidth, _windowHeight, SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL);

    if(_window == 0) {

        std::cout << "Something went wrong when creating the window : " << SDL_GetError() << std::endl;
        SDL_Quit();

        return false;
    }

    // -------------------------- OPENGL CONTEXT ------------------------ //

    _OpenGLContext = SDL_GL_CreateContext(_window);
    checkGLErrors();
    if(_OpenGLContext == 0) {

        std::cout << SDL_GetError() << std::endl;
        SDL_DestroyWindow(_window);
        SDL_Quit();

        return false;
    }

    // -------------------------- INITALIZING GLEW ---------------------- //

    // checking openGL extensions, versions, etc.
    glewExperimental = true;
    checkGLErrors();
    GLenum err = glewInit();
    checkGLErrors();
    if (err != GLEW_OK) {

        std::cout << "Problem: glewInit failed, something is seriously wrong." << std::endl;
        return false; // or handle the error in a nicer way
    }
    if (!GLEW_VERSION_2_1) {  // check that the machine supports the 2.1 API.

        std::cout << "Problem: your machine does not support the 2.1 API" << std::endl;
        return false; // or handle the error in a nicer way
    }

    return true;
}

bool Scene::initGL() {

    // Depth Buffer activation
    glEnable(GL_DEPTH_TEST);

    // Init Frame Buffer
    _FBO = new FrameBuffer( _camWidth, _camHeight, _depthFocal );

    // Init shaders

    _fromRadial2OrthoShader = new Shader( "sources/openGL/shaders/2Dtexture.vert", "sources/openGL/shaders/fromRadial2Ortho.frag" );
    _addDepthScaleShader = new Shader( "sources/openGL/shaders/2Dtexture.vert", "sources/openGL/shaders/addDepthScale.frag" );
    _tauWarpShader = new Shader( "sources/openGL/shaders/2Dtexture.vert", "sources/openGL/shaders/tauWarp.frag" );
    _tauPartialShader = new Shader( "sources/openGL/shaders/2Dtexture.vert", "sources/openGL/shaders/tauPartial.frag" );
    _warpVkShader = new Shader( "sources/openGL/shaders/2Dtexture.vert", "sources/openGL/shaders/warpVk.frag" );
    _normalizationShader = new Shader( "sources/openGL/shaders/2Dtexture.vert", "sources/openGL/shaders/normalization.frag" );
    _addTexturesShader = new Shader( "sources/openGL/shaders/2Dtexture.vert", "sources/openGL/shaders/addTextures.frag" );
    _textureShader = new Shader( "sources/openGL/shaders/2Dtexture.vert", "sources/openGL/shaders/2Dtexture.frag" );
    _depthFromMeshShader = new Shader( "sources/openGL/shaders/depthFromMesh.vert", "sources/openGL/shaders/depthFromMesh.frag" );

    _depthSplattingShader = new ShaderGeometry( "sources/openGL/shaders/depthSplatting.vert", "sources/openGL/shaders/depthSplatting.frag", "sources/openGL/shaders/depthSplatting.geom" );

    if(_depthFocal == 0) { // general IBR

        _forwardWarpingShader = new ShaderGeometry( "sources/openGL/shaders/forwardWarping.vert", "sources/openGL/shaders/imageSplatting.frag", "sources/openGL/shaders/imageSplatting.geom" );

    } else { // refocussing

        _forwardWarpingShader = new ShaderGeometry( "sources/openGL/shaders/forwardWarping.vert", "sources/openGL/shaders/imageSplatting.frag", "sources/openGL/shaders/planeSplatting.geom" );
    }

    init2DTextureShader( _fromRadial2OrthoShader );
    init2DTextureShader( _addDepthScaleShader );
    init2DTextureShader( _tauWarpShader );
    init2DTextureShader( _tauPartialShader );
    init2DTextureShader( _warpVkShader );
    init2DTextureShader( _normalizationShader );
    init2DTextureShader( _addTexturesShader );
    init2DTextureShader( _textureShader );
    init2DTextureShader( _depthFromMeshShader );

    initQuadGeomShader( _depthSplattingShader );
    initQuadGeomShader( _forwardWarpingShader );

    return true;
}

void Scene::mainLoop() {

    checkGLErrors();

    assert( _meshes[0]->isMeshOK() );

    unsigned int frameRate(1000/50);
    Uint32 beginingLoop(0), endLoop(0), elapsedTime(0);

    glm::mat4 projection(1.0);
    glm::mat4 modelview(1.0);
    glm::mat4 renderMatrix(1.0);

    // default render camera for free viewpoint
    projection = glm::perspective(70.0, (double) _windowWidth / _windowHeight, 0.1, 100.0);
    Camera camera(glm::vec3(-3.46416, 0.165908, 0.65109), glm::vec3(0, 0, 0), glm::vec3(0, -1, 0), 0.5, 0.5);

    bool freeViewpoint = true;

    // _uCam->displayMatrix();
    if( freeViewpoint ) {

        std::cout << "Free viewpoint" << std::endl;

        _renderIndex = 5;

    } else {

        _renderIndex = 0;
        std::cout << "Render view " << _renderIndex << std::endl;
        _vCam[_renderIndex]->getPinholeCamera().setAsRenderCam(renderMatrix);
    }

    // Boucle principale

    //    Crate crate(2.0, "openGL/texture.vert", "openGL/texture.frag", "openGL/Textures/Caisse2.jpg");
    //    crate.charger();

    std::cout << "Display...";

    _input.showCursor(false);
    _input.catchCursor(true);

    glClearColor(1.0, 1.0, 1.0, 1.0);
    glViewport( 0, 0, _windowWidth, _windowHeight );

    while(!_input.end()) {

        // Define time at begining of the loop

        beginingLoop = SDL_GetTicks();

        // Event managment

        _input.updateEvents();

        if(_input.getKey(SDL_SCANCODE_ESCAPE)) {

            break;
        }

        camera.deplacer(_input);

        // Clean buffers

        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

        // Camera managment

        camera.lookAt(modelview);

        // Display triangulated mesh

        checkGLErrors();

        // render with Camera, otherwise render with input cam

        if( freeViewpoint ) {

            // update the render matrix
            renderMatrix = projection * modelview;
        }

        // don't set the second parameter to disable texture mapping
        _meshes[0]->display( renderMatrix, _vCam[_renderIndex] );

        for( uint k = 0 ; k < _nbCameras ; ++k ) {

            _vCam[k]->display( renderMatrix );
        }

        // Refresh window

        SDL_GL_SwapWindow(_window);

        // Compute elapsed time

        endLoop = SDL_GetTicks();
        elapsedTime = endLoop - beginingLoop;

        // If necessary, pause the program

        if(elapsedTime < frameRate) {

            SDL_Delay(frameRate - elapsedTime);
        }
    }

    std::cout << "...done!" << std::endl;
}

void Scene::renderingLoop() {

    checkGLErrors();

    //    unsigned int frameRate(1000/50);

    // first view initializes rendercam
    //_uCam->setPinholeCamera(_vCam[0]->getPinholeCamera());

    // render splatted target view
    // renderSplatting();

    _input.showCursor(false);
    _input.catchCursor(true);

    //    createGaussianPyramid();
    // createGaussianPyramidCPU();
    createGaussianScaleSpace();
    // createDepthPyramidCPU(_pyramidHeight);

    glClearColor(1.0, 1.0, 1.0, 1.0);
    glViewport( 0, 0, _windowWidth, _windowHeight );

    std::cout << "Render Frame with Laplacian IBR" << std::endl;
    // pyramidSplatting(_camWidth, _camHeight);
    // scaleSpaceSplatting(_camWidth, _camHeight);

    // warp a scale level of an input view and decompose it
    //    const uint viewIndex = 4;
    //    const uint inputScale = 5;
    //    const bool verbose = true;
    //    singleViewForwardWarping( _camWidth, _camHeight, inputScale, viewIndex, &Pyramid::_laplacianPyramidTex, &Pyramid::_laplacianPyramidArray,
    //                              verbose, "/warpedScale%02iview%02i.pfm" );
    //    _uCam->createDecomposeWarpedImage(inputScale, _uCam->_pyramid->_laplacianPyramidTex[inputScale]);

    const int nbFrames = 1;
    const int nbChannels = 3;
    for(int count = 0 ; count < nbFrames ; ++count) {

        Uint32 beginingLoop(0), endLoop(0), elapsedTime(0);
        beginingLoop = SDL_GetTicks();

        scaleSpaceWarping(_camWidth, _camHeight); // render target view

        endLoop = SDL_GetTicks();
        elapsedTime = endLoop - beginingLoop;
        std::cout << "elapsedTime: " << elapsedTime << std::endl;

        if(true){
            std::string splattedGaussianName = _outdir + "/frame_%02i.pfm";
            char tmpNameChar[500];
            sprintf( tmpNameChar, splattedGaussianName.c_str(), count );
            std::cout << "Save frame" << count << " in " << tmpNameChar << std::endl;
            _uCam->_pyramid->_gaussianPyramidTex[0]->saveRGBAFloatTexture(_camWidth, _camHeight, nbChannels, std::string(tmpNameChar), false);
        }

        _uCam->move(_vCam[5]->getPinholeCamera(), _vCam[4]->getPinholeCamera(), count); // update target camera position
    }

    uint count = 0;

    while(false) {
        //    while(!_input.end()) {

        //        beginingLoop = SDL_GetTicks();

        _input.updateEvents();

        if(_input.getKey(SDL_SCANCODE_ESCAPE)) {
            break;
        }

        // ------------------------------------------------
        // MOVE CAMERA
        // ------------------------------------------------

        // compute new camera matrices

        // renderCam.setAsRenderCam(renderMatrix);
        _uCam->move(_vCam[5]->getPinholeCamera(), _vCam[4]->getPinholeCamera(), count);

        // ------------------------------------------------
        //
        // ------------------------------------------------


        // we don't export the warps and weights, just compute them
        //        computeTargetDepthMap();
        //        exportWarps();

        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

        checkGLErrors();

        // render with Camera, otherwise render with input cam

        // ------------------------------------------------
        // RENDER VIEW
        // ------------------------------------------------

        // compute view and display
        // pyramidSplatting(_camWidth, _camHeight);
        // scaleSpaceSplatting(_camWidth, _camHeight);
        scaleSpaceWarping(_camWidth, _camHeight);
        _uCam->display();

        // ------------------------------------------------
        //
        // ------------------------------------------------

        SDL_GL_SwapWindow(_window);

        //        endLoop = SDL_GetTicks();
        //        elapsedTime = endLoop - beginingLoop;

        //        if(elapsedTime < frameRate) {

        //            SDL_Delay(frameRate - elapsedTime);
        //        }

        ++count;
    }

    std::cout << "...done!" << std::endl;
}

void Scene::renderingTest() {

    checkGLErrors();

    _input.showCursor(false);
    _input.catchCursor(true);

    createGaussianScaleSpace();

    //    createDepthScaleSpaceCPU(6);

    glClearColor(1.0, 1.0, 1.0, 1.0);
    glViewport( 0, 0, _windowWidth, _windowHeight );

    std::cout << "Render Test Image with Laplacian IBR" << std::endl;

    const int nbChannels = 3;

    Uint32 beginingLoop(0), endLoop(0), elapsedTime(0);
    beginingLoop = SDL_GetTicks();

    scaleSpaceWarping(_camWidth, _camHeight); // render target view

    endLoop = SDL_GetTicks();
    elapsedTime = endLoop - beginingLoop;
    std::cout << "elapsedTime: " << elapsedTime << std::endl;

    std::string splattedGaussianName = _outdir + "/testImage.pfm";
    std::cout << "Save test image in " << splattedGaussianName << std::endl;
    _uCam->_pyramid->_gaussianPyramidTex[0]->saveRGBAFloatTexture(_camWidth, _camHeight, nbChannels, splattedGaussianName, false);
}

void Scene::refocussingLoop() {

    checkGLErrors();

    _input.showCursor(false);
    _input.catchCursor(true);

    createGaussianScaleSpace();

    glClearColor(1.0, 1.0, 1.0, 1.0);
    glViewport( 0, 0, _windowWidth, _windowHeight );

    std::cout << "Render Frame with Laplacian Refocussing" << std::endl;

    const uint firstFrame = 1;
    const uint lastFrame = 1;
    const int nbChannels = 3;
    //    _depthFocal += 0.01*firstFrame;
    for(uint count = firstFrame ; count <= lastFrame ; ++count) {

        Uint32 beginingLoop(0), endLoop(0), elapsedTime(0);
        beginingLoop = SDL_GetTicks();

        scaleSpaceWarping(_camWidth, _camHeight); // render target view

        endLoop = SDL_GetTicks();
        elapsedTime = endLoop - beginingLoop;
        std::cout << "elapsedTime: " << elapsedTime << " --- " << "depth of the focal plane: " << _depthFocal << std::endl;

        std::string finalName = _outdir + "/final_%02i.pfm";
        char tmpNameChar[500];
        sprintf( tmpNameChar, finalName.c_str(), count );
        std::cout << "Save final frame" << count << " in " << tmpNameChar << std::endl;
        _uCam->_pyramid->_gaussianPyramidTex[0]->saveRGBAFloatTexture(_camWidth, _camHeight, nbChannels, std::string(tmpNameChar), false);

        //        // update refocussing plane
        //        _depthFocal += 0.01;
    }

    std::cout << "...done!" << std::endl;
}

void Scene::renderSplatting() {

    // -------------------------------- CREATE BUFFERS -------------------------------- //

    Texture* tempTex = new Texture(0, _camWidth, _camHeight, GL_RGBA, GL_FLOAT, GL_RGBA32F, false);
    tempTex->loadEmptyTexture();
    Texture* splattedVk = new Texture(0, _camWidth, _camHeight, GL_RGBA, GL_FLOAT, GL_RGBA32F, false);
    splattedVk->loadEmptyTexture();

    _FBO->attachSplattingBuffers(splattedVk, tempTex, _camWidth, _camHeight);

    const bool verbose = false;

    if(verbose) {
        std::cout << "Visibility pass" << std::endl;
    }

    for(uint i = 0 ; i < _nbCameras ; ++i) {

        if(i != _renderIndex) {

            if(verbose) {
                std::cout << "Splat view " << i << "... ";
            }

            _uCam->forwardWarping( _vCam[i],
                                   _FBO,
                                   true,
                                   _vCam[i]->getDepthMap(),
                                   _vCam[i]->getTexture(),
                                   _vCam[i]->getMask(),
                                   _forwardWarpingShader );

            if(verbose) {
                char tempNameChar[500];
                std::string tempName = _outdir + "/visibility%02i.pfm";
                sprintf( tempNameChar, tempName.c_str(), i );

                std::cout << "Save visibility in " << tempNameChar << std::endl;
                tempTex->saveRGBAFloatTexture( _camWidth, _camHeight, 3, std::string(tempNameChar) );
            }

        } else {

            if(verbose) {
                std::cout << "View " << i << " is target view, skipping it" << "... ";
            }
        }
    }

    if(verbose) {
        std::cout << "Blending pass" << std::endl;
    }

    _FBO->clearAttachment(1, _camWidth, _camHeight); // clear temp buffer

    for(uint i = 0 ; i < _nbCameras ; ++i) {

        if(i != _renderIndex) {

            if(verbose) {
                std::cout << "Splat view " << i << "... ";
            }

            _uCam->forwardWarping( _vCam[i],
                                   _FBO,
                                   false,
                                   _vCam[i]->getDepthMap(),
                                   _vCam[i]->getTexture(),
                                   _vCam[i]->getMask(),
                                   _forwardWarpingShader );

            if(verbose) {
                char tempNameChar[500];
                std::string tempName = _outdir + "/splattedVk%02i.pfm";
                sprintf( tempNameChar, tempName.c_str(), i );

                std::cout << "Save warped view in " << tempNameChar << std::endl;
                tempTex->saveRGBAFloatTexture( _camWidth, _camHeight, 3, std::string(tempNameChar) );
            }

        } else {

            if(verbose) {
                std::cout << "View " << i << " is target view, skipping it" << "... ";
            }
        }
    }
    if(verbose) {
        std::cout << std::endl;
    }

    _FBO->clearAttachment(1, _camWidth, _camHeight); // clear temp buffer

    _FBO->splatNormalisation( tempTex, _normalizationShader );

    std::string splattedVkName = _outdir + "/splattedVk.pfm";
    std::cout << "Save warped view in " << splattedVkName << std::endl;
    splattedVk->saveRGBAFloatTexture( _camWidth, _camHeight, 3, splattedVkName );

    delete tempTex;
    delete splattedVk;
}

void Scene::singleViewForwardWarping( uint W, uint H,
                                      uint scale,
                                      uint viewIndex,
                                      std::vector< Texture* > Pyramid::*texVec,
                                      std::vector< float* > Pyramid::*arrayVec,
                                      bool verbose, std::string outfile ) {

    FrameBuffer* splattingBuffer = new FrameBuffer( W, H, _depthFocal );

    // -------------------------------- CREATE BUFFERS -------------------------------- //

    (_uCam->_pyramid->*texVec)[scale]->loadEmptyTexture();
    Texture* tempTex = new Texture(0, W, H, GL_RGBA, GL_FLOAT, GL_RGBA32F, false);
    tempTex->loadEmptyTexture();

    splattingBuffer->attachSplattingBuffers((_uCam->_pyramid->*texVec)[scale], tempTex, W, H );

    // --------------------------- PERFORM FORWARD WARPING ---------------------------- //

    _uCam->unscaledForwardWarping( _vCam[viewIndex],
                                   splattingBuffer,
                                   true,
                                   _vCam[viewIndex]->getDepthMap(),
                                   (_vCam[viewIndex]->_pyramid->*texVec)[scale],
                                   _vCam[viewIndex]->getMask(),
                                   _forwardWarpingShader );

    splattingBuffer->clearAttachment(1, W, H); // clear temp buffer

    _uCam->unscaledForwardWarping( _vCam[viewIndex],
                                   splattingBuffer,
                                   false,
                                   _vCam[viewIndex]->getDepthMap(),
                                   (_vCam[viewIndex]->_pyramid->*texVec)[scale],
                                   _vCam[viewIndex]->getMask(),
                                   _forwardWarpingShader );

    splattingBuffer->splatNormalisation( tempTex, _normalizationShader );

    if(verbose && !_outdir.empty() && !outfile.empty()) {

        std::string tmpNameStr = _outdir + outfile;
        char tmpNameChar[500];
        sprintf( tmpNameChar, tmpNameStr.c_str(), scale, viewIndex );
        std::cout << "Save warped scale " << scale << "of view " << viewIndex << " in " << tmpNameChar << std::endl;
        (_uCam->_pyramid->*texVec)[scale]->saveRGBAFloatTexture(W, H, 3, std::string(tmpNameChar), false);
    }

    delete tempTex;
    delete splattingBuffer;

    glBindTexture(GL_TEXTURE_RECTANGLE, (_uCam->_pyramid->*texVec)[scale]->getID());
    glGetTexImage( GL_TEXTURE_RECTANGLE, 0, (_uCam->_pyramid->*texVec)[scale]->getFormat(), (_uCam->_pyramid->*texVec)[scale]->getType(), (_uCam->_pyramid->*arrayVec)[scale] );
    glBindTexture(GL_TEXTURE_RECTANGLE, 0);
}

void createRenderBuffer( GLuint *id, GLenum internalFormat, uint W, uint H ) {

    if( glIsRenderbuffer(*id) == GL_TRUE ) {

        glDeleteRenderbuffers( 1, id );
    }

    glGenRenderbuffers( 1, id );

    glBindRenderbuffer( GL_RENDERBUFFER, *id );

    glRenderbufferStorage( GL_RENDERBUFFER, internalFormat, W, H );

    glBindRenderbuffer( GL_RENDERBUFFER, 0 );
}

void initQuadGeomShader( ShaderGeometry *shader) {

    checkGLErrors();

    assert( shader->add() );

    // Bind shader inputs
    shader->bindAttribLocation( 0, "in_pixel" );

    assert( shader->link() );

    checkGLErrors();
}

void init2DTextureShader( Shader *shader) {

    checkGLErrors();

    assert( shader->add() );

    // Bind shader inputs
    shader->bindAttribLocation( 0, "in_pixel" );
    shader->bindAttribLocation( 1, "in_textureCoord" );

    assert( shader->link() );

    checkGLErrors();
}

void deleteFrameBuffer(GLuint *id, GLuint *depthBufferID, GLuint *verticesVBO, GLuint *texCoordVBO, GLuint *VAO, GLuint *EBO,
                       GLuint *sensorPlaneVAO, GLuint *sensorPlaneVBO) {

    checkGLErrors();

    // frame buffer
    glDeleteFramebuffers( 1, id );
    glDeleteFramebuffers( 1, depthBufferID );

    // vao
    glDeleteVertexArrays( 1, VAO );
    glDeleteBuffers( 1, verticesVBO );
    glDeleteBuffers( 1, texCoordVBO );
    glDeleteBuffers( 1, EBO );

    // sensor plane vao
    glDeleteVertexArrays( 1, sensorPlaneVAO );
    glDeleteBuffers( 1, sensorPlaneVBO );

    checkGLErrors();
}

void collapsePyramid(int W, int H, int nbChannels, const float* const input, float* const output) {

    for(int i = 0 ; i < H ; ++i) {
        for(int j = 0 ; j < W ; ++j) {
            for(int c = 0 ; c < nbChannels ; ++c) {

                output[i*W*nbChannels+j*nbChannels + c] += input[i*W*nbChannels+j*nbChannels + c];
            }
        }
    }
}

void Scene::forwardWarping( uint originalW, uint originalH,
                            uint scale,
                            std::vector< Texture* > InputView::*inTexVec,
                            std::vector< Texture* > TargetView::*outTexVec,
                            std::vector< float* > TargetView::*outArrayVec,
                            bool originalDepth,
                            bool verbose, std::string outfile ) {

    uint outW = originalW / (uint)pow(2.0, (double)scale);
    uint outH = originalH / (uint)pow(2.0, (double)scale);
    uint inW, inH;

    if(originalDepth) {
        inW = originalW;
        inH = originalH;
    } else {
        inW = outW;
        inH = outH;
    }

    FrameBuffer* splattingBuffer = new FrameBuffer( inW, inH, _depthFocal );

    // -------------------------------- CREATE BUFFERS -------------------------------- //

    (_uCam->*outTexVec)[scale]->loadEmptyTexture();
    Texture* tempTex = new Texture(0, outW, outH, GL_RGBA, GL_FLOAT, GL_RGBA32F, false);
    tempTex->loadEmptyTexture();

    splattingBuffer->attachSplattingBuffers((_uCam->*outTexVec)[scale], tempTex, outW, outH );

    // --------------------------- PERFORM FORWARD WARPING ---------------------------- //

    if(verbose) {
        std::cout << "Visibility pass" << std::endl;
    }

    for(uint i = 0 ; i < _nbCameras ; ++i) {

        if(i != _renderIndex) {

            if(verbose) {
                std::cout << "Splat view " << i << "... ";
            }

            _uCam->forwardWarping( _vCam[i],
                                   splattingBuffer,
                                   true,
                                   _vCam[i]->_depthPyramidTex[0],
                    (_vCam[i]->*inTexVec)[scale],
                    _vCam[i]->getMask(),
                    _forwardWarpingShader,
                    scale,
                    originalDepth );

        } else {

            if(verbose) {
                std::cout << "View " << i << " is target view, skipping it" << "... ";
            }
        }
    }

    if(verbose) {
        std::cout << "Blending pass" << std::endl;
    }

    splattingBuffer->clearAttachment(1, outW, outH); // clear temp buffer

    for(uint i = 0 ; i < _nbCameras ; ++i) {

        if(i != _renderIndex) {

            if(verbose) {
                std::cout << "Splat view " << i << "... ";
            }

            _uCam->forwardWarping( _vCam[i],
                                   splattingBuffer,
                                   false,
                                   _vCam[i]->_depthPyramidTex[0],
                    (_vCam[i]->*inTexVec)[scale],
                    _vCam[i]->getMask(),
                    _forwardWarpingShader,
                    scale,
                    originalDepth );

        } else {

            if(verbose) {
                std::cout << "View " << i << " is target view, skipping it" << "... ";
            }
        }
    }
    if(verbose) {
        std::cout << "Normalization pass" << std::endl;
    }

    splattingBuffer->splatNormalisation( tempTex, _normalizationShader );

    delete tempTex;
    delete splattingBuffer;

    if(verbose && !_outdir.empty() && !outfile.empty()) {

        std::string tmpNameStr = _outdir + outfile;
        char tmpNameChar[500];
        sprintf( tmpNameChar, tmpNameStr.c_str(), scale );
        std::cout << "Save image scale " << scale << " in " << tmpNameChar << std::endl;
        (_uCam->*outTexVec)[scale]->saveRGBAFloatTexture(outW, outH, 3, std::string(tmpNameChar), false);
    }

    glBindTexture(GL_TEXTURE_RECTANGLE, (_uCam->*outTexVec)[scale]->getID());
    glGetTexImage( GL_TEXTURE_RECTANGLE, 0, (_uCam->*outTexVec)[scale]->getFormat(), (_uCam->*outTexVec)[scale]->getType(), (_uCam->*outArrayVec)[scale] );
    glBindTexture(GL_TEXTURE_RECTANGLE, 0);
}

void Scene::unscaledForwardWarping( uint W, uint H,
                                    uint scale,
                                    std::vector< Texture* > Pyramid::*texVec,
                                    std::vector< float* > Pyramid::*arrayVec,
                                    bool verbose, std::string outfile ) {

    FrameBuffer* splattingBuffer = new FrameBuffer( W, H, _depthFocal );

    // -------------------------------- CREATE BUFFERS -------------------------------- //

    (_uCam->_pyramid->*texVec)[scale]->loadEmptyTexture();
    Texture* tempTex = new Texture(0, W, H, GL_RGBA, GL_FLOAT, GL_RGBA32F, false);
    tempTex->loadEmptyTexture();

    splattingBuffer->attachSplattingBuffers((_uCam->_pyramid->*texVec)[scale], tempTex, W, H );

    // --------------------------- PERFORM FORWARD WARPING ---------------------------- //

    for(uint i = 0 ; i < _nbCameras ; ++i) {

        if(i != _renderIndex) {

            _uCam->unscaledForwardWarping( _vCam[i],
                                           splattingBuffer,
                                           true,
                                           _vCam[i]->getDepthMap(),
                                           (_vCam[i]->_pyramid->*texVec)[scale],
                                           _vCam[i]->getMask(),
                                           _forwardWarpingShader );
        }
    }

    if(verbose && !_outdir.empty()) {

        std::string tmpNameStr = _outdir + "/visibility.tiff";
        char tmpNameChar[500];
        sprintf( tmpNameChar, tmpNameStr.c_str() );
        std::cout << "Visibility stage: save warped view in " << tmpNameChar << std::endl;
        tempTex->saveRGBAFloatTexture(W, H, 4, std::string(tmpNameChar), false);
    }

    splattingBuffer->clearAttachment(1, W, H); // clear temp buffer

    for(uint i = 0 ; i < _nbCameras ; ++i) {

        if(i != _renderIndex) {

            _uCam->unscaledForwardWarping( _vCam[i],
                                           splattingBuffer,
                                           false,
                                           _vCam[i]->getDepthMap(),
                                           (_vCam[i]->_pyramid->*texVec)[scale],
                                           _vCam[i]->getMask(),
                                           _forwardWarpingShader );

            if(verbose && !_outdir.empty()) {

                std::string tmpNameStr = _outdir + "/blending_%02lu.tiff";
                char tmpNameChar[500];
                sprintf( tmpNameChar, tmpNameStr.c_str(), i );
                std::cout << "Blending stage: save warped view " << i << " in " << tmpNameChar << std::endl;
                tempTex->saveRGBAFloatTexture(W, H, 4, std::string(tmpNameChar), false);
            }
        }
    }

    splattingBuffer->splatNormalisation( tempTex, _normalizationShader );

    delete tempTex;
    delete splattingBuffer;

    if(verbose && !_outdir.empty() && !outfile.empty()) {

        std::string tmpNameStr = _outdir + outfile;
        char tmpNameChar[500];
        sprintf( tmpNameChar, tmpNameStr.c_str(), scale );
        std::cout << "Save image scale " << scale << " in " << tmpNameChar << std::endl;
        (_uCam->_pyramid->*texVec)[scale]->saveRGBAFloatTexture(W, H, 4, std::string(tmpNameChar), false);
    }

    if(verbose && !_outdir.empty()) {

        std::string tmpNameStr = _outdir + "/normalized.tiff";
        std::cout << "Save image scale " << scale << " in " << tmpNameStr << std::endl;
        (_uCam->_pyramid->*texVec)[scale]->saveRGBAFloatTexture(W, H, 4, tmpNameStr, false);
    }

    glBindTexture(GL_TEXTURE_RECTANGLE, (_uCam->_pyramid->*texVec)[scale]->getID());
    glGetTexImage( GL_TEXTURE_RECTANGLE, 0, (_uCam->_pyramid->*texVec)[scale]->getFormat(), (_uCam->_pyramid->*texVec)[scale]->getType(), (_uCam->_pyramid->*arrayVec)[scale] );
    glBindTexture(GL_TEXTURE_RECTANGLE, 0);
}

void Scene::collapseScaleSpace(uint W, uint H, bool verbose, uint nbChannels) {

    // -------------------------------- LOWEST SCALE SPLATTING -------------------------------- //
    // In the case this part is not commented, it ignores the previous splatting of the largest scale images
    // Performs iterative reduction starting form the original image splatting

    unscaledForwardWarping( W, H,
                            0, // scale
                            &Pyramid::_gaussianPyramidTex,
                            &Pyramid::_gaussianPyramidArray,
                            verbose, "/splattedGaussian2_%02i.tiff" ); // verbose

    // -------------------------------- COMPUTE GAUSSIAN PYRAMID -------------------------------- //

    // _uCam->_pyramid->createTestImage(W, H, nbChannels);

    if(true) {
        _uCam->_pyramid->_gaussianPyramidTex[0]->loadFromData(_uCam->_pyramid->_gaussianPyramidArray[0]);
        std::string splattedGaussianName = _outdir + "/foundamentalGaussian_%02i.pfm";
        char tmpNameChar[500];
        sprintf( tmpNameChar, splattedGaussianName.c_str(), 0 );
        std::cout << "Save fundamental Gaussian scale " << 0 << " in " << tmpNameChar << std::endl;
        _uCam->_pyramid->_gaussianPyramidTex[0]->saveRGBAFloatTexture(W, H, 3, std::string(tmpNameChar), false);
    }

    for(uint scale = 1 ; scale <= (uint)_pyramidHeight ; ++scale) {

        _uCam->_pyramid->oddHDCreduced(W, H, nbChannels, scale);

        const int outR = (int)pow((double)2.0, (double)(scale));
        const int outW = W/outR;
        const int outH = H/outR;

        if(verbose) {
            _uCam->_pyramid->_gaussianPyramidTex[scale]->loadFromData(_uCam->_pyramid->_gaussianPyramidArray[scale]);
            std::string splattedGaussianName = _outdir + "/foundamentalGaussian_%02i.tiff";
            char tmpNameChar[500];
            sprintf( tmpNameChar, splattedGaussianName.c_str(), scale );
            std::cout << "Save fundamental Gaussian scale " << scale << " in " << tmpNameChar << std::endl;
            _uCam->_pyramid->_gaussianPyramidTex[scale]->saveRGBAFloatTexture(outW, outH, nbChannels, std::string(tmpNameChar), false);
        }
    }

    // -------------------------------- COLLAPSE LAPLACIAN PYRAMID -------------------------------- //

    for(uint scale = _pyramidHeight ; 0 < scale ; --scale) {

        const int outR = (int)pow((double)2.0, (double)(scale-1));
        const int outW = W/outR;
        const int outH = H/outR;

        _uCam->_pyramid->collapse(nbChannels, scale-1);

        if(verbose) {
            _uCam->_pyramid->_gaussianPyramidTex[scale-1]->loadFromData(_uCam->_pyramid->_gaussianPyramidArray[scale-1]);
            std::string splattedGaussianName = _outdir + "/finalGaussian_%02i.tiff";
            char tmpNameChar[500];
            sprintf( tmpNameChar, splattedGaussianName.c_str(), scale-1 );
            std::cout << "Save splatted Gaussian scale " << scale-1 << " in " << tmpNameChar << std::endl;
            _uCam->_pyramid->_gaussianPyramidTex[scale-1]->saveRGBAFloatTexture(outW, outH, nbChannels, std::string(tmpNameChar), false);
        }
    }

    // -------------------------------- DISPLAY TEXTURE -------------------------------- //

    _uCam->_pyramid->_gaussianPyramidTex[0]->loadFromData(_uCam->_pyramid->_gaussianPyramidArray[0]);
}

void Scene::scaleSpaceSplatting(uint W, uint H) {

    const uint nbChannels = 3;
    const bool verbose = false;

    // create a frame buffer per scale
    for (uint scale = 0 ; scale < (uint)_pyramidHeight ; ++scale) {

        if(verbose) {
            std::cout << "Scale " << scale << " Laplacian splatting." << std::endl;
        }

        unscaledForwardWarping( W, H,
                                scale, // scale
                                &Pyramid::_laplacianPyramidTex,
                                &Pyramid::_laplacianPyramidArray,
                                verbose, "/splattedLaplacian2_%02i.pfm" ); // verbose
    }

    collapseScaleSpace( W, H, verbose, nbChannels);

    // test laplacian warping by decomposing at warped laplacian at given scale into its laplacian pyramid (multiple scales)
    //    uint inputScale = 5;
    //    _uCam->_pyramid->collapse(W, H, nbChannels, inputScale, inputScale); // target image for a specific frequency band
    //    _uCam->createDecomposeWarpedImage(inputScale, _uCam->_pyramid->_gaussianPyramidTex[inputScale]);
}

void Scene::scaleSpaceWarping(uint W, uint H) {

    const uint nbChannels = 4;
    const bool verbose = false;

    FrameBuffer* warpingBuffer = new FrameBuffer( W, H, _depthFocal );

    const uint tempTexIndex1 = GL_COLOR_ATTACHMENT0 + 0;
    const uint tempTexIndex2 = GL_COLOR_ATTACHMENT0 + 1;

    // INIT BUFFERS AND TEXTURES

    _uCam->initPyramid();

    Texture* tempTex1 = new Texture(0, W, H, GL_RGBA, GL_FLOAT, GL_RGBA32F, false);
    Texture* tempTex2 = new Texture(0, W, H, GL_RGBA, GL_FLOAT, GL_RGBA32F, false);
    tempTex1->loadEmptyTexture();
    tempTex2->loadEmptyTexture();

    checkGLErrors();

    // VISIBILITY PASS: COMPUTE DEPTH BUFFER (COMMON FOR ALL VIEWS, ALL SCALES)

    // attach and clear buffers
    glBindFramebuffer( GL_FRAMEBUFFER, warpingBuffer->getID() );

    glFramebufferTexture2D( GL_FRAMEBUFFER, tempTexIndex1, GL_TEXTURE_RECTANGLE, tempTex1->getID(), 0 );
    glDrawBuffer(tempTexIndex1);
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClearDepth(1.0);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glViewport( 0, 0, W, H );

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    for(uint l = 0 ; l < _nbCameras ; ++l) {

        if(l != _renderIndex) {

            _uCam->unscaledForwardWarping( _vCam[l],
                                           warpingBuffer,
                                           true,
                                           _vCam[l]->getDepthMap(),
                                           tempTex2,
                                           _vCam[l]->getMask(),
                                           _forwardWarpingShader );
        }
    }

    checkGLErrors();

    // for all scales, warp/blend independently each source image, then add image contribution to final level
    for (uint inputScale = 0 ; inputScale < (uint)_pyramidHeight ; ++inputScale) {

        // scales to select after decomposition of the warped input scale
        const int radius1 = 1;
        const int radius2 = 1;
        uint scale1 = (int)inputScale<radius1?0:inputScale-radius1;
        uint scale2 = ((int)inputScale+radius2)<(_pyramidHeight-1)?inputScale+radius2:_pyramidHeight-1;

        // buffers for contributions of each input scale
        // as many as selected scales
        std::vector< std::vector< float > > scaleBuffer(scale2 - scale1 + 1);

        for(uint s = scale1 ; s <= scale2 ; ++s) {

            // indices for pyramid
            const uint outR = (int)pow((double)2.0, (double)s);
            const uint outW = W/outR;
            const uint outH = H/outR;

            scaleBuffer[s - scale1].resize(outW*outH*nbChannels);

            // intialize the weights (target pyramid alpha channel)
            for(uint i = 0 ; i < outH ; ++i) {
                for(uint j = 0 ; j < outW ; ++j) {
                    for(uint c = 0 ; c < nbChannels ; ++c) {

                        uint o = i*outW*nbChannels+j*nbChannels + c;
                        (scaleBuffer[s - scale1])[o] = 0.0;
                    }
                }
            }
        }

        for(uint k = 0 ; k < _nbCameras ; ++k) {

            if(k != _renderIndex) {

                // BLENDING PASS: COMPUTE COLOR BUFFER

                // attach and clear buffers
                glBindFramebuffer( GL_FRAMEBUFFER, warpingBuffer->getID() );

                glFramebufferTexture2D( GL_FRAMEBUFFER, tempTexIndex1, GL_TEXTURE_RECTANGLE, tempTex1->getID(), 0 );
                glDrawBuffer(tempTexIndex1);
                glClearColor(0.0, 0.0, 0.0, 0.0);
                glClear( GL_COLOR_BUFFER_BIT );
                glViewport( 0, 0, W, H );

                glFramebufferTexture2D( GL_FRAMEBUFFER, tempTexIndex2, GL_TEXTURE_RECTANGLE, tempTex2->getID(), 0 );
                glDrawBuffer(tempTexIndex2);
                glClearColor(0.0, 0.0, 0.0, 0.0);
                glClear( GL_COLOR_BUFFER_BIT );
                glViewport( 0, 0, W, H );

                glBindFramebuffer( GL_FRAMEBUFFER, 0 );

                checkGLErrors();

                _uCam->unscaledForwardWarping( _vCam[k],
                                               warpingBuffer,
                                               false,
                                               _vCam[k]->getDepthMap(),
                                               (_vCam[k]->_pyramid->_laplacianPyramidTex)[inputScale],
                                               _vCam[k]->getMask(),
                                               _forwardWarpingShader,
                                               tempTexIndex1 );

                if(verbose) {
                    std::string splattedGaussianName = _outdir + "/blendedLaplacian_%02i_%02i.tiff";
                    char tmpNameChar[500];
                    sprintf( tmpNameChar, splattedGaussianName.c_str(), k, inputScale );
                    std::cout << "Save blended Laplacian scale " << inputScale << " view " << k << " in " << tmpNameChar << std::endl;
                    tempTex1->saveRGBAFloatTexture(W, H, nbChannels, std::string(tmpNameChar), false);
                }

                warpingBuffer->splatNormalisation( tempTex1, _normalizationShader, tempTexIndex2 );

                if(verbose) {
                    std::string splattedGaussianName = _outdir + "/normedLaplacian_%02i_%02i.tiff";
                    char tmpNameChar[500];
                    sprintf( tmpNameChar, splattedGaussianName.c_str(), k, inputScale );
                    std::cout << "Save normalized Laplacian scale " << inputScale << " view " << k << " in " << tmpNameChar << std::endl;
                    tempTex2->saveRGBAFloatTexture(W, H, nbChannels, std::string(tmpNameChar), false);
                }

                // DECOMPOSE warped image and select some levels
                //                _uCam->createDecomposeWarpedImage(s, tempTex2);

                Pyramid* tempPyramid = new Pyramid(W, H); // temp pyramid

                float *floatBuffer = new float[nbChannels*W*H];

                glBindTexture(GL_TEXTURE_RECTANGLE, tempTex2->getID());
                glGetTexImage( GL_TEXTURE_RECTANGLE, 0, tempTex2->getFormat(), tempTex2->getType(), floatBuffer );
                glBindTexture(GL_TEXTURE_RECTANGLE, 0);

                tempPyramid->_gaussianPyramidArray.push_back(floatBuffer);

                for(uint s = 1 ; s <= (uint)_pyramidHeight ; ++s) {

                    const int inR = (int)pow((double)2.0, (double)(s-1));
                    const int outR = (int)pow((double)2.0, (double)(s));
                    const int inW = W/inR;
                    const int inH = H/inR;
                    const int outW = W/outR;
                    const int outH = H/outR;

                    float *gaussianArray = new float[nbChannels*outW*outH];

                    memset(gaussianArray, 0, nbChannels*outW*outH*sizeof(float));
                    tempPyramid->_gaussianPyramidArray.push_back(gaussianArray);

                    tempPyramid->oddHDCreduced(W, H, nbChannels, s);

                    float *laplacianArray = new float[nbChannels*inW*inH]; // scale (s-1)

                    memset(laplacianArray, 0, nbChannels*inW*inH*sizeof(float));
                    tempPyramid->_laplacianPyramidArray.push_back(laplacianArray);

                    tempPyramid->gaussianToLaplacian(nbChannels, s-1);
                }

                // add contribution to buffers
                for( uint s = scale1 ; s <= scale2 ; ++s ) {

                    const int outR = (int)pow((double)2.0, (double)(s));
                    const int outW = W/outR;
                    const int outH = H/outR;

                    for(int i = 0 ; i < outH ; ++i) {
                        for(int j = 0 ; j < outW ; ++j) {

                            uint oa = i*outW*nbChannels+j*nbChannels + nbChannels-1;
                            float alpha = (tempPyramid->_laplacianPyramidArray[s])[oa];

                            for(uint c = 0 ; c < (nbChannels-1) ; ++c) {

                                uint o = i*outW*nbChannels+j*nbChannels + c;

                                (scaleBuffer[s - scale1])[o] += (tempPyramid->_laplacianPyramidArray[s])[o] * alpha;
                            }
                            (scaleBuffer[s - scale1])[oa] += alpha;
                        }
                    }
                }

                delete tempPyramid;
            }
        }

        // normalize the buffers with the alpha channel

        for( uint s = scale1 ; s <= scale2 ; ++s ) {

            const uint outR = (int)pow((double)2.0, (double)(s));
            const uint outW = W/outR;
            const uint outH = H/outR;
            for(uint i = 0 ; i < outH ; ++i) {
                for(uint j = 0 ; j < outW ; ++j) {

                    uint oa = i*outW*nbChannels+j*nbChannels + nbChannels-1;
                    float alpha = (scaleBuffer[s - scale1])[oa];

                    if(alpha != 0) {
                        for(uint c = 0 ; c < nbChannels-1 ; ++c) {

                            uint o = i*outW*nbChannels+j*nbChannels + c;
                            // add normalized buffers to target pyramid (selected levels)
                            (_uCam->_pyramid->_laplacianPyramidArray[s])[o] += (scaleBuffer[s - scale1])[o]/alpha*1.5;
                        }
                    } else {
                        for(uint c = 0 ; c < nbChannels-1 ; ++c) {

                            uint o = i*outW*nbChannels+j*nbChannels + c;
                            assert((scaleBuffer[s - scale1])[o] == 0);
                        }
                    }

                    (_uCam->_pyramid->_laplacianPyramidArray[s])[oa] += alpha;
                }
            }
        }
        // end normalization

        if(verbose && inputScale == 0) {

            for (uint s = 0 ; s < (uint)_pyramidHeight ; ++s) {

                const int outR = (int)pow((double)2.0, (double)(s));
                const int outW = W/outR;
                const int outH = H/outR;

                _uCam->_pyramid->_laplacianPyramidTex[s]->loadFromData(_uCam->_pyramid->_laplacianPyramidArray[s]);
                std::string splattedGaussianName = _outdir + "/scale0Laplacian_%02i.tiff";
                char tmpNameChar[500];
                sprintf( tmpNameChar, splattedGaussianName.c_str(), s );
                std::cout << "Save Laplacian from warped scale 0 " << s << " in " << tmpNameChar << std::endl;
                (_uCam->_pyramid->_laplacianPyramidTex)[s]->saveRGBAFloatTexture(outW, outH, nbChannels, std::string(tmpNameChar), false);
            }
        }
    }

    if(verbose) {

        for (uint inputScale = 0 ; inputScale < (uint)_pyramidHeight ; ++inputScale) {

            const int outR = (int)pow((double)2.0, (double)(inputScale));
            const int outW = W/outR;
            const int outH = H/outR;

            _uCam->_pyramid->_laplacianPyramidTex[inputScale]->loadFromData(_uCam->_pyramid->_laplacianPyramidArray[inputScale]);
            std::string splattedGaussianName = _outdir + "/finalLaplacian_%02i.tiff";
            char tmpNameChar[500];
            sprintf( tmpNameChar, splattedGaussianName.c_str(), inputScale );
            std::cout << "Save final Laplacian scale " << inputScale << " in " << tmpNameChar << std::endl;
            (_uCam->_pyramid->_laplacianPyramidTex)[inputScale]->saveRGBAFloatTexture(outW, outH, nbChannels, std::string(tmpNameChar), false);
        }
    }

    collapseScaleSpace( W, H, verbose, nbChannels );

    checkGLErrors();

    delete tempTex1;
    delete tempTex2;
    delete warpingBuffer;
}

void Scene::pyramidSplatting(uint originalW, uint originalH) {

    const uint nbChannels = 3;
    const bool verbose = false;

    // create a frame buffer per scale
    for (uint scale = 0 ; scale < (uint)_pyramidHeight ; ++scale) {

        if(verbose) {
            std::cout << "Scale " << scale << " Laplacian splatting." << std::endl;
        }

        forwardWarping( originalW, originalH,
                        scale, // scale
                        &InputView::_laplacianPyramidTex,
                        &TargetView::_laplacianPyramidTex,
                        &TargetView::_laplacianPyramidArray,
                        true, // use original depth map
                        verbose, "/splattedLaplacian_%02i.pfm" ); // verbose
    }

    const bool hack = true;

    if(!hack) {
        // -------------------------------- HIGHEST SCALE SPLATTING -------------------------------- //

        forwardWarping( originalW, originalH,
                        _pyramidHeight,
                        &InputView::_gaussianPyramidTex,
                        &TargetView::_gaussianPyramidTex,
                        &TargetView::_gaussianPyramidArray,
                        false, // use original depth map
                        verbose, "/splattedGaussian_%02i.pfm" ); // verbose

    } else {

        // -------------------------------- LOWEST SCALE SPLATTING -------------------------------- //
        // In the case this part is not commented, it ignores the previous splatting of the largest scale images
        // Performs iterative reduction starting form the original image splatting

        forwardWarping( originalW, originalH,
                        0, // scale
                        &InputView::_gaussianPyramidTex,
                        &TargetView::_gaussianPyramidTex,
                        &TargetView::_gaussianPyramidArray,
                        false, // use original depth map
                        verbose, "/splattedGaussian_%02i.pfm" ); // verbose

        // -------------------------------- COMPUTE GAUSSIAN PYRAMID -------------------------------- //

        for(uint scale = 1 ; scale <= (uint)_pyramidHeight ; ++scale) {

            uint W = originalW / (uint)pow(2.0, (double)(scale-1));
            uint H = originalH / (uint)pow(2.0, (double)(scale-1));
            uint w = originalW / (uint)pow(2.0, (double)scale);
            uint h = originalH / (uint)pow(2.0, (double)scale);

            reduceGaussian(W, H, w, h, nbChannels, _uCam->_gaussianPyramidArray[scale-1], _uCam->_gaussianPyramidArray[scale]);

            //            if(verbose) {
            //                Texture* gaussianTex(0);
            //                if(nbChannels == 3) {
            //                    gaussianTex = new Texture(0, w, h, GL_RGB, GL_FLOAT, GL_RGB32F, false);
            //                } else {
            //                    gaussianTex = new Texture(0, w, h, GL_RGBA, GL_FLOAT, GL_RGBA32F, false);
            //                }
            //                gaussianTex->loadFromData(_uCam->_gaussianPyramidArray[scale]);
            //                char tempNameChar[500];
            //                std::string tempName = _outdir + "/test_gaussian_%02i_%02i.pfm";
            //                sprintf( tempNameChar, tempName.c_str(), scale );
            //                std::cout << "Save test gaussian image in " << tempNameChar << std::endl;
            //                gaussianTex->saveRGBAFloatTexture( w, h, nbChannels, std::string(tempNameChar) );
            //                delete gaussianTex;
            //            }
        }

        //        float topPyramid[3*2*3] = {0.1, 0.2, 0.9, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.8, 0.7, 0.45, 0.2, 0.45, 0.7, 0.8, 0.1, 0.45};
        //        for(int i = 0 ; i < 3*2*3 ; ++i) {
        //            _uCam->_gaussianPyramidArray[_pyramidHeight][i] = topPyramid[i];
        //        }
    }
    // -------------------------------- COLLAPSE LAPLACIAN PYRAMID -------------------------------- //

    for(uint scale = _pyramidHeight ; 0 < scale ; --scale) {

        uint W = originalW / (uint)pow(2.0, (double)(scale-1));
        uint H = originalH / (uint)pow(2.0, (double)(scale-1));
        uint w = originalW / (uint)pow(2.0, (double)scale);
        uint h = originalH / (uint)pow(2.0, (double)scale);

        expandGaussian(W, H, w, h, nbChannels, _uCam->_gaussianPyramidArray[scale], _uCam->_gaussianPyramidArray[scale-1]);
        collapsePyramid(W, H, nbChannels, _uCam->_laplacianPyramidArray[scale-1], _uCam->_gaussianPyramidArray[scale-1]);

        if(verbose) {
            _uCam->_gaussianPyramidTex[scale-1]->loadFromData(_uCam->_gaussianPyramidArray[scale-1]);
            std::string splattedGaussianName = _outdir + "/splattedGaussian_%02i_%02i.pfm";
            char tmpNameChar[500];
            sprintf( tmpNameChar, splattedGaussianName.c_str(), scale-1 );
            std::cout << "Save splatted Gaussian scale " << scale-1 << " in " << tmpNameChar << std::endl;
            _uCam->_gaussianPyramidTex[scale-1]->saveRGBAFloatTexture(W, H, nbChannels, std::string(tmpNameChar), false);
        }
    }

    // -------------------------------- DISPLAY TEXTURE -------------------------------- //

    _uCam->_gaussianPyramidTex[0]->loadFromData(_uCam->_gaussianPyramidArray[0]);
}

void Scene::computeTargetDepthMap( std::string uDepthName ) {

    std::cout << "Compute the depth map of the target view " << uDepthName << std::endl;

    checkGLErrors();

    glBindFramebuffer( GL_FRAMEBUFFER, _FBO->getID() );
    glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE, _uCam->getDepthMap()->getID(), 0 );

    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClearDepth(INVALID_DEPTH);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glViewport( 0, 0, _camWidth, _camHeight );
    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    checkGLErrors();

    for(uint i = 0 ; i < _nbCameras ; ++i) {

        if(i != _renderIndex) {

            std::cout << "Warp view " << i << "... ";
            _uCam->depthMapSplatting(_vCam[i], _FBO, _depthSplattingShader);

        } else {

            std::cout << "View " << i << " is target view, skipping it" << "... ";
        }
    }
    std::cout << std::endl;

    if(!uDepthName.empty()) {
        std::cout << "Save target map in " << uDepthName << std::endl;
        _uCam->getDepthMap()->saveRGBAFloatTexture(_camWidth, _camHeight, 3, uDepthName, false);
    } else {
        std::cout << "Target depth map path is missing, can't save it" << std::endl;
    }
}

void Scene::exportWarps( std::string lfName, std::string tauName, std::string dpartName, std::string uWarpedName ) {

    for(uint i = 0 ; i < _nbCameras ; ++i) {

        char *lfNameChar = new char[500];
        char *tauNameChar = new char[500];
        char *dpartNameChar = new char[500];
        char *uWarpedNameChar = new char[500];

        if(!lfName.empty()) {
            sprintf( lfNameChar, lfName.c_str(), i );
        } else {
            lfNameChar = 0;
        }
        if(!tauName.empty()) {
            sprintf( tauNameChar, tauName.c_str(), i );
        } else {
            tauNameChar = 0;
        }
        if(!dpartName.empty()) {
            sprintf( dpartNameChar, dpartName.c_str(), i );
        } else {
            dpartNameChar = 0;
        }
        if(!uWarpedName.empty()) {
            sprintf( uWarpedNameChar, uWarpedName.c_str(), i );
        } else {
            uWarpedNameChar = 0;
        }

        _vCam[i]->exportWarp(lfNameChar, tauNameChar, dpartNameChar,
                             _uCam, uWarpedNameChar, _vCam[_renderIndex]->getTexture(), _FBO,
                             _tauWarpShader, _tauPartialShader, _warpVkShader);

        delete[] lfNameChar;
        delete[] tauNameChar;
        delete[] dpartNameChar;
        delete[] uWarpedNameChar;
    }
}

//void Scene::computeDepthMap( ) {

//    std::cout << "Compute depth maps for each view..." << std::endl;
//    int k = 0;
//    for ( int s = _sMin ; s <= _sMax ; ++s ) {

////            assert(_meshes.size() >= 2);

////            _vCam[k]->renderDepth( _meshes[0]->getVAO(), // mesh
////                    _meshes[1]->getVAO(), // point cloud
////                    _meshes[0]->getNbTriangles() );

//        _vCam[k]->renderDepth( 0, 0, 0 );
//        ++k;
//    }
//    std::cout << "done!" << std::endl;
//}

void Scene::filterDepthMaps( ) {

    // std::cout << "_nbCameras: " << _nbCameras << std::endl;

    const bool verbose = false;

    for(uint i = 0 ; i < _nbCameras ; ++i) {

        // _vCam[i]->filterDepthmap(_fromRadial2OrthoShader, _FBO, filteredDepthNameChar);
        _vCam[i]->filterDepthmap(_FBO, _fromRadial2OrthoShader);

        // _vCam[i]->createDepthPyramidCPU(i, _pyramidHeight);

        if(verbose && !_outdir.empty()) {

            char filteredDepthChar[500];
            std::string filteredDepth = _outdir + "/filteredDepth_%02i.pfm";
            sprintf( filteredDepthChar, filteredDepth.c_str(), i );
            std::cout << "Export depth map " << i << " in " << filteredDepthChar << std::endl;
            _vCam[i]->getDepthMap()->saveRGBAFloatTexture(_camWidth, _camHeight, 3, std::string(filteredDepthChar), false);
        }
    }
}

//void Scene::createGaussianPyramid() {

//    for(uint i = 0 ; i < _nbCameras ; ++i) {

//        if(i != _renderIndex) {

//            _vCam[i]->createGaussianPyramid(i);
//        }
//    }
//}

void Scene::createGaussianPyramidCPU() {

    for(uint i = 0 ; i < _nbCameras ; ++i) {

        _vCam[i]->createGaussianPyramidCPU(i);
    }
}

void Scene::createGaussianScaleSpace() {

    for(uint i = 0 ; i < _nbCameras ; ++i) {

        _vCam[i]->createGaussianScaleSpace(i);
    }
}

void Scene::createDepthPyramidCPU(int depthPyramidHeight) {

    for(uint i = 0 ; i < _nbCameras ; ++i) {

        if(i != _renderIndex) {

            _vCam[i]->createDepthPyramidCPU(i, depthPyramidHeight);
        }
    }
}

void Scene::createDepthScaleSpaceCPU(int depthScale) {

    for(uint i = 0 ; i < _nbCameras ; ++i) {

        if(i != _renderIndex) {

            _vCam[i]->createDepthScaleSpaceCPU(i, depthScale);
        }
    }
}

void Scene::createTestViews( ) {

    // hard-coded cube and views

    checkGLErrors();

    const bool verbose = false;
    glm::mat4 renderMatrix;

    Cube cube(0.5, "sources/openGL/shaders/simpleColor.vert", "sources/openGL/shaders/simpleColor.frag");
    cube.charger();

    uint viewIndex = 0;
    for ( int s = _sMin ; s <= _sMax ; ++s ) {

        InputView *inputView = new InputView( _camWidth, _camHeight, _outdir, _pyramidHeight );

        if(s == 0) {
            inputView->createTestInputView(90);
        } else {
            inputView->createTestInputView(60);
        }

        // INPUT VIEW

        PinholeCamera inputCam(inputView->getPinholeCamera());
        inputCam.setAsRenderCam( renderMatrix );

        _FBO->clearTexture( inputView->getTexture(), 0.0 );

        glm::mat4 offset(1.0);

        offset[0][0] = s*0.5+1.0;
        offset[1][1] = s*0.5+1.0;
        offset[2][2] = s*0.5+1.0;

        // render view in texture
        cube.renderInFB(renderMatrix*offset, _FBO, inputView->getTexture());

        // init depth map
        inputView->initDepthMap();
        // clear depth (set first channel to invalid value)
        _FBO->clearTexture( inputView->getDepthMap(), 10.0 );

        // render depth map to texture
        cube.renderDepth( _FBO, inputView->getDepthMap(), _depthFromMeshShader,
                          renderMatrix, inputCam._K, inputCam._R, inputCam._t,
                          0.0f );

        if(verbose && !_outdir.empty()) {

            char outDepthNameChar[500];
            const std::string outDepthName = _outdir + "/depthMap_%02i.pfm";
            sprintf( outDepthNameChar, outDepthName.c_str(), s );
            std::cout << "Export depth map " << s << " in " << outDepthNameChar << std::endl;
            inputView->saveDepthMap( std::string(outDepthNameChar) );

            char outImageNameChar[500];
            const std::string outImageName = _outdir + "/image_%02i.png";
            sprintf( outImageNameChar, outImageName.c_str(), s );
            std::cout << "Export image " << s << " in " << outImageNameChar << std::endl;
            inputView->getTexture()->saveRGBAIntTexture( _camWidth, _camHeight, 3, std::string(outImageNameChar), false );
        }

        _vCam.push_back( inputView );

        // import the view to synthetize also, since we remove it only when running cocolib
        if ( s == _sRmv ) {

            _uCam = new TargetView( _camWidth, _camHeight, inputView->getPinholeCamera(), _outdir, _pyramidHeight );
            // empty depth map
            _uCam->initDepthMap();
            // clear depth (set first channel to invalid value)
            _FBO->clearTexture( _uCam->getDepthMap(), INVALID_DEPTH );

            _renderIndex = viewIndex;
        }
        ++viewIndex;
    }

    assert(viewIndex == _vCam.size());
    _nbCameras = viewIndex;

    // RENDERING ON SCREEN

    //    unsigned int frameRate(1000/50);
    //    Uint32 beginingLoop(0), endLoop(0), elapsedTime(0);
    //    beginingLoop = SDL_GetTicks();
    //    endLoop = SDL_GetTicks();
    //    elapsedTime = endLoop - beginingLoop;

    //    uint count = 0;

    //    while(!_input.end()) {

    //        beginingLoop = SDL_GetTicks();

    //        _input.updateEvents();

    //        if(_input.getKey(SDL_SCANCODE_ESCAPE)) {
    //            break;
    //        }

    //        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    //        // render texture to screen
    //        // inputView->drawTexture( _FBO, _textureShader );
    //        _FBO->drawTexture( _vCam[1]->getTexture(), _textureShader );

    //        SDL_GL_SwapWindow(_window);

    //        endLoop = SDL_GetTicks();
    //        elapsedTime = endLoop - beginingLoop;

    //        if(elapsedTime < frameRate) {

    //            SDL_Delay(frameRate - elapsedTime);
    //        }

    //        ++count;
    //    }
}

// PRIVATE

void Scene::moveTargetCam(int frame) {

    _uCam->move(_vCam[5]->getPinholeCamera(), _vCam[3]->getPinholeCamera(), frame);
}

void Scene::init2DTextureShader( Shader *shader ) {

    checkGLErrors();

    assert( shader->add() );

    // Bind shader inputs
    shader->bindAttribLocation( 0, "in_pixel" );
    shader->bindAttribLocation( 1, "in_textureCoord" );

    assert( shader->link() );

    checkGLErrors();
}

void Scene::initQuadGeomShader( ShaderGeometry *shader ) {

    checkGLErrors();

    assert( shader->add() );

    // Bind shader inputs
    shader->bindAttribLocation( 0, "in_pixel" );

    assert( shader->link() );

    checkGLErrors();
}
