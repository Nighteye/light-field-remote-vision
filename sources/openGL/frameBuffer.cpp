#include "frameBuffer.h"
#include "shader.h"

//#include "glm/ext.hpp" // to_string

//#include <iostream>
//#include <cstdio>

FrameBuffer::FrameBuffer(int W, int H , float depthFocal) :
    _id(0),
    _verticesVBO(0), _texCoordVBO(0), _VAO(0), _EBO(0),
    _sensorPlaneVAO(0),_sensorPlaneVBO(0),
    _W( W ), _H( H ),
    _depthFocal(depthFocal),
    _depthBufferID(0) {

    checkGLErrors();

    _refocus = false;
    if(depthFocal != 0) {

        _refocus = true;

    }

    assert( load() );

    checkGLErrors();
}

FrameBuffer::~FrameBuffer() {

    checkGLErrors();

    // frame buffer
    glDeleteFramebuffers( 1, &_id );
    glDeleteFramebuffers( 1, &_depthBufferID );

    // vao
    glDeleteVertexArrays( 1, &_VAO );
    glDeleteBuffers( 1, &_verticesVBO );
    glDeleteBuffers( 1, &_texCoordVBO );
    glDeleteBuffers( 1, &_EBO );

    // sensor plane vao
    glDeleteVertexArrays( 1, &_sensorPlaneVAO );
    glDeleteBuffers( 1, &_sensorPlaneVBO );

    checkGLErrors();
}

void FrameBuffer::createRenderBuffer( GLuint &id, GLenum internalFormat ) {

    if( glIsRenderbuffer(id) == GL_TRUE ) {

        glDeleteRenderbuffers( 1, &id );
    }

    glGenRenderbuffers( 1, &id );

    glBindRenderbuffer( GL_RENDERBUFFER, id );

    glRenderbufferStorage( GL_RENDERBUFFER, internalFormat, _W, _H );

    glBindRenderbuffer( GL_RENDERBUFFER, 0 );
}

void FrameBuffer::loadSimpleVAVBEBO() {

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

    if( glIsVertexArray( _VAO ) == GL_TRUE ) {
        glDeleteVertexArrays( 1, &_VAO );
    }
    if(glIsBuffer( _verticesVBO ) == GL_TRUE ) {
        glDeleteBuffers(1, &_verticesVBO);
    }
    if(glIsBuffer( _texCoordVBO ) == GL_TRUE ) {
        glDeleteBuffers(1, &_texCoordVBO);
    }
    if(glIsBuffer ( _EBO ) == GL_TRUE ) {
        glDeleteBuffers( 1, &_EBO );
    }
    glGenVertexArrays( 1, &_VAO );
    glGenBuffers( 1, &_verticesVBO );
    glGenBuffers( 1, &_texCoordVBO );
    glGenBuffers( 1, &_EBO );

    glBindVertexArray( _VAO );

    glBindBuffer( GL_ARRAY_BUFFER, _verticesVBO );
    glBufferData( GL_ARRAY_BUFFER, nbVertices * sizeof(GLfloat), vertices, GL_STATIC_DRAW );
    glVertexAttribPointer( 0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0 );
    glEnableVertexAttribArray(0);

    glBindBuffer( GL_ARRAY_BUFFER, _texCoordVBO );
    glBufferData( GL_ARRAY_BUFFER, nbVertices * sizeof(GLfloat), texCoord, GL_STATIC_DRAW );
    glVertexAttribPointer( 1, 2, GL_FLOAT, GL_FALSE, 0, (void*)0 );
    glEnableVertexAttribArray(1);

    glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, _EBO );
    glBufferData( GL_ELEMENT_ARRAY_BUFFER, nbIndices * sizeof(GLushort), indices, GL_STATIC_DRAW );

    glBindVertexArray(0);
}

void FrameBuffer::loadSensorPlaneVAVBO() {

    std::vector< GLfloat > planeVertices(2*_W*_H);

    assert(_W*_H != 0);

    for(int i = 0 ; i < _H ; ++i) {
        for(int j = 0 ; j < _W ; ++j) {

            planeVertices[i*_W*2 + j*2 + 0] = (GLfloat)(j + 0.5);
            planeVertices[i*_W*2 + j*2 + 1] = (GLfloat)(i + 0.5);
            // planeVertices[i*_W*2 + j*2 + 1] = (GLfloat)((_H - i - 1)+0.5); // flip y-coordinates
        }
    }

    if( glIsVertexArray( _sensorPlaneVAO ) == GL_TRUE ) {
        glDeleteVertexArrays( 1, &_sensorPlaneVAO );
    }
    if(glIsBuffer( _sensorPlaneVBO ) == GL_TRUE ) {
        glDeleteBuffers(1, &_sensorPlaneVBO);
    }
    glGenVertexArrays( 1, &_sensorPlaneVAO );
    glGenBuffers( 1, &_sensorPlaneVBO );

    glBindVertexArray( _sensorPlaneVAO );

    glBindBuffer( GL_ARRAY_BUFFER, _sensorPlaneVBO );
    glBufferData( GL_ARRAY_BUFFER, 2*_W*_H * sizeof(GLfloat), planeVertices.data(), GL_STATIC_DRAW );
    glVertexAttribPointer( 0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0 );
    glEnableVertexAttribArray(0);

    glBindVertexArray( 0 );

    checkGLErrors();
}

bool FrameBuffer::load() {

    checkGLErrors();

    // ------------------- LOAD FRAMEBUFFER ------------------- //

    if( glIsFramebuffer(_id) == GL_TRUE ) {

        glDeleteFramebuffers( 1, &_id );
    }

    glGenFramebuffers( 1, &_id );

    glBindFramebuffer( GL_FRAMEBUFFER, _id );

    createRenderBuffer( _depthBufferID, GL_DEPTH_COMPONENT24 );

    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, _depthBufferID);

    if( glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE ) {

        glDeleteFramebuffers( 1, &_id );
        glDeleteRenderbuffers( 1, &_depthBufferID );

        std::cout << "Error while loading the FBO" << std::endl;

        return false;
    }

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    // ------------------- LOAD VAO/VBO/EBO ------------------- //

    loadSimpleVAVBEBO();

    // ------------------- LOAD SENSOR PLANE VAO/VBO ------------------- //

    loadSensorPlaneVAVBO();

    return true;
}

void FrameBuffer::addDepthScale( const Texture* inDepthMap, Texture* outDepthMap, const uint scale, Shader* addDepthScaleShader ) {

    checkGLErrors();

    glBindFramebuffer( GL_FRAMEBUFFER, _id );

    // ------------------------------- ATTACH TMP BUFFER ------------------------------- //

    Texture *tmp = new Texture(0, _W, _H, GL_RGB, GL_FLOAT, GL_RGB32F, true);
    tmp->loadEmptyTexture();
    uint tmpIdx = GL_COLOR_ATTACHMENT0;
    glFramebufferTexture2D( GL_FRAMEBUFFER, tmpIdx, GL_TEXTURE_RECTANGLE, tmp->getID(), 0 );

    // ---------------------------- WRITE IN TEMP BUFFER ---------------------------- //

    glDrawBuffer(tmpIdx); // temp texture

    // clear window and buffers
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glViewport( 0, 0, _W, _H );

    glUseProgram( addDepthScaleShader->getProgramID() );

    glBindVertexArray( _VAO );

    addDepthScaleShader->setUniformi( "accuDepthMap", 0 ); // sum of previous scales
    addDepthScaleShader->setUniformi( "inDepthMap", 1 ); // new scale
    addDepthScaleShader->setUniformf( "scale", pow(2.0, (double)scale) );
    addDepthScaleShader->setUniformf( "invalidDepth", INVALID_DEPTH );

    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, outDepthMap->getID() );
    glActiveTexture(GL_TEXTURE1);
    glBindTexture( GL_TEXTURE_RECTANGLE, inDepthMap->getID() );

    glDrawElements( GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_SHORT, (void*)0 );

    glActiveTexture(GL_TEXTURE1);
    glBindTexture( GL_TEXTURE_RECTANGLE, 0 );

    // ---------------- READ AND COPY TEMP BUFFER TO OUTPUT DEPTH MAP --------------- //

    glActiveTexture(GL_TEXTURE0);checkGLErrors();
    glReadBuffer(tmpIdx);checkGLErrors();
    glCopyTexSubImage2D(GL_TEXTURE_RECTANGLE, 0, 0, 0, 0, 0, _W, _H);checkGLErrors();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_RECTANGLE, 0);

    glBindVertexArray(0);

    glUseProgram(0);

    delete tmp;

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    checkGLErrors();
}

void FrameBuffer::fromRadial2Ortho( const Texture* vkDepthMap, const glm::mat3 &vk_K, Shader* fromRadial2OrthoShader ) {

    checkGLErrors();

    glBindFramebuffer( GL_FRAMEBUFFER, _id );

    // ------------------------------- ATTACH TMP BUFFER ------------------------------- //

    Texture *tmp = new Texture(0, _W, _H, GL_RGB, GL_FLOAT, GL_RGB32F, true);
    tmp->loadEmptyTexture();
    uint tmpIdx = GL_COLOR_ATTACHMENT0;
    glFramebufferTexture2D( GL_FRAMEBUFFER, tmpIdx, GL_TEXTURE_RECTANGLE, tmp->getID(), 0 );

    // ---------------------------- WRITE IN TEMP BUFFER ---------------------------- //

    glDrawBuffer(tmpIdx); // temp texture

    // clear window and buffers
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glViewport( 0, 0, _W, _H );

    glUseProgram( fromRadial2OrthoShader->getProgramID() );

    glBindVertexArray( _VAO );

    fromRadial2OrthoShader->setUniformi( "vkDepthMap", 0 );

    glm::mat3 vk_K_inv = glm::inverse(vk_K);
    fromRadial2OrthoShader->setUniformMat3( "vk_K_inv", vk_K_inv );

    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, vkDepthMap->getID() );

    glDrawElements( GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_SHORT, (void*)0 );

    // read and copy temp buffer in depth map
    glReadBuffer(tmpIdx);
    glCopyTexSubImage2D(GL_TEXTURE_RECTANGLE, 0, 0, 0, 0, 0, _W, _H);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_RECTANGLE, 0);

    glBindVertexArray(0);

    glUseProgram(0);

    delete tmp;

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    checkGLErrors();
}

void FrameBuffer::attachSplattingBuffers( const Texture* inputTex, const Texture* tempTex, int W, int H ) {

    checkGLErrors();

    glBindFramebuffer( GL_FRAMEBUFFER, _id );

    glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE, inputTex->getID(), 0 );
    glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_RECTANGLE, tempTex->getID(), 0 );

    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClearDepth(1.0);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glViewport( 0, 0, W, H );

    glDrawBuffer(GL_COLOR_ATTACHMENT1);
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClearDepth(1.0);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glViewport( 0, 0, W, H );

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    checkGLErrors();
}

void FrameBuffer::attachWarpingBuffers( int W, int H, const Texture* tex0, const Texture* tex1, const Texture* tex2 ) {

    checkGLErrors();

    glBindFramebuffer( GL_FRAMEBUFFER, _id );

    glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE, tex0->getID(), 0 );
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClearDepth(1.0);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glViewport( 0, 0, W, H );

    if( tex1 != 0 ) {

        glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_RECTANGLE, tex1->getID(), 0 );
        glDrawBuffer(GL_COLOR_ATTACHMENT1);
        glClearColor(0.0, 0.0, 0.0, 0.0);
        glClearDepth(1.0);
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
        glViewport( 0, 0, W, H );
    }

    if( tex2 != 0 ) {

        glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_RECTANGLE, tex2->getID(), 0 );
        glDrawBuffer(GL_COLOR_ATTACHMENT2);
        glClearColor(0.0, 0.0, 0.0, 0.0);
        glClearDepth(1.0);
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
        glViewport( 0, 0, W, H );
    }

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    checkGLErrors();
}

void FrameBuffer::forwardWarping( const bool visibilityPass,
                                  const uint ratioDepth,
                                  const uint ratioImage,
                                  const Texture* depthMap,
                                  const Texture* inputTex,
                                  const Texture* inputMask,
                                  const glm::mat4 &renderMatrix,
                                  const glm::mat3 &vk_K, const glm::mat3 &vk_R, const glm::vec3 &vk_t,
                                  const glm::mat3 &u_R, const glm::vec3 &u_t,
                                  ShaderGeometry* imageSplattingShader,
                                  uint texIndex ) {

    // -------------------------------- VISIBILITY/BLENDING PASS -------------------------------- //

    if(visibilityPass) {

        // enable writing in depth buffer
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);

    } else {

        // enable blending
        glEnable(GL_BLEND);
        glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE, GL_ONE, GL_ONE);

        // smooth z-buffer
        glDepthMask(GL_FALSE);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(-10.0, -5.0); // TODO tune
    }

    glBindFramebuffer( GL_FRAMEBUFFER, _id );
    glDrawBuffer(texIndex); // tempTex
    checkGLErrors();
    glUseProgram( imageSplattingShader->getProgramID() );
    checkGLErrors();
    glBindVertexArray( _sensorPlaneVAO );

    imageSplattingShader->setUniformi( "vkImage", 1 );
    checkGLErrors();

    imageSplattingShader->setUniformi( "ratioImage", ratioImage );
    imageSplattingShader->setUniformMat4( "renderMatrix", renderMatrix );
    imageSplattingShader->setUniformMat3( "vk_K", vk_K );
    imageSplattingShader->setUniformMat3( "vk_R", vk_R );
    imageSplattingShader->setUniform3fv( "vk_t", vk_t );

    if(_refocus) {

        // std::cout << "_depthFocal: " << _depthFocal << std::endl;
        assert(_depthFocal != 0);
        imageSplattingShader->setUniformf( "depthFocal", _depthFocal );
        imageSplattingShader->setUniformMat3( "u_R", u_R );
        imageSplattingShader->setUniform3fv( "u_t", u_t );

    } else {

        // std::cout << "_depthFocal: " << _depthFocal << std::endl;
        imageSplattingShader->setUniformf( "invalidDepth", INVALID_DEPTH );
        imageSplattingShader->setUniformi( "ratioDepth", ratioDepth );
        imageSplattingShader->setUniformi( "vkDepthMap", 0 );
    }

    checkGLErrors();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, depthMap->getID() );
    glActiveTexture(GL_TEXTURE1);
    glBindTexture( GL_TEXTURE_RECTANGLE, inputTex->getID() );
    checkGLErrors();
    if(inputMask != 0) {
        imageSplattingShader->setUniformi( "vkMask", 2 );
        checkGLErrors();
        glActiveTexture(GL_TEXTURE2);
        checkGLErrors();
        glBindTexture( GL_TEXTURE_RECTANGLE, inputMask->getID() );
    } else {
        assert(false);
    }
    checkGLErrors();
    glDrawArrays( GL_POINTS, 0, _W * _H );
    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, 0 );
    glActiveTexture(GL_TEXTURE1);
    glBindTexture( GL_TEXTURE_RECTANGLE, 0 );
    checkGLErrors();
    glActiveTexture(GL_TEXTURE2);
    glBindTexture( GL_TEXTURE_RECTANGLE, 0 );
    checkGLErrors();

    glBindVertexArray(0);

    glUseProgram(0);

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );
    checkGLErrors();
    glDisable(GL_BLEND);
    glDisable(GL_POLYGON_OFFSET_FILL);
    glDepthMask(GL_TRUE);
    checkGLErrors();
}

// Normalised the splatted Laplacian
void FrameBuffer::splatNormalisation( const Texture* tempTex, Shader* normalizationShader, uint texIndex ) {

    // -------------------------------- NORMALIZATION PASS --------------------------- //
    checkGLErrors();
    glBindFramebuffer( GL_FRAMEBUFFER, _id );
    glDrawBuffer(texIndex); // splattedVk texture

    // disable depth buffer testing and writing while normalizing
    glDepthMask(GL_FALSE);
    glDisable(GL_DEPTH_TEST);

    // clear window
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear( GL_COLOR_BUFFER_BIT );
    glViewport( 0, 0, _W, _H );

    glUseProgram( normalizationShader->getProgramID() );

    glBindVertexArray( _VAO ); // we draw an image-sized rectangle

    normalizationShader->setUniformi( "myTexture", 0 );
    checkGLErrors();
    // input texture is blended temp tex
    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, tempTex->getID() );

    glDrawElements( GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_SHORT, (void*)0 );

    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, 0 );

    glBindVertexArray(0);

    glUseProgram(0);

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    checkGLErrors();
}

void FrameBuffer::addTextures( const Texture* tex1, const Texture* tex2, uint outIndex, Shader* addTexturesShader ) {

    // -------------------------------- NORMALIZATION PASS --------------------------- //
    checkGLErrors();
    glBindFramebuffer( GL_FRAMEBUFFER, _id );
    glDrawBuffer(outIndex); // output texture

    // clear window

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glViewport( 0, 0, _W, _H );

    glUseProgram( addTexturesShader->getProgramID() );

    glBindVertexArray( _VAO ); // we draw an image-sized rectangle

    addTexturesShader->setUniformi( "tex1", 0 );
    addTexturesShader->setUniformi( "tex2", 1 );
    checkGLErrors();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, tex1->getID() );
    glActiveTexture(GL_TEXTURE1);
    glBindTexture( GL_TEXTURE_RECTANGLE, tex2->getID() );

    glDrawElements( GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_SHORT, (void*)0 );

    // read and copy result in tex1
    glActiveTexture(GL_TEXTURE0);
    glReadBuffer(outIndex);
    glCopyTexSubImage2D(GL_TEXTURE_RECTANGLE, 0, 0, 0, 0, 0, _W, _H);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, 0 );
    glActiveTexture(GL_TEXTURE1);
    glBindTexture( GL_TEXTURE_RECTANGLE, 0 );

    glBindVertexArray(0);

    glUseProgram(0);

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    checkGLErrors();
}

void FrameBuffer::computeTauWarps( const Texture* uDepthMap, const Texture* vkDepthMap, const Texture* vkTauWarp,
                                   const glm::mat3 &u_K, const glm::mat3 &u_R, const glm::vec3 &u_t,
                                   const glm::mat3 &vk_K, const glm::mat3 &vk_R, const glm::vec3 &vk_t,
                                   Shader* tauWarpShader ) {

    checkGLErrors();

    glBindFramebuffer( GL_FRAMEBUFFER, _id );

    uint vkTauWarpIdx = GL_COLOR_ATTACHMENT0;
    glFramebufferTexture2D( GL_FRAMEBUFFER, vkTauWarpIdx, GL_TEXTURE_RECTANGLE, vkTauWarp->getID(), 0 );
    glDrawBuffer(vkTauWarpIdx);

    // clear buffer
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glViewport( 0, 0, _W, _H );

    glUseProgram( tauWarpShader->getProgramID() );

    glBindVertexArray( _VAO );

    tauWarpShader->setUniformi( "uDepthMap", 0 );
    tauWarpShader->setUniformi( "vkDepthMap", 1 );

    tauWarpShader->setUniformi( "W", _W );
    tauWarpShader->setUniformi( "H", _H );

    tauWarpShader->setUniformMat3( "u_K", u_K );
    tauWarpShader->setUniformMat3( "u_R", u_R );
    tauWarpShader->setUniform3fv( "u_t", u_t );
    tauWarpShader->setUniformMat3( "vk_K", vk_K );
    tauWarpShader->setUniformMat3( "vk_R", vk_R );
    tauWarpShader->setUniform3fv( "vk_t", vk_t );
    tauWarpShader->setUniformf( "invalidDepth", INVALID_DEPTH );

    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, uDepthMap->getID() );
    glActiveTexture(GL_TEXTURE1);
    glBindTexture( GL_TEXTURE_RECTANGLE, vkDepthMap->getID() );

    glDrawElements( GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_SHORT, (void*)0 );

    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, 0 );
    glActiveTexture(GL_TEXTURE1);
    glBindTexture( GL_TEXTURE_RECTANGLE, 0 );

    glBindVertexArray(0);

    glUseProgram(0);

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    checkGLErrors();
}

void FrameBuffer::computeTauPartial( const Texture* vkDepthMap, const Texture* vkTauPartial,
                                     const glm::mat3 &u_K, const glm::mat3 &u_R, const glm::vec3 &u_t,
                                     const glm::mat3 &vk_K, const glm::mat3 &vk_R, const glm::vec3 &vk_t,
                                     Shader* tauPartialShader ) {

    checkGLErrors();

    glBindFramebuffer( GL_FRAMEBUFFER, _id );

    uint vkTauPartialIdx = GL_COLOR_ATTACHMENT0;
    glFramebufferTexture2D( GL_FRAMEBUFFER, vkTauPartialIdx, GL_TEXTURE_RECTANGLE, vkTauPartial->getID(), 0 );
    glDrawBuffer(vkTauPartialIdx);

    // clear window
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glViewport( 0, 0, _W, _H );

    glUseProgram( tauPartialShader->getProgramID() );

    glBindVertexArray( _VAO );

    tauPartialShader->setUniformi( "vkDepthMap", 0 );

    tauPartialShader->setUniformMat3( "u_K", u_K );
    tauPartialShader->setUniformMat3( "u_R", u_R );
    tauPartialShader->setUniform3fv( "u_t", u_t );
    tauPartialShader->setUniformMat3( "vk_K", vk_K );
    tauPartialShader->setUniformMat3( "vk_R", vk_R );
    tauPartialShader->setUniform3fv( "vk_t", vk_t );

    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, vkDepthMap->getID() );

    glDrawElements( GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_SHORT, (void*)0 );

    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, 0 );

    glBindVertexArray(0);

    glUseProgram(0);

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    checkGLErrors();
}

void FrameBuffer::warpVk( const Texture* vkTauWarp, const Texture* sourceImage, const Texture* warpedVk,
                          Shader* warpVkShader ) {

    checkGLErrors();

    glBindFramebuffer( GL_FRAMEBUFFER, _id );

    uint warpedVkIdx = GL_COLOR_ATTACHMENT0;
    glFramebufferTexture2D( GL_FRAMEBUFFER, warpedVkIdx, GL_TEXTURE_RECTANGLE, warpedVk->getID(), 0 );
    glDrawBuffer(warpedVkIdx);

    // clear window
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glViewport( 0, 0, _W, _H );

    glUseProgram( warpVkShader->getProgramID() );

    glBindVertexArray( _VAO );
    warpVkShader->setUniformi( "tauk", 0 );
    warpVkShader->setUniformi( "sourceImage", 1 );

    warpVkShader->setUniformi( "W", _W );
    warpVkShader->setUniformi( "H", _H );

    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, vkTauWarp->getID() );
    glActiveTexture(GL_TEXTURE1);
    glBindTexture( GL_TEXTURE_RECTANGLE, sourceImage->getID() );

    glDrawElements( GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_SHORT, (void*)0 );

    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, 0 );
    glActiveTexture(GL_TEXTURE1);
    glBindTexture( GL_TEXTURE_RECTANGLE, 0 );

    glBindVertexArray(0);

    glUseProgram(0);

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    checkGLErrors();
}

void FrameBuffer::splatDepth( const Texture* vkDepthMap,
                              const glm::mat3 &u_K, const glm::mat3 &u_R, const glm::vec3 &u_t,
                              const glm::mat4 &renderMatrix, const glm::mat3 &vk_K, const glm::mat3 &vk_R, const glm::vec3 &vk_t,
                              Shader* depthSplattingShader ) {

    checkGLErrors();

    // enable writing in depth buffer to store the closest quads
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);

    glBindFramebuffer( GL_FRAMEBUFFER, _id );
    glDrawBuffer(GL_COLOR_ATTACHMENT0); // uDepthMap

    glUseProgram( depthSplattingShader->getProgramID() );

    glBindVertexArray( _sensorPlaneVAO );

    depthSplattingShader->setUniformi( "vkDepthMap", 0 );

    depthSplattingShader->setUniformMat4( "renderMatrix", renderMatrix );
    depthSplattingShader->setUniformMat3( "vk_K", vk_K );
    depthSplattingShader->setUniformMat3( "vk_R", vk_R );
    depthSplattingShader->setUniform3fv( "vk_t", vk_t );
    depthSplattingShader->setUniformMat3( "u_K", u_K );
    depthSplattingShader->setUniformMat3( "u_R", u_R );
    depthSplattingShader->setUniform3fv( "u_t", u_t );
    depthSplattingShader->setUniformf( "invalidDepth", INVALID_DEPTH );

    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, vkDepthMap->getID() );

    glDrawArrays( GL_POINTS, 0, _W*_H ); // np points = W*H, number of pixels in texture

    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, 0 );

    glBindVertexArray(0);

    glUseProgram(0);

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    checkGLErrors();
}

void FrameBuffer::drawTexture( Texture* texture, Shader* textureShader ) {

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glViewport( 0, 0, _W, _H );

    glUseProgram( textureShader->getProgramID() );

    glBindVertexArray( _VAO ); // we draw an image-sized rectangle

    textureShader->setUniformi( "myTexture", 0 );
    textureShader->setUniformi( "H", _H );

    // input texture is blended temp tex
    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, texture->getID() );

    glDrawElements( GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_SHORT, (void*)0 );

    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, 0 );

    glBindVertexArray(0);

    glUseProgram(0);
}

void FrameBuffer::clearAttachment( uint n, int W, int H ) {

    glBindFramebuffer( GL_FRAMEBUFFER, _id );

    glDrawBuffer(GL_COLOR_ATTACHMENT0+n);
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear( GL_COLOR_BUFFER_BIT );
    glViewport( 0, 0, W, H );

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );
}

void FrameBuffer::clearTexture( Texture* texture, float value ) {

    glBindFramebuffer( GL_FRAMEBUFFER, _id );

    glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE, texture->getID(), 0 );
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glClearColor(value, 0.0, 0.0, 0.0);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glViewport( 0, 0, _W, _H );

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );
}

void FrameBuffer::clearBuffers() {

    glBindFramebuffer( GL_FRAMEBUFFER, _id );

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glViewport( 0, 0, _W, _H );

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );
}

// GETTERS

GLuint FrameBuffer::getID() {

    return _id;
}
