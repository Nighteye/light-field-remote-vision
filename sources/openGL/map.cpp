#include "map.h"

#include "glm/ext.hpp" // to_string

#include <iostream>
#include <cstdio>

// Map is a frame buffer class

Map::Map( int W, int H, bool useStencilBuffer ) :
    _id(0),
    _verticesVBO(0), _texCoordVBO(0), _VAO(0), _EBO(0),
    _sensorPlaneVAO(0),_sensorPlaneVBO(0),
    _W( W ), _H( H ),
    //    _sparseDepth( W, H, GL_RED, GL_FLOAT, GL_R32F, true ),
    //    _depthFromPointCloud( W, H, GL_RED, GL_FLOAT, GL_R32F, true ),
    //    _depthFromMesh( W, H, GL_RED, GL_FLOAT, GL_R32F, true ),
    //    _temp( W, H, GL_RGBA, GL_FLOAT, GL_RGBA32F, true ),
    //    _lowResDepth( W, H, GL_RED, GL_FLOAT, GL_R32F, true ),
    //    _highResDepth( W, H, GL_RED, GL_FLOAT, GL_R32F, true ),
    //    _normalMapTex( W, H, GL_RG, GL_FLOAT, GL_RG32F, true ),
    _depthBufferID(0),
    _useStencilBuffer( useStencilBuffer ),
    // _trilateralFilterShader( "sources/openGL/shaders/trilateralFilter.vert", "sources/openGL/shaders/trilateralFilter.frag", "sources/openGL/shaders/trilateralFilter.geom" ),
    // _depthFromMeshShader( "sources/openGL/shaders/depthFromMesh.vert", "sources/openGL/shaders/depthFromMesh.frag" ),
    _bilateralShader( "sources/openGL/shaders/2Dtexture.vert", "sources/openGL/shaders/bilateral.frag" ),
    _normalizationShader( "sources/openGL/shaders/2Dtexture.vert", "sources/openGL/shaders/normalization.frag" ),
    _tauWarpShader( "sources/openGL/shaders/2Dtexture.vert", "sources/openGL/shaders/tauWarp.frag" ),
    _tauPartialShader( "sources/openGL/shaders/2Dtexture.vert", "sources/openGL/shaders/tauPartial.frag" ),
    _warpVkShader( "sources/openGL/shaders/2Dtexture.vert", "sources/openGL/shaders/warpVk.frag" ),
    _fromRadial2OrthoShader( "sources/openGL/shaders/2Dtexture.vert", "sources/openGL/shaders/fromRadial2Ortho.frag" ),
    _addDepthScaleShader( "sources/openGL/shaders/2Dtexture.vert", "sources/openGL/shaders/addDepthScale.frag" ),
    _depthSplattingShader( "sources/openGL/shaders/depthSplatting.vert", "sources/openGL/shaders/depthSplatting.frag", "sources/openGL/shaders/depthSplatting.geom" ),
    _imageSplattingShader( "sources/openGL/shaders/imageSplatting.vert", "sources/openGL/shaders/imageSplatting.frag", "sources/openGL/shaders/imageSplatting.geom" ) {

    checkGLErrors();

    // initTrilateralFilterShader( &_trilateralFilterShader );
    // initDepthFromMeshShader( &_depthFromMeshShader );
    init2DTextureShader( &_bilateralShader );
    init2DTextureShader( &_normalizationShader );
    init2DTextureShader( &_tauWarpShader );
    init2DTextureShader( &_tauPartialShader );
    init2DTextureShader( &_warpVkShader );
    init2DTextureShader( &_fromRadial2OrthoShader );
    init2DTextureShader( &_addDepthScaleShader );
    initQuadGeomShader( &_depthSplattingShader );
    initQuadGeomShader( &_imageSplattingShader );

    assert( load() );

    checkGLErrors();
}

Map::~Map() {

    checkGLErrors();

    // color buffers
    for(uint i = 0 ; i < _mapVector.size() ; ++i) {

        delete _mapVector[i];
    }

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

void Map::createRenderBuffer( GLuint &id, GLenum internalFormat ) {

    if( glIsRenderbuffer(id) == GL_TRUE ) {

        glDeleteRenderbuffers( 1, &id );
    }

    glGenRenderbuffers( 1, &id );

    glBindRenderbuffer( GL_RENDERBUFFER, id );

    glRenderbufferStorage( GL_RENDERBUFFER, internalFormat, _W, _H );

    glBindRenderbuffer( GL_RENDERBUFFER, 0 );
}

void Map::clearDepthBuffer(uint bufferIndex) {

    checkGLErrors();

    glBindFramebuffer( GL_FRAMEBUFFER, _id );
    glDrawBuffer(GL_COLOR_ATTACHMENT0+bufferIndex);

    glClearColor(INVALID_DEPTH, 0.0, 0.0, 0.0);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glViewport( 0, 0, _W, _H );

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    checkGLErrors();
}

void Map::initBuffer(uint bufferIndex) {

    checkGLErrors();

    glBindFramebuffer( GL_FRAMEBUFFER, _id );
    glDrawBuffer(GL_COLOR_ATTACHMENT0+bufferIndex);

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClearDepth(1.0);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glViewport( 0, 0, _W, _H );

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    checkGLErrors();
}

void Map::clearColorBuffer(uint bufferIndex) {

    checkGLErrors();

    glBindFramebuffer( GL_FRAMEBUFFER, _id );
    glDrawBuffer(GL_COLOR_ATTACHMENT0+bufferIndex);

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear( GL_COLOR_BUFFER_BIT );
    glViewport( 0, 0, _W, _H );

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    checkGLErrors();
}

void Map::computeTauWarps( GLuint uDepthMapID, GLuint vkDepthMapID, uint vkTauWarpIndex,
                           const glm::mat3 &u_K, const glm::mat3 &u_R, const glm::vec3 &u_t,
                           const glm::mat3 &vk_K, const glm::mat3 &vk_R, const glm::vec3 &vk_t) {

    checkGLErrors();

    glBindFramebuffer( GL_FRAMEBUFFER, _id );
    glDrawBuffer(GL_COLOR_ATTACHMENT0+vkTauWarpIndex); // tau warp texture

    // clear window

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glViewport( 0, 0, _W, _H );

    glUseProgram( _tauWarpShader.getProgramID() );

    glBindVertexArray( _VAO );

    _tauWarpShader.setUniformi( "uDepthMap", 0 );
    _tauWarpShader.setUniformi( "vkDepthMap", 1 );

    _tauWarpShader.setUniformi( "W", _W );
    _tauWarpShader.setUniformi( "H", _H );

    _tauWarpShader.setUniformMat3( "u_K", u_K );
    _tauWarpShader.setUniformMat3( "u_R", u_R );
    _tauWarpShader.setUniform3fv( "u_t", u_t );
    _tauWarpShader.setUniformMat3( "vk_K", vk_K );
    _tauWarpShader.setUniformMat3( "vk_R", vk_R );
    _tauWarpShader.setUniform3fv( "vk_t", vk_t );
    _tauWarpShader.setUniformf( "invalidDepth", INVALID_DEPTH );

    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, uDepthMapID );
    glActiveTexture(GL_TEXTURE1);
    glBindTexture( GL_TEXTURE_RECTANGLE, vkDepthMapID );

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

void Map::computeTauPartial( GLuint vkDepthMapID, uint vkTauPartialIndex,
                             const glm::mat3 &u_K, const glm::mat3 &u_R, const glm::vec3 &u_t,
                             const glm::mat3 &vk_K, const glm::mat3 &vk_R, const glm::vec3 &vk_t) {

    checkGLErrors();

    glBindFramebuffer( GL_FRAMEBUFFER, _id );
    glDrawBuffer(GL_COLOR_ATTACHMENT0+vkTauPartialIndex); // tau partial texture

    // clear window

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glViewport( 0, 0, _W, _H );

    glUseProgram( _tauPartialShader.getProgramID() );

    glBindVertexArray( _VAO );

    _tauPartialShader.setUniformi( "vkDepthMap", 0 );

    _tauPartialShader.setUniformMat3( "u_K", u_K );
    _tauPartialShader.setUniformMat3( "u_R", u_R );
    _tauPartialShader.setUniform3fv( "u_t", u_t );
    _tauPartialShader.setUniformMat3( "vk_K", vk_K );
    _tauPartialShader.setUniformMat3( "vk_R", vk_R );
    _tauPartialShader.setUniform3fv( "vk_t", vk_t );

    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, vkDepthMapID );

    glDrawElements( GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_SHORT, (void*)0 );

    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, 0 );

    glBindVertexArray(0);

    glUseProgram(0);

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    checkGLErrors();
}

void Map::warpVk( GLuint taukID, GLuint sourceImageID, uint warpedVkIndex ) {

    checkGLErrors();

    glBindFramebuffer( GL_FRAMEBUFFER, _id );
    glDrawBuffer(GL_COLOR_ATTACHMENT0+warpedVkIndex); // warped view vk texture

    // clear window

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glViewport( 0, 0, _W, _H );

    glUseProgram( _warpVkShader.getProgramID() );

    glBindVertexArray( _VAO );
    _warpVkShader.setUniformi( "tauk", 0 );
    _warpVkShader.setUniformi( "sourceImage", 1 );

    _tauWarpShader.setUniformi( "W", _W );
    _tauWarpShader.setUniformi( "H", _H );

    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, taukID );
    glActiveTexture(GL_TEXTURE1);
    glBindTexture( GL_TEXTURE_RECTANGLE, sourceImageID );

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

void Map::fromRadial2Ortho( uint vkDepthMapIndex, uint tempIndex, const glm::mat3 &vk_K ) {

    checkGLErrors();

    // ---------------------------- WRITE IN TEMP BUFFER ---------------------------- //

    glBindFramebuffer( GL_FRAMEBUFFER, _id );
    glDrawBuffer(GL_COLOR_ATTACHMENT0+tempIndex); // temp texture

    // clear window and buffers
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glViewport( 0, 0, _W, _H );

    glUseProgram( _fromRadial2OrthoShader.getProgramID() );

    glBindVertexArray( _VAO );

    _fromRadial2OrthoShader.setUniformi( "vkDepthMap", 0 );

    glm::mat3 vk_K_inv = glm::inverse(vk_K);
    _fromRadial2OrthoShader.setUniformMat3( "vk_K_inv", vk_K_inv );

    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, getTexID(vkDepthMapIndex) );

    glDrawElements( GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_SHORT, (void*)0 );

    // read and copy temp buffer in depth map
    glReadBuffer(GL_COLOR_ATTACHMENT0+tempIndex);
    glCopyTexSubImage2D(GL_TEXTURE_RECTANGLE, 0, 0, 0, 0, 0, _W, _H);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_RECTANGLE, 0);

    glBindVertexArray(0);

    glUseProgram(0);

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    checkGLErrors();
}

void Map::splatDepth( GLuint vkDepthMapID, uint uDepthMapIndex,
                      const glm::mat3 &u_K, const glm::mat3 &u_R, const glm::vec3 &u_t,
                      const glm::mat4 &renderMatrix, const glm::mat3 &vk_K, const glm::mat3 &vk_R, const glm::vec3 &vk_t) {

    checkGLErrors();

    // enable writing in depth buffer to store the closest quads
    glEnable( GL_DEPTH_TEST );
    glDepthMask(GL_TRUE);

    glBindFramebuffer( GL_FRAMEBUFFER, _id );
    glDrawBuffer(GL_COLOR_ATTACHMENT0+uDepthMapIndex); // u depth map

    glUseProgram( _depthSplattingShader.getProgramID() );

    glBindVertexArray( _sensorPlaneVAO );

    _depthSplattingShader.setUniformi( "vkDepthMap", 0 );

    _depthSplattingShader.setUniformMat4( "renderMatrix", renderMatrix );
    _depthSplattingShader.setUniformMat3( "vk_K", vk_K );
    _depthSplattingShader.setUniformMat3( "vk_R", vk_R );
    _depthSplattingShader.setUniform3fv( "vk_t", vk_t );
    _depthSplattingShader.setUniformMat3( "u_K", u_K );
    _depthSplattingShader.setUniformMat3( "u_R", u_R );
    _depthSplattingShader.setUniform3fv( "u_t", u_t );

    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, vkDepthMapID );

    glDrawArrays( GL_POINTS, 0, _W*_H ); // np points = W*H, number of pixels in texture

    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, 0 );

    glBindVertexArray(0);

    glUseProgram(0);

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    checkGLErrors();
}

void Map::splatVk( GLuint vkDepthMapID, GLuint vkImageID, uint tempIndex,
                   const glm::mat4 &renderMatrix, const glm::mat3 &vk_K, const glm::mat3 &vk_R, const glm::vec3 &vk_t, bool visibilityPass) {

    checkGLErrors();

    // ratio is for multiscale splatting
    const uint ratio = 1;

    // -------------------------------- BLENDING PASS -------------------------------- //

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
        glPolygonOffset(-1.0, -1.0);
    }

    glBindFramebuffer( GL_FRAMEBUFFER, _id );
    glDrawBuffer(GL_COLOR_ATTACHMENT0+tempIndex); // _temp tex

    glUseProgram( _imageSplattingShader.getProgramID() );

    glBindVertexArray( _sensorPlaneVAO );

    _imageSplattingShader.setUniformi( "vkDepthMap", 0 );
    _imageSplattingShader.setUniformi( "vkImage", 1 );

    _imageSplattingShader.setUniformi( "ratio", ratio );
    _imageSplattingShader.setUniformMat4( "renderMatrix", renderMatrix );
    _imageSplattingShader.setUniformMat3( "vk_K", vk_K );
    _imageSplattingShader.setUniformMat3( "vk_R", vk_R );
    _imageSplattingShader.setUniform3fv( "vk_t", vk_t );
    _imageSplattingShader.setUniformf( "invalidDepth", INVALID_DEPTH );

    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, vkDepthMapID );
    glActiveTexture(GL_TEXTURE1);
    glBindTexture( GL_TEXTURE_RECTANGLE, vkImageID );

    glDrawArrays( GL_POINTS, 0, _W*_H );

    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, 0 );
    glActiveTexture(GL_TEXTURE1);
    glBindTexture( GL_TEXTURE_RECTANGLE, 0 );

    glBindVertexArray(0);

    glUseProgram(0);

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    glDisable(GL_BLEND);
    glDisable(GL_POLYGON_OFFSET_FILL);
    glDepthMask(GL_TRUE);
    checkGLErrors();
}

void Map::splatNormalisation( uint splattedVkIndex, uint tempIndex ) {

    // -------------------------------- NORMALIZATION PASS --------------------------- //
    checkGLErrors();
    glBindFramebuffer( GL_FRAMEBUFFER, _id );
    glDrawBuffer(GL_COLOR_ATTACHMENT0+splattedVkIndex); // splattedVk texture

    // clear window

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glViewport( 0, 0, _W, _H );

    glUseProgram( _normalizationShader.getProgramID() );

    glBindVertexArray( _VAO ); // we draw an image-sized rectangle

    _normalizationShader.setUniformi( "myTexture", 0 );
    checkGLErrors();
    // input texture is blended temp tex
    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, getTexID(tempIndex) );

    glDrawElements( GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_SHORT, (void*)0 );

    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, 0 );

    glBindVertexArray(0);

    glUseProgram(0);

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    checkGLErrors();
}


void Map::bilateralFiltering( GLuint imageTexID, uint tempIndex, uint depthMapIndex ) {

    checkGLErrors();

    glBindFramebuffer( GL_FRAMEBUFFER, _id );

    // bilateral filtering, horizontal

    glDrawBuffer(GL_COLOR_ATTACHMENT0+tempIndex); // _temp texture

    // clear temp buffer and resize window
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glViewport( 0, 0, _W, _H );

    glUseProgram( _bilateralShader.getProgramID() );

    glBindVertexArray( _VAO );

    _bilateralShader.setUniformi( "myTexture", 0 );
    _bilateralShader.setUniformi( "sparseDepthMap", 1 );
    _bilateralShader.setUniformi( "horizontal", 1 ); // blur in x

    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, imageTexID );
    glActiveTexture(GL_TEXTURE1);// input tex is sparse depth map (not filtered yet)
    glBindTexture( GL_TEXTURE_RECTANGLE, getTexID(depthMapIndex) );

    glDrawElements( GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_SHORT, (void*)0 );

    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, 0 );
    glActiveTexture(GL_TEXTURE1);
    glBindTexture( GL_TEXTURE_RECTANGLE, 0 );

    // bilateral filtering, vertical

    glDrawBuffer(GL_COLOR_ATTACHMENT0+depthMapIndex); // depth map

    // clear depth map and resize window
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glViewport( 0, 0, _W, _H );

    _bilateralShader.setUniformi( "myTexture", 0 );
    _bilateralShader.setUniformi( "sparseDepthMap", 1 );
    _bilateralShader.setUniformi( "horizontal", 0 ); // blur in y

    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, imageTexID );
    glActiveTexture(GL_TEXTURE1); // input tex is horizontally filtered sparse depth map
    glBindTexture( GL_TEXTURE_RECTANGLE, getTexID(tempIndex) );

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

// add inDepthMapIndex to outDepthMapIndex where outDepthMapIndex is 0
void Map::addDepthScale( uint inDepthMapIndex, uint outDepthMapIndex, uint tempIndex, uint scale ) {

    checkGLErrors();

    // ---------------------------- WRITE IN TEMP BUFFER ---------------------------- //

    glBindFramebuffer( GL_FRAMEBUFFER, _id );
    glDrawBuffer(GL_COLOR_ATTACHMENT0+tempIndex); // temp texture

    // clear window and buffers
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glViewport( 0, 0, _W, _H );

    glUseProgram( _addDepthScaleShader.getProgramID() );

    glBindVertexArray( _VAO );

    _addDepthScaleShader.setUniformi( "accuDepthMap", 0 ); // sum of previous scales
    _addDepthScaleShader.setUniformi( "inDepthMap", 1 ); // new scale
    _addDepthScaleShader.setUniformf( "scale", pow(2.0, (double)scale) );
    _addDepthScaleShader.setUniformf( "invalidDepth", INVALID_DEPTH );

    glActiveTexture(GL_TEXTURE0);
    glBindTexture( GL_TEXTURE_RECTANGLE, getTexID(outDepthMapIndex) );
    glActiveTexture(GL_TEXTURE1);
    glBindTexture( GL_TEXTURE_RECTANGLE, getTexID(inDepthMapIndex) );

    glDrawElements( GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_SHORT, (void*)0 );

    glActiveTexture(GL_TEXTURE1);
    glBindTexture( GL_TEXTURE_RECTANGLE, 0 );

    // ---------------- READ AND COPY TEMP BUFFER TO OUTPUT DEPTH MAP --------------- //

    glActiveTexture(GL_TEXTURE0);
    glReadBuffer(GL_COLOR_ATTACHMENT0+tempIndex);
    glCopyTexSubImage2D(GL_TEXTURE_RECTANGLE, 0, 0, 0, 0, 0, _W, _H);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_RECTANGLE, 0);

    glBindVertexArray(0);

    glUseProgram(0);

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    checkGLErrors();
}

//// render depth map from point cloud using trilateral filtering (space, intensity and depth)
//void Map::trilateralFiltering( GLuint imageTexID, // source image
//                               const glm::mat4 &renderMatrix,
//                               const glm::mat3 &vi_Kinv,
//                               const glm::mat3 &vi_R,
//                               const glm::vec3 &vi_C,
//                               const GLuint pointCloudVAO,
//                               const uint nbPoints ) {

//    checkGLErrors();

//    // -------------------------------- BLENDING PASS -------------------------------- //

//    glDisable( GL_DEPTH_TEST );
//    glDepthMask(GL_FALSE);
//    glEnable(GL_BLEND);

//    glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
//    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE, GL_ONE, GL_ONE);

//    glBindFramebuffer( GL_FRAMEBUFFER, _id );
//    glDrawBuffer(GL_COLOR_ATTACHMENT3); // _temp

//    // clear window

//    glClearColor(0.0, 0.0, 0.0, 0.0);
//    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
//    glViewport( 0, 0, _W, _H );

//    glUseProgram( _trilateralFilterShader.getProgramID() );

//    glBindVertexArray( pointCloudVAO );

//    _trilateralFilterShader.setUniformi( "sourceTex", 0 );
//    _trilateralFilterShader.setUniformMat4( "renderMatrix", renderMatrix );
//    _trilateralFilterShader.setUniformMat3( "vi_Kinv", vi_Kinv );
//    _trilateralFilterShader.setUniformMat3( "vi_R", vi_R );
//    _trilateralFilterShader.setUniform3fv( "vi_C", vi_C );

//    // input texture is source image
//    glBindTexture( GL_TEXTURE_RECTANGLE, imageTexID );

//    glDrawArrays( GL_POINTS, 0, nbPoints );

//    glBindTexture( GL_TEXTURE_RECTANGLE, 0 );

//    glBindVertexArray(0);

//    glUseProgram(0);

//    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

//    glDisable(GL_BLEND);
//    glDepthMask(GL_TRUE);

//    // -------------------------------- NORMALIZATION PASS --------------------------- //

//    glBindFramebuffer( GL_FRAMEBUFFER, _id );
//    glDrawBuffer(GL_COLOR_ATTACHMENT2); // _depthFromPointCloud texture

//    // clear window

//    glClearColor(0.0, 0.0, 0.0, 0.0);
//    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
//    glViewport( 0, 0, _W, _H );

//    glUseProgram( _normalizationShader.getProgramID() );

//    glBindVertexArray( _VAO ); // we draw an image-sized rectangle

//    _normalizationShader.setUniformi( "myTexture", 0 );

//    // input texture is blended temp tex
//    glBindTexture( GL_TEXTURE_RECTANGLE, _temp.getID() );

//    glDrawElements( GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_SHORT, (void*)0 );

//    glBindTexture( GL_TEXTURE_RECTANGLE, 0 );

//    glBindVertexArray(0);

//    glUseProgram(0);

//    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

//    checkGLErrors();
//}

//void Map::projectPointCloud( const glm::mat4 &renderMatrix,
//                                     const glm::mat3 &vi_R,
//                                     const glm::vec3 &vi_C,
//                                     const GLuint pointCloudVAO,
//                                     const uint nbPoints ) {

//    checkGLErrors();

//    glEnable( GL_DEPTH_TEST );
//    glCullFace(GL_BACK);
//    glEnable(GL_CULL_FACE);

//    glBindFramebuffer( GL_FRAMEBUFFER, _id );
//    glDrawBuffer(GL_COLOR_ATTACHMENT1); // _sparseDepth texture

//    // clear window

//    glClearColor(0.0, 0.0, 0.0, 0.0);
//    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
//    glViewport( 0, 0, _W, _H );

//    // point cloud projection

//    glUseProgram( _depthFromMeshShader.getProgramID() );

//    glBindVertexArray( pointCloudVAO );

//    _depthFromMeshShader.setUniformMat4( "renderMatrix", renderMatrix );
//    _depthFromMeshShader.setUniformMat3( "vi_R", vi_R );
//    _depthFromMeshShader.setUniform3fv( "vi_C", vi_C );

//    glDrawArrays( GL_POINTS, 0, nbPoints );

//    glBindVertexArray(0);

//    glUseProgram(0);

//    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

//    glDisable(GL_CULL_FACE);
//    glDisable( GL_DEPTH_TEST );

//    checkGLErrors();
//}

//void Map::renderDepthFromMesh( const glm::mat4 &renderMatrix, const glm::mat3 &vi_R, const glm::vec3 &vi_C, const GLuint meshVAO, const uint nbTriangles ) {

//    checkGLErrors();

//    glEnable( GL_DEPTH_TEST );
//    glCullFace( GL_BACK );
//    glEnable( GL_CULL_FACE );

//    glBindFramebuffer( GL_FRAMEBUFFER, _id );
//    glDrawBuffer(GL_COLOR_ATTACHMENT0); // _depthFromMesh texture

//    glClearColor(-1.0, -1.0, 0.0, 0.0);
//    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
//    glViewport( 0, 0, _W, _H );

//    glUseProgram( _depthFromMeshShader.getProgramID() );

//    glBindVertexArray( meshVAO );

//    _depthFromMeshShader.setUniformMat4( "renderMatrix", renderMatrix );
//    _depthFromMeshShader.setUniformMat3( "vi_R", vi_R );
//    _depthFromMeshShader.setUniform3fv( "vi_C", vi_C );

//    glDrawElements( GL_TRIANGLES, nbTriangles, GL_UNSIGNED_INT, 0 );

//    glBindVertexArray(0);

//    glUseProgram(0);

//    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

//    glDisable( GL_CULL_FACE );
//    glDisable( GL_DEPTH_TEST );

//    checkGLErrors();
//}

//void Map::saveDepthFromPointCloud( const std::string &depthMapName ) {

//    _depthFromPointCloud.saveRGBAFloatTexture( _W, _H, 1, depthMapName, false );
//}

//void Map::saveDepthFromMesh( const std::string &depthMapName ) {

//    _depthFromMesh.saveRGBAFloatTexture( _W, _H, 1, depthMapName, false );
//}

//void Map::saveLowResDepth( const std::string &depthMapName ) {

//    _lowResDepth.saveRGBAFloatTexture( _W, _H, 1, depthMapName, false );
//}

//void Map::saveHighResDepth( const std::string &depthMapName ) {

//    _highResDepth.saveRGBAFloatTexture( _W, _H, 1, depthMapName, false );
//}

void Map::saveMap( const std::string &mapName, uint channels, uint id ) {

    assert(id < _mapVector.size());
    _mapVector[id]->saveRGBAFloatTexture( _W, _H, channels, mapName, false );
}

// GETTERS

//Texture Map::getDepthTex() const {

//    return _lowResDepth;
//}

//Texture Map::getNormalTex() const {

//    return _normalMapTex;
//}

Texture* Map::getMap(uint index) const {

    return _mapVector[index];
}

GLuint Map::getTexID(uint index) const {

    return _mapVector[index]->getID();
}

GLuint Map::getID() const {

    return _id;
}

//GLuint Map::getDepthFromMeshID( ) const {

//    return _depthFromMesh.getID();
//}

int Map::getWidth() const {

    return _W;
}

int Map::getHeight() const {

    return _H;
}

uint* Map::addEmptyBuffer(uint channels) {

    assert(_mapVector.size() < NB_COLOR_BUFFER_MAX-1);

    Texture *map(0);

    uint newIndex = _mapVector.size();

    if(channels == 1) {
        map = new Texture(newIndex, _W, _H, GL_RED, GL_FLOAT, GL_R32F, true);
    } else if(channels == 2) {
        map = new Texture(newIndex, _W, _H, GL_RG, GL_FLOAT, GL_RG32F, true);
    } else if(channels == 3) {
        map = new Texture(newIndex, _W, _H, GL_RGB, GL_FLOAT, GL_RGB32F, true);
    } else {
        map = new Texture(newIndex, _W, _H, GL_RGBA, GL_FLOAT, GL_RGBA32F, true);
    }

    glBindFramebuffer( GL_FRAMEBUFFER, _id );
    map->loadEmptyTexture();
    glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0+newIndex, GL_TEXTURE_RECTANGLE, map->getID(), 0 );

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    // std::cout << "newIndex: " << newIndex << std::endl;

    _mapVector.push_back(map);

    return &(map->_index);
}

uint* Map::addBufferFromData(const std::vector< std::vector< float > > &data, uint w, uint h, uint channels) {

    assert(_mapVector.size() < NB_COLOR_BUFFER_MAX-1);
    assert(_W/w == _H/h);
    assert(_W >= (int)w && _H >= (int)h);

    Texture *map(0);

    uint newIndex = _mapVector.size();

    if(channels == 1) {
        map = new Texture(newIndex, _W, _H, GL_RED, GL_FLOAT, GL_R32F, true);
    } else if(channels == 2) {
        map = new Texture(newIndex, _W, _H, GL_RG, GL_FLOAT, GL_RG32F, true);
    } else if(channels == 3) {
        map = new Texture(newIndex, _W, _H, GL_RGB, GL_FLOAT, GL_RGB32F, true);
    } else {
        map = new Texture(newIndex, _W, _H, GL_RGBA, GL_FLOAT, GL_RGBA32F, true);
    }

    glBindFramebuffer( GL_FRAMEBUFFER, _id );
    map->loadFromData(data, _W/w, (float)INVALID_DEPTH, false);
    glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0+newIndex, GL_TEXTURE_RECTANGLE, map->getID(), 0 );

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    // std::cout << "newIndex: " << newIndex << std::endl;

    _mapVector.push_back(map);

    return &(map->_index);
}

void Map::deleteBuffer(uint index) {

    assert(!_mapVector.empty());
    assert(index < _mapVector.size());

    delete _mapVector[index];
    _mapVector.erase(_mapVector.begin() + index);

    // decrement indices
    for(uint i(index) ; i < _mapVector.size() ; ++i ) {

        --_mapVector[i]->_index;
    }

    // std::cout << "_mapVector.size(): " << _mapVector.size() << std::endl;
}

bool Map::load() {

    checkGLErrors();

    // ------------------- LOAD FRAMEBUFFER ------------------- //

    if( glIsFramebuffer(_id) == GL_TRUE ) {

        glDeleteFramebuffers( 1, &_id );
    }

    glGenFramebuffers( 1, &_id );

    glBindFramebuffer( GL_FRAMEBUFFER, _id );

    if( _useStencilBuffer == true ) {

        createRenderBuffer( _depthBufferID, GL_DEPTH24_STENCIL8 );

    } else {

        createRenderBuffer( _depthBufferID, GL_DEPTH_COMPONENT24 );
    }

    if( _useStencilBuffer == true ) {

        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, _depthBufferID);

    } else {

        glFramebufferRenderbuffer( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, _depthBufferID );
    }

    if( glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE ) {

        glDeleteFramebuffers( 1, &_id );
        glDeleteRenderbuffers( 1, &_depthBufferID );

        std::cout << "Error while loading the FBO" << std::endl;

        return false;
    }

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    // ------------------- LOAD VAO/VBO/EBO ------------------- //

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

    // ------------------- LOAD SENSOR PLANE VAO/VBO ------------------- //

    std::vector< GLfloat > planeVertices(2*_W*_H);

    assert(_W*_H != 0);

    for(int i = 0 ; i < _H ; ++i) {
        for(int j = 0 ; j < _W ; ++j) {

            planeVertices[i*_W*2 + j*2 + 0] = (GLfloat)(j + 0.5);
            planeVertices[i*_W*2 + j*2 + 1] = (GLfloat)(i + 0.5);
            //            planeVertices[i*_W*2 + j*2 + 1] = (GLfloat)((_H - i - 1)+0.5); // flip y-coordinates
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

    return true;
}

//void Map::initTrilateralFilterShader( ShaderGeometry *shader) {

//    checkGLErrors();

//    assert( shader->add() );

//    // Bind shader inputs
//    shader->bindAttribLocation( 0, "in_vertex" );
//    shader->bindAttribLocation( 1, "in_normal" );

//    assert( shader->link() );

//    checkGLErrors();
//}

//void Map::initDepthFromMeshShader( Shader *shader ) {

//    assert( shader->add() );

//    // Bind shader inputs
//    shader->bindAttribLocation( 0, "in_vertex" );

//    assert( shader->link() );
//}

void Map::init2DTextureShader( Shader *shader) {

    checkGLErrors();

    assert( shader->add() );

    // Bind shader inputs
    shader->bindAttribLocation( 0, "in_pixel" );
    shader->bindAttribLocation( 1, "in_textureCoord" );

    assert( shader->link() );

    checkGLErrors();
}

void Map::initQuadGeomShader( ShaderGeometry *shader) {

    checkGLErrors();

    assert( shader->add() );

    // Bind shader inputs
    shader->bindAttribLocation( 0, "in_pixel" );

    assert( shader->link() );

    checkGLErrors();
}

