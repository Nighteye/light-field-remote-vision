#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <cocolib/cocolib/common/gsl_image.h>
#include <cocolib/cocolib/common/debug.h>

#define cimg_display 0
#define cimg_use_tiff
#define cimg_use_png
#include "CImg.h"

#include "openglFunctions.h"
#include "config.h"
#include "assert.h"
#include "pinholeCamera.h"
#include "shaderHandler.h"

#define checkGLErrors() {\
    GLenum error = glGetError(); \
    if(error != GL_NO_ERROR) { \
    std::cout << "GL_ERROR :" << __FILE__ << " "<< __LINE__ << " " << error << std::endl; \
    } \
    }

void saveRGBAFloatTexture(GLuint textureId, int render_w, int render_h, int depth, const std::string &name, bool flip) {

    checkGLErrors();

    cimg_library::CImg<float> backwardMap;
    if (depth == 4) {
        backwardMap.resize(depth, render_w, render_h, 1);
    } else {
        backwardMap.resize(render_w, render_h, 1, depth);
    }

    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glBindTexture(GL_TEXTURE_RECTANGLE, textureId);

    checkGLErrors();

    if (depth == 4) {
        glGetTexImage(GL_TEXTURE_RECTANGLE, 0, GL_RGBA, GL_FLOAT, &backwardMap(0,0,0,0));
        checkGLErrors();
        backwardMap.permute_axes("yzcx");
    } else {
        if (depth>0) {
            glGetTexImage(GL_TEXTURE_RECTANGLE, 0, GL_RED, GL_FLOAT, &backwardMap(0,0,0,0));
            checkGLErrors();
        }
        if (depth>1 ) {
            glGetTexImage(GL_TEXTURE_RECTANGLE, 0, GL_GREEN, GL_FLOAT, &backwardMap(0,0,0,1));
            checkGLErrors();
        }
        if (depth>2) {
            glGetTexImage(GL_TEXTURE_RECTANGLE, 0, GL_BLUE, GL_FLOAT, &backwardMap(0,0,0,2));
            checkGLErrors();
        }
    }
    checkGLErrors();

    glBindTexture(GL_TEXTURE_RECTANGLE, 0);

    checkGLErrors();
    if(flip) {
        backwardMap.mirror('y');
    }
    try {
        backwardMap.save(name.c_str());
    } catch (cimg_library::CImgIOException) {
        printf("Exception COUGHT: file not saved\n");
    }
}

// Convert Mat4 to col-major array
void setMat4_to_Ptr(const glm::mat4 &Mat, float ptr[16]) {

    for (int i=0; i<4; ++i) {
        for (int j=0; j<4; ++j) {
            ptr[4*j+i] = Mat[i][j];
        }
    }
}

// Convert Mat3 to col-major array
void setMat3_to_Ptr(const glm::mat3 &Mat, float ptr[9]) {

    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j) {
            ptr[3*j+i] = Mat[i][j];
        }
    }
}

// Convert Mat34 to 4x4 col-major array
void setMat34_to_Ptr(const glm::mat4x3 &Mat, float ptr[16]) {

    for (int i=0; i<4; ++i) {
        for (int j=0; j<4; ++j) {
            if (i == 3 && j == 3) {
                ptr[4*j+i] = 1.;
            } else if (i == 3 ) {
                ptr[4*j+i] = 0.;
            } else {
                ptr[4*j+i] = Mat[j][i];
            }
        }
    }
}

void setCurrentCamFrustum(const PinholeCamera &camera, int w, int h,
                          glm::mat4 &projMatrix, glm::mat4 &modelViewMat) {

    checkGLErrors();

    GLfloat zNear = 1e-2;
    GLfloat zFar = 1e5;

    GLfloat focal_x = camera._K[0][0];
    GLfloat focal_y = camera._K[1][1];

    GLfloat fW = w/focal_x * zNear /2;
    GLfloat fH = h/focal_y * zNear /2;

    GLfloat pp_x = camera._K[2][0] - w/2.;
    GLfloat pp_y = camera._K[2][1] - h/2.;

    GLfloat pp_offset_x = pp_x/focal_x * zNear;
    GLfloat pp_offset_y = pp_y/focal_y * zNear;

    //glFrustum( -fW-pp_offset_x, fW-pp_offset_x, fH-pp_offset_y, -fH-pp_offset_y, zNear, zFar );
    // https://www.opengl.org/sdk/docs/man2/xhtml/glFrustum.xml
    GLfloat A = -pp_offset_x/ fW; //(fW-pp_offset_x) + (-fW-pp_offset_x) / ((fW-pp_offset_x)-(-fW-pp_offset_x));
    GLfloat B =  pp_offset_y/ fH;//(-fH-pp_offset_y + (fH-pp_offset_y)) / (-fH-pp_offset_y - (fH-pp_offset_y));
    GLfloat C = - (zFar + zNear) / (zFar-zNear);
    GLfloat D = -2* zFar * zNear / (zFar-zNear);

    projMatrix = glm::mat4(1.0f);
    projMatrix[0][0] =  zNear / fW;
    projMatrix[1][1] = -zNear / fH;

    projMatrix[2][0] = A;
    projMatrix[2][1] = B;
    projMatrix[2][2] = C;
    projMatrix[2][3] = -1;
    projMatrix[3][2] = D;

    projMatrix[3][3] = 0;

    // reverse the z of the camera. document cameras point to (0,0,1) whil opengl cameras point to (0,0,-1)
    //glMultMatrixf((GLfloat*)m_z_invert);
    glm::mat4 z_invert(1.0f);
    z_invert[2][2] = -1.;
    projMatrix = projMatrix * z_invert;
    checkGLErrors();

    // apply render camera transformation
    modelViewMat[0] = glm::vec4(camera._R[0], 0.0f);
    modelViewMat[1] = glm::vec4(camera._R[1], 0.0f);
    modelViewMat[2] = glm::vec4(camera._R[2], 0.0f);
    modelViewMat[3] = glm::vec4(camera._t, 0.0f);
}

WarpGL::WarpGL() {

    m_depth_epsilon = 0.01; // How to estimate this parameter? Depends on the scene?
    m_nb_displacement = 5;
    m_depth_mapping = false;
    m_mapping_cut_depth = 24.0;
    m_mapping_factor = 3.0;

    window_width = 1000;
    window_height = 600;
}

WarpGL::~WarpGL() {

}

// Compute the warp tau from the render camera to the current camera
// x = tau(y), where x is in target camera units, and y in current camera units
void WarpGL::computeWarps( Config_data *config_data,
                           unsigned int nview,
                           int nbTriangles,
                           const PinholeCamera &targetCam, // - has the output resolution
                           const PinholeCamera &currentCam ){ // - has the input resolution

    checkGLErrors();

    GLuint fbo = m_render_fbo[0];
    GLuint vaoID = m_vaoID[0];
    GLuint vi_texture = m_imageVector[nview].texture; // in - input resolution
    GLuint per_pixel_sigma = m_per_pixel_sigma[nview]; // in - output resolution: we compute it in the same image
    GLuint lookup_texture = m_lookup_texture[nview];
    GLuint depth_vi = m_depth_vi[nview][m_nb_displacement/2]; // in - input resolution
    float depth_epsilon = m_depth_epsilon; // in
    GLuint buffer = m_buffer[0]; // target image viewed from current camera v_k
    GLuint tau = m_backwards[0]; // output tau warps
    GLuint partial_tau = m_partial_tau_grad_v[0]; // output tau derivatives
    bool depth_mapping = m_depth_mapping;
    float mapping_cut_depth = m_mapping_cut_depth;
    float mapping_factor = m_mapping_factor;

    int w = config_data->_w;
    int h = config_data->_h;
    assert( w * h > 0 );

    glViewport( 0, 0, w, h );
    glm::mat4 modelViewMat, projectionMat;
    setCurrentCamFrustum( currentCam, w, h, projectionMat, modelViewMat );

//    TRACE(projectionMat(0, 0) << " " << projectionMat(0, 1) << " " << projectionMat(0, 2) << " " << projectionMat(0, 3) << std::endl <<
//          projectionMat(1, 0) << " " << projectionMat(1, 1) << " " << projectionMat(1, 2) << " " << projectionMat(1, 3) << std::endl <<
//          projectionMat(2, 0) << " " << projectionMat(2, 1) << " " << projectionMat(2, 2) << " " << projectionMat(2, 3) << std::endl <<
//          projectionMat(3, 0) << " " << projectionMat(3, 1) << " " << projectionMat(3, 2) << " " << projectionMat(3, 3) << std::endl << std::endl);

//    TRACE(modelViewMat(0, 0) << " " << modelViewMat(0, 1) << " " << modelViewMat(0, 2) << " " << modelViewMat(0, 3) << std::endl <<
//          modelViewMat(1, 0) << " " << modelViewMat(1, 1) << " " << modelViewMat(1, 2) << " " << modelViewMat(1, 3) << std::endl <<
//          modelViewMat(2, 0) << " " << modelViewMat(2, 1) << " " << modelViewMat(2, 2) << " " << modelViewMat(2, 3) << std::endl <<
//          modelViewMat(3, 0) << " " << modelViewMat(3, 1) << " " << modelViewMat(3, 2) << " " << modelViewMat(3, 3) << std::endl << std::endl);

    // Bind rendering FBO
    glBindFramebuffer( GL_FRAMEBUFFER, fbo );

    checkGLErrors();

    // configure framebuffer attachments
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, buffer, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, tau, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, partial_tau, 0);
    checkGLErrors();

    const GLenum DrawBuffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
    glDrawBuffers(3, DrawBuffers);
    checkGLErrors();

    int status = glCheckFramebufferStatus(GL_FRAMEBUFFER);

    if (status != GL_FRAMEBUFFER_COMPLETE ) {
        TRACE( "     Framebuffer KO: PROBLEM" << std::endl );
        return;
    }

    glClear(GL_DEPTH_BUFFER_BIT);
    checkGLErrors();

    GLfloat color[4]={0.,0.,0.,0.};
    glClearBufferfv(GL_COLOR, 0, color);

    // Invalid values are:
    // enourmous vector for invalid warp
    // 0 weight for differential of the warp
    GLfloat warp_init[4]={(float)-w, (float)-h, 0, 0};
    glClearBufferfv(GL_COLOR, 1, warp_init);

    // Invalid values are:
    // 0 for per pixel sigma
    // zero vector for invalid  tau/ partial z (epipolar line)
    GLfloat partial_init[4]={0.,0,0,0};
    glClearBufferfv(GL_COLOR, 2, partial_init);

    checkGLErrors();

    warpShader.useProgram();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_RECTANGLE, vi_texture);
    warpShader.setUniformi("myTexture", 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_RECTANGLE, lookup_texture);
    //warpShader.setUniformi("myCoordTexture", 1);


    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_RECTANGLE, depth_vi);
    warpShader.setUniformi("myDepthTexture", 2);

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_RECTANGLE, per_pixel_sigma);
    warpShader.setUniformi("myPerPixelSigmaTexture", 3);

    warpShader.setUniformf("depth_epsilon", depth_epsilon);

    warpShader.setUniformi("depth_mapping", depth_mapping);
    warpShader.setUniformf("mapping_cut_depth", mapping_cut_depth);
    warpShader.setUniformf("mapping_factor", mapping_factor);

    // Current camera matrices
    warpShader.setUniformf("vi_width", w);
    warpShader.setUniformf("vi_height", h);

    float vi_P[16];
    setMat34_to_Ptr(currentCam._P, vi_P);
    warpShader.setUniformMat4f("vi_P", vi_P);

    float vi_C[3] = {(float)currentCam._C[0], (float)currentCam._C[1],(float)currentCam._C[2]};
    warpShader.setUniform3fv("vi_C", vi_C);

    float vi_R[9];
    setMat3_to_Ptr(currentCam._R, vi_R);
    warpShader.setUniformMat3f("vi_R", vi_R);

    // Target camera matrices
    warpShader.setUniformf("u_width", w);
    warpShader.setUniformf("u_height", h);

    float u_P[16];
    setMat34_to_Ptr(targetCam._P, u_P);
    warpShader.setUniformMat4f("u_P", u_P);

    float u_C[3] = {(float)targetCam._C[0], (float)targetCam._C[1], (float)targetCam._C[2]};
    warpShader.setUniform3fv("u_C", u_C);

    float u_R[9];
    setMat3_to_Ptr(targetCam._R, u_R);
    warpShader.setUniformMat3f("u_R", u_R);

    float u_t[3] = {(float)targetCam._t[0], (float)targetCam._t[1], (float)targetCam._t[2]};
    warpShader.setUniform3fv("u_t", u_t);

    float u_pp[2] = {(float)targetCam._K[2][0], (float)targetCam._K[2][1]};
    warpShader.setUniform2fv("u_pp", u_pp);

    warpShader.setUniformf("u_f", targetCam._K[0][0]);

    // Transition cameras for closed form computations of deformation
    // A = (K_u * R_u * (R_vi)^t * (K_vi)^(-1) )
    glm::mat3 A = targetCam._K * targetCam._R * glm::transpose(currentCam._R) * glm::inverse(currentCam._K);
    // b = K_u * t_u - K_u * R_u * (R_vi)t * t_vi
    glm::vec3 b = targetCam._K * targetCam._t - targetCam._K * targetCam._R * glm::transpose(currentCam._R) * currentCam._t;

    float A_def[9];
    setMat3_to_Ptr(A, A_def);
    float b_def[3] = {(float)b[0], (float)b[1], (float)b[2]};

    warpShader.setUniformMat3f("A_def", A_def);
    warpShader.setUniform3fv("b_def", b_def);

    warpShader.setUniformMat3f("A_tau", A_def);
    warpShader.setUniform3fv("b_tau", b_def);

    checkGLErrors();

    float ptr[16];
    setMat4_to_Ptr(modelViewMat, ptr);
    warpShader.setUniformMat4f("modelViewMat", ptr);

    setMat4_to_Ptr(projectionMat, ptr);
    warpShader.setUniformMat4f("projectionMat", ptr);
    // ---- finished uniforms

    glEnable(GL_DEPTH_TEST);
    glCullFace(GL_BACK);
    glEnable(GL_CULL_FACE);

    glBindVertexArray(vaoID);

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(3);

    glDrawElements(GL_TRIANGLES, nbTriangles, GL_UNSIGNED_INT, 0);

    glDisableVertexAttribArray(3);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);

    glBindVertexArray(0);

    warpShader.stopUsingProgram();

    checkGLErrors();

    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_RECTANGLE, 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_RECTANGLE, 0);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_RECTANGLE, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    checkGLErrors();
}

void WarpGL::createVAOs( const unsigned int nbCams ) {

    // alloc variable
    m_vaoID = new GLuint[nbCams];
    m_vertBufferID = new GLuint[nbCams];
    m_normalsBufferID = new GLuint[nbCams];
    m_indexBufferID = new GLuint[nbCams];

    glGenVertexArrays(nbCams, m_vaoID);

    glGenBuffers(nbCams, m_vertBufferID);
    glGenBuffers(nbCams, m_normalsBufferID);
    glGenBuffers(nbCams, m_indexBufferID);

    // to draw normals
    m_vaoNormalsID = new GLuint[nbCams];
    m_drawNormalsBufferID = new GLuint[nbCams];
    m_drawNormalsNormalsBufferID = new GLuint[nbCams];

    glGenVertexArrays(nbCams, m_vaoNormalsID);
    glGenBuffers(nbCams, m_drawNormalsBufferID);
    glGenBuffers(nbCams, m_drawNormalsNormalsBufferID);

    checkGLErrors();
}

void WarpGL::deleteVAOs( const unsigned int nbCams ) {

    glDeleteVertexArrays(nbCams, m_vaoID);

    glDeleteBuffers(nbCams, m_vertBufferID);
    glDeleteBuffers(nbCams, m_normalsBufferID);
    glDeleteBuffers(nbCams, m_indexBufferID);

    // delete variable
    delete[] m_vaoID;
    delete[] m_vertBufferID;
    delete[] m_normalsBufferID;
    delete[] m_indexBufferID;

    checkGLErrors();
}

void WarpGL::initVAOs( const std::vector<float> &vec_points,
                       const std::vector<float> &sigma_points,
                       const std::vector<unsigned int> &triangles,
                       const std::vector<float> &vertex_normals,
                       double scale ) {

    TRACE( "Initing VAOs: " << std::endl );
    checkGLErrors();

    int i = 0;

    glBindVertexArray( m_vaoID[i] );

    // vertices
    glBindBuffer(GL_ARRAY_BUFFER, m_vertBufferID[i]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * (vec_points.size() + sigma_points.size()), 0, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * vec_points.size(), vec_points.data());
    glBufferSubData(GL_ARRAY_BUFFER, sizeof(float) * vec_points.size(), sizeof(float) * sigma_points.size(), sigma_points.data());
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL); //3*sizeof(float)
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, (GLvoid*) (sizeof(float) * vec_points.size())); //sizeof(float)

    // normals
    glBindBuffer(GL_ARRAY_BUFFER, m_normalsBufferID[i]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertex_normals.size(), vertex_normals.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, NULL); // 3* sizeof(float)

    checkGLErrors();

    // triangles
    std::vector<unsigned int> indices_triangles(triangles.size());
    for ( unsigned int j = 0 ; j < triangles.size() ; ++j ) {

        int pointID = triangles[j];
        int trackID = pointID; // when using only one geometry, the point indexes are the trackID
        indices_triangles[j] = trackID;

    }
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBufferID[i]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * indices_triangles.size(), indices_triangles.data(), GL_STATIC_DRAW);
    checkGLErrors();

    { // create normals vao
        // the lines will go from p -> p+ scale* n
        // each point (begin, end) will have the same normal (duplicated)
        std::vector<float> normalVector(2*vec_points.size());
        std::vector<float> sigmaVector(2*vec_points.size()/3);
        std::vector<float> normalNormalVector(2*vertex_normals.size());

        for ( unsigned int iVert = 0 ; iVert < vec_points.size()/3 ; ++iVert ) {

            // start point is the vertex
            normalVector[6*iVert+0] = vec_points[3*iVert+0];
            normalVector[6*iVert+1] = vec_points[3*iVert+1];
            normalVector[6*iVert+2] = vec_points[3*iVert+2];

            // end point is the vertex + normal
            normalVector[6*iVert+3] = vec_points[3*iVert+0] + scale * vertex_normals[3*iVert+0];
            normalVector[6*iVert+4] = vec_points[3*iVert+1] + scale * vertex_normals[3*iVert+1];
            normalVector[6*iVert+5] = vec_points[3*iVert+2] + scale * vertex_normals[3*iVert+2];

            // same sigma for both points
            sigmaVector[2*iVert+0] = sigma_points[iVert];
            sigmaVector[2*iVert+1] = sigma_points[iVert];

            // same normal for both points
            normalNormalVector[6*iVert+0] = vertex_normals[3*iVert+0];
            normalNormalVector[6*iVert+1] = vertex_normals[3*iVert+1];
            normalNormalVector[6*iVert+2] = vertex_normals[3*iVert+2];

            normalNormalVector[6*iVert+3] = vertex_normals[3*iVert+0];
            normalNormalVector[6*iVert+4] = vertex_normals[3*iVert+1];
            normalNormalVector[6*iVert+5] = vertex_normals[3*iVert+2];
        }

        glBindVertexArray(m_vaoNormalsID[i]);

        glBindBuffer(GL_ARRAY_BUFFER, m_drawNormalsBufferID[i]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * (normalVector.size() + sigmaVector.size()), 0, GL_STATIC_DRAW);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * normalVector.size(), normalVector.data());
        glBufferSubData(GL_ARRAY_BUFFER, sizeof(float) * normalVector.size(), sizeof(float) * sigmaVector.size(), sigmaVector.data());
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL); //3*sizeof(float)
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, (GLvoid*) (sizeof(float) * normalVector.size()));

        glBindBuffer(GL_ARRAY_BUFFER, m_drawNormalsNormalsBufferID[i]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * normalNormalVector.size(), normalNormalVector.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, NULL); // 3* sizeof(float)
    }

    checkGLErrors();
    glBindVertexArray(0);
}

void WarpGL::createQuadVAO() {

    // Generate VAO
    glGenVertexArrays(1, &m_Quad_VAO);

    glGenBuffers(1, &m_Quad_VertexBO);
    glGenBuffers(1, &m_Quad_TextCoordBO);
    glGenBuffers(1, &m_Quad_indexBO);
}

int WarpGL::initRenderFBOs( unsigned int nbCams, int render_w, int render_h ) {

    checkGLErrors();

    // Generate Render Framebufer
    m_render_fbo = new GLuint[nbCams];
    glGenFramebuffers(nbCams, m_render_fbo);

    // alloc render textures
    m_buffer = new GLuint[nbCams];
    m_depth = new GLuint[nbCams];
    m_visibility = new GLuint[nbCams];
    m_backwards = new GLuint[nbCams];
    m_partial_tau_grad_v = new GLuint[nbCams];
    m_weights = new GLuint[nbCams];

    glGenTextures(nbCams, m_buffer);
    glGenRenderbuffers(nbCams, m_depth);
    glGenTextures(nbCams, m_visibility);

    glGenTextures(nbCams, m_backwards);
    glGenTextures(nbCams, m_partial_tau_grad_v);
    glGenTextures(nbCams, m_weights);

    checkGLErrors();
    glGenTextures(1, &m_final);
    glGenTextures(1, &m_colorStats);
    glGenTextures(1, &m_lumigraphTexture);
    glGenTextures(1, &m_u_depth);

    TRACE( "    Gen Textures OK" << std::endl );

    checkGLErrors();

    for( unsigned int i = 0 ; i < nbCams ; ++i ) {

        glBindFramebuffer(GL_FRAMEBUFFER, m_render_fbo[i]);

        /// Generate Texture for the rendered image
        glBindTexture(GL_TEXTURE_RECTANGLE, m_buffer[i]);
        glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA8, render_w, render_h, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);

        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        checkGLErrors();

        // Generate Depth buffer for FBO
        glBindRenderbuffer(GL_RENDERBUFFER, m_depth[i]);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, render_w, render_h);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_depth[i]);

        // Visibility textures
        glBindTexture(GL_TEXTURE_RECTANGLE, m_visibility[i]);
        glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_R16F, render_w, render_h, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);

        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        /// Generate Texture for the backward maps
        glBindTexture(GL_TEXTURE_RECTANGLE, m_backwards[i]);
        glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA32F, render_w, render_h, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);

        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        checkGLErrors();

        /// Generate Texture for partial tau and image gradient
        glBindTexture(GL_TEXTURE_RECTANGLE, m_partial_tau_grad_v[i]);
        glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA32F, render_w, render_h, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);

        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        checkGLErrors();

        /// Generate Texture for the weight image
        glBindTexture(GL_TEXTURE_RECTANGLE, m_weights[i]);
        glTexImage2D(GL_TEXTURE_RECTANGLE, 0, /*GL_R32F*/GL_RGBA32F, render_w, render_h, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);

        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        checkGLErrors();
    }
    TRACE( "    Allocate " << nbCams << " textures OK" << std::endl );

    glGenFramebuffers(1, &m_fbo_final);
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo_final);

    // allocate final image
    glBindTexture(GL_TEXTURE_RECTANGLE, m_final);
    glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA8, render_w, render_h, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);

    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    checkGLErrors();
    TRACE( "    Final image OK" << std::endl );

    // statistics over color distribution
    glBindTexture(GL_TEXTURE_RECTANGLE, m_colorStats);
    glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA32F, render_w, render_h, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);

    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    checkGLErrors();
    TRACE( "    Statics image OK" << std::endl );

    // indexes of the contribution views
    glBindTexture(GL_TEXTURE_RECTANGLE, m_lumigraphTexture);
    glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA8, render_w, render_h, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);

    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    checkGLErrors();
    TRACE( "    Lumigraph image OK" << std::endl );

    // depth as seen from the rendered image
    glBindTexture(GL_TEXTURE_RECTANGLE, m_u_depth);
    glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_R32F, render_w, render_h,
                 0, GL_RED, GL_UNSIGNED_BYTE, NULL);

    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    checkGLErrors();
    TRACE( "    Depth image OK" << std::endl );

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glBindTexture(GL_TEXTURE_RECTANGLE, 0);

    checkGLErrors();

    return 0;
}

int WarpGL::initWarpShader() {

    checkGLErrors();

    GLchar *warp_vert, *warp_frag;

    // load shader code
    unsigned long len;
    int res = loadshader("/scratch/gnieto/lfremotevision/cocolib_unstructured/unstructured_lf/warp_and_weights.vert", &warp_vert, len);
    if(res) {
        return res;
    }

    res = loadshader("/scratch/gnieto/lfremotevision/cocolib_unstructured/unstructured_lf/warp_and_weights.frag", &warp_frag, len);
    if(res) {
        return res;
    }

    // add shader code
    bool ok = warpShader.addShader(Vertex,warp_vert);
    if(!ok) {
        return -4;
    }

    ok = warpShader.addShader(Fragment,warp_frag);
    if(!ok) {
        return -5;
    }

    warpShader.bindAttribLocation(0, "in_position");
    warpShader.bindAttribLocation(1, "sigma_vertex");
    warpShader.bindAttribLocation(2, "in_texture_coord");

    ok = warpShader.link();
    if(!ok) {
        return -6;
    }

    checkGLErrors();

    return 0;
}

int WarpGL::initViewFBOs( ) {

    checkGLErrors();

    int nbCams = m_imageVector.size();

    // Generate views Framebuffer
    m_view_fbo = new GLuint[nbCams];
    glGenFramebuffers(nbCams, m_view_fbo);

    m_lookup_texture = new GLuint[nbCams];
    m_per_pixel_sigma = new GLuint[nbCams];
    m_depth_vi = new GLuint*[nbCams];

    glGenTextures(nbCams, m_lookup_texture);
    glGenTextures(nbCams, m_per_pixel_sigma);
    for(int iCam=0; iCam<nbCams; ++iCam) {
        m_depth_vi[iCam] = new GLuint[m_nb_displacement];
        glGenTextures(m_nb_displacement, m_depth_vi[iCam]);
    }

    // Auxiliary buffer for the opengl z-test
    m_view_depth = new GLuint[nbCams];
    glGenRenderbuffers(nbCams, m_view_depth);
    checkGLErrors();

    for( int iCam = 0 ; iCam < nbCams ; ++iCam ) {

        int i_cam_w = m_imageVector[iCam].width;
        int i_cam_h = m_imageVector[iCam].height;

        //printf("Framebuffer %d\n", iCam);
        glBindFramebuffer(GL_FRAMEBUFFER, m_view_fbo[iCam]);

        glBindTexture(GL_TEXTURE_RECTANGLE, m_lookup_texture[iCam]);
        glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGB32F, i_cam_w, i_cam_h,
                     0, GL_RED, GL_UNSIGNED_BYTE, NULL);

        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        // per pixel sigma
        glBindTexture(GL_TEXTURE_RECTANGLE, m_per_pixel_sigma[iCam]);
        glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RG32F, i_cam_w, i_cam_h,
                     0, GL_RED, GL_UNSIGNED_BYTE, NULL);

        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        // WARNING: Do not use the openGL depth buffer for depth computation.
        //          We will need to compute the depth "on our own"
        //          in some frag shaders. Depth should be computed everywhere consistently:
        //          Using  depth = (cam_R*(P - cam_C)).z;
        for( unsigned int iDisp = 0 ; iDisp < m_nb_displacement ; ++iDisp ) {

            glBindTexture(GL_TEXTURE_RECTANGLE, m_depth_vi[iCam][iDisp]);
            glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_R32F, i_cam_w, i_cam_h,
                         0, GL_RED, GL_UNSIGNED_BYTE, NULL);

            glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }
        // Generate Depth buffer for FBO
        glBindRenderbuffer(GL_RENDERBUFFER, m_view_depth[iCam]);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, i_cam_w, i_cam_h);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_view_depth[iCam]);

        checkGLErrors();
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glBindTexture(GL_TEXTURE_RECTANGLE, 0);

    checkGLErrors();

    return 0;
}

void WarpGL::resizeImageVector( const unsigned int nbCams ) {

    m_imageVector.resize( nbCams );
}

void WarpGL::loadTextures( Config_data *config_data, unsigned int s, unsigned int nview ) {

    std::string image_name = config_data->_mve_name + "/views" + "/view_%04i.mve" + "/undistorted.png";

    char str[500];
    sprintf( str, image_name.c_str(), s );
    coco::gsl_image *image = coco::gsl_image_load( str );

    // Image
    unsigned int N = config_data->_w * config_data->_h;
    std::vector<unsigned char> img;
    img.resize(N * config_data->_nchannels);

    for ( int n = 0 ; n < config_data->_nchannels ; ++n ) {
        // load view to device
        coco::gsl_matrix *channel = coco::gsl_image_get_channel( image, (coco::gsl_image_channel)n );

        for ( size_t i=0; i<N; i++ ) {
            img[N*n+i] = (unsigned char)channel->data[i];
        }
    }

    coco::gsl_image_free( image );

    int w = config_data->_w;
    int h = config_data->_h;

    glDeleteTextures(1, &m_imageVector[nview].texture);

    checkGLErrors();

    // Create texture
    glGenTextures( 1, &m_imageVector[nview].texture);

    // select our current texture
    glBindTexture(GL_TEXTURE_RECTANGLE, m_imageVector[nview].texture);
    checkGLErrors();

    glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    checkGLErrors();

    m_imageVector[nview].width = w;
    m_imageVector[nview].height = h;

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGB,  m_imageVector[nview].width,
                 m_imageVector[nview].height, 0, GL_RGB, GL_UNSIGNED_BYTE,
                 &img[0]);

    checkGLErrors();
    glBindTexture(GL_TEXTURE_RECTANGLE, 0);
}

void WarpGL::saveFloatTexture( Config_data *config_data, unsigned int s ) {

    char tauName[500];
    char vTestName[500];
    char partialName[500];

    const int w = config_data->_w;
    const int h = config_data->_h;
    assert( w * h > 0 );

    TRACE( "    Saving tau warps for camera " << s << std::endl );
    sprintf( tauName, config_data->_tau_name.c_str(), s );
    saveRGBAFloatTexture( m_backwards[0], w, h, 3, tauName );

    TRACE( "    Saving target image viewed from camera " << s << std::endl );
    sprintf( vTestName, ( config_data->_outdir + "/v_test_%02i.png").c_str(), s );
    saveRGBAFloatTexture( m_buffer[0], w, h, 3, vTestName );

    TRACE( "    Saving partial derivative of tau warp w.r.t. z for camera " << s << std::endl );
    sprintf( partialName, config_data->_dpart_name.c_str(), s );
    saveRGBAFloatTexture( m_partial_tau_grad_v[0], w, h, 3, partialName );
}

void WarpGL::removeRenderFBOs( unsigned int nbCams ) {

    // RENDER textures
    glDeleteFramebuffers(nbCams, m_render_fbo);
    glDeleteTextures(nbCams, m_buffer);
    glDeleteRenderbuffers(nbCams, m_depth);
    glDeleteTextures(nbCams, m_visibility);
    glDeleteTextures(nbCams, m_weights);
    glDeleteTextures(nbCams, m_backwards);
    glDeleteTextures(nbCams, m_partial_tau_grad_v);

    // Buffers for final rendering
    glDeleteFramebuffers(1, &m_fbo_final);
    glDeleteTextures(1, &m_final);
    glDeleteTextures(1, &m_colorStats);
    glDeleteTextures(1, &m_lumigraphTexture);
    glDeleteTextures(1, &m_u_depth);

    TRACE( "Remove RenderFBOs Ok" << std::endl );

    delete[] m_render_fbo;
    delete[] m_buffer;
    delete[] m_depth;
    delete[] m_visibility;
    delete[] m_backwards;
    delete[] m_partial_tau_grad_v;
    delete[] m_weights;
}

void WarpGL::removeViewFBOs( unsigned int nbCams ) {

    checkGLErrors();

    glDeleteFramebuffers(nbCams, m_view_fbo);

    glDeleteTextures(nbCams, m_lookup_texture);
    glDeleteTextures(nbCams, m_per_pixel_sigma);

    for( unsigned int iCam = 0 ; iCam < nbCams ; ++iCam ) {

        glDeleteTextures(m_nb_displacement, m_depth_vi[iCam]);
        delete[] m_depth_vi[iCam];
    }
    glDeleteRenderbuffers(nbCams, m_view_depth);

    delete[] m_view_fbo;
    delete[] m_lookup_texture;
    delete[] m_per_pixel_sigma;
    delete[] m_depth_vi;
    delete[] m_view_depth;

    TRACE( "Remove ViewFBOs Ok" << std::endl );

    checkGLErrors();
}

void WarpGL::createGLcontext() {

    //-- Create the GL window context

    if( !glfwInit() ) {

        fprintf( stderr, "Failed to initialize GLFW\n" );
        exit( EXIT_FAILURE );
    }

    //glfwWindowHint(GLFW_DEPTH_BITS, 16);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);

    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow( window_width, window_height, "Compute warps and weights from mesh", NULL, NULL );
    if (!window) {

        fprintf( stderr, "Failed to open GLFW window\n" );
        glfwTerminate();
        exit( EXIT_FAILURE );
    }
    checkGLErrors();

    // Set callback functions
//    glfwSetWindowCloseCallback(window, window_close_callback);
//    glfwSetWindowSizeCallback(window, reshape);
//    glfwSetKeyCallback(window, key);
//    glfwSetErrorCallback(error_callback);

    glfwMakeContextCurrent(window);
    glfwSwapInterval( 1 );

    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LEQUAL);
    glDepthRange(0.0f, 1.0f);

    // We modified the Triangles library to give triangles indexes in counter-clock-wise order
    glFrontFace(GL_CCW); // this is the default, make it explicit

//    if (reverse_ply_triangle_orientation) {
//        glFrontFace(GL_CW);
//    }

    checkGLErrors();
}

void WarpGL::GLterminate() {

    // Terminate GLFW
    glfwTerminate();

    TRACE( "GLFW Terminate OK" << std::endl );
}

void checkError() {

    checkGLErrors();
}

//// This functions computes the depth map in the vi frame
//void render_vi_depth(GLuint fbo,
//                     GLuint vaoID,
//                     int nbTriangles,
//                     float displacement,
//                     GLuint depthText,
//                     ShaderHandler &depthShader,
//                     const openMVG::PinholeCamera & camera,
//                     int w, int h) {
//    checkGLErrors();

//    glViewport(0,0,w,h);
//    openMVG::Mat4 projectionMat, modelViewMat;
//    setCurrentCamFrustum(camera, w,h, projectionMat, modelViewMat);

//    // Bind rendering FBO
//    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

//    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, depthText, 0);
//    glDrawBuffer(GL_COLOR_ATTACHMENT0);

//    glClear(GL_DEPTH_BUFFER_BIT);

//    GLfloat color[4]={-1.,-1.,0.,0};
//    glClearBufferfv(GL_COLOR, 0, color);

//    int status = glCheckFramebufferStatus(GL_FRAMEBUFFER);

//    if (status != GL_FRAMEBUFFER_COMPLETE ) {
//        printf("\tFramebuffer KO: PROBLEM\n");
//        return;
//    }

//    glEnable(GL_DEPTH_TEST);

//    glCullFace(GL_BACK);
//    glEnable(GL_CULL_FACE);

//    depthShader.useProgram();

//    checkGLErrors();

//    float ptr[16];
//    //glGetFloatv(GL_MODELVIEW_MATRIX, ptr);
//    setMat4_to_Ptr(modelViewMat, ptr);
//    depthShader.setUniformMat4f("modelViewMat", ptr);

//    //glGetFloatv(GL_PROJECTION_MATRIX, ptr);
//    setMat4_to_Ptr(projectionMat, ptr);
//    depthShader.setUniformMat4f("projectionMat", ptr);

//    depthShader.setUniformf("displacement", displacement);

//    float vi_C[3] = {(float)camera._C(0), (float)camera._C(1), (float)camera._C(2)};
//    depthShader.setUniform3fv("vi_C", vi_C);

//    float vi_R[9] /*= {camera._R(0,0), camera._R(1,0), camera._R(2,0),
//                                   camera._R(0,1), camera._R(1,1), camera._R(2,1),
//                                   camera._R(0,2), camera._R(1,2), camera._R(2,2)  }*/;
//    setMat3_to_Ptr(camera._R, vi_R);
//    depthShader.setUniformMat3f("vi_R", vi_R);

//    checkGLErrors();

//    glBindVertexArray(vaoID);
//    glEnableVertexAttribArray(0);
//    glEnableVertexAttribArray(1);
//    glEnableVertexAttribArray(3);

//    glDrawElements(GL_TRIANGLES, nbTriangles, GL_UNSIGNED_INT, 0);
//    checkGLErrors();

//    depthShader.stopUsingProgram();

//    glDisableVertexAttribArray(0);
//    glDisableVertexAttribArray(1);
//    glDisableVertexAttribArray(3);

//    glBindVertexArray(0);

//    checkGLErrors();

//    glDisable(GL_CULL_FACE);
//    glDisable(GL_DEPTH_TEST);

//    glBindTexture(GL_TEXTURE_RECTANGLE, 0);
//    glBindFramebuffer(GL_FRAMEBUFFER, 0);

//    checkGLErrors();
//}


