#ifndef OPENGL_FUNCTIONS_H
#define OPENGL_FUNCTIONS_H

#include <glm/glm.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <vector>

#include "shaderHandler.h"

struct PinholeCamera;
struct Config_data;

struct GLWImage {
    int width, height;
    GLuint texture;
};

class WarpGL {

public:

    WarpGL();
    ~WarpGL();

    void createVAOs( const unsigned int nb_cams );
    void deleteVAOs( const unsigned int nb_cams );
    void initVAOs( const std::vector<float> &vec_points,
                   const std::vector<float> &sigma_points,
                   const std::vector<unsigned int> &triangles,
                   const std::vector<float> &vertex_normals,
                   double scale );
    void createQuadVAO();
    int initRenderFBOs( unsigned int nb_cams, int render_w, int render_h );
    int initWarpShader( );
    int initViewFBOs( );
    void resizeImageVector( const unsigned int nbCams );
    void loadTextures( Config_data *config_data, unsigned int s, unsigned int nview );

    // Compute the warp tau from the render camera to the current camera
    // x = tau(y), where x is in target camera units, and y in current camera units
    void computeWarps( Config_data *config_data,
                       unsigned int nview,
                       int nbTriangles,
                       const PinholeCamera &targetCam, // - has the output resolution
                       const PinholeCamera &currentCam ); // - has the input resolution

    void saveFloatTexture( Config_data *config_data, unsigned int s );
    void removeRenderFBOs( unsigned int nbCams );
    void removeViewFBOs( unsigned int nbCams );
    void createGLcontext();
    void GLterminate();

private:

    GLFWwindow* window;
    int window_width;
    int window_height;

    std::vector< GLWImage > m_imageVector;

    ShaderHandler warpShader;

    float m_depth_epsilon; // How to estimate this parameter? Depends on the scene?
    unsigned int m_nb_displacement;
    bool m_depth_mapping;
    float m_mapping_cut_depth;
    float m_mapping_factor;

    // Vertex array object with the 3D Vertices, normals and triangles
    GLuint* m_vaoID;
    GLuint* m_vertBufferID;
    GLuint* m_normalsBufferID;
    GLuint* m_indexBufferID;

    // Vertex array object to DRAW the normals of each vertex
    GLuint *m_vaoNormalsID;
    GLuint *m_drawNormalsBufferID;
    GLuint *m_drawNormalsNormalsBufferID;

    // Quad VAO for pixel operations
    GLuint m_Quad_VAO;
    GLuint m_Quad_VertexBO, m_Quad_TextCoordBO, m_Quad_indexBO;

    // INPUT VIEW textures: input images size (each input may be different)
    GLuint *m_view_fbo;
    GLuint *m_lookup_texture; // a lookup table to go from undistorded coordinates to distorded images
    GLuint *m_per_pixel_sigma; // the geometric uncertainty as seen by cam i
    GLuint **m_depth_vi; // the depth map corresponding to the view i
    GLuint *m_view_depth; // auxiliary buffer for opengl z-buffering

    // RENDER textures: render image size (render size is the same)
    GLuint *m_render_fbo;
    GLuint *m_buffer; // the warped image
    GLuint *m_depth; // a buffer for z-buffering in the fbo
    GLuint *m_visibility; // the visibility of the i-th camera in the render frame (not binary)
    GLuint *m_weights;  // the associated
    GLuint *m_backwards; // the backwards map to go from the warped image to the original (distorded image)
    GLuint *m_partial_tau_grad_v; // the derivative of the warp with respecto to the geometry

    // Buffers for final rendering (render size buffers)
    GLuint m_fbo_final;
    GLuint m_final;
    GLuint m_colorStats;
    GLuint m_lumigraphTexture;
    GLuint m_u_depth;
};

void saveRGBAFloatTexture(GLuint textureId, int render_w, int render_h, int depth, const std::string &name, bool flip = false);

// Convert Mat4 to col-major array
void setMat4_to_Ptr(const glm::mat4 &Mat, float ptr[16]);

// Convert Mat3 to col-major array
void setMat3_to_Ptr(const glm::mat3 &Mat, float ptr[9]);

// Convert Mat34 to 4x4 col-major array
void setMat34_to_Ptr(const glm::mat4x3 &Mat, float ptr[16]);

void setCurrentCamFrustum(const PinholeCamera &camera, int w, int h,
                          glm::mat4 &projMatrix, glm::mat4 &modelViewMat);

void checkError();

#endif /* #ifndef OPENGL_FUNCTIONS_H */
