#ifndef FRAME_BUFFER_H
#define FRAME_BUFFER_H

//#include <string>
#include <GL/glew.h>
#include <iostream>
#include <glm/glm.hpp>
//#include <vector>

#include "texture.h"

#define NB_COLOR_BUFFER_MAX 16
#define INVALID_DEPTH 1000.0f

#define checkGLErrors() {\
    GLenum error = glGetError(); \
    if(error != GL_NO_ERROR) { \
    std::cout << "GL_ERROR :" << __FILE__ << " "<< __LINE__ << " " << error << std::endl; \
    } \
    }

class Shader;
class ShaderGeometry;

class FrameBuffer {

public:

    FrameBuffer( int W, int H, float depthFocal = 0 );
    ~FrameBuffer();

    // add inDepthMap to outDepthMap where outDepthMap is 0
    void addDepthScale( const Texture* inDepthMap, Texture* outDepthMap, uint scale, Shader* addDepthScaleShader );

    // convert depth from radial (distance to optical center) to orthogonal (distance along the principal axis)
    void fromRadial2Ortho( const Texture* vkDepthMap, const glm::mat3 &vk_K, Shader* fromRadial2OrthoShader );

    // attach input tex and temp tex to frame buffer object for splatting
    void attachSplattingBuffers( const Texture* inputTex, const Texture* tempTex, int W, int H );

    // attach input tex and temp tex to frame buffer object for forward warping
    void attachWarpingBuffers( int W, int H, const Texture* tex0, const Texture* tex1 = 0, const Texture* tex2 = 0 );

    // perform forward warping aka splatting pass (either for visibility computation or not)
    void forwardWarping( const bool visibilityPass,
                         const uint ratioDepth,
                         const uint ratioImage,
                         const Texture* depthMap, const Texture* inputTex, const Texture* inputMask,
                         const glm::mat4 &renderMatrix,
                         const glm::mat3 &vk_K, const glm::mat3 &vk_R, const glm::vec3 &vk_t,
                         const glm::mat3 &u_R, const glm::vec3 &u_t,
                         ShaderGeometry* imageSplattingShader,
                         uint texIndex = GL_COLOR_ATTACHMENT1 );

    // Normalised the splatted Laplacian
    void splatNormalisation( const Texture* tempTex, Shader* normalizationShader, uint texIndex = GL_COLOR_ATTACHMENT0 );

    // Add up two textures
    void addTextures( const Texture* tex1, const Texture* tex2, uint outIndex, Shader* addTexturesShader );

    // Compute the tau warps and deformation weights
    void computeTauWarps( const Texture* uDepthMap, const Texture* vkDepthMap, const Texture* vkTauWarp,
                          const glm::mat3 &u_K, const glm::mat3 &u_R, const glm::vec3 &u_t,
                          const glm::mat3 &vk_K, const glm::mat3 &vk_R, const glm::vec3 &vk_t,
                          Shader* tauWarpShader );

    // Compute tau partial for geometry weights
    void computeTauPartial( const Texture* vkDepthMap, const Texture* vkTauPartial,
                            const glm::mat3 &u_K, const glm::mat3 &u_R, const glm::vec3 &u_t,
                            const glm::mat3 &vk_K, const glm::mat3 &vk_R, const glm::vec3 &vk_t,
                            Shader* tauPartialShader );

    // warped vk view to test tau warp
    void warpVk( const Texture* vkTauWarp, const Texture* sourceImage, const Texture* warpedVk,
                 Shader* warpVkShader );

    // target depth map splatting
    void splatDepth( const Texture* vkDepthMap,
                     const glm::mat3 &u_K, const glm::mat3 &u_R, const glm::vec3 &u_t,
                     const glm::mat4 &renderMatrix, const glm::mat3 &vk_K, const glm::mat3 &vk_R, const glm::vec3 &vk_t,
                     Shader* depthSplattingShader );

    // draw texture on screen (don't really need the framebuffer actually)
    void drawTexture( Texture* texture, Shader* textureShader );

    // clear color buffer
    void clearAttachment( uint n, int W, int H );

    // set texture's first channel to value
    void clearTexture( Texture* texture, float value );

    // clear color and depth buffer
    void clearBuffers();

    // getters

    GLuint getID();

private:

    void createRenderBuffer( GLuint &id, GLenum internalFormat );
    void loadSimpleVAVBEBO();
    void loadSensorPlaneVAVBO();
    bool load();

    GLuint _id;
    GLuint _verticesVBO, _texCoordVBO, _VAO, _EBO;
    GLuint _sensorPlaneVAO, _sensorPlaneVBO;
    GLuint _meshVAO;
    int _W, _H;
    float _depthFocal;
    bool _refocus;

    GLuint _depthBufferID;
};

#endif /* #ifndef FRAME_BUFFER_H */
