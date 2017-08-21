#ifndef MAP_H
#define MAP_H

#include <string>
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <vector>

#include "texture.h"
#include "shader.h"

#define NB_COLOR_BUFFER_MAX 16
#define INVALID_DEPTH 1000.0f

#define checkGLErrors() {\
    GLenum error = glGetError(); \
    if(error != GL_NO_ERROR) { \
    std::cout << "GL_ERROR :" << __FILE__ << " "<< __LINE__ << " " << error << std::endl; \
    } \
    }

class Map {

public:

    Map( int W, int H, bool useStencilBuffer = false );
    ~Map();

    void createRenderBuffer( GLuint &id, GLenum internalFormat );

    // target depth map splatting
    void splatDepth( GLuint vkDepthMapID, uint uDepthMapIndex,
                     const glm::mat3 &u_K, const glm::mat3 &u_R, const glm::vec3 &u_t,
                     const glm::mat4 &renderMatrix, const glm::mat3 &vk_K, const glm::mat3 &vk_R, const glm::vec3 &vk_t);

    void clearDepthBuffer(uint bufferIndex);
    void initBuffer(uint bufferIndex);
    void clearColorBuffer(uint bufferIndex);

    // Compute the tau warps and deformation weights
    void computeTauWarps( GLuint uDepthMapID, GLuint vkDepthMapID, uint vkTauWarpIndex,
                          const glm::mat3 &u_K, const glm::mat3 &u_R, const glm::vec3 &u_t,
                          const glm::mat3 &vk_K, const glm::mat3 &vk_R, const glm::vec3 &vk_t);

    // Compute tau partial for geometry weights
    void computeTauPartial( GLuint vkDepthMapID, uint vkTauPartialIndex,
                            const glm::mat3 &u_K, const glm::mat3 &u_R, const glm::vec3 &u_t,
                            const glm::mat3 &vk_K, const glm::mat3 &vk_R, const glm::vec3 &vk_t);

    // warped vk view to test tau warp
    void warpVk( GLuint vkDepthMapID, GLuint sourceImageID, uint warpedVkIndex );

    // convert depth from radial (distance to optical center) to orthogonal (distance along the principal axis)
    void fromRadial2Ortho( uint vkDepthMapIndex, uint tempIndex, const glm::mat3 &vk_K );

    // splat vk view to test tau warp
    void splatVk( GLuint vkDepthMapID, GLuint vkImageID, uint tempIndex,
                  const glm::mat4 &renderMatrix, const glm::mat3 &vk_K, const glm::mat3 &vk_R, const glm::vec3 &vk_t, bool visibilityPass);

    // normalization after all view splatting
    void splatNormalisation( uint splattedVkIndex, uint tempIndex );

    // render depth map from point cloud using trilateral filtering (space, intensity and depth)
    void trilateralFiltering( GLuint imageTexID, // source image
                              const glm::mat4 &renderMatrix,
                              const glm::mat3 &vi_Kinv,
                              const glm::mat3 &vi_R,
                              const glm::vec3 &vi_C,
                              const GLuint pointCloudVAO,
                              const uint nbPoints );

    void projectPointCloud( const glm::mat4 &renderMatrix, const glm::mat3 &vi_R, const glm::vec3 &vi_C, const GLuint pointCloudVAO, const uint nbPoints );

    void bilateralFiltering( GLuint imageTexID, uint tempIndex, uint depthMapIndex );

    // add inDepthMapIndex to outDepthMapIndex where outDepthMapIndex is 0
    void addDepthScale( uint inDepthMapIndex, uint outDepthMapIndex, uint tempIndex, uint scale = 0 );

    void renderDepthFromMesh( const glm::mat4 &renderMatrix, const glm::mat3 &vi_R, const glm::vec3 &vi_C, const GLuint meshVAO, const uint nbTriangles );
    uint *addBufferFromData(const std::vector< std::vector< float > > &data, uint w, uint h, uint channels);
    uint *addEmptyBuffer(uint channels);
    void deleteBuffer(uint index);

    // savers
    //    void saveDepthFromPointCloud( const std::string &depthMapName );
    //    void saveDepthFromMesh( const std::string &depthMapName );
    //    void saveLowResDepth( const std::string &depthMapName );
    //    void saveHighResDepth( const std::string &depthMapName );
    void saveMap( const std::string &mapName, uint channels, uint id );

    // getters
    Texture getDepthTex() const;
    Texture getNormalTex() const;
    Texture* getMap(uint index) const;
    GLuint getTexID(uint index) const;
    GLuint getID() const;
    GLuint getDepthFromMeshID() const;
    int getWidth() const;
    int getHeight() const;

private:

    bool load();

    // init shaders
//    void initTrilateralFilterShader( ShaderGeometry *shader);
//    void initDepthFromMeshShader( Shader *shader );

    // Shader to process a 2D texture
    void init2DTextureShader( Shader *shader );
    // Shader to splat quads
    void initQuadGeomShader( ShaderGeometry *shader);

    GLuint _id;
    GLuint _verticesVBO, _texCoordVBO, _VAO, _EBO;
    GLuint _sensorPlaneVAO, _sensorPlaneVBO;
    GLuint _meshVAO;
    int _W, _H;

    //    Texture _sparseDepth;
    //    Texture _depthFromPointCloud;
    //    Texture _depthFromMesh;
    //    Texture _temp;
    //    Texture _lowResDepth;
    //    Texture _highResDepth;
    //    Texture _normalMapTex;

    std::vector< Texture* > _mapVector;

    GLuint _depthBufferID;
    bool _useStencilBuffer;
//    ShaderGeometry _trilateralFilterShader;
//    Shader _depthFromMeshShader;
    Shader _bilateralShader;
    Shader _normalizationShader;
    Shader _tauWarpShader;
    Shader _tauPartialShader;
    Shader _warpVkShader;
    Shader _fromRadial2OrthoShader;
    Shader _addDepthScaleShader;
    ShaderGeometry _depthSplattingShader;
    ShaderGeometry _imageSplattingShader;
};

#endif /* #ifndef MAP_H */
