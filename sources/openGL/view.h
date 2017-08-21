#ifndef INPUT_VIEW_H
#define INPUT_VIEW_H

#include <vector>
#include <string>
#include <GL/glew.h>
#include <iostream>

// GLM includes
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "shader.h"
#include "../pinholeCamera.h"
#include "texture.h"

// Macro for VBO
#ifndef BUFFER_OFFSET
#define BUFFER_OFFSET(offset) ((char*)NULL + (offset))
#endif

#define checkGLErrors() {\
    GLenum error = glGetError(); \
    if(error != GL_NO_ERROR) { \
    std::cout << "GL_ERROR :" << __FILE__ << " "<< __LINE__ << " " << error << std::endl; \
    } \
    }

class TargetView;
class FrameBuffer;
class Pyramid;

class InputView {

public:

    InputView( uint W, uint H, std::string outdir, int pyramidHeight = 0 );
    ~InputView();

    bool importTexture( const char* imageName );
    bool importMask( const char* maskName, float &count );
    bool importCamParametersStanford( double centerX, double centerY );
    bool importCamParameters( char *cameraName );
    bool importCamParametersStanford(char *cameraName);
    void display( const glm::mat4 &renderMatrix );
    void setTexMappingUniform( Shader *triangleMeshShader );
    void renderDepth( const GLuint meshVAO, const GLuint pointCloudVAO, const uint nbTriangles );
    Texture* setDepthAndNormals(char *depthName, char *normalName);
    // when no depth data is available (refocussing, no preliminary reconstruction), standard initialization
    Texture* initDepthAndNormals();
    void exportWarp(char* lfName, char* tauName, char* dpartName,
                    TargetView *uCam, char* uWarpedName, Texture* textureToWarp, FrameBuffer* FBO,
                    Shader* tauWarpShader, Shader* tauPartialShader, Shader* warpVkShader);
    void filterDepthmap(FrameBuffer* FBO, Shader* fromRadial2OrthoShader, char *filteredDepthNameChar = 0);

    // Multi-scale depth map accumulation
    void addDepthScale(Texture* depthMapToAdd, FrameBuffer* FBO, uint scale, Shader* addDepthScaleShader);

    // Create Gaussian Pyramid from input original image
    void createGaussianPyramid(uint k);
    // Create Gaussian Pyramid from input original image, CPU version
    void createGaussianPyramidCPU(uint k);
    // create Gaussian Pyramid wihout downsampling
    void createGaussianScaleSpace(uint k);
    // create Gaussian Pyramid from input depth maps, CPU version
    void createDepthPyramidCPU(uint k, int depthPyramidHeight);
    // reduce depthmap according to the scale parameter
    void createDepthScaleSpaceCPU(uint k, int depthScale);

    // test input view for laplacian warping
    void createTestInputView( float fov );

    // draw view on screen
    void drawTexture( FrameBuffer* FBO, Shader* textureShader );

    void saveDepthMap( const std::string &texName );

    // setters
    void initDepthMap();
    void initViewTexture();
    void setPinholeCamera(const PinholeCamera &pinholeCamera);

    // getters
    GLuint getTextureID() const;
    PinholeCamera getPinholeCamera() const;
    Texture* getTexture() const;
    std::vector<float>* getTextureVec() const;
    void getTextureRGB(std::vector<cv::Point3f>& view) const;
    void getMask(std::vector< std::vector<float> >& view, uint nbChannels) const;
    Texture* getDepthMap() const;
    Texture* getMask() const;

    // Laplacian Pyramid
    std::vector< Texture* > _gaussianPyramidTex;
    std::vector< float* > _gaussianPyramidArray;
    std::vector< Texture* > _laplacianPyramidTex;
    std::vector< float* > _laplacianPyramidArray;
    std::vector< Texture* > _depthPyramidTex;
    std::vector< float* > _depthPyramidArray;

    Pyramid* _pyramid;

private:

    // Methods
    bool importMVEIFile( char *fileName, std::vector< std::vector<float> > &data, uint &W, uint &H );
    void load();
    void initCameraWireShader( Shader *shader );
    void initCameraTextureShader( Shader *shader );

    uint _W; uint _H;
    std::string _outdir;
    int _pyramidHeight;
    Shader _cameraWireShader;
    Shader _cameraTextureShader;
    PinholeCamera _pinholeCamera;
    GLuint _cameraFrustumVAO, _cameraFrustumVBO, _cameraFrustumEBO;
    GLuint _cameraCentersVAO, _cameraCentersVBO;
    GLuint _cameraImagetteVAO, _imagetteVertVBO, _imagetteTexCoordVBO, _imagetteEBO;

    // Textures
    Texture *_depthMap;
    Texture *_texture;
    Texture *_mask;
};

class TargetView {

public:

    TargetView( uint W, uint H, const PinholeCamera &pinholeCamera, std::string outdir, int pyramidHeight = 0 );
    ~TargetView();

    // init pyramid buffers
    void initPyramid();

    // pyramid decomposition of a warped Laplacian at a specific scale
    void createDecomposeWarpedImage(uint inputScale, Texture* inputTex);

    // Project vk depth map onto the target view, writing in the z-buffer
    void depthMapSplatting(InputView *vCam, FrameBuffer* FBO, Shader* depthSplattingShader);

    // warp view vk with depth map, performing a visibility pass or not, no resolution change
    void unscaledForwardWarping( InputView *vCam,
                                 FrameBuffer* splattingBuffer,
                                 bool visibilityPass,
                                 Texture* vkDepthMap,
                                 Texture* vkImage,
                                 Texture* vkMask,
                                 ShaderGeometry* imageSplattingShader,
                                 uint texIndex = GL_COLOR_ATTACHMENT1 );

    // warp view vk with depth map, performing a visibility pass or not, with optional change of scale
    void forwardWarping( InputView *vCam,
                         FrameBuffer* splattingBuffer,
                         bool visibilityPass,
                         Texture* vkDepthMap,
                         Texture* vkImage,
                         Texture* vkMask,
                         ShaderGeometry* imageSplattingShader,
                         uint scale = 0,
                         bool originalDepth = false );

    // display on screen the Laplacian blended view
    void display();

    // move the target view along a predefined path for video
    void move( const PinholeCamera &pinholeCamera1, const PinholeCamera &pinholeCamera2, uint count);

    // test target view for laplacian warping
    void createTestTargetView();

    void saveDepthMap( const std::string &texName );

    // setters
    void initDepthMap();

    // getters
    PinholeCamera getPinholeCamera() const;
    Texture* getDepthMap() const;

    // Laplacian Pyramid
    std::vector< Texture* > _gaussianPyramidTex;
    std::vector< float* > _gaussianPyramidArray;
    std::vector< Texture* > _laplacianPyramidTex;
    std::vector< float* > _laplacianPyramidArray;

    Pyramid* _pyramid;

private:

    // Textures
    Texture *_depthMap;

    uint _W; uint _H;
    std::string _outdir;
    int _pyramidHeight;
    PinholeCamera _pinholeCamera;
};

#endif /* #ifndef INPUT_VIEW_H */
