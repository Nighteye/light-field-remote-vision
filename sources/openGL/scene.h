#ifndef SCENE_OPENGL_H
#define SCENE_OPENGL_H

#include <SDL2/SDL.h>
#include <string>
#include <vector>

#include "input.h"

class Shader;
class ShaderGeometry;
class Mesh;
class InputView;
class TargetView;
class PinholeCamera;
class FrameBuffer;
class Texture;
class Pyramid;

class Scene {

public:

    Scene( std::string outdir,
           std::string winTitle,
           int winWidth, int winHeight,
           uint camWidth, uint camHeight,
           int sMin, int sMax, int sRmv,
           int tMin, int tMax, int tRmv,
           int pyramidHeight = 0, float depthFocal = 0 );
    ~Scene();

    bool initWindow();
    bool initGL();
    void mainLoop();

    // render the target view u with the selected IBR technique
    void renderingLoop();

    // render a test image with laplacian IBR technique
    void renderingTest();

    // render a set of frames via Laplacian refocussing and moving the focal plane iteratively
    void refocussingLoop();

    // IMPORT_SCENE
    // ---------------------------------------------------------------------------------------------------------- //

    void importMesh( std::string mveName );
    void importPointCloud( std::string mveName );
    void importViews( std::string mveName, uint scale_min, uint scale_max = 0 );

    // load standford lightfield dataset (images only)
    void importStanfordViews( std::string imageName = 0);

    // load standford lightfield dataset (image and camera matrices)
    // camera matrices are obainted thank to openMVG calibration
    // STANFORD FORMAT (rows and columns)
    void importStanfordOriginalViews( std::string cameraName, std::string imageName = 0 );

    // IBR_laplacian
    // ---------------------------------------------------------------------------------------------------------- //

    void renderSplatting();

    // forward warp an input image at a specific scale
    void singleViewForwardWarping( uint W, uint H,
                                   uint scale,
                                   uint viewIndex,
                                   std::vector< Texture* > Pyramid::*texVec,
                                   std::vector< float* > Pyramid::*arrayVec,
                                   bool verbose, std::string outfile );

    // forward warp all view of resolutino (inW, inH) to target view of resolution (outW, outH)
    void forwardWarping( uint originalW, uint originalH,
                         uint scale,
                         std::vector< Texture* > InputView::*inTexVec,
                         std::vector< Texture* > TargetView::*outTexVec,
                         std::vector< float* > TargetView::*outArrayVec,
                         bool originalDepth,
                         bool verbose = false, std::string outfile = "" );

    // forward warp all views at the same resolution
    void unscaledForwardWarping( uint W, uint H,
                                 uint scale,
                                 std::vector< Texture* > Pyramid::*texVec,
                                 std::vector< float* > Pyramid::*arrayVec,
                                 bool verbose, std::string outfile = "" );

    // compute fondamental and collapse scale space by adding up Laplacians
    void collapseScaleSpace(uint W, uint H, bool verbose, uint nbChannels);

    // perform forward warping of all pyramid levels but at the same resolution
    void scaleSpaceSplatting(uint W, uint H);

    // perform forward warping of all pyramid levels but at the same resolution (new version)
    void scaleSpaceWarping(uint W, uint H);

    void pyramidSplatting(uint originalW, uint originalH);
    void computeTargetDepthMap( std::string uDepthName = "" );
    void exportWarps( std::string lfName = "", std::string tauName = "", std::string dpartName = "", std::string uWarpedName = "" );
    void computeDepthMap( );
    void filterDepthMaps();
    void createGaussianPyramid();
    void createGaussianPyramidCPU();
    void createGaussianScaleSpace();
    void createDepthPyramidCPU(int depthPyramidHeight);
    // create gaussian scale space of depth maps, and replace the current dm by the lowest scale. delete the scale space
    void createDepthScaleSpaceCPU(int depthScale);

    // update target camera position and orientation
    void moveTargetCam(int frame);

    // test laplacian warping on a simple cube
    void createTestViews();

private:

    // Shader to process a 2D texture
    void init2DTextureShader( Shader *shader );
    // Shader to splat quads
    void initQuadGeomShader( ShaderGeometry *shader );

    std::string _outdir;
    std::string _windowTitle;
    int _windowWidth, _windowHeight;
    uint _camWidth, _camHeight;
    int _sMin, _sMax, _sRmv;
    int _tMin, _tMax, _tRmv;
    int _pyramidHeight;
    int _S, _T; // LF angular range
    int _centralS, _centralT; // central view angular coordinates
    uint _nbCameras;
    uint _renderIndex;
    uint _centralIndex; // index of the central view (for optical flow)
    float _depthFocal; // depth of the focal plane when refocussing
    SDL_Window* _window;
    SDL_GLContext _OpenGLContext;
    Input _input;
    std::vector<Mesh*> _meshes;
    std::vector<InputView*> _vCam;
    TargetView* _uCam;
    FrameBuffer* _FBO;

    // shaders

    Shader* _fromRadial2OrthoShader;
    Shader* _addDepthScaleShader;
    Shader* _tauWarpShader;
    Shader* _tauPartialShader;
    Shader* _warpVkShader;
    Shader* _normalizationShader;
    Shader* _addTexturesShader;
    Shader* _textureShader;
    Shader* _depthFromMeshShader;

    ShaderGeometry* _depthSplattingShader;
    ShaderGeometry* _forwardWarpingShader;
};

#endif /* #ifndef SCENE_OPENGL_H */

