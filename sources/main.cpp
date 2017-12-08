#include "config.h"
#include "import.h"
#include "fromMVE2coco.h" // fromMVE2coco
#include "super-resolution.h" // IBR_color
#include "gradientIBR.h" // IBR_gradient
#include "openGL/scene.h" // IBR_direct
#include "lfScene.h" // IBR_optical

#include <cocolib/cocolib/common/debug.h>
#include <cocolib/cocolib/common/parse_config.h>

using namespace std;
using namespace coco;

void checkRequirement(Config_data *config_data)
{
    // test required parameters
    assert(config_data->_w != 0);
    assert(config_data->_h != 0);
    assert(config_data->_s_min >= 0);
    assert(config_data->_s_max >= 0);
    assert(config_data->_s_rmv >= 0);
    assert(config_data->_pyramidHeight >= 0);

    assert(!config_data->_imageName.empty());
    assert(!config_data->_lf_name.empty());
    assert(!config_data->_tau_name.empty());
    assert(!config_data->_dpart_name.empty());

    // _uWarped_name is not necessarily required

    // print parameters
    std::cout << "Parameters: " << std::endl;

    std::cout << "- image width: " << config_data->_w << std::endl;
    std::cout << "- image height: " << config_data->_h << std::endl;
    std::cout << "- s min: " << config_data->_s_min << std::endl;
    std::cout << "- s max: " << config_data->_s_max << std::endl;
    std::cout << "- s to remove: " << config_data->_s_rmv << std::endl;
    std::cout << "- scale min: " << config_data->_scale_min << std::endl;
    std::cout << "- scale max: " << config_data->_scale_max << std::endl;
    std::cout << "- pyramid height: " << config_data->_pyramidHeight << std::endl;
    std::cout << "- optical flow algorithm: " << config_data->_flowAlg << std::endl;

    if(!config_data->_cameraName.empty()) {
        std::cout << "- camera name: " << config_data->_cameraName << std::endl;
    } else {
        std::cout << "no camera parameter files, loading stanford-like dataset" << std::endl;
    }
    std::cout << "- image name: " << config_data->_imageName << std::endl;
    std::cout << "- lf name: " << config_data->_lf_name << std::endl;
    std::cout << "- tau warp name: " << config_data->_tau_name << std::endl;
    std::cout << "- tau partial name: " << config_data->_dpart_name << std::endl;
}

int main( int argc, char **argv ) {

    setTraceLevel( 0 );

    // parse configuration
    config cfg;
    cfg.parse_command_line( argc, argv );

    // Pre-parse config data to input for algorithms
    Config_data *config_data = new Config_data;
    config_data->parse_config( cfg );
    assert( config_data != NULL );

    // Write configuration to output dir
    cfg.dump( config_data->_outdir + "/config" );

    // Check which algorithm to use
    string algo = "";
    cfg.get_switch( "algorithm", algo );

    if (algo == "IBR_direct") {

        assert(config_data->_w != 0 && config_data->_h != 0);

        TRACE("DIRECT IBR METHODS" << endl);

        Scene scene(config_data->_outdir,
                    "OpenGL",
                    config_data->_w, config_data->_h,
                    config_data->_w, config_data->_h,
                    config_data->_s_min, config_data->_s_max, config_data->_s_rmv,
                    config_data->_t_min, config_data->_t_max, config_data->_t_rmv);

        if(scene.initWindow() == false) {
            return -1;
        }

        if(scene.initGL() == false) {
            return -1;
        }

        scene.importViews( config_data->_mve_name, config_data->_scale_min, config_data->_scale_max );
        scene.importMesh( config_data->_mve_name );
        scene.mainLoop();

    } else if(algo == "IBR_laplacian") {

        TRACE("LAPLACIAN IBR" << endl);

        // unecessary parameters
        config_data->_lf_name = " ";
        config_data->_tau_name = " ";
        config_data->_dpart_name = " ";
        config_data->_imageName = " ";
        checkRequirement(config_data);

        // load scene
        Scene scene(config_data->_outdir,
                    "OpenGL",
                    config_data->_w, config_data->_h,
                    config_data->_w, config_data->_h,
                    config_data->_s_min, config_data->_s_max, config_data->_s_rmv,
                    config_data->_t_min, config_data->_t_max, config_data->_t_rmv,
                    config_data->_pyramidHeight);
        if(scene.initWindow() == false) {
            return -1;
        }

        if(scene.initGL() == false) {
            return -1;
        }

        std::cout << "Import Views" << std::endl;
        scene.importViews( config_data->_mve_name, config_data->_scale_min, config_data->_scale_max );
        std::cout << "Filter Depth Maps" << std::endl;
        scene.filterDepthMaps();
        // std::cout << "Rendering Loop" << std::endl;
        // scene.renderingLoop();
        std::cout << "Rendering Test" << std::endl;
        scene.renderingTest();

    } else if(algo == "refocus") {

        TRACE("REFOCUSSING / IBR" << endl);

        // unecessary parameters
        config_data->_lf_name = " ";
        config_data->_tau_name = " ";
        config_data->_dpart_name = " ";
        config_data->_imageName = " ";
        checkRequirement(config_data);
        const bool stanford = false; // TODO config param
        //        const float depthFocal = 2.5;// TODO config param
        const float depthFocal = 0;

        // load scene
        Scene scene(config_data->_outdir,
                    "OpenGL",
                    config_data->_w, config_data->_h,
                    config_data->_w, config_data->_h,
                    config_data->_s_min, config_data->_s_max, config_data->_s_rmv,
                    config_data->_t_min, config_data->_t_max, config_data->_t_rmv,
                    config_data->_pyramidHeight, depthFocal);
        if(scene.initWindow() == false) {
            return -1;
        }

        if(scene.initGL() == false) {
            return -1;
        }

        if(stanford) {

            std::cout << "Import source images, camera parameters and init empty depth maps (stanford dataset)" << std::endl;
            scene.importStanfordOriginalViews(config_data->_cameraName, config_data->_imageName);

        } else {

            std::cout << "Import Views" << std::endl;
            scene.importViews(config_data->_mve_name, config_data->_scale_min, config_data->_scale_max);
            std::cout << "Filter Depth Maps" << std::endl;
            scene.filterDepthMaps();
        }

        std::cout << "Rendering Loop" << std::endl;
        scene.refocussingLoop();

    } else if(algo == "IBR_optical") {

        TRACE("OPTICAL FLOW / IBR" << endl);

        // unecessary parameters
        config_data->_lf_name = " ";
        config_data->_tau_name = " ";
        config_data->_dpart_name = " ";

        // necessary parameters
        const bool stanford = false; // TODO config param
        if(stanford) {

            assert(!config_data->_mve_name.empty());
            config_data->_imageName = config_data->_mve_name + "/views" + "/view_%04i.mve" + "/undistorted.png";
            config_data->_cameraName = config_data->_mve_name + "/views" + "/view_%04i.mve" + "/meta.ini";

        } else {

            assert(!config_data->_imageName.empty());
            assert(!config_data->_cameraName.empty());
            config_data->_mve_name = " ";
        }

        if(config_data->_windowW1 == 0 && config_data->_windowW2 == 0) {
            config_data->_windowW1 = 0;
            config_data->_windowW2 = config_data->_w;
        }
        if(config_data->_windowH1 == 0 && config_data->_windowH2 == 0) {
            config_data->_windowH1 = 0;
            config_data->_windowH2 = config_data->_h;
        }

        std::cout << "Window width: from " << config_data->_windowW1 << " to " << config_data->_windowW2 << std::endl;
        std::cout << "Window height: from " << config_data->_windowH1 << " to " << config_data->_windowH2 << std::endl;

        std::cout << "Remove view (" << config_data->_s_rmv << " , " << config_data->_t_rmv << ")" << std::endl;

        // load lfScene
        LFScene lfScene(config_data->_unitTest,
                        config_data->_outdir,
                        "Plenoptic Space Linearization",
                        config_data->_mve_name, config_data->_imageName, config_data->_cameraName,
                        config_data->_windowW1, config_data->_windowW2,
                        config_data->_windowH1, config_data->_windowH2,
                        config_data->_w, config_data->_h,
                        config_data->_s_min, config_data->_s_max, config_data->_s_rmv,
                        config_data->_t_min, config_data->_t_max, config_data->_t_rmv,
                        stanford);

        if(stanford) {

            std::cout << "Import STANFORD views, MVE format" << std::endl;
            //            lfScene.importStanfordMVEViews();
            lfScene.importCustomMVEViews();

        } else {

            std::cout << "Import TOLF views, TOLF format" << std::endl;
            lfScene.importCustomTOLFViews();
        }

        if(!lfScene.checkExistenceAllViews(config_data->_outdir + "/flow%02lu.pfm")) {

            lfScene.computePerPixelCorrespCustomConfig(config_data->_flowAlg);

        } else {
            std::cout << "Optical flow already computed" << std::endl;
        }

        std::cout << "Compute flowed lightfield" << std::endl;
        lfScene.computeFlowedLFCustomConfig();

        //            if(!lfScene.checkExistence(config_data->_outdir + "/model_3g_IHM_%02lu.pfm", _renderIndex) ||
        //                    !lfScene.checkExistence(config_data->_outdir + "/model_4g_IHM_%02lu_a.pfm", _renderIndex) ||
        //                    !lfScene.checkExistence(config_data->_outdir + "/model_4g_IHM_%02lu_b.pfm", _renderIndex) ||
        //                    !lfScene.checkExistence(config_data->_outdir + "/model_6g_IHM_%02lu_au.pfm", _renderIndex) ||
        //                    !lfScene.checkExistence(config_data->_outdir + "/model_6g_IHM_%02lu_av.pfm", _renderIndex) ||
        //                    !lfScene.checkExistence(config_data->_outdir + "/model_6g_IHM_%02lu_b.pfm", _renderIndex))

        //                if(!lfScene.checkExistenceNoArg(config_data->_outdir + "/model_3g_IHM_allViews.pfm") ||
        //                !lfScene.checkExistenceNoArg(config_data->_outdir + "/model_4g_IHM_allViews_a.pfm") ||
        //                !lfScene.checkExistenceNoArg(config_data->_outdir + "/model_4g_IHM_allViews_b.pfm") ||
        //                !lfScene.checkExistenceNoArg(config_data->_outdir + "/model_6g_IHM_allViews_au.pfm") ||
        //                !lfScene.checkExistenceNoArg(config_data->_outdir + "/model_6g_IHM_allViews_av.pfm") ||
        //                !lfScene.checkExistenceNoArg(config_data->_outdir + "/model_6g_IHM_allViews_b.pfm"))

        std::cout << "Fit geometric models to light flow samples, with inhomogeneous method (IHM) initialization" << std::endl;
        lfScene.curveFitting(); // OK

        std::cout << "Fit photometric models to light flow samples" << std::endl;
        lfScene.curveFittingColor();

        //        if(!lfScene.checkExistence(config_data->_outdir + "/model_3g_IHM_%02lu.pfm", _renderIndex) ||
        //                !lfScene.checkExistence(config_data->_outdir + "/model_4g_IHM_%02lu_a.pfm", _renderIndex) ||
        //                !lfScene.checkExistence(config_data->_outdir + "/model_4g_IHM_%02lu_b.pfm", _renderIndex) ||
        //                !lfScene.checkExistence(config_data->_outdir + "/model_6g_IHM_%02lu_au.pfm", _renderIndex) ||
        //                !lfScene.checkExistence(config_data->_outdir + "/model_6g_IHM_%02lu_av.pfm", _renderIndex) ||
        //                !lfScene.checkExistence(config_data->_outdir + "/model_6g_IHM_%02lu_b.pfm", _renderIndex))

        std::cout << "Perform model selection via BIC" << std::endl;
        lfScene.bic();

        std::cout << "Render image by interpolating the light flow" << std::endl;

        //        lfScene.renderLightFlow(); // NOT OK
        //        lfScene.renderLightFlowLambertianModel(); // NOT OK
        lfScene.renderLightFlowLambertianVideo(); // OK

        // DEPRECATED FUNCTIONS

        //            std::cout << "Import source images, camera parameters and init empty depth maps" << std::endl;
        //            lfScene.importViewsNoDepth();

        //            if(config_data->_computeFlow != 0) {
        //                std::cout << "Compute optical flow" << std::endl;
        //                lfScene.computePerPixelCorresp(config_data->_flowAlg);
        //            } else {
        //                std::cout << "Optical flow already computed" << std::endl;
        //            }

        //            std::cout << "Compute optical flow" << std::endl;
        //            lfScene.computeFlowedLightfield();

        //            std::cout << "Compute point cloud from optical flow, given a target view" << std::endl;
        //            lfScene.curveFitting();

        //            //        std::cout << "Rendering Loop" << std::endl;
        //            //        lfScene.renderingLoop();

    } else if(algo == "fromMVE2coco") {

        TRACE("FROM MVE TO COCOLIB" << endl);

        // unecessary parameters
        config_data->_imageName = " ";
        config_data->_cameraName = " ";
        checkRequirement(config_data);

        // load scene
        Scene scene(config_data->_outdir,
                    "OpenGL",
                    config_data->_w, config_data->_h,
                    config_data->_w, config_data->_h,
                    config_data->_s_min, config_data->_s_max, config_data->_s_rmv,
                    config_data->_t_min, config_data->_t_max, config_data->_t_rmv);
        if(scene.initWindow() == false) {
            return -1;
        }

        if(scene.initGL() == false) {
            return -1;
        }

        // compute and export warps
        std::cout << "Import Views" << std::endl;
        scene.importViews( config_data->_mve_name, config_data->_scale_min, config_data->_scale_max );
        std::cout << "Filter Depth Maps" << std::endl;
        scene.filterDepthMaps();
        std::cout << "Compute Target Depth Map" << std::endl;
        scene.computeTargetDepthMap( config_data->_uDepth_name );
        std::cout << "Export Warps" << std::endl;
        scene.exportWarps( config_data->_lf_name, config_data->_tau_name, config_data->_dpart_name, config_data->_uWarped_name );
        //        std::cout << "Test Splatting" << std::endl;
        //        scene.renderSplatting();

    } else if ( algo == "IBR_color" ) {

        TRACE("IMAGE-BASED RENDERING WITH GRADIENT CONSTRAINTS" << endl);
        sr_synthesize_view( config_data );

    } else if ( algo == "IBR_gradient" ) {

        TRACE("IMAGE-BASED RENDERING IN THE GRADIENT DOMAIN" << endl);
        IBR_gradient( config_data );

    } else if ( algo == "IBR_video" ) {

        TRACE("IMAGE-BASED RENDERING, VIDEO" << endl);

        // unecessary parameters
        config_data->_imageName = " ";
        config_data->_cameraName = " ";
        checkRequirement(config_data);

        int nbFrames = 300;

        for(int frame = 0 ; frame < nbFrames ; ++frame) {

            // load scene
            Scene* scene = new Scene(config_data->_outdir,
                                     "OpenGL",
                                     1600, 1200,
                                     config_data->_w, config_data->_h,
                                     config_data->_s_min, config_data->_s_max, config_data->_s_rmv,
                                     config_data->_t_min, config_data->_t_max, config_data->_t_rmv);
            if(scene->initWindow() == false) {
                return -1;
            }

            if(scene->initGL() == false) {
                return -1;
            }

            // compute and export warps
            std::cout << "Import Views" << std::endl;
            scene->importViews( config_data->_mve_name, config_data->_scale_min, config_data->_scale_max );
            std::cout << "Filter Depth Maps" << std::endl;
            scene->filterDepthMaps();
            std::cout << "Update target view" << std::endl;
            scene->moveTargetCam(frame);
            std::cout << "Compute Target Depth Map" << std::endl;
            scene->computeTargetDepthMap( config_data->_uDepth_name );
            std::cout << "Export Warps" << std::endl;
            scene->exportWarps( config_data->_lf_name, config_data->_tau_name, config_data->_dpart_name, config_data->_uWarped_name );

            delete scene;

            // render view
            sr_synthesize_view( config_data, frame );
        }

    } else if( algo == "" ) {

        // load scene
        Scene * scene = new Scene(config_data->_outdir,
                                  "OpenGL",
                                  config_data->_w, config_data->_h,
                                  config_data->_w, config_data->_h,
                                  config_data->_s_min, config_data->_s_max, config_data->_s_rmv,
                                  config_data->_t_min, config_data->_t_max, config_data->_t_rmv,
                                  config_data->_pyramidHeight);
        if(scene->initWindow() == false) {
            return -1;
        }

        if(scene->initGL() == false) {
            return -1;
        }

        std::cout << "Create Views" << std::endl;
        scene->createTestViews();
        // no need to filter the depth maps (already orthogonal depth thank to depthFromMesh shader)
        std::cout << "Rendering Loop" << std::endl;
        scene->renderingLoop();

        delete scene;

    } else {

        TRACE("DEFAULT: IMAGE-BASED RENDERING WITH GRADIENT CONSTRAINTS" << endl);
        sr_synthesize_view( config_data );
        //        TRACE( "no algorithm specified - set switch 'algorithm' to one of the following:" << std::endl );
        //        TRACE( "  IBR_color           : variational IBR with gradient constraints." << std::endl );
        //        TRACE( "  IBR_gradient        : variational IBR in the gradient domain." << std::endl );
    }

    delete config_data;

    return 0;
}
