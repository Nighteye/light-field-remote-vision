#include <vector>
#include <string>
#include <fstream>

#include "openglFunctions.h"
#include "ply_io.h"
#include "config.h"
#include "pinholeCamera.h"
#include <cocolib/cocolib/common/gsl_image.h>

// load the first image just to get its size
void getSize( Config_data *config_data ) {

    std::string image_name = config_data->_mve_name + "/views" + "/view_%04i.mve" + "/undistorted.png";

    char str[500];
    sprintf( str, image_name.c_str(), config_data->_s_min );
    coco::gsl_image *image = coco::gsl_image_load( str );
    assert( image != 0 );

    config_data->_w = image->_w;
    config_data->_h = image->_h;

    coco::gsl_image_free( image );
}

// load the (undistorded) images from mve scene and export them in PNG format
void exportImages( Config_data *config_data, unsigned int s ) {

    std::string image_name = config_data->_mve_name + "/views" + "/view_%04i.mve" + "/undistorted.png";

    char str[500];
    sprintf( str, image_name.c_str(), s );
    // TRACE( str << std::endl );
    coco::gsl_image *image = coco::gsl_image_load( str );

    config_data->_w = image->_w;
    config_data->_h = image->_h;

    sprintf( str, config_data->_lf_name.c_str(), s );
    coco::gsl_image_save( str, image );
    TRACE( "Save image " << str << std::endl );

    coco::gsl_image_free( image );
}

// Read the camera matrices from INI file (MVE format)
PinholeCamera importINICam( Config_data *config_data, unsigned int s ) {

    std::string camera_name = config_data->_mve_name + "/views" + "/view_%04i.mve" + "/meta.ini";

    char str[500];
    sprintf( str, camera_name.c_str(), s );
    TRACE( "Import camera " << str << std::endl );

    std::ifstream in( str, std::ifstream::in );
    assert( in.is_open() );
    assert( in );

    std::string tmp; // PMVS header

    float focal_length = 0.0;
    float pixel_aspect = 0.0;
    double principal_point[] = {0.0, 0.0};
    glm::mat3 R(1.0);
    glm::vec3 t(0.0);
    glm::mat3 K(0.0);

    while( strcmp( "[camera]", tmp.c_str() ) ) {

        in >> tmp;
    }

    in >> tmp >> tmp >> focal_length
            >> tmp >> tmp >> pixel_aspect
            >> tmp >> tmp >> principal_point[0] >> principal_point[1]
            >> tmp >> tmp >> R[0][0] >> R[1][0] >> R[2][0] >> R[0][1] >> R[1][1] >> R[2][1] >> R[0][2] >> R[1][2] >> R[2][2]
                                     >> tmp >> tmp >> t[0] >> t[1] >> t[2];

    in.close();

    assert( pixel_aspect != 0 );

    // focal_length = f1 in pixels divided by larger side
    // pixel_aspect = pixel width divided by pixel height
    // principal_point is also normalized and independent of the image size
    if( config_data->_w >= config_data->_h) {
        K[0][0] = config_data->_w * focal_length;
    } else {
        K[0][0] = config_data->_h * focal_length;
    }
    K[1][1] = K[0][0] / pixel_aspect;
    K[2][2] = 1.0;
    K[2][0] = config_data->_w * principal_point[0];
    K[2][1] = config_data->_h * principal_point[1];

    return PinholeCamera( K, R, t );
}

// Read the ply file (reconstructed surface)
void importMesh( Config_data *config_data, std::vector< float > &vec_points,
                 std::vector< float > &vec_normals,
                 std::vector< unsigned int > &vec_triangles ) {

    std::string mesh_name = config_data->_mve_name + "/surface-L2-clean.ply";

    TRACE( "Import mesh " << mesh_name << std::endl );

    assert( read_ply( mesh_name.c_str(), vec_points, vec_normals, vec_triangles ) );
}

// Compute data structures needed for IBR from an MVE scene (with .ply mesh)
void fromMVE2coco( Config_data *config_data ) {

    // check for required data
    assert( config_data != NULL );
    // we assume that the target view is one of the input views
    assert( config_data->_s_min <= config_data->_s_rmv &&
            config_data->_s_rmv <= config_data->_s_max );
    // no super-resolution
    assert( config_data->_dsf == 1 );

    getSize( config_data );
    const int w = config_data->_w;
    const int h = config_data->_h;
    assert( w * h > 0 );

    std::vector< float > vec_points;
    std::vector< float > vec_normals;
    std::vector< float > vec_sigma_vertex;
    std::vector< unsigned int > vec_triangles;

    importMesh( config_data, vec_points, vec_normals, vec_triangles );

    const unsigned int nb_points = vec_points.size()/3;
    // const unsigned int nb_normals = vec_normals.size()/3;
    // const unsigned int nb_triangles = vec_triangles.size()/3;
    const unsigned int nbCams = config_data->_s_max - config_data->_s_min + 1;

    const float uniform_vertex_sigma = 0.04;
    const float normal_scale = 0.01;

    PinholeCamera u_cam;

    WarpGL warpGL;

    warpGL.resizeImageVector( nbCams );

    unsigned int nview = 0;

    //    TRACE( "Load and export input views" << std::endl );
    //    for ( int s = config_data->_s_min ; s <= config_data->_s_max ; ++s ) {

    //        // load the (undistorded) images from mve scene and export them in PNG format
    //        exportImages( config_data, s );

    //        // warpGL.loadTextures( config_data, s, nview );

    //        ++nview;
    //    }

    // -------------------------- INITALIZING GLEW ---------------------- //

    // Create GL context
    warpGL.createGLcontext();

    // checking openGL extensions, versions, etc.
    glewExperimental=true;
    GLenum err = glewInit();

    if (err != GLEW_OK) {
        TRACE( "Problem: glewInit failed, something is seriously wrong." << std::endl );
        exit(1); // or handle the error in a nicer way
    }
    checkError();
    if (!GLEW_VERSION_2_1) {  // check that the machine supports the 2.1 API.
        TRACE( "Problem: your machine does not support the 2.1 API" << std::endl );
        exit(1); // or handle the error in a nicer way
    }

    // ------------------ CREATE AND INIT SHADERS ---------------------- //
    TRACE( "Initialize shaders" << std::endl );

    int res;

    res = warpGL.initWarpShader( );
    if (res != 0) {
        TRACE( "Problem initializing the warp and weights shader (" << res << ")" << std::endl );
        exit( EXIT_FAILURE );
    }

    TRACE( "Shaders initialized" << std::endl );

    // ------------------- INIT FBOs ---------------------------------- //

    res = warpGL.initRenderFBOs( nbCams, w, h );
    if (res != 0) {
        TRACE( "Error in initRenderFBOs" << std::endl );
        exit( EXIT_FAILURE );
    }

    res = warpGL.initViewFBOs( );
    if (res != 0) {
        TRACE( "Error in initViewFBOs" << std::endl );
        exit( EXIT_FAILURE );
    }

    TRACE( nbCams << " FBO and textures initialized" << std::endl );

    //------------- CREATE AND INIT VAOs -------------------------------//
    TRACE("Creating Vertex VAOs" << std::endl);
    // Use vertex array for better drawing

    // Vertex array object with the 3D Vertices, normals and triangles

    warpGL.createVAOs(1);

    // Init sigma without computing it, visiblity is not yet handled
    vec_sigma_vertex.resize(nb_points);
    std::fill(vec_sigma_vertex.begin(), vec_sigma_vertex.end(), uniform_vertex_sigma);

    warpGL.initVAOs(vec_points, vec_sigma_vertex, vec_triangles, vec_normals, normal_scale);
    TRACE("   Vertex VAOs OK" << std::endl);
    TRACE("   Creating Quad VAOs" << std::endl);
    TRACE("   Quad VAOs OK" << std::endl);

    warpGL.createQuadVAO();

    //TODO: createCameraCenterVAO(doc);
    //TODO: createCameraFrustumVAO(doc);

    // ------------------- MAIN LOOP ---------------------------------- //

    // for all views
    nview = 0;
    for ( int s = config_data->_s_min ; s <= config_data->_s_max ; ++s ) {

        // import camera parameters
        PinholeCamera v_i_cam = importINICam( config_data, s );
        // v_i_cam.display();

        // import the view to synthetize also, since we remove it only when running cocolib
        if ( s == config_data->_s_rmv ) {

            u_cam = v_i_cam;
        }

        // FIRST PASS: compute the beta warp: from Gamma to omega_i
        // we obtain beta vi: the image vi seen from the viewpoint u

        TRACE("Compute warps and weights of camera " << s << std::endl );
        warpGL.computeWarps( config_data,
                             nview,
                             vec_triangles.size(),
                             u_cam,
                             v_i_cam );

        warpGL.saveFloatTexture( config_data, s );

        ++nview;

        continue;

        //        // first compute the depth maps of the triangulation as seen from u
        //        render_vi_depth(m_render_fbo[0],
        //                m_vaoID[0], vec_triangles.size(), 0,
        //                m_u_depth,
        //                depthShader, render_cam.cam, render_cam.width, render_cam.height);

        //        // compute the tau warp: from omega_i to Gamma
        //        // we obtain tau u : the image "u" seen from the viewpoint vi
        //        // now the image per_pixel_sigma is used
        //        compute_warps_and_weights(m_view_fbo[0], m_vaoID[0], vec_triangles.size(), 0,
        //                image_vector_dist[render_cam_index].texture,
        //                m_lookup_texture[render_cam_index],
        //                m_per_pixel_sigma[render_cam_index],
        //                m_u_depth, m_depth_epsilon,
        //                m_buffer[0], // output image: may be useless if we do not have the u image
        //                m_backwards[0], // the tau warp
        //                m_partial_tau_grad_v[0], // // the z derivative of the tau warp
        //                warp_and_weightsShader,
        //                current_cam, render_cam,
        //                m_depth_mapping, m_mapping_cut_depth, m_mapping_factor, ref_cam);
    }

    //    // if lumigraph

    //    printf("Storing %d Warp Buffers for cocolib transfer\n", render_indices.size());

    //    for (int r_index = 0; r_index<render_indices.size(); ++r_index) {

    //        render_cam_index = render_indices[r_index];
    //        bool use_render_cam = false;
    //        if (render_cam_index == -1) { // USE CURRENT CAM
    //            render_cam_index = 0;
    //            use_render_cam = true;
    //        } else if (render_cam_index <0 || render_cam_index >= nbCams) {
    //            printf("Invalid index in list: %d. Index range [0,%d]\n", render_cam_index, nbCams);
    //            continue;
    //        } else {
    //            render_cam.cam = openMVG::PinholeCamera(doc._map_brown_camera[render_cam_index]._P);
    //            render_cam.width = doc._map_imageSize[render_cam_index].first;
    //            render_cam.height = doc._map_imageSize[render_cam_index].second;
    //        }

    //        std::ostringstream cam_index_str;
    //        cam_index_str << std::setfill('0') << std::setw(2) << render_cam_index;
    //        std::string outFolderRender = stlplus::folder_append_separator(out_folder) + stlplus::folder_append_separator(cam_index_str.str());

    //        printf("Storing Warp Buffers for cocolib transfer for render image %d ...\n", render_cam_index);

    //        std::vector< cimg_library::CImg<float> > vec_img(nbCams);

    //        for (int i=0; i<nbCams; ++i) {


    //            //BrownPinholeCamera current_cam = doc._map_brown_camera.find(i)->second;
    //            IbrCamera current_cam;
    //            current_cam.cam = doc._map_camera.find(i)->second;
    //            current_cam.width = image_vector_dist[i].width;
    //            current_cam.height = image_vector_dist[i].height;

    //            // set all penalties to max for the current cam -> weights to 0 will follow
    //            if (render_cam_index == i && !use_render_cam) {
    //                vec_img[i].assign(current_cam.width, current_cam.height, 1, 4);
    //                cimg_forXY(vec_img[i], x, y) {
    //                    vec_img[i](x,y,0,0) = 0.;
    //                    vec_img[i](x,y,0,1) = 0.;
    //                    vec_img[i](x,y,0,2) = -180.;
    //                    vec_img[i](x,y,0,3) = -1.;
    //                }
    //                continue;
    //            }

    //            render_new_view(m_render_fbo[0], m_vaoID[0], vec_triangles.size(), m_displacement,
    //                    image_vector_dist[i].texture,
    //                    m_lookup_texture[i],
    //                    m_per_pixel_sigma[0], // in this computation per_pixel_sigma is not needed
    //                    m_depth_vi[i][m_nb_displacement/2], m_depth_epsilon,
    //                    m_buffer[0], m_backwards[0], m_partial_tau_grad_v[0],
    //                    simpleShader, render_cam,
    //                    current_cam,
    //                    m_deformation_weight_threshold,
    //                    m_depth_mapping, m_mapping_cut_depth, m_mapping_factor, ref_cam);

    //            std::ostringstream number;
    //            number << std::setfill('0') << std::setw(IBR_PADDING) << i;

    //            // take care, setColorTexture2CImg initializes the image and transfers the 4 channels
    //            setColorTexture2CImg(m_backwards[0], vec_img[i], render_cam.width, render_cam.height);
    //        }

    //        std::string weights_filename = outFolderRender + std::string("lumi_weight_");
    //        lumigraph_weights_cpu(vec_img, // third and fourth component have angle and norm
    //                              weights_filename);
    //    }

    // delete
    warpGL.removeRenderFBOs(nbCams);
    warpGL.removeViewFBOs(nbCams);
    warpGL.deleteVAOs(1);

    warpGL.GLterminate();
}
