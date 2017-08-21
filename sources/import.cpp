#include <QtGui/QImage>
#include <fstream>
#include <vector>

// GLM
#include <glm/gtx/norm.hpp>
#include "glm/ext.hpp"

// OpenEXR
#include <OpenEXR/ImfRgbaFile.h>
#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfMatrixAttribute.h>
#include <OpenEXR/ImfArray.h>
#include <cocolib/cocolib/common/gsl_image.h>

#include "pinholeCamera.h"
#include "config.h"
#include "super-resolution.h"
#include "import.h"
#include "ply_io.h"

#define INVALID_DEPTH 1000.0

using namespace std;
using namespace coco; // goldluecke's files

// fill the visibility vector with the indices of the cameras that see the points
bool read_patches( const char* filename, vector< vector<int> > &visibility ) {

    ifstream in(filename, ifstream::in);
    assert(in.is_open());
    assert(in);

    string header = "";
    size_t nb_patches = 0;
    in >> header >> nb_patches;
    assert( nb_patches == visibility.size() );
    TRACE( "Reading " << nb_patches << " patches from file " << filename << endl );

    float tmp = 0;
    float tmp2 = 0;

    for( size_t n = 0 ; n < nb_patches ; ++n ) {

        in >> header >>
                tmp >> tmp >> tmp >> tmp >>
                tmp >> tmp >> tmp >> tmp >>
                tmp >> tmp >> tmp >>
                tmp;

        visibility[n].resize( (int)tmp );

        for( int i = 0 ; i < tmp ; ++i ) {

            in >> visibility[n][i];
        }

        in >> tmp;
        for( int i = 0 ; i < tmp ; ++i ) {

            in >> tmp2;
        }
    }
    in.close();

    return true;
}

// load the first channel of an EXR image with OpenEXR
float* import_exr( int w, int h, string filename ) {

    Imf::RgbaInputFile file(filename.c_str());
    Imath::Box2i dw = file.dataWindow();
    const int
            inwidth = dw.max.x - dw.min.x + 1,
            inheight = dw.max.y - dw.min.y + 1;

    assert( w == inwidth && h == inheight );

    Imf::Array2D<Imf::Rgba> pixels;
    pixels.resizeErase(inheight,inwidth);
    file.setFrameBuffer(&pixels[0][0] - dw.min.x - dw.min.y*inwidth, 1, inwidth);
    file.readPixels(dw.min.y, dw.max.y);

    float *ptr_r = new float[inwidth*inheight];
    for (int i(0) ; i < inwidth*inheight ; ++i) {
        ptr_r[i] = (float)pixels[i/inwidth][i%inwidth].r;
    }

    return ptr_r;
}

// write exr image
void save_exr( int w, int h, float* image, string filename, int hint ) {

    char str[500];
    sprintf( str, filename.c_str(), hint );

    Imf::Rgba *const ptrd0 = new Imf::Rgba[w*h], *ptrd = ptrd0, rgba;

    for (const float *ptr_r = image, *const ptr_e = ptr_r + w*h; ptr_r<ptr_e;) {
        rgba.r = rgba.g = rgba.b = (half)(*(ptr_r++));
        rgba.a = (half)1;
        *(ptrd++) = rgba;
    }

    Imf::RgbaOutputFile outFile(str,w,h,Imf::WRITE_Y);
    outFile.setFrameBuffer(ptrd0,1,w);
    outFile.writePixels(h);

    delete[] ptrd0;
}

// load low res images and setup the dimensions
vector<gsl_image*> import_lightfield( Config_data *config_data ) {

    // input file mask
    const char *mask = config_data->_lf_name.c_str();

    // output lightfield
    vector<gsl_image*> lightfield;

    // loop over all directory entries
    string lf_name, lf_dir;
    breakupFileName( mask, lf_dir, lf_name );
    Directory::cd( lf_dir );
    TRACE( "  searching for images in subdirectory " << lf_dir << ", format " << lf_name << endl );

    char first_image[500];
    sprintf( first_image, lf_name.c_str(), 0 );
    if ( config_data->_w * config_data->_h == 0 ) { // config the image dimensions (low res)
        QImage im( first_image );
        config_data->_w = im.width();
        config_data->_h = im.height();
    }

    TRACE( "    Image dimensions (low res): ( " <<  config_data->_w << " , " <<  config_data->_h << " )" << endl);

    size_t nview = 0;
    for ( int s = config_data->_s_min; s <= config_data->_s_max ; s++ ) {

        if ( s != config_data->_s_rmv ) { // we don't import the view to synthesize

            char str[500];
            sprintf( str, lf_name.c_str(), s );
            TRACE( "   reading input file " << str << endl);

            // read file
            gsl_image *image = gsl_image_load( str );
            // check if the dimensions of the files are low res
            assert( config_data->_w * config_data->_h > 0 );
            assert( config_data->_dsf > 0 );
            assert( config_data->_w == image->_w && config_data->_h == image->_h );

            lightfield.push_back(image);
            nview++;
        }
    }

    config_data->_nviews = nview;
    config_data->_W = config_data->_w * config_data->_dsf;
    config_data->_H = config_data->_h * config_data->_dsf;

    // reset work dir
    Directory::cd( Directory::base() );
    TRACE( "switched back to dir " << Directory::current() << endl );

    return lightfield;
}

// Read warps and weights from files
void import_warps( Config_data *config_data, gsl_image** warps, string path ) {

    // loop over all directory entries
    string file_name, file_dir;
    breakupFileName( path.c_str(), file_dir, file_name );
    Directory::cd( file_dir );
    TRACE( "  searching for input files in subdirectory " << file_dir << ", format " << file_name << endl );

    // Read files
    int nview = 0;
    for ( int s = config_data->_s_min ; s <= config_data->_s_max ; s++ ) {

        if ( s != config_data->_s_rmv ) { // we don't import the view to synthesize

            char str[500];
            sprintf( str, file_name.c_str(), s );
            TRACE( "   reading input file " << str << std::endl);

            // read file
            gsl_image* warp = gsl_image_load_pfm( str );
            // check if the dimensions of the files are low res
            assert( config_data->_w * config_data->_h > 0 );
            assert( config_data->_dsf > 0 );
            assert( (size_t)config_data->_w == warp->_w && (size_t)config_data->_h == warp->_h );
            warps[nview] = warp;

            nview++;
        }
    }

    // reset work dir
    Directory::cd( Directory::base() );
    TRACE( "switched back to dir " << Directory::current() << endl );
}

// Read the input depth maps
double * import_depth(string path, size_t nview) {

    char filename[500];
    sprintf( filename, path.c_str(), nview );
    TRACE( "   reading depth map " << filename << std::endl );

    return gsl_image_load_pfm( filename )->_r->data;
}

// Read the camera matrices
glm::mat4x3 import_cam( string path, size_t nview ) {

    char filename[500];
    sprintf( filename, path.c_str(), nview );

    ifstream in(filename, ifstream::in);
    assert(in.is_open());
    assert(in);

    glm::mat4x3 P;
    string header; // PMVS header

    in >> header >> P[0][0] >> P[1][0] >> P[2][0] >> P[3][0]
                 >> P[0][1] >> P[1][1] >> P[2][1] >> P[3][1]
                 >> P[0][2] >> P[1][2] >> P[2][2] >> P[3][2];

    assert( strcmp( "CONTOUR", header.c_str() ) == 0 );

    in.close();

    return P;
}

// Compute warps and using depth map and camera matrices
void import_warps_from_depth( Config_data *config_data, gsl_image** warps ) {

    // check for required data
    assert( config_data != NULL );
    size_t w = config_data->_w;
    size_t h = config_data->_h;
    assert( w * h > 0 );
    size_t W = config_data->_W;
    size_t H = config_data->_H;
    assert( W*H > 0 );

    //read the last camera matrix: view to synthsize
    size_t u_index = config_data->_nviews;
    if ( config_data->_s_rmv != -1 ) {
        ++u_index;
    }

    PinholeCamera u_cam(import_cam( config_data->_cam_name, u_index ));
    //u_cam.display();

    float u_depth[W*H];
    for ( size_t p = 0 ; p < W*H ; ++p ) {

        u_depth[p] = INVALID_DEPTH;
    }

    TRACE("Compute warps from depth map and camera matrices..." << endl);

    //Compute u depth map for visibility
    size_t nview = 0;
    for ( int s = config_data->_s_min ; s <= config_data->_s_max ; ++s ) {

        if ( s == config_data->_s_rmv ) {
            continue;
        }

        warps[nview] = gsl_image_alloc( w, h );

        // reading the depth map of the currrent view
        double *depth = import_depth( config_data->_depth_name, s );

        // read the camera matrix of the current view
        PinholeCamera v_i_cam(import_cam( config_data->_cam_name, s ));
        //v_i_cam.display();

        for ( size_t m = 0 ; m < w * h ; ++m ) { // for each pixel of the depth map

            float d = depth[m];

            if ( d < 1.0/INVALID_DEPTH || INVALID_DEPTH < d ) {
                warps[nview]->_r->data[m] = -(double)w;
                warps[nview]->_g->data[m] = -(double)h;
                continue;
            }

            float x = m % w;
            float y = m / w;

            glm::vec3 X(x+0.5, y+0.5, 1);
            glm::mat3 K_inv(1.0);
            assert(v_i_cam._K[0][0] != 0);
            assert(v_i_cam._K[1][1] != 0);
            K_inv[0][0] = 1.0/v_i_cam._K[0][0];
            K_inv[1][1] = 1.0/v_i_cam._K[1][1];
            K_inv[2][0] = -v_i_cam._K[2][0]/v_i_cam._K[0][0];
            K_inv[2][1] = -v_i_cam._K[2][1]/v_i_cam._K[1][1];

            X = d * K_inv * X;
            glm::mat4x3 Rt_inv;
            glm::mat3 R_inv = glm::transpose(v_i_cam._R);
            Rt_inv[0] = R_inv[0]; Rt_inv[1] = R_inv[1]; Rt_inv[2] = R_inv[2]; Rt_inv[3] = v_i_cam._C;
            glm::vec4 X_tmp(X, 1.0);
            X = Rt_inv*X_tmp;

            X_tmp = glm::vec4(X, 1.0);
            X = u_cam._P * X_tmp;
            warps[nview]->_r->data[m] = (float)(X[0]/X[2]);
            warps[nview]->_g->data[m] = (float)(X[1]/X[2]);
            // forward visibility is automatically handled by the warp value being outside the image frame

            // create u map for backward visibility
            for ( int i = 0 ; i < config_data->_dsf ; ++i ) {
                for ( int j = 0 ; j < config_data->_dsf ; ++j ) {
                    // round(x-0.5) = floor(x-0.5+0.5) = floor(x)
                    int px = int(X[0]/X[2]) - config_data->_dsf/2 + j;
                    int py = int(X[1]/X[2]) - config_data->_dsf/2 + i;
                    int p = px + py*W;
                    if ( 0 <= px && px <= (int)(W-1) &&
                         0 <= py && py <= (int)(H-1) ) {

                        if ( X[2] < u_depth[p] ) {
                            u_depth[p] = X[2];
                        }
                    }
                }
            }
        }

        delete[] depth;

        ++nview;
    }
    TRACE("...done!" << endl);

    TRACE("Write u depth map" << endl);
    write_pfm_image( w, h, u_depth, "./in/blender/skull/u_depth.pfm", 0 );

    // cope with backward visibility
    TRACE("Check for visibility..." << endl);

    nview = 0;
    for ( int s = config_data->_s_min ; s <= config_data->_s_max ; ++s ) {

        if ( s == config_data->_s_rmv ) {
            continue;
        }

        // read the camera matrix of the current view
        PinholeCamera v_i_cam(import_cam( config_data->_cam_name, s ));

        for ( size_t m = 0 ; m < w*h ; ++m ) { // for each pixel of v_i depth map

            float u_d = 0.0;
            float u_norm = 0.0;
            for ( int i = 0 ; i < config_data->_dsf+1 ; ++i ) {
                for ( int j = 0 ; j < config_data->_dsf+1 ; ++j ) {

                    int px = (int)floor(warps[nview]->_r->data[m] - float(config_data->_dsf)*0.5 + 0.5) + j;
                    int py = (int)floor(warps[nview]->_g->data[m] - float(config_data->_dsf)*0.5 + 0.5) + i;
                    if ( 0 <= px && px <= (int)W-1 &&
                         0 <= py && py <= (int)H-1 ) {

                        int p = py * W + px;
                        if ( u_depth[p] == INVALID_DEPTH ) {
                            continue;
                        }

                        float dx = 1.0;
                        if ( j == 0 ) {
                            dx = (float(px) + 0.5) - (warps[nview]->_r->data[m] - 0.5*float(config_data->_dsf));
                        }
                        else if ( j == config_data->_dsf+1 - 1 ) {
                            dx = (warps[nview]->_r->data[m] + 0.5*float(config_data->_dsf)) - (float(px) - 0.5);
                        }
                        float dy = 1.0;
                        if ( i == 0 ) {
                            dy = (float(py) + 0.5) - (warps[nview]->_g->data[m] - 0.5*float(config_data->_dsf));
                        }
                        else if ( i == config_data->_dsf+1 - 1 ) {
                            dy = (warps[nview]->_g->data[m] + 0.5*float(config_data->_dsf)) - (float(py) - 0.5);
                        }

                        u_d += dx*dy*u_depth[p];
                        u_norm += dx*dy;
                    }
                }
            }

            if ( u_norm == 0.0 ) {

                warps[nview]->_r->data[m] = -(double)w;
                warps[nview]->_g->data[m] = -(double)h;
                continue;

            } else {

                u_d /= u_norm;
            }

            glm::vec3 X(int(warps[nview]->_r->data[m])+0.5, int(warps[nview]->_g->data[m])+0.5, 1); // nearest
            glm::mat3 K_inv(1.0);
            assert(u_cam._K[0][0] != 0);
            assert(u_cam._K[1][1] != 0);
            K_inv[0][0] = 1/u_cam._K[0][0];
            K_inv[1][1] = 1/u_cam._K[1][1];
            K_inv[2][0] = -u_cam._K[2][0]/u_cam._K[0][0];
            K_inv[2][1] = -u_cam._K[2][1]/u_cam._K[1][1];

            X = u_d * K_inv * X;
            glm::mat4x3 Rt_inv;
            glm::mat3 R_inv = glm::transpose(v_i_cam._R);
            Rt_inv[0] = R_inv[0]; Rt_inv[1] = R_inv[1]; Rt_inv[2] = R_inv[2]; Rt_inv[3] = v_i_cam._C;
            glm::vec4 X_tmp(X, 1.0);
            X = Rt_inv*X_tmp;

            X_tmp = glm::vec4(X, 1.0);

            X = v_i_cam._P * X_tmp;
            // round(x-0.5) = floor(x-0.5+0.5) = floor(x)
            assert(X[2] != 0);
            int ox = int(X[0]/X[2]);
            int oy = int(X[1]/X[2]);

            if ( ox - int(m%w) < -10 || 10 < ox - int(m%w) ||
                 oy - int(m/w) < -10 || 10 < oy - int(m/w) ) {

                warps[nview]->_r->data[m] = -(double)w*2.0;
                warps[nview]->_g->data[m] = -(double)h*2.0;
            }
        }

        TRACE("     Write pfm tau warp, view " << s << endl);
        write_pfm_image( w, h, warps[nview]->_r->data, warps[nview]->_g->data, warps[nview]->_g->data, config_data->_tau_name, s );

        ++nview;
    }
    TRACE("...done!" << endl);
}

// compute the 3x3 covariance matrix of a 3D point
glm::mat3 compute_3D_covariance( Config_data *config_data, glm::vec3 X, vector<int> XVisibility ) {

    // for the moment the covariance of the image point is identity which means that sigma_x=sigma_y=1

    assert(XVisibility.size() > 1); // point must be visible from at least 2 cameras

    glm::vec4 Xh(X, 1.0f);

    glm::mat3x2 Jx(0.0f);
    glm::mat3 CXX(0.0f);

    for ( size_t i = 0 ; i < XVisibility.size() ; ++i ) {

        int s = XVisibility[i];

        PinholeCamera v_i_cam(import_cam( config_data->_cam_name, s ));

        glm::vec3 xh = v_i_cam._P * Xh;

        Jx[0][0] = 1; Jx[1][0] = 0; Jx[2][0] = -xh[0]/xh[2];
        Jx[0][1] = 0; Jx[1][1] = 1; Jx[2][1] = -xh[1]/xh[2];
        Jx /= xh[2];

        Jx = Jx * v_i_cam._K * v_i_cam._R;
        assert(false);
        // TODO to compile:
//        CXX = CXX + (const glm::mat2x3&) (glm::transpose(Jx)) * (const glm::mat3x2&) Jx;
    }

    CXX = glm::inverse(CXX);
    TRACE2("CXX: " << endl << glm::to_string(CXX) << endl);

    return CXX;
}

// Compute depth using ply and camera matrices
void import_depth_from_ply( Config_data *config_data, vector<gsl_image*> lightfield ) {

    // check for required data
    assert( config_data != NULL );
    size_t w = config_data->_w;
    size_t h = config_data->_h;
    assert( w * h > 0 );

    vector< float > vec_points;
    vector< float > vec_normals;
    vector< unsigned int > vec_triangles;

    // the ith element contains the cameras from which the ith point is visible
    vector< vector<int> > visibility;

    assert( read_ply( config_data->_ply_name.c_str(), vec_points, vec_normals, vec_triangles ) );
    visibility.resize( vec_points.size()/3 );
    if ( !config_data->_patch_name.empty() ) {
        assert( read_patches( config_data->_patch_name.c_str(), visibility ) );
    }

    TRACE("Compute depth maps from .ply and .patch files..." << endl);

    //read the last camera matrix: view to synthsize
    size_t u_index = config_data->_nviews;
    if ( config_data->_s_rmv != -1 ) {
        ++u_index;
    }

    PinholeCamera u_cam(import_cam( config_data->_cam_name, u_index ));

    // for all views
    size_t nview = 0;
    for ( int s = config_data->_s_min ; s <= config_data->_s_max ; ++s ) {

        if ( s == config_data->_s_rmv ) {
            continue;
        }

        gsl_image* I = lightfield[nview];

        float *buffer_f = new float[ w*h*config_data->_nchannels ];

        for ( int n = 0 ; n < config_data->_nchannels ; ++n ) {

            gsl_matrix *channel = gsl_image_get_channel( I, (gsl_image_channel)n );

            for ( size_t i = 0 ; i < w*h ; ++i ) {
                buffer_f[w*h*n + i] = (float)channel->data[i];
            }
        }

        // read the camera matrix of the current view
        PinholeCamera v_i_cam(import_cam( config_data->_cam_name, s ));
        //v_i_cam.display();

        vector<float> depth(w*h, 0.0);
        vector<float> tauPartialX(w*h, 0.0);
        vector<float> tauPartialY(w*h, 0.0);
        vector<float> sigmaZ(w*h, 0.0);
        vector<float> norm(w*h, 0.0);

        // for each 3D point
        for ( size_t n = 0 ; n < vec_points.size()/3 ; ++n ) {

            bool visible = visibility[n].empty();
            for( size_t i = 0 ; i < visibility[n].size() ; ++i ) {

                visible = s == visibility[n][i];
                if( visible ) break;
            }
            if( !visible ) {

                continue;
            }

            glm::vec3 X(vec_points[3*n+0], vec_points[3*n+1], vec_points[3*n+2]);
            glm::vec3 N(vec_normals[3*n+0], vec_normals[3*n+1], vec_normals[3*n+2]);
            N = N / glm::l2Norm(N);

            glm::vec3 x = v_i_cam._K * ( v_i_cam._R * X + v_i_cam._t);
            glm::mat3 K_inv(1.0);
            K_inv[0][0] = 1/v_i_cam._K[0][0]; K_inv[1][1] = 1/v_i_cam._K[1][1];
            K_inv[2][0] = -v_i_cam._K[2][0]/v_i_cam._K[0][0]; K_inv[2][1] = -v_i_cam._K[2][1]/v_i_cam._K[1][1];

            //            // projection of a sphere on the camera sensor plane
            //            Matrix<double, 2, 3> Je_i;
            //            Je_i(0, 0) = 1; Je_i(0, 1) = 0; Je_i(0, 2) = -x(0)/x(2);
            //            Je_i(1, 0) = 0; Je_i(1, 1) = 1; Je_i(1, 2) = -x(1)/x(2);
            //            Je_i /= x(2);
            //            Je_i = Je_i * v_i_cam._K * v_i_cam._R;

//            Matrix<double, 3, 3> CXX = Matrix<double, 3, 3>::Identity();
            glm::mat3 CXX = compute_3D_covariance( config_data, X, visibility[n] );

            //            Matrix<double, 2, 2> C = Je_i * CXX * Je_i.transpose();
            //            C = C.inverse().eval();

            glm::mat3 Cdd = 0.1f*glm::mat3(1.0);
            glm::mat3 Cdd_inv = glm::inverse(Cdd);

            int k = 10;
            // round(x-0.5) = floor(x-0.5+0.5) = floor(x)
            int xp = int(x[0]/x[2]);
            int yp = int(x[1]/x[2]);
            int p = xp + yp*w;

            for ( int i = 0 ; i < k ; ++i ) {
                for ( int j = 0 ; j < k ; ++j ) {
                    int mx = xp - k/2 + j;
                    int my = yp - k/2 + i;
                    if ( 0 <= mx && mx <= (int)(w-1) &&
                         0 <= my && my <= (int)(h-1) ) {

                        // compute the affine projection of X parallel to the patch of normal N
                        glm::vec3 xm(mx+0.5, my+0.5, 1);
                        glm::vec3 r = glm::transpose(v_i_cam._R) * K_inv * xm; // world coordinates
                        glm::vec3 Xr = v_i_cam._C + (glm::dot(N, X - v_i_cam._C) / glm::dot(N, r)) * r; // world coordinates
                        //                        if ( N.dot(r)/r.norm()*N.dot(r)/r.norm() < 0.000000001 ) {
                        //                            continue;
                        //                        }
                        float F = exp( - glm::dot(Xr - X, Cdd_inv*(Xr - X) ));
                        //                        TRACE( "N :" << N << endl <<
                        //                               "X :" << X << endl <<
                        //                               "xm: " << xm << endl <<
                        //                               "r: " << r << endl <<
                        //                               "Xr: " << Xr << endl );

                        int m = mx + my*w;
////                        Vector2d dm(mx - xp, my - yp);

                        // luminance
                        float Ym = 0.2126 * buffer_f[m + 0] + 0.7152 * buffer_f[m + config_data->_nchannels] + 0.0722 * buffer_f[m + 2*config_data->_nchannels];
                        float Yp = 0.2126 * buffer_f[p + 0] + 0.7152 * buffer_f[p + config_data->_nchannels] + 0.0722 * buffer_f[p + 2*config_data->_nchannels];

                        //float F = exp( - dm.transpose()*C*dm );
                        float G = exp( - (Ym - Yp)*(Ym - Yp) / (2*config_data->_sigma_sensor*config_data->_sigma_sensor) );
                        float H = exp( - 10*(x[2] - 20) );

                        //                        TRACE("F: " << F << endl);
                        //                        TRACE("G: " << G << endl);
                        //                        TRACE("H: " << H << endl);

                        Xr = v_i_cam._R * Xr + v_i_cam._t; // cam coordinates

                        if ( F > -1 ) {
                            depth[m] += F * G * H * Xr[2];
                            norm[m] += F * G * H;
                            sigmaZ[m] += (F * G * H)*(F * G * H)*glm::dot(N, CXX*N)/(glm::dot(N, r)*glm::dot(N, r));
                        }
                    }
                }
            }

            //            break;
        }

        for ( size_t m = 0 ; m < w*h ; ++m ) {

            sigmaZ[m] = sqrt(sigmaZ[m]);

            if ( norm[m] > 0 ) {
                depth[m] /= norm[m];
                sigmaZ[m] /= norm[m];
            } else {
                //depth[m] = 20;
            }
        }

        delete[] buffer_f;

        // Compute the partial derivatives of tau with respect to estimated depth

        for ( size_t m = 0 ; m < w*h ; ++m ) { // for each pixel of the depth map

            tauPartialX[m] = 0;
            tauPartialY[m] = 0;

            if ( depth[m] != 0 ) {

                // homogeneous image point in u
                glm::vec3 xh = u_cam._K * (u_cam._R * ((float)depth[m] * glm::transpose(v_i_cam._R) * glm::inverse(v_i_cam._K) * glm::vec3(m%w + 0.5f, m/w + 0.5f, 1.0f) + v_i_cam._C) + u_cam._t);
                glm::vec3 temp_vec = u_cam._K * u_cam._R * glm::transpose(v_i_cam._R) * glm::inverse(v_i_cam._K) * glm::vec3(m%w + 0.5, m/w + 0.5, 1);

                // Jacobian matrix of the euclidean normalization
                glm::mat3x2 Je;
                Je[0][0] = 1; Je[1][0] = 0; Je[2][0] = -xh[0]/xh[2];
                Je[0][1] = 0; Je[1][1] = 1; Je[2][1] = -xh[1]/xh[2];
                Je /= xh[2];

                glm::vec2 res = Je * temp_vec;

                tauPartialX[m] = res[0];
                tauPartialY[m] = res[1];
            }
        }

        write_pfm_image( w, h, sigmaZ.data(), tauPartialX.data(), tauPartialY.data(), config_data->_dpart_name, s );
//        write_pfm_image( w, h, sigmaZ.data(), "./in/blender/skull/sigmaZ_%02i.pfm", s );
        write_pfm_image( w, h, depth.data(), config_data->_depth_name, s );
        //save_exr( w, h, depth, config_data->_depth_name, s );
        ++nview;
    }
    TRACE("...done!" << endl);
}

// vertically flip the image because pfm (0, 0) is at bottom left
template<typename T>
void reverse_buffer(vector<T> &ptr, size_t w, size_t h, int depth) {

    for ( size_t i = 0 ; i < h/2 ; ++i ) {
        swap_ranges(ptr.begin() + i*w*depth, ptr.begin()+ i*w*depth + w*depth, ptr.end() - w*depth - i*w*depth);
    }
}

// write double array to image file, three channels
template<typename T>
void write_pfm_image( size_t W, size_t H, const T *r, const T *g, const T *b, const string &spattern, int hint, bool reverse ) {

    char str[500];
    sprintf( str, spattern.c_str(), hint );
    FILE *const nfile = fopen(str, "wb");

    size_t N = W*H;
    assert( N > 0 );
    vector<float> buffer_r(N);
    vector<float> buffer_g(N);
    vector<float> buffer_b(N);

    for(size_t i=0; i<N; ++i) {

        buffer_r[i] = float(r[i]); // find a way to write double
        buffer_g[i] = float(g[i]);
        buffer_b[i] = float(b[i]);
    }

    // reverse buffer:
    // cuda image 0,0 is at the TOP left corner of the image
    // pfm format has the 0,0 at the BOTTOM left corner of the image
    // lines must be swapped
    if ( reverse ) {
        reverse_buffer( buffer_r, W, H, 1 );
        reverse_buffer( buffer_g, W, H, 1 );
        reverse_buffer( buffer_b, W, H, 1 );
    }

    fprintf(nfile, "P%c\n%lu %lu\n%d.0\n", 'F', W, H, endianness() ? 1 : -1);
    for(size_t i=0; i<N; ++i) {

        fwrite(&(buffer_r[i]), sizeof(float), 1, nfile);
        fwrite(&(buffer_g[i]), sizeof(float), 1, nfile);
        fwrite(&(buffer_b[i]), sizeof(float), 1, nfile);
    }

    fclose(nfile);
}

// write double array to image file, one channel
template<typename T>
void write_pfm_image( size_t W, size_t H, const T *image, const string &spattern, int hint, bool reverse ) {

    char str[500];
    sprintf( str, spattern.c_str(), hint );
    FILE *const nfile = fopen(str, "wb");

    size_t N = W*H;
    assert( N > 0 );
    vector<float> buffer(N);

    for(size_t i=0; i<N; ++i) {

        buffer[i] = float(image[i]); // find a way to write double
    }

    // reverse buffer:
    // cuda image 0,0 is at the TOP left corner of the image
    // pfm format has the 0,0 at the BOTTOM left corner of the image
    // lines must be swapped
    if ( reverse ) {
        reverse_buffer( buffer, W, H, 1 );
    }

    fprintf(nfile, "P%c\n%lu %lu\n%d.0\n", 'f', W, H, endianness() ? 1 : -1);
    fwrite(buffer.data(), sizeof(float), N, nfile);

    fclose(nfile);
}

