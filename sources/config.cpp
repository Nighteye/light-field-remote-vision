#include <cocolib/cocolib/common/parse_config.h>
#include <cocolib/cocolib/common/debug.h>
#include <string>

#include "config.h"

using namespace coco;
using namespace std;

// Set data configuration to default values
Config_data::Config_data() {

    _nviews = 0;
    _s_min = -1;
    _s_max = -1;
    _s_rmv = -1;
    _t_min = 1;
    _t_max = 1;
    _t_rmv = -1;
    _mve_name = "";
    _cameraName = "";
    _imageName = "";
    _flowAlg = "";
    _computeFlow = 0;
    _lf_name = "";
    _init_name = "";
    _tau_name = "";
    _dpart_name = "";
    _uWarped_name = "";
    _uDepth_name = "";
    _scale_min = -1;
    _scale_max = -1;
    _depth_name = "";
    _depthFromPC_name = "";
    _depthFromMVE_name = "";
    _depthSuper_name = "";
    _pyramidHeight = 0;
    _cam_name = "";
    _ply_name = "";
    _patch_name = "";
    _outdir = "";
    _outfile = "";
    _dsf = 0;
    _w = 0;
    _h = 0;
    _W = 0;
    _H = 0;
    _windowW1 = 0;
    _windowW2 = 0;
    _windowH1 = 0;
    _windowH2 = 0;
    _nchannels = 0;
    _sigma_sensor = 1./255.0; // default for 8-bit normalized
    _ugrad_threshold = 1.0;
    _lambda = 0.0;
    _lambda_max_factor = 0.0;
    _dw_type = 0;
    _gw_type = 0;
    _dt_alpha = 1.0;
    _dt_beta = 0.0;
    _niter = 0;
    _nmeta_iter = 1;
    _gradient_step = 0.1;
}

// Parse configuration data
void Config_data::parse_config( coco::config &cfg ) {

    // directory and light field
    cfg.get_switch( "outdir", _outdir );
    Directory::create( _outdir );
    size_t i = _outdir.rfind( "/" );
    if ( i == string::npos || i != _outdir.length() - 1 ) {
        _outdir += "/";
    }
    cfg.get_switch( "outfile", _outfile );
    _outfile = _outdir + _outfile;

    cfg.get_switch( "s_min", _s_min );
    cfg.get_switch( "s_max", _s_max );
    cfg.get_switch( "s_rmv", _s_rmv );
    cfg.get_switch( "t_min", _t_min );
    cfg.get_switch( "t_max", _t_max );
    cfg.get_switch( "t_rmv", _t_rmv );
    cfg.get_switch( "mve_name", _mve_name );
    cfg.get_switch( "cameraName", _cameraName );
    cfg.get_switch( "imageName", _imageName );
    cfg.get_switch( "flowAlg", _flowAlg );
    cfg.get_switch( "computeFlow", _computeFlow );
    cfg.get_switch( "lf_name", _lf_name );
    cfg.get_switch( "init_name", _init_name );
    cfg.get_switch( "tau_name", _tau_name );
    cfg.get_switch( "dpart_name", _dpart_name );
    cfg.get_switch( "uWarped_name", _uWarped_name );
    cfg.get_switch( "uDepth_name", _uDepth_name );
    cfg.get_switch( "scale_min", _scale_min );
    cfg.get_switch( "scale_max", _scale_max );
    cfg.get_switch( "depth_name", _depth_name );
    cfg.get_switch( "depthFromPC_name", _depthFromPC_name );
    cfg.get_switch( "depthFromMVE_name", _depthFromMVE_name );
    cfg.get_switch( "depthSuper_name", _depthSuper_name );
    cfg.get_switch( "pyramidHeight", _pyramidHeight );
    cfg.get_switch( "cam_name", _cam_name );
    cfg.get_switch( "ply_name", _ply_name );
    cfg.get_switch( "patch_name", _patch_name );
    cfg.get_switch( "outdir", _outdir );
    cfg.get_switch( "outfile", _outfile );
    cfg.get_switch( "dsf", _dsf );
    cfg.get_switch( "width", _w );
    cfg.get_switch( "height", _h );
    cfg.get_switch( "window_width_1", _windowW1 );
    cfg.get_switch( "window_width_2", _windowW2 );
    cfg.get_switch( "window_height_1", _windowH1 );
    cfg.get_switch( "window_height_2", _windowH2 );
    cfg.get_switch( "nchannels", _nchannels );
    assert( _nchannels == 3 ); // grayscale is not handled for the moment
    cfg.get_switch( "ugrad_threshold", _ugrad_threshold );
    cfg.get_switch( "sigma_sensor", _sigma_sensor );
    _sigma_sensor /= 255.0;
    cfg.get_switch( "lambda", _lambda );
    cfg.get_switch( "lambda_max_factor", _lambda_max_factor );
    cfg.get_switch( "dw_type", _dw_type );
    cfg.get_switch( "gw_type", _gw_type );
    cfg.get_switch( "dt_alpha", _dt_alpha );
    cfg.get_switch( "dt_beta", _dt_beta );
    cfg.get_switch( "niter", _niter );
    cfg.get_switch( "nmeta_iter", _nmeta_iter );
    cfg.get_switch( "gradient_step", _gradient_step );
}

