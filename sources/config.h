#ifndef __CONFIG_H
#define __CONFIG_H

#include <vector>
#include <string>

namespace coco {
class config;
}

class Config_data {

public:

    Config_data();

    // Parse configuration data
    void parse_config( coco::config &cfg );

    int _unitTest;

    size_t _nviews; // number of input views
    int _s_min; // index of first view
    int _s_max; // index of last view
    int _s_rmv; // view to remove
    int _t_min; // index of first view
    int _t_max; // index of last view
    int _t_rmv; // view to remove

    // Compute warps from mesh (mve scene)

    std::string _mve_name;

    std::string _imageName;
    std::string _cameraName;

    std::string _flowAlg;

    // lightfield name (complete path)
    std::string _lf_name;
    // starting image name (complete path)
    std::string _init_name;
    // tau warp name (complete path)
    std::string _tau_name;
    // tau partial name (complete path)
    std::string _dpart_name;
    // warped target u name (complete path)
    std::string _uWarped_name;
    // depth map of target view (complete path)
    std::string _uDepth_name;

    //output depth
    // scale is depth map downsampling factor (0: original, 1: ds x2, 2: ds x4, etc)
    // we take depth maps of scale in [scale_min ; scale_max]
    int _scale_min;
    int _scale_max;
    std::string _depth_name; // depth map (complete path) (optional)
    std::string _depthFromPC_name; // depth map from point cloud (complete path) (optional)
    std::string _depthFromMVE_name; // depth map from MVE, low resolution (complete path) (optional)
    std::string _depthSuper_name; // bilateraly filtered and resolution enhanced depth (complete path) (optional)

    int _pyramidHeight;

    // camera matrix (complete path) (optional)
    std::string _cam_name;
    // ply file path, (from PMVS) (complete path) (optional)
    std::string _ply_name;
    // patch file path, (from PMVS) (complete path) (optional)
    std::string _patch_name;
    // output directory
    std::string _outdir;
    // output file
    std::string _outfile;

    // input view downscale factor
    int _dsf;
    // input view downscaled resolution
    unsigned int _w;
    unsigned int _h;
    // output view resolution
    unsigned int _W;
    unsigned int _H;
    // window dimensions (croped images)
    int _windowW1, _windowW2, _windowH1, _windowH2;

    int _nchannels; // only rgb is handled for now

    // sigma of the sensor noise, in the same units as _image
    // could be moved to view_data and different for each input image
    float _sigma_sensor;

    // threshold for the u gradient.
    // Values bigger than this threshold will be set to the threshold
    // This is to avoid too low weights
    float _ugrad_threshold;

    // Smoothness parameter
    float _lambda;
    // Smoothness multiplicator for adaptive algorithms
    float _lambda_max_factor;

    // 0: Wanner's deformation weights
    // 1: Sergi's deformation weights
    // 2: Nieto's deformation weights
    int _dw_type;

    // 0: no weights for gradient term
    // 1: Sergi's weights for gradient term
    // 2: Nieto's weights for gradient term
    int _gw_type;

    // Dataterm parameters. alpha: intensity term. beta: gradient term.
    float _dt_alpha;
    float _dt_beta;

    float _gradient_step;

    // Number of inner iterations (fista)
    int _niter;
    // Number of meta-iterations (reweighted optimization)
    int _nmeta_iter;
};

#endif
