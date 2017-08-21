#ifndef IMPORT_H
#define IMPORT_H

#include <glm/glm.hpp>
#include <vector>

#include <cocolib/cocolib/common/gsl_image.h>

class Config_data;

struct PinholeCamera;

// fill the visibility vector with the indices of the cameras that see the points
bool read_patches( const char* filename, std::vector< std::vector<int> > &visibility );

// load the first channel of an EXR image with OpenEXR
float* import_exr( int w, int h, std::string filename );

// write exr image
void save_exr( int w, int h, float* image, std::string filename, int hint );

// load low res images
std::vector<coco::gsl_image*> import_lightfield( Config_data *data );

// Read warps and weights from files
void import_warps( Config_data *config_data, coco::gsl_image** warps, std::string path );

// Read the input depth maps
double * import_depth(std::string path, size_t nview);

// Read the camera matrices
glm::mat4x3 import_cam( std::string path, size_t nview );

// Compute warps using depth map and camera matrices
void import_warps_from_depth( Config_data *data, coco::gsl_image** warps );

// compute the 3x3 covariance matrix of a 3D point
glm::mat3 compute_3D_covariance( Config_data *config_data, glm::vec3 X, std::vector<int> XVisibility );

// Compute warps using ply and camera matrices
void import_depth_from_ply( Config_data *config_data, std::vector<coco::gsl_image*> lightfield );

// vertically flip the image because pfm (0, 0) is at bottom left
template<typename T>
void reverse_buffer(std::vector<T> &ptr, size_t w, size_t h, int depth);

// write double array to image file, three channels
template<typename T>
void write_pfm_image( size_t W, size_t H, const T *r, const T *g, const T *b, const std::string &spattern, int hint, bool reverse = true );

// write double array to image file, one channel
template<typename T>
void write_pfm_image( size_t W, size_t H, const T *image, const std::string &spattern, int hint, bool reverse = true );

#endif /* #ifndef IMPORT_H */
