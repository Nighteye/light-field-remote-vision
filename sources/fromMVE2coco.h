#ifndef FROM_MVE_2_COCO_H
#define FROM_MVE_2_COCO_H

struct Config_data;
struct PinholeCamera;

// load the first image just to get its size
void getSize( Config_data *config_data );

// load the (undistorded) images from mve scene and export them in PNG format
void exportImages( Config_data *config_data, unsigned int s );

// Read the camera matrices from INI file (MVE format)
PinholeCamera importINICam( Config_data *config_data, unsigned int s );

// Read the ply file (reconstructed surface)
void importMesh( Config_data *config_data, std::vector< float > &vec_points,
                                           std::vector< float > &vec_normals,
                                           std::vector< unsigned int > &vec_triangles );

void fromMVE2coco( Config_data *config_data );

#endif /* #ifndef FROM_MVE_2_COCO_H */
