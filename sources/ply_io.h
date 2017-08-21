//
//  ply_io.h
//  openMVG
//
//  Created by Sergi Pujades on 05/01/15.
//
//

#ifndef __openMVG__ply_io__
#define __openMVG__ply_io__

#include <vector>

bool read_ply(const char *filename,
             std::vector< float > &vec_points,
             std::vector< float > &vec_normals,
             std::vector< unsigned int > &vec_triangles);

bool writePly(const char *filename,
              const std::vector< std::vector<double> > &vec_points);


bool writePly(const char *filename,
              const std::vector< std::vector<float> > &vec_points);

#endif /* defined(__openMVG__ply_io__) */

