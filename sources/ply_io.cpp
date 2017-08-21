//
//  ply_io.cpp
//  openMVG
//
//  Created by Sergi Pujades on 05/01/15.
//
//

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>

#include "ply_io.h"
#include "../rply/rply.h"

using namespace std;

static int vertex_cb(p_ply_argument argument) {
    void *data;
    long index;
    ply_get_argument_user_data(argument, &data, &index);

    long instance_index;
    ply_get_argument_element(argument, NULL, &instance_index);

    ((float*)data)[instance_index*3+index] = ply_get_argument_value(argument);

    //printf("%ld %ld, %g\n", instance_index, index, ply_get_argument_value(argument));

    return 1;
}

static int face_cb(p_ply_argument argument) {
    void *data;
    ply_get_argument_user_data(argument, &data, NULL);

    long instance_index;
    ply_get_argument_element(argument, NULL, &instance_index);

    long length, value_index;
    ply_get_argument_property(argument, NULL, &length, &value_index);

    switch (value_index) {
    case 0:
    case 1:
    case 2:
        ((unsigned int*)data)[instance_index*length+value_index] = ply_get_argument_value(argument);
        break;
    default:
        // There is a callback for the size of the list
        break;
    }
    //printf("%ld %ld %ld, %g\n", instance_index, length, value_index, ply_get_argument_value(argument));

    return 1;
}

bool read_ply(const char *filename,
              std::vector< float > &vec_points,
              std::vector< float > &vec_normals,
              std::vector< unsigned int > &vec_triangles) {
    long nvertices, ntriangles;
    p_ply ply = ply_open(filename, NULL, 0, NULL);
    if (!ply) {
        return false;
    }
    if (!ply_read_header(ply)) {
        return false;
    }

    ntriangles = 0;
    nvertices = 0;

    // Alloc vector with correct lenght
    p_ply_element element = NULL;
    while ((element = ply_get_next_element(ply, element))) {
        //p_ply_property property = NULL;
        long ninstances = 0;
        const char *element_name;
        ply_get_element_info(element, &element_name, &ninstances);

        if (strcmp(element_name, "vertex") == 0) {
            vec_points.resize(3*ninstances);
            vec_normals.resize(3*ninstances);
            nvertices = ninstances;
        } else if (strcmp(element_name, "face") == 0) {
            vec_triangles.resize(3*ninstances);
            ntriangles = ninstances;
        } else {
        }
    }

    ply_set_read_cb(ply, "vertex", "x", vertex_cb, (void*)&(vec_points[0]), 0);
    ply_set_read_cb(ply, "vertex", "y", vertex_cb, (void*)&(vec_points[0]), 1);
    ply_set_read_cb(ply, "vertex", "z", vertex_cb, (void*)&(vec_points[0]), 2);
    ply_set_read_cb(ply, "vertex", "nx", vertex_cb, (void*)&(vec_normals[0]), 0);
    ply_set_read_cb(ply, "vertex", "ny", vertex_cb, (void*)&(vec_normals[0]), 1);
    ply_set_read_cb(ply, "vertex", "nz", vertex_cb, (void*)&(vec_normals[0]), 2);

    ply_set_read_cb(ply, "face", "vertex_indices", face_cb, (void*)&(vec_triangles[0]), 0);

    if (!ply_read(ply)) {
        return false;
    }

    ply_close(ply);

    printf("Reading %ld Vertices and %ld triangles\n in %s", nvertices, ntriangles, filename);

    /*for (int i=0; i<vec_points.size()/3; ++i) {
    printf("Point %g %g %g, with normal %g %g %g\n", vec_points[i*3], vec_points[i*3+1], vec_points[i*3+2],
           vec_normals[i*3], vec_normals[i*3+1], vec_normals[i*3+2]);
  }
  for (int i=0; i<vec_triangles.size()/3; ++i) {
    printf("Triangle %d %d %d\n", vec_triangles[i*3], vec_triangles[i*3+1], vec_triangles[i*3+2]);
  }*/

    return true;
}

bool writePly(const char *filename,
               const std::vector< std::vector<double> > &vec_points) {

    ofstream plyFile;
    plyFile.open(string(filename));
    plyFile << "ply" << std::endl;
    plyFile << "format ascii 1.0" << std::endl;
    plyFile << "element vertex " << vec_points.size() << std::endl;
    plyFile << "property float x" << std::endl;
    plyFile << "property float y" << std::endl;
    plyFile << "property float z" << std::endl;
    plyFile << "end_header" << std::endl;

    for(uint i = 0 ; i < vec_points.size() ; ++i) {

        plyFile << vec_points[i][0] << " " << vec_points[i][1] << " " << vec_points[i][2] << std::endl;
    }

    plyFile.close();

    return true;
}

bool writePly(const char *filename,
               const std::vector< std::vector<float> > &vec_points) {

    ofstream plyFile;
    plyFile.open(string(filename));
    plyFile << "ply" << std::endl;
    plyFile << "format ascii 1.0" << std::endl;
    plyFile << "element vertex " << vec_points.size() << std::endl;
    plyFile << "property float x" << std::endl;
    plyFile << "property float y" << std::endl;
    plyFile << "property float z" << std::endl;
    plyFile << "end_header" << std::endl;

    for(uint i = 0 ; i < vec_points.size() ; ++i) {

        plyFile << vec_points[i][0] << " " << vec_points[i][1] << " " << vec_points[i][2] << std::endl;
    }

    plyFile.close();

    return true;
}
