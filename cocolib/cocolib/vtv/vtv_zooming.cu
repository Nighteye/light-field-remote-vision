/* -*-c++-*- */
#include <iostream>
#include <algorithm>

#include "vtv.h"
#include "vtv.cuh"
#include "../common/debug.h"

// Init functional
bool coco::coco_vtv_zooming_init( coco_vtv_data *data, gsl_image *source )
{
  size_t fx = data->_W / source->_w;
  if ( fx * source->_w != data->_W ) {
    ERROR( "Zooming: target width is not a multiple of source width" << std::endl );
    return false;
  }
  size_t fy = data->_H / source->_h;
  if ( fy * source->_h != data->_H ) {
    ERROR( "Zooming: target height is not a multiple of source height" << std::endl );
    return false;
  }
  if ( fx != fy ) {
    ERROR( "VTV zooming currently only supported for equal x and y scaling." << std::endl );
    return false;
  }
  // Init underlying SR model
  coco_vtv_sr_init( data, 1, fx );
  // Init image as single view
  float *disparity = new float[ source->_w * source->_h ];
  memset( disparity,0, sizeof(float) * source->_w * source->_h );
  coco_vtv_sr_create_view( data, 0, 0.0, 0.0,
			   source, disparity );

  // Finalize
  delete[] disparity;
  coco_vtv_sr_end_view_creation( data );
  return true;
}

// Perform one full iteration of FISTA
bool coco::coco_vtv_zooming_iteration_fista( coco_vtv_data *data )
{
  return coco_vtv_sr_iteration_chambolle_pock_1( data );
}
