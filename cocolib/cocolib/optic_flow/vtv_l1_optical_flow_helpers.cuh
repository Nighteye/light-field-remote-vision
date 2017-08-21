/* -*-c++-*- */
/** \of_tv_l1_helpers.cuh

	optical flow detection
	Implemented from Zach/Pock,Bischof 2007,
	"A Duality Based Approach for Realtime TV-L1 Optical Flow"
	combined with vectorial total variation

    Copyright (C) 2012 Ole Johannsen,
    <first name>.<last name>ATberlin.de

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef OF_TV_L1_HELPERS_CUH_
#define OF_TV_L1_HELPERS_CUH_
namespace coco{

  void of_tv_l1_bilinear_interpolation( float *input, //image to be warped
					float *u,     //x component of the vector field
					float *v,     //y component of the vector field
					float *output,      //warped output image
					int nx,       //width of the image
					int ny,       //height of the image
					float *mask         //mask to detect the motions outside the image
					);

  void of_tv_l1_zoom_size( int nx,             //width of the orignal image
			   int ny,             //height of the orignal image
			   int *nxx,           //width of the zoomed image
			   int *nyy,           //height of the zoomed image
			   float factor  //zoom factor between 0 and 1
			   );

  bool of_tv_l1_zoom( float *I,    //input image
		      float *Iout,       //output image
		      int nx,            //image width
		      int ny,            //image height
		      float factor //zoom factor between 0 and 1
		      );

  __global__ void of_tv_l1_centered_gradient( float *input,  //input image
						    float *dx,           //computed x derivative
						    float *dy,           //computed y derivative
						    int nx,        //image width
						    int ny         //image height
						    );

  __global__ void of_tv_l1_calculate_rho_( float *I1w,
					   float *I0s,
					   float *I1wx,
					   float *I1wy,
					   float *grad,
					   float *rho_c,
					   float *u1,
					   float *u2,
					   int nx,
					   int ny
					   );

  __global__ void of_tv_l1_calculate_TH( float *I1wx,
					 float *I1wy,
					 float *grad,
					 float *rho_c,
					 float *u1,
					 float *u2,
					 float *v1,
					 float *v2,
					 float* mask,
					 int nx,
					 int ny,
					 float l_t
					 );

  __global__ void of_tv_l1_calculate_error( float *u1,
					    float *u2,
					    float *v1,
					    float *v2,
					    float *div_p1,
					    float *div_p2,
					    int nx,
					    int ny,
					    float theta,
					    float *error
					    );

  __global__ void of_tv_l1_calculate_rho_const( float* I1w,
						float* I0,
						float* I1wx,
						float* I1wy,
						float* grad,
						float* rho_c,
						float* u1,
						float* u2,
						int nx,
						int ny
						);

  __global__ void of_tv_l1_calculate_dual( float *u1x,
					   float *u1y,
					   float *u2x,
					   float *u2y,
					   float *p11,
					   float *p12,
					   float *p21,
					   float *p22,
					   int nx,
					   int ny,
					   float taut
					   );

  __global__ void of_tv_l1_zoom_out( float *I, //input image
				     float *Iout,    //output image
				     int nx,         //width of the original image
				     int ny,         //height of the original image
				     int nxx,        //width of the zoomed image
				     int nyy         //height of the zoomed image
				     );

  __global__ void of_tv_l1_multiply_by_float( float *I,
					      int nx,
					      int ny,
					      float factor
					      );

  __global__ void of_tv_l1_set_to_float( float *I,
					 int nx,
					 int ny,
					 float value
					 );

  __global__ void of_tv_l1_cpy( float *u,
				float *i,
				int nx,
				int ny
				);

  __global__ void of_vtv_l1_dual_step( int W, int H,
				       float tstep,
				       float *u,
				       float *px,
				       float *py
				       );

  __global__ void of_vtv_l1_calculate_dual_TVJ( float *u1x,
						float *u1y,
						float *u2x,
						float *u2y,
						float *p11,
						float *p12,
						float *p21,
						float *p22,
						int nx,
						int ny,
						float tau
						);

  __global__ void of_vtv_l1_calculate_dual_TVF(	float *u1x,
						float *u1y,
						float *u2x,
						float *u2y,
						float *p11,
						float *p12,
						float *p21,
						float *p22,
						int nx,
						int ny,
						float taut
						);
}

#endif /* OF_TV_L1_HELPERS_CUH_ */
