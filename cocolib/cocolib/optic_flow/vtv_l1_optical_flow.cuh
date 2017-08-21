/* -*-c++-*- */
/** \of_tv_l1.cuh

    Example code for COCOLIB: call solvers for the optical flow models
    with TV, VTV_J or VTV_F regularizer term.

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
namespace coco {

  struct of_tv_l1_workspace
  {
	float* *I0s; //scaled *im0
	float* *I1s; //scaled *im1
	float* *U1s; //scaled U1
	float* *U2s; //scaled U2
	float* *U1qs; //scaled U1
	float* *U2qs; //scaled U2
	int *Ws; //scaled width
	int *Hs; //scaled height
  };


  // Auxiliary functions
  bool of_tv_l1_step(of_tv_l1_data *data,int cs); //workspace + current scale
  void of_tv_l1_zoom_size(
  		int nx,             //width of the orignal image
  		int ny,             //height of the orignal image
  		int *nxx,           //width of the zoomed image
  		int *nyy,           //height of the zoomed image
  		float factor  //zoom factor between 0 and 1
  	      );
}
