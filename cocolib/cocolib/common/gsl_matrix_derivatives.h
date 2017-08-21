/** \file gsl_matrix_derivatives.h

    File imported from "common" lib, use if this library is not available.
    Uses "gov" namespace (Graphics-optics-vision, MPI).

    Compute numerical derivatices of matrices using finite differences.
    For all functions, h is the grid cell size.
    
    Copyright (C) 2008 Bastian Goldluecke,
    <first name>AT<last name>.net

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

#ifndef __GOV_GSL_MATRIX_DERIVATIVES_H
#define __GOV_GSL_MATRIX_DERIVATIVES_H

#include "gsl_matrix_helper.h"
#include "linalg3d.h"

namespace coco {

  // Derivative in x-direction
  bool gsl_matrix_dx( double hx, const gsl_matrix *in, gsl_matrix *out );
  // Derivative in x-direction, improved rotation invariance (3x3 filter)
  bool gsl_matrix_dx_roi( const gsl_matrix *u, gsl_matrix *ux );

  // Derivative in y-direction
  bool gsl_matrix_dy( double hy, const gsl_matrix *in, gsl_matrix *out );
  // Derivative in y-direction, improved rotation invariance (3x3 filter)
  bool gsl_matrix_dy_roi( const gsl_matrix *u, gsl_matrix *uy );

  // Derivative in x-direction, forward differences, Neumann
  bool gsl_matrix_dx_forward( double hx, const gsl_matrix *in, gsl_matrix *out );

  // Derivative in y-direction, forward differences, Neumann
  bool gsl_matrix_dy_forward( double hy, const gsl_matrix *in, gsl_matrix *out );

  // Derivative in x-direction, backward differences (Dirichlet)
  bool gsl_matrix_dx_backward( double hx, const gsl_matrix *in, gsl_matrix *out );

  // Derivative in y-direction, backward differencing (Dirichlet)
  bool gsl_matrix_dy_backward( double hy, const gsl_matrix *in, gsl_matrix *out );

  // Normalize gradient
  bool gsl_matrix_norm_grad( gsl_matrix *dx, gsl_matrix *dy, gsl_matrix *norm );

  // Laplacian directly
  bool gsl_matrix_laplacian( double hx, double hy, const gsl_matrix *in, gsl_matrix *out );

  // Divergence of a vector field
  bool gsl_matrix_divergence( double hx, double hy,
			      const gsl_matrix *dx, const gsl_matrix *dy,
			      gsl_matrix *out );

  // Compute gradient at cell boundaries for non-negativity scheme
  // Dirichlet boundary conditions.
  void gsl_matrix_gradient_halfway_x_dx( double hx, const gsl_matrix *u, gsl_matrix *dx_hx );
  void gsl_matrix_gradient_halfway_x_dy( double hy, const gsl_matrix *u, gsl_matrix *dx_hy );
  void gsl_matrix_gradient_halfway_y_dx( double hx, const gsl_matrix *u, gsl_matrix *dy_hx );
  void gsl_matrix_gradient_halfway_y_dy( double hy, const gsl_matrix *u, gsl_matrix *dy_hy );
  // Neumann boundary conditions.
  void gsl_matrix_gradient_halfway_x_dx_neumann( double hx, const gsl_matrix *u, gsl_matrix *dx_hx );
  void gsl_matrix_gradient_halfway_x_dy_neumann( double hy, const gsl_matrix *u, gsl_matrix *dx_hy );
  void gsl_matrix_gradient_halfway_y_dx_neumann( double hx, const gsl_matrix *u, gsl_matrix *dy_hx );
  void gsl_matrix_gradient_halfway_y_dy_neumann( double hy, const gsl_matrix *u, gsl_matrix *dy_hy );
    
  // Computes Euler-Lagrange equation for total variation term
  // Neumann boundary conditions
  double gsl_matrix_tv_derivative( double hx, double hy, const gsl_matrix *u, gsl_matrix *td );

  // Compute Eigenvectors / values of a 2x2 matrix
  int gsl_matrix_eigensystem( Mat22d &M, Mat22d &E );

  // Compute square root of a (symmetric, positive semi-definite) matrix
  bool gsl_matrix_sqrt( const Mat22d &M, Mat22d &R );

  // Multichannel structure tensor computation
  bool gsl_matrix_multichannel_structure_tensor( const std::vector<gsl_matrix*> &U,
						 float tau,
						 gsl_matrix *edgedir_x, gsl_matrix *edgedir_y,
						 gsl_matrix *ev_max, gsl_matrix *ev_min );


  // Diffusion tensor for reaction-diffusion schemes
  typedef Vec2d( *diffusion_tensor )( double dx, double dy, const std::vector<double> &params );

  // Diffusion tensor which applies PM-Diffusivity to
  // Eigenvalues of structure tensor
  Vec2d diffusion_perona_malik_anisotropic( double dx, double dy, const std::vector<double> &params );


  // Diffusion tensor which applies PM-Diffusivity to gradient squared
  Vec2d diffusion_perona_malik_isotropic( double dx, double dy, const std::vector<double> &params );


  // Diffusion tensor which applies TV-Diffusivity (essentially leading to TV-EL equation)
  Vec2d diffusion_tv_isotropic( double dx, double dy, const std::vector<double> &params );



  // Compute the structure tensor coefficients
  bool gsl_matrix_structure_tensor( const gsl_matrix *dx, const gsl_matrix *dy,
				    gsl_matrix *a, gsl_matrix *b, gsl_matrix *c );

  // Computes diffusion tensor for Perona-Malik anisotropic or Weickert coherence enhancing flow
  bool gsl_matrix_diffusion_tensor_perona_malik( gsl_matrix *a, gsl_matrix *b, gsl_matrix *c,
						 const std::vector<double> &params,
						 gsl_matrix* edgedir_x=NULL, gsl_matrix *edgedir_y = NULL,
						 gsl_matrix *ev_max=NULL, gsl_matrix *ev_min=NULL );

  //
  // Computes anisotropic diffusion flow field
  // Neumann boundary conditions
  //
  // See
  // PDE-Based Deconvolution with Forward-Backward Diffusivities and Diffusion Tensors
  // Brox/Weickert et al
  //
  double gsl_matrix_anisotropic_diffusion_flow( double hx, double hy,
						const gsl_matrix *u,
						diffusion_tensor D,
						const std::vector<double> &params,
						gsl_matrix *flow );



  // Anisotropic diffusion, improved rotation invariance
  double gsl_matrix_anisotropic_diffusion_flow_roi( double hx, double hy,
						    const gsl_matrix *u,
						    gsl_matrix *flow );
  


  // Anisotropic diffusion, improved rotation invariance. Multichannel version.
  double gsl_matrix_anisotropic_diffusion_flow_roi_multichannel( double hx, double hy,
								 const std::vector<gsl_matrix*> &U,
								 std::vector<gsl_matrix*> &F,
								 const std::vector<double> &params,
								 gsl_matrix *lambda = NULL,
								 gsl_matrix* edgedir_x=NULL, gsl_matrix *edgedir_y = NULL,
								 gsl_matrix* ev_max=NULL, gsl_matrix *ev_min = NULL );

}


#endif
