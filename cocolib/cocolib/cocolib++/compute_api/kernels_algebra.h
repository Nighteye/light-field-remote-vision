/* -*-c++-*- */
/** \file kernels_basic.h

    Basic computations on grids

    Copyright (C) 2011-2014 Bastian Goldluecke,
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

#ifndef __COCO_COMPUTE_API_KERNELS_BASIC_H
#define __COCO_COMPUTE_API_KERNELS_BASIC_H

#include "compute_grid.h"

namespace coco {

  ////////////////////////////////////////////////////////////////////////////////
  // Grid initialization
  ////////////////////////////////////////////////////////////////////////////////
  void kernel_set_all( const compute_grid *G, compute_buffer &dst, const float value );

  ////////////////////////////////////////////////////////////////////////////////
  // Add first argument to second
  ////////////////////////////////////////////////////////////////////////////////
  void kernel_add_to( const compute_grid *G, const compute_buffer &src, compute_buffer &dst );
  void kernel_add_to( const compute_grid* G, const float v, compute_buffer &dst );

  ////////////////////////////////////////////////////////////////////////////////
  // Subtract first argument from second
  ////////////////////////////////////////////////////////////////////////////////
  void kernel_subtract_from( const compute_grid* G, const compute_buffer &src, compute_buffer &dst );

  ////////////////////////////////////////////////////////////////////////////////
  // Divide first argument by second
  ////////////////////////////////////////////////////////////////////////////////
  void kernel_divide_by( const compute_grid* G, compute_buffer &dst, const compute_buffer &src );

  ////////////////////////////////////////////////////////////////////////////////
  // Multiply first argument with second
  ////////////////////////////////////////////////////////////////////////////////
  void kernel_multiply_with( const compute_grid* G, compute_buffer &dst, const compute_buffer &src );

  ////////////////////////////////////////////////////////////////////////////////
  // Multiply first argument with second, store in third
  ////////////////////////////////////////////////////////////////////////////////
  void kernel_multiply( const compute_grid* G, const compute_buffer &m1, const compute_buffer &m2, compute_buffer &r );

  ////////////////////////////////////////////////////////////////////////////////
  // Add scaled first argument to second
  ////////////////////////////////////////////////////////////////////////////////
  void kernel_multiply_and_add_to( const compute_grid* G, const compute_buffer &src, const compute_buffer &t, compute_buffer &dst );

  ////////////////////////////////////////////////////////////////////////////////
  // Compute linear combination of two arguments
  ////////////////////////////////////////////////////////////////////////////////
  void kernel_linear_combination( const compute_grid* G,
				  const float w0, const compute_buffer &src0,
				  const float w1, const compute_buffer &src1,
				  compute_buffer &dst );

  ////////////////////////////////////////////////////////////////////////////////
  // Scale array by argument
  ////////////////////////////////////////////////////////////////////////////////
  void kernel_scale( const compute_grid* G, compute_buffer &dst, const float t );


  ////////////////////////////////////////////////////////////////////////////////
  // Square array element-wise
  ////////////////////////////////////////////////////////////////////////////////
  void kernel_square( const compute_grid* G, compute_buffer &dst );


  ////////////////////////////////////////////////////////////////////////////////
  // Arbitrary power element-wise
  ////////////////////////////////////////////////////////////////////////////////
  void kernel_pow( const compute_grid* G, compute_buffer &dst, float exponent );


  ////////////////////////////////////////////////////////////////////////////////
  // Pointwise absolute value
  ////////////////////////////////////////////////////////////////////////////////
  void kernel_abs( const compute_grid* G, compute_buffer &dst );
  

  ////////////////////////////////////////////////////////////////////////////////
  // Clamp to range element-wise
  ////////////////////////////////////////////////////////////////////////////////
  void kernel_clamp( const compute_grid* G, compute_buffer &dst, float m, float M );
  

  ////////////////////////////////////////////////////////////////////////////////
  // Threshold element-wise
  ////////////////////////////////////////////////////////////////////////////////
  void kernel_threshold( const compute_grid* G,
			 compute_buffer &target, float threshold,
			 float min_val, float max_val );
  

  /*
  /////////////////////////////////////////////////////////////
  //  Standard derivative kernels
  /////////////////////////////////////////////////////////////
  void kernel_compute_gradient( const compute_grid* G, 
				compute_buffer &u,
				compute_buffer &px, compute_buffer &py );

  /////////////////////////////////////////////////////////////
  //  Norms
  /////////////////////////////////////////////////////////////
  void kernel_compute_norm( const compute_grid* G,
			    compute_buffer &x, compute_buffer &y,
			    compute_buffer &norm);
  void kernel_compute_norm( const compute_grid* G,
			    compute_buffer &x, compute_buffer &y, compute_buffer &z,
			    compute_buffer &norm);
  void kernel_compute_norm( const compute_grid* G,
			    compute_buffer &x, compute_buffer &y,
			    compute_buffer &z, compute_buffer &w,
			    compute_buffer &norm);
  void kernel_compute_norm( const compute_grid* G,
			    compute_buffer &px1, compute_buffer &py1,
			    compute_buffer &px2, compute_buffer &py2,
			    compute_buffer &px3, compute_buffer &py3,
			    compute_buffer &norm);
  
  /////////////////////////////////////////////////////////////
  //  STANDARD ROF KERNELS
  /////////////////////////////////////////////////////////////
  void kernel_rof_primal_prox_step( int W, 
				    int H,
				    float tau,
				    float lambda,
				    compute_buffer &u,
				    compute_buffer &uq,
				    compute_buffer &f,
				    compute_buffer &px, compute_buffer &py );
  
  void kernel_rof_primal_descent_step( const compute_grid* G,
				       float tau,
				       float lambda,
				       compute_buffer &u,
				       compute_buffer &uq,
				       compute_buffer &f,
				       compute_buffer &px, compute_buffer &py );


  void tv_l2_dual_step( const compute_grid* G, float tstep,
			compute_buffer &u,
			compute_buffer &px, compute_buffer &py );
  
  
  void tv_primal_descent_step( const compute_grid* G,
			       float tau,
			       compute_buffer &u,
			       compute_buffer &v,
			       compute_buffer &px, compute_buffer &py );
  
  
  /////////////////////////////////////////////////////////////
  //  STANDARD LINEAR KERNELS
  /////////////////////////////////////////////////////////////
  void kernel_linear_primal_prox_step( const compute_grid* G,
				       float tau,
				       compute_buffer &u,
				       compute_buffer &uq,
				       compute_buffer &a,
				       compute_buffer &px, compute_buffer &py );



  ///////////////////////////////////////////////////////////////////////////////////////////
  // Multi-channel reprojections
  ///////////////////////////////////////////////////////////////////////////////////////////
  
  void kernel_reproject_to_unit_ball_1d( const compute_grid* G, compute_buffer &p );
  void kernel_reproject_to_unit_ball_2d( const compute_grid* G, compute_buffer &px, compute_buffer &py );
  void kernel_reproject_to_ball_3d( const compute_grid* G,
				    float r, compute_buffer &p1, compute_buffer &p2, compute_buffer &p3 );
  void kernel_reproject_to_ball_2d( const compute_grid* G,
				    float r, compute_buffer &p1, compute_buffer &p2 );
  void kernel_reproject_to_ball_1d( const compute_grid* G,
				    float r, compute_buffer &p1 );
  void kernel_reproject_to_ball_2d( const compute_grid* G, compute_buffer &g, compute_buffer &p1, compute_buffer &p2 );
  void kernel_reproject_to_ellipse( const compute_grid* G,
				    float nx, float ny, // main axis direction
				    float r,            // small axis scale
				    compute_buffer &px, compute_buffer &py );
  void kernel_reproject_to_ellipse( const compute_grid* G,
				    float nx, float ny, // main axis direction
				    float r,            // small axis scale
				    compute_buffer &a,            // main axis length (variable)
				    compute_buffer &px, compute_buffer &py );
  void kernel_compute_largest_singular_value( const compute_grid* G,
					      compute_buffer &px1, compute_buffer &py1,
					      compute_buffer &px2, compute_buffer &py2,
					      compute_buffer &px3, compute_buffer &py3,
					      compute_buffer &lsv );
  

  /////////////////////////////////////////////////////////////
  //  Full primal-dual algorithm kernel (one full iteration)
  //  Adaptive maximum step size
  /////////////////////////////////////////////////////////////
  void kernel_linear_dual_prox( const compute_grid* G,
				float lambda,
				compute_buffer &uq,
				compute_buffer &px, compute_buffer &py );
  void kernel_linear_dual_prox_weighted( const compute_grid* G,
					 compute_buffer &g,
					 compute_buffer &uq,
					 compute_buffer &px, compute_buffer &py );
  void kernel_linear_primal_prox_extragradient( const compute_grid* G,
						compute_buffer &u,
						compute_buffer &uq,
						compute_buffer &a,
						compute_buffer &px, compute_buffer &py );
  void kernel_linear_primal_prox_weighted_extragradient( const compute_grid* G,
							 compute_buffer &g,
							 compute_buffer &u,
							 compute_buffer &uq,
							 compute_buffer &a,
							 compute_buffer &px, compute_buffer &py );



  ///////////////////////////////////////////////////////////////////////////////////////////
  // Interpolation and resampling
  ///////////////////////////////////////////////////////////////////////////////////////////
  
  // Matrix upsampling
  void kernel_upsample_matrix( const compute_grid* G, // Hi-res size
			       int w, int h, // Lo-res size
			       float F,        // Scale factor
			       compute_buffer &m,     // lo-res matrix
			       compute_buffer &M );    // hi-res result
  

  ////////////////////////////////////////////////////////////////////////////////
  // Copy inside mask region
  ////////////////////////////////////////////////////////////////////////////////
  void kernel_masked_copy_to( const compute_grid* G, compute_buffer &s, compute_buffer &mask, compute_buffer &r );
  

  ////////////////////////////////////////////////////////////////////////////////
  // Compute weighted average of two channels
  ////////////////////////////////////////////////////////////////////////////////
  void kernel_weighted_average( const compute_grid* G, compute_buffer &s1, compute_buffer &s2, compute_buffer &mask, compute_buffer &r );
  



  ////////////////////////////////////////////////////////////////////////////////
  // EIGENSYSTEMS
  ////////////////////////////////////////////////////////////////////////////////
  void kernel_eigenvalues_symm( const compute_grid* G,
				compute_buffer &a, compute_buffer &b, compute_buffer &c,
				compute_buffer &lmin, compute_buffer &lmax );




  ////////////////////////////////////////////////////////////////////////////////
  // SLICES OF IMAGE STACKS
  ////////////////////////////////////////////////////////////////////////////////
  
  // Horizontal slice, size W x N at scanline y
  // Attn: set block and grid size to accomodate W x N threads
  void kernel_stack_slice_H( const compute_grid* G, int N,
			     int y,
			     compute_buffer &stack, compute_buffer &slice );
  
  
  // Vertical slice, size N x H at column x
  // Attn: set block and grid size to accomodate N x H threads
  void kernel_stack_slice_W( const compute_grid* G, int N,
			     int x,
			     compute_buffer &stack, compute_buffer &slice );
  */

 

} // namespace
#endif

