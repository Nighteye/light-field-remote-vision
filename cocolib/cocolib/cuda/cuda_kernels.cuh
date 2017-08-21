/* -*-c++-*- */
/** \file cuda_kernels.cuh

    Some standard CUDA kernels.

    Copyright (C) 2011 Bastian Goldluecke,
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


#ifdef CUDA_DOUBLE
typedef double cuflt;
#else
typedef float cuflt;
#endif


////////////////////////////////////////////////////////////////////////////////
// Array initialization
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_set_all_device( int W, int H, cuflt *dst, float value );

__global__ void cuda_set_all_bool_device( int W, int H, bool *dst, bool value );


////////////////////////////////////////////////////////////////////////////////
// Transpose matrix
// Grid layout given for target
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_transpose_device( int W, int H, cuflt *src, cuflt *dst );


////////////////////////////////////////////////////////////////////////////////
// Add first argument to second
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_add_to_device( int W, int H, cuflt *src, cuflt *dst );
__global__ void cuda_add_scalar_to_device( int W, int H, cuflt v, cuflt *dst );


////////////////////////////////////////////////////////////////////////////////
// Subtract first argument from second
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_subtract_from_device( int W, int H, const cuflt *src, cuflt *dst );


////////////////////////////////////////////////////////////////////////////////
// Divide first argument by second
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_divide_by_device( int W, int H, cuflt *src, cuflt *dst );


////////////////////////////////////////////////////////////////////////////////
// Multiply first argument with second
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_multiply_with_device( int W, int H, cuflt *dst, cuflt *src );


////////////////////////////////////////////////////////////////////////////////
// Multiply first argument with second, store in third
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_multiply_device( int W, int H, cuflt *m1, cuflt *m2, cuflt *r );


////////////////////////////////////////////////////////////////////////////////
// Add scaled first argument to second
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_add_scaled_to_device( int W, int H, const cuflt *src, cuflt t, cuflt *dst );
__global__ void cuda_add_scaled_to_device( int W, int H, const cuflt *src, cuflt *t, cuflt *dst );


////////////////////////////////////////////////////////////////////////////////
// Compute linear combination of two arguments
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_linear_combination_device( int W, int H,
						cuflt w0, cuflt *src0,
						cuflt w1, cuflt *src1,
						cuflt *dst );

////////////////////////////////////////////////////////////////////////////////
// Scale array by argument
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_scale_device( int W, int H, cuflt *dst, cuflt t );


////////////////////////////////////////////////////////////////////////////////
// Square array element-wise
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_square_device( int W, int H, cuflt *dst );


////////////////////////////////////////////////////////////////////////////////
// Arbitrary power element-wise
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_pow_device( int W, int H, cuflt *dst, cuflt exponent );


////////////////////////////////////////////////////////////////////////////////
// Pointwise absolute value
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_abs_device( int W, int H, cuflt *dst );


////////////////////////////////////////////////////////////////////////////////
// Truncate with value from below (max operation)
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_max_truncate_device( int W, int H, cuflt *dst, cuflt value );


////////////////////////////////////////////////////////////////////////////////
// Clamp to range element-wise
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_clamp_device( int W, int H, cuflt *dst, cuflt m, cuflt M );


////////////////////////////////////////////////////////////////////////////////
// Normalize element-wise (zero weight target is set to zero)
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_normalize_device( int W, int H,
				       float *target, const float *weight );


////////////////////////////////////////////////////////////////////////////////
// Threshold element-wise
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_threshold_device( int W, int H,
				       float *target, float threshold,
				       float min_val, float max_val );

/////////////////////////////////////////////////////////////
//  Standard derivative kernels
/////////////////////////////////////////////////////////////
__global__ void cuda_compute_gradient_device( int W, int H, 
					      cuflt *u,
					      cuflt *px, cuflt *py );

__global__ void cuda_compute_divergence_device( int W, 
						int H,
						cuflt *px, cuflt *py,
						cuflt *d );

/////////////////////////////////////////////////////////////
//  Norms
/////////////////////////////////////////////////////////////
__global__ void cuda_compute_norm_device( int W, int H,
					  cuflt *x, cuflt *y,
					  cuflt *norm);
__global__ void cuda_compute_norm_device( int W, int H,
					  cuflt *x, cuflt *y, cuflt *z,
					  cuflt *norm);
__global__ void cuda_compute_norm_device( int W, int H,
					  cuflt *x, cuflt *y,
					  cuflt *z, cuflt *w,
					  cuflt *norm);
__global__ void cuda_compute_norm_device( int W, int H,
					  cuflt *px1, cuflt *py1,
					  cuflt *px2, cuflt *py2,
					  cuflt *px3, cuflt *py3,
					  cuflt *norm);

/////////////////////////////////////////////////////////////
//  STANDARD ROF KERNELS
/////////////////////////////////////////////////////////////
__global__ void cuda_rof_primal_prox_step_device( int W, 
						  int H,
						  cuflt tau,
						  cuflt lambda,
						  cuflt *u,
						  cuflt *uq,
						  cuflt *f,
						  cuflt *px, cuflt *py );

__global__ void cuda_rof_primal_descent_step_device( int W, int H,
						     cuflt tau,
						     cuflt lambda,
						     cuflt *u,
						     cuflt *uq,
						     cuflt *f,
						     cuflt *px, cuflt *py );


__global__ void tv_l2_dual_step_device( int W, int H, float tstep,
					float *u,
					float *px, float *py );


__global__ void tv_primal_descent_step_device( int W, int H,
					       cuflt tau,
					       cuflt *u,
					       cuflt *v,
					       cuflt *px, cuflt *py );


/////////////////////////////////////////////////////////////
//  STANDARD LINEAR KERNELS
/////////////////////////////////////////////////////////////
__global__ void cuda_linear_primal_prox_step_device( int W, int H,
						     float tau,
						     float *u,
						     float *uq,
						     float *a,
						     float *px, float *py );



///////////////////////////////////////////////////////////////////////////////////////////
// Multi-channel reprojections
///////////////////////////////////////////////////////////////////////////////////////////

__global__ void cuda_reproject_to_unit_ball_1d( int W, int H, cuflt *p );
__global__ void cuda_reproject_to_unit_ball_2d( int W, int H, cuflt *px, cuflt *py );
__global__ void cuda_reproject_to_ball_3d( int W, int H,
					   cuflt r, cuflt *p1, cuflt *p2, cuflt *p3 );
__global__ void cuda_reproject_to_ball_2d( int W, int H,
					   cuflt r, cuflt *p1, cuflt *p2 );
__global__ void cuda_reproject_to_ball_1d( int W, int H,
					   cuflt r, cuflt *p1 );
__global__ void cuda_reproject_to_ball_2d( int W, int H, cuflt *g, cuflt *p1, cuflt *p2 );
__global__ void cuda_reproject_to_ellipse( int W, int H,
					   float nx, float ny, // main axis direction
					   float r,            // small axis scale
					   float *px, float *py );
__global__ void cuda_reproject_to_ellipse( int W, int H,
					   float nx, float ny, // main axis direction
					   float r,            // small axis scale
					   float *a,            // main axis length (variable)
					   float *px, float *py );
__global__ void cuda_compute_largest_singular_value_device( int W, int H,
							    cuflt *px1, cuflt *py1,
							    cuflt *px2, cuflt *py2,
							    cuflt *px3, cuflt *py3,
							    cuflt *lsv );


/////////////////////////////////////////////////////////////
//  Full primal-dual algorithm kernel (one full iteration)
//  Adaptive maximum step size
/////////////////////////////////////////////////////////////
__global__ void cuda_linear_dual_prox_device( int W, int H,
					      float lambda,
					      float *uq,
					      float *px, float *py );
__global__ void cuda_linear_dual_prox_weighted_device( int W, int H,
						       float *g,
						       float *uq,
						       float *px, float *py );
__global__ void cuda_linear_primal_prox_extragradient_device( int W, int H,
							      float *u,
							      float *uq,
							      float *a,
							      float *px, float *py );
__global__ void cuda_linear_primal_prox_weighted_extragradient_device( int W, int H,
								       float *g,
								       float *u,
								       float *uq,
								       float *a,
								       float *px, float *py );



///////////////////////////////////////////////////////////////////////////////////////////
// Interpolation and resampling
///////////////////////////////////////////////////////////////////////////////////////////

// Matrix upsampling
__global__ void cuda_upsample_matrix_device( int W, int H, // Hi-res size
					     int w, int h, // Lo-res size
					     float F,        // Scale factor
					     float *m,     // lo-res matrix
					     float *M );    // hi-res result


////////////////////////////////////////////////////////////////////////////////
// Copy inside mask region
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_masked_copy_to_device( int W, int H, cuflt *s, cuflt *mask, cuflt *r );


////////////////////////////////////////////////////////////////////////////////
// Compute weighted average of two channels
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_weighted_average_device( int W, int H, cuflt *s1, cuflt *s2, cuflt *mask, cuflt *r );




////////////////////////////////////////////////////////////////////////////////
// EIGENSYSTEMS
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_eigenvalues_symm_device( int W, int H,
					      float *a, float *b, float *c,
					      float *lmin, float *lmax );




////////////////////////////////////////////////////////////////////////////////
// SLICES OF IMAGE STACKS
////////////////////////////////////////////////////////////////////////////////

// Horizontal slice, size W x N at scanline y
// Attn: set block and grid size to accomodate W x N threads
__global__ void cuda_stack_slice_H_device( int W, int H, int N,
					   int y,
					   float *stack, float *slice );


// Vertical slice, size N x H at column x
// Attn: set block and grid size to accomodate N x H threads
__global__ void cuda_stack_slice_W_device( int W, int H, int N,
					   int x,
					   float *stack, float *slice );

