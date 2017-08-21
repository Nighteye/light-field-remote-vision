/** \file vectorial_multilabel.h
   Algorithms to solve the vectorial multilabel model
   with kD label space

   argmin_u J(u) + \int_\Omega \rho( u(x), x ) \dx

   where u:\Omega\to\Gamma\subset\R{k}, and
   Gamma is a rectangle.

   Copyright (C) 2012 Bastian Goldluecke,
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

#ifndef __COCO_VML_H
#define __COCO_VML_H

#include <map>
#include <vector>
#include <assert.h>

#include "../common/gsl_image.h"



namespace coco {


  // Maximum dimension
  enum {
    VML_MAX_DIM = 10,
  };

  // Regularizer types: 1 defined per label space dimension
  // (label space is separable)
  enum vml_regularizer {
    VML_POTTS,
    VML_LINEAR,
    VML_CYCLIC,
    VML_TRUNCATED_LINEAR,
    VML_COST_MATRIX,
    VML_MUMFORD_SHAH,
    VML_HUBER,
    VML_REG_N,
  };
  // Regularizer names
  extern const char* vml_regularizer_name[];
  // Get regularizer ID from name
  vml_regularizer vml_regularizer_id( const char *reg_name );

  // Holds all data for one dimension of the label space
  struct vml_dimension_data
  {
    // image size
    size_t _W;
    size_t _H;
    size_t _N;

    // number of labels
    size_t _G;
    // label <-> float translation
    std::vector<double> _values;
    // inverse label ordering (changes some signs)
    bool _order_invert;

    // regularizer type
    vml_regularizer _type;
    // regularizer weight
    double _lambda;

    // params
    // cutoff value for truncated linear
    double _t;
    // huber cutoff constant
    double _alpha;
    // penalty matrix
    gsl_matrix *_cost;

    // Workspace
    struct vml_dim_workspace *_w;
  };

  // Data structure for problem processing
  struct vml_data
  {
    // Image dimension
    size_t _W;
    size_t _H;
    // Label space dimension
    size_t _K;
    // Total label count
    size_t _G;

    // Data for each dimension
    vml_dimension_data** _dim;

    // Data term on CPU?
    bool _cpu_data_term;
    // Multiple chunks? If yes, this is the width per chunk in warps
    size_t _chunk_width_warps;
    // Number of inner iterations for primal prox projection
    size_t _inner_iterations;

    // List of all labels (as index array)
    std::vector<int*> _labels;

    // Workspace
    struct vml_workspace *_w;
  };


  /*****************************************************************************
       Dimension data creation / access
  *****************************************************************************/

  // Alloc data for different types of regularizers
  vml_dimension_data *vml_dimension_data_alloc_potts( size_t G, double lambda );
  // Alloc data for different types of regularizers
  vml_dimension_data *vml_dimension_data_alloc_linear( size_t G, double lambda,
						       double range_min=0.0, double range_max=1.0,
						       bool order_invert = false );
  // Alloc data for different types of regularizers
  vml_dimension_data *vml_dimension_data_alloc_cyclic( size_t G, double lambda,
						       double range_min=0.0, double range_max=1.0 );
  // Alloc data for different types of regularizers
  vml_dimension_data *vml_dimension_data_alloc_truncated_linear( size_t G, double lambda, double t,
								 double range_min=0.0, double range_max=1.0 );

  // Free data
  bool vml_dimension_data_free( vml_dimension_data *ddata );
  // Set label range
  bool vml_dimension_data_set_label_range( vml_dimension_data *ddata, double vmin, double vmax );


  /*****************************************************************************
       Workspace creation / access
  *****************************************************************************/

  // Alloc multilabel problem structure for vectorial multilabel model
  vml_data* vml_data_alloc( size_t W, size_t H, size_t K,
			    bool data_term_cpu = false,
			    size_t chunk_width_warps = 0 ); 

  // Set data for a single label space dimension (can be done only once)
  bool vml_init_dimension_data( vml_data *data, size_t g,
				vml_dimension_data *ddata );

  // Finalize initialization after all dimensions have been defined
  bool vml_alloc_finalize( vml_data *data );

  // Free multilabel problem structure
  bool vml_data_free( vml_data* data );

  // Get total mem usage
  size_t vml_total_mem( vml_data *data );

  // Init algorithm with zero solution
  bool vml_init( vml_data* data );

  // Set current solution data for a single label space dimension
  // if cpu_data_term is true, data term is stored memory efficiently on
  // CPU memory, but needs to be transferred to GPU in each iteration
  // (which is really slow)
  bool vml_set_data_term( vml_data* data, float *a );

  // Set data term to on-the-fly computation.
  // No normal data term is required then, except if one wants to 
  // compute the energies (a CPU data term is sufficient in that case)
  bool vml_set_data_term_on_the_fly_segmentation( vml_data* data, gsl_image *I );

  // Set current solution data for a single label space dimension
  bool vml_set_solution( vml_data* data,
			 size_t g, int *s );

  // Project current relaxed solution onto integer values
  bool vml_project_solution( vml_data *data );

  // Project current relaxed solution onto integer values
  // (separate for p and q, required for some bounds computations)
  bool vml_project_p( vml_data *D );
  bool vml_project_q( vml_data *D );
  bool vml_project_solution_separate_q( vml_data *D );

  // Get current solution of the relaxation
  bool vml_get_solution_relaxation( vml_data *D,
				    size_t k, int *ur );

  // Get current solution
  bool vml_get_solution( vml_data *data,
			 size_t g, int *s );

  // Compute current energy
  // Only valid after final iteration, see below.
  double vml_energy( vml_data *data, bool data_q=false );



  /*****************************************************************************
  Vectorial multilabel 2D algorithm: Strekalovskiy/Goldluecke/Cremers ICCV 2011
  *****************************************************************************/

  // Perform one multilabel iteration
  bool vml_iteration( vml_data *data );

  // Somewhat of a hack: the last iteration needs to do something slightly
  // different for energy computations to work. The reason is a highly optimized
  // iteration scheme to get rid of the need to store some extragradient variables.
  // NOTE: AFTER A CALL TO THIS FUNCTION, SUBSEQUENT ITERATIONS WILL RETURN INVALID
  // RESULTS.
  bool vml_final_iteration( vml_data *data );

}


#endif
