/** \file compute_array.h

    Structure for (vector) arrays allocated on GPU, many helper functions
    Simplifies standard allocation and data transfer operations

    Copyright (C) 2014 Bastian Goldluecke.

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

#ifndef __COCO_COMPUTE_ARRAY_H
#define __COCO_COMPUTE_ARRAY_H

#include "compute_grid.h"
#include "../../common/gsl_image.h"


namespace coco {

  /***************************************************************
  vector of 2D float arrays over a compute grid
  ****************************************************************/
  struct vector_valued_function_2D
  {
    // Construction and destruction
    vector_valued_function_2D();
    virtual ~vector_valued_function_2D();

    // Alloc memory on compute grid
    // given number of channels N, width W and height H
    virtual bool alloc( compute_grid *G, int N );
    // Release memory
    virtual bool free();

    // Debug check and output
    bool dump() const;

    // Access to compute buffer
    compute_buffer& buffer();
    const compute_buffer& buffer() const;
    // Access to individual channel
    compute_buffer &channel( int dim );
    const compute_buffer &channel( int dim ) const;

    // Check for compatibility (same dimension)
    bool equal_dim( const vector_valued_function_2D *U ) const;

    // Set all values in array to zero
    bool set_zero();

    // Copy from and to various sources (for convenience)
    // CPU to this array
    bool copy_from_cpu( const gsl_matrix* );
    bool copy_from_cpu( const std::vector<gsl_matrix*> & );
    bool copy_from_cpu( const gsl_image * );
    bool copy_from_cpu( const std::vector<float*> & );
    bool copy_from_cpu( const float ** );
    bool copy_from_cpu( const float * );
    bool copy_from_cpu( int channel, const float * );
    bool copy_from_cpu( const std::vector<double*> & );
    bool copy_from_cpu( const double ** );
    bool copy_from_cpu( const double * );
    bool copy_from_cpu( int channel, const double * );

    // GPU to this array
    bool copy_from_gpu( const vector_valued_function_2D* );
    bool copy_from_gpu( const std::vector<compute_buffer*> & );
    bool copy_from_gpu( const compute_buffer* );
    
    // This array to CPU
    bool copy_to_cpu( std::vector<gsl_matrix*> & ) const;
    bool copy_to_cpu( gsl_image * ) const;
    bool copy_to_cpu( std::vector<float*> & ) const;
    bool copy_to_cpu( float ** ) const;
    bool copy_to_cpu( float * ) const;
    bool copy_to_cpu( int channel, float * ) const;
    bool copy_to_cpu( std::vector<double*> & ) const;
    bool copy_to_cpu( double ** ) const;
    bool copy_to_cpu( double * ) const;
    bool copy_to_cpu( int channel, double * ) const;
    bool copy_to_gpu( vector_valued_function_2D* ) const;

    // Simple queries
    int N() const;
    int W() const;
    int H() const;
    int bytes_per_dim() const;
    int total_bytes() const;
    compute_grid *grid() const;


    // Debugging
    bool trace_pixel( int x, int y ) const;

  private:
    // Internal data (for now, consecutive engine float buffers
    // - might change someday, so kept private)
    compute_buffer *_buffer;
    std::vector<compute_buffer*> _channels;
    int _N;
    compute_grid *_G;
    compute_engine *_CE;
    // CPU buffer memory for transfers, used temporarily during function calls
    float *_fbuffer;

    // Copying and assigning forbidden.
    // These are inherently inefficient, or, if implemented efficiently via pointer copies,
    // can easily lead to unintended side effects. Copy operations must always be initiated
    // explicitly.
    virtual vector_valued_function_2D &operator= ( const vector_valued_function_2D & );
    vector_valued_function_2D( const vector_valued_function_2D &V );
  };

};



#endif
