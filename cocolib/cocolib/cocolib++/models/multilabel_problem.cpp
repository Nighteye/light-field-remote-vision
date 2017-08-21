/* -*-c++-*- */
/** \file multilabel_problem.cpp

    Base data structure for a multilabel problem.
      - Each component is interpreted as indicator function
      - Data term is usually linear (assignment costs)

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

#include "multilabel_problem.h"
#include "../compute_api/kernels_multilabel.h"


using namespace coco;
using namespace std;

// Construction and destruction
multilabel_problem::multilabel_problem( regularizer *J, data_term *F )
  : variational_model( J,F )
{
}

multilabel_problem::~multilabel_problem()
{
}


// Set/access solution vectors
bool multilabel_problem::set_pointwise_optimal_labeling( const vector_valued_function_2D *data_term,
							 vector_valued_function_2D *U )
{
  // check compatibility
  assert( data_term->grid()->is_compatible( U->grid() ));
  assert( data_term->N() == U->N() );
  kernel_assign_optimum_label( data_term->grid(),
			       U->N(),
			       data_term->buffer(), 
			       U->buffer() );
  return true;
}


/*
  bool project_labeling( const vector_valued_function_2D *U,
			   const vector<float> &labels,
			   compute_buffer *solution );
*/
bool multilabel_problem::project_labeling_to_cpu( const vector_valued_function_2D *U,
						  const vector<float> &labels,
						  float *solution )
{
  int W = U->W();
  int H = U->H();
  int N = U->N();
  assert( (int)labels.size() == N );
  float *u = new float[ W*H*N ];
  U->copy_to_cpu( u );

  // Reconstruct u by finding the maximum label variable
  // TODO: Implement better scheme from Lelmann et al., TR
  size_t index = 0;
  for ( int y=0; y<H; y++ ) {
    for ( int x=0; x<W; x++ ) {
      TRACE6( "Solution @" << x << ", " << y << ": " << endl );

      // u: argmax over G
      int v = -1; float umax = -1.0;
      float sum = 0.0;
      TRACE6( "  dim1: " );
      for ( int g=0; g<N; g++ ) {
	float uv = u[ index + g*W*H ];
	sum += uv;
	TRACE6( uv << " " );
	if ( uv > umax ) {
	  v = g; umax = uv;
	}
      }

      solution[index] = labels[v];
      TRACE6( "  " << v << " = " << labels[v] << "  sum=" << sum << endl );

      // Next pixel
      index++;
    }
  }

  delete[] u;
  return true;
}

