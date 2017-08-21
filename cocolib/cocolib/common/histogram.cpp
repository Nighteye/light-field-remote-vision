/** \file histogram.cpp

    Compute simple histogram for double values.
    
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

#include "histogram.h"
#include "math.h"

using namespace std;

// Initialize histogram
bool coco::histogram_init( histogram &H, double vmin, double vmax, size_t bins )
{
  // Range and number of bins
  H._min = vmin;
  H._max = vmax;
  H._N = bins;
  H._bin.resize( bins );
  return histogram_reset( H );
}

// Reset histogram to zero values
bool coco::histogram_reset( histogram &H )
{
  for ( size_t i=0; i<H._N; i++ ) {
    H._bin[i] = 0.0;
  }
  return true;
}

// Add value to histogram (i.e. increase bin of v by 1)
bool coco::histogram_add( histogram &H, double v )
{
  size_t i = histogram_bin( H,v );
  if ( i >= H._N ) {
    // Index out of range.
    return false;
  }
  H._bin[i] += 1.0;
  return true;
}

// Normalize a histogram (so that sum of bin values is 1)
bool coco::histogram_normalize( histogram &H )
{
  double s = 0.0;
  for ( size_t i=0; i<H._N; i++ ) {
    s += H._bin[i];
  }
  if ( s==0.0 ) {
    // Sum is zero, no normalization possible.
    return false;
  }
  for ( size_t i=0; i<H._N; i++ ) {
    H._bin[i] /= s;
  }
  return true;
}

// Get bin index for value
size_t coco::histogram_bin( const histogram &H, double v )
{
  if ( v<H._min || v>H._max ) {
    // Value out of range
    ERROR( "histogram::value " << v << " out of range." << endl );
    return H._N;
  }
  size_t i = (size_t)((v-H._min) / (H._max - H._min) * H._N );
  i = min( i, H._N-1 );
  return i;
}

// Get bin entry for value
double coco::histogram_bin_value( const histogram &H, double v )
{
  size_t i = histogram_bin( H,v );
  if ( i >= H._N ) {
    return 0.0;
  }
  return H._bin[i];
}




size_t bin_total( vector<size_t> &Ns )
{
  size_t N = 1;
  for ( size_t i=0; i<Ns.size(); i++ ) {
    N *= Ns[i];
  }
  return N;
}



// Initialize histogram
bool coco::md_histogram_init( md_histogram &H, size_t d )
{
  H._d = d;
  H._min.resize( d );
  H._max.resize( d );
  H._N.resize( d );
  H._sigma.resize( d );

  for ( size_t i=0; i<d; i++ ) {
    H._min[i] = 0.0;
    H._max[i] = 1.0;
    H._N[i] = 1;
    H._sigma[i] = 0.0;
  }

  return true;
}

// Reset histogram to zero values
bool coco::md_histogram_reset( md_histogram &H )
{
  H._bin.clear();
  return true;
}

// Set range of a bin
bool coco::md_histogram_set_range( md_histogram &H, size_t dim, size_t N, double min, double max )
{
  assert( dim < H._d );
  H._min[dim] = min;
  H._max[dim] = max;
  H._N[dim] = N;

  size_t Nb = bin_total( H._N );
  H._bin.resize( Nb );
  for ( size_t i=0; i<Nb; i++ ) {
    H._bin[i] = 0.0;
  }

  return true;
}


void compute_modulus_vector( vector<size_t> &m, const vector<size_t> &N, size_t j )
{
  m.resize( N.size() );
  for ( size_t i=0; i<N.size(); i++ ) {
    size_t n = N[i];
    m[i] = j % n;
    j = (j - m[i]) / n;
  }
}


// Add value to histogram (i.e. increase bin of v by 1)
bool coco::md_histogram_insert( md_histogram &H, vector<double> &v )
{
  size_t N = bin_total( H._N );
  for ( size_t j=0; j<N; j++ ) {
    vector<size_t> idx;
    compute_modulus_vector( idx, H._N, j );
    double w = 1.0;
    for ( size_t k=0; k<idx.size(); k++ ) {
      double dv = (double(idx[k]) + 0.5) / double(H._N[k]);
      double bin_width = H._max[k] - H._min[k];
      dv = H._min[k] + bin_width * dv;
      double dist2 = fabs( v[k] - dv );
      double sigma = H._sigma[k];
      if ( sigma == 0.0 ) {
	if ( dist2 < 1.0 / (2.0 * bin_width) ) {
	  w *= 1.0;
	}
	else {
	  w *= 0.0;
	}
      }
      else {
	w *= exp( - dv*dv / (sigma*sigma) );
      }
    }

    //    TRACE( "Adding " << w << " to " << j << endl );
    H._bin[j] += w;
  }

  return true;
}



// Normalize a histogram (so that sum of bin values is 1)
bool coco::md_histogram_normalize( md_histogram &H )
{
  size_t N = bin_total( H._N );
  double sum = 0.0;
  for ( size_t i=0; i<N; i++ ) {
    sum += H._bin[i];
  }
  if ( sum > 0.0 ) {
    for ( size_t i=0; i<N; i++ ) {
      H._bin[i] /= sum;
    }
  }
  return true;
}

// Get bin index for value
size_t coco::md_histogram_bin( const md_histogram &H, vector<double> &vs )
{
  size_t bin = 0;
  for ( size_t j=0; j<H._d; j++ ) {
    size_t i = H._d - j - 1;
    double v = vs[i];
    if ( v<H._min[i] || v>H._max[i] ) {
      // Value out of range
      ERROR( "histogram::value " << v << " dimension " << i << " out of range." << endl );
      return bin;
    }
    size_t idx = (size_t)((v-H._min[i]) / (H._max[i] - H._min[i]) * H._N[i] );
    idx = min( idx, H._N[i]-1 );
    bin = H._N[i] * bin + idx;
  }

  return bin;
}

// Get bin entry for value
double coco::md_histogram_bin_value( const md_histogram &H, vector<double> &v )
{
  size_t bin = md_histogram_bin( H, v );
  if ( bin >= H._bin.size() ) {
    TRACE( "invalid bin " << bin << " / " << H._bin.size() << endl );
    return 0.0;
  }
  return H._bin[bin];
}
