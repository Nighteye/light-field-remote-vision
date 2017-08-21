/** \file histogram.h

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

#ifndef __COCO_COMMON_HISTOGRAM_H
#define __COCO_COMMON_HISTOGRAM_H

#include <string.h>
#include <vector>

#include "../common/debug.h"

namespace coco {

  // Histogram structure
  struct histogram {
    // Range and number of bins
    double _min;
    double _max;
    size_t _N;
    // Bin values
    std::vector<double> _bin;
  };

  // Initialize histogram
  bool histogram_init( histogram &H, double vmin, double vmax, size_t bins );
  // Reset histogram to zero values
  bool histogram_reset( histogram &H );
  // Add value to histogram (i.e. increase bin of v by 1)
  bool histogram_add( histogram &H, double v );
  // Normalize a histogram (so that sum of bin values is 1)
  bool histogram_normalize( histogram &H );
  // Get bin index for value
  size_t histogram_bin( const histogram &H, double v );
  // Get bin entry for value
  double histogram_bin_value( const histogram &H, double );



  // Multidim-histogram structure
  struct md_histogram {
    // Range and number of bins
    size_t _d;
    std::vector<double> _min;
    std::vector<double> _max;
    std::vector<size_t> _N;
    // Bin values
    std::vector<double> _bin;
    // Insert spill standard deviation
    std::vector<double> _sigma;
  };

  // Initialize histogram
  bool md_histogram_init( md_histogram &H, size_t d );
  // Reset histogram to zero values
  bool md_histogram_reset( md_histogram &H );
  // Set range of a bin
  bool md_histogram_set_range( md_histogram &H, size_t bin, size_t N, double min, double max );

  // Add value to histogram (i.e. increase bin of v by 1)
  bool md_histogram_insert( md_histogram &H, std::vector<double> &v );
  // Normalize a histogram (so that sum of bin values is 1)
  bool md_histogram_normalize( md_histogram &H );
  // Get bin index for value
  size_t md_histogram_bin( const md_histogram &H, std::vector<double> &v );
  // Get bin entry for value
  double md_histogram_bin_value( const md_histogram &H, std::vector<double> &v );
}

    
#endif
