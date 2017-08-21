/** \file hdf5_tools.h

    Functions which help with reading hdf5 files

    Copyright (C) 2001 Bastian Goldluecke,
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

#ifndef __HDF5_TOOLS_H
#define __HDF5_TOOLS_H

#include <time.h>
#include <map>
#include <set>

#include <hdf5.h>
#include <hdf5_hl.h>

#include "debug.h"



// Try to read 4D dataset, return as float array
// WARNING: hard-coded to certain ordering
float *hdf5_read_data_set( hid_t file, const std::string &dset_name,
			   size_t &W, size_t &H, size_t &S, size_t &T );

// Try to read 4D dataset, return as float array
// WARNING: hard-coded to certain ordering
float *hdf5_read_data_set( hid_t file, const std::string &dset_name,
			   size_t &W, size_t &H, size_t &L );

// Try to read 2D dataset, return as float array
// WARNING: hard-coded to certain ordering
float *hdf5_read_data_set( hid_t file, const std::string &dset_name,
			   size_t &W, size_t &H );

// Obtain a list of all datasets in the file
bool hdf5_dataset_names( hid_t file, std::set<std::string> &datasets );

#endif
