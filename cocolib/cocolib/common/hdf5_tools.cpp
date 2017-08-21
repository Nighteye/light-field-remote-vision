/** \file hdf5_tools.cpp

    Functions which help with reading hdf5 files

    Copyright (C) 2011 Bastian Goldluecke

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

#define H5_USE_16_API
#include "hdf5_tools.h"

using namespace std;

float *hdf5_read_data_set( hid_t file, const string &dset_name,
			   size_t &W, size_t &H, size_t &S, size_t &T )
{
  // Open the light field data sets
  // HDF5 library differs across Ubuntu versions.
  // Try to uncomment if build fails for you.
  //hid_t dset = H5Dopen (file, dset_name.c_str(), H5P_DEFAULT );
  hid_t dset = H5Dopen (file, dset_name.c_str() );
  if ( dset == 0 ) {
    ERROR( "could not open light field data set '" << dset_name << endl );
    return NULL;
  }
  TRACE1( "  data set '" << dset_name << "' opened." << endl );

  // Retrieve dimension attributes
  int ndims;
  H5LTget_dataset_ndims(file, dset_name.c_str(), &ndims );
  if ( ndims != 4 ) {
    ERROR( "light field data set '" << dset_name << "' is not four-dimensional." << endl );
    return NULL;
  }
  hsize_t dims[4];
  H5LTget_dataset_info(file, dset_name.c_str(), dims, NULL,NULL);
  S = dims[0];
  T = dims[1];
  W = dims[3];
  H = dims[2];
  TRACE1( "  data set '" << dset_name << "'  dim " << S << " x " << T << " views, " << W << " x " << H << endl );

  // create buffer for light field data
  float *data = new float[ S*T*W*H ];
  H5LTread_dataset_float( file, dset_name.c_str(), data );
  H5Dclose( dset );
  return data;
}


float *hdf5_read_data_set( hid_t file, const string &dset_name,
			   size_t &W, size_t &H, size_t &L )
{
  // Open the light field data sets
  // HDF5 library differs across Ubuntu versions.
  // Try to uncomment if build fails for you.
  //hid_t dset = H5Dopen (file, dset_name.c_str(), H5P_DEFAULT );
  hid_t dset = H5Dopen (file, dset_name.c_str() );
  if ( dset == 0 ) {
    ERROR( "could not open light field data set '" << dset_name << endl );
    return NULL;
  }
  TRACE1( "  data set '" << dset_name << "' opened." << endl );

  // Retrieve dimension attributes
  int ndims;
  H5LTget_dataset_ndims(file, dset_name.c_str(), &ndims );
  if ( ndims != 3 ) {
    ERROR( "light field data set '" << dset_name << "' is not three-dimensional." << endl );
    return NULL;
  }
  hsize_t dims[3];
  H5LTget_dataset_info(file, dset_name.c_str(), dims, NULL,NULL);
  L = dims[0];
  W = dims[1];
  H = dims[2];
  TRACE1( "  data set '" << dset_name << "'  dim " << W << " x " << H << " pixels, " << L << " layers." << endl );

  // create buffer for light field data
  float *data = new float[ W*H*L ];
  H5LTread_dataset_float( file, dset_name.c_str(), data );
  H5Dclose( dset );
  return data;
}


float *hdf5_read_data_set( hid_t file, const string &dset_name,
			   size_t &W, size_t &H )
{
  // Open the light field data sets
  // HDF5 library differs across Ubuntu versions.
  // Try to uncomment if build fails for you.
  //hid_t dset = H5Dopen (file, dset_name.c_str(), H5P_DEFAULT );
  hid_t dset = H5Dopen (file, dset_name.c_str() );
  if ( dset == 0 ) {
    ERROR( "could not open light field data set '" << dset_name << endl );
    return NULL;
  }
  TRACE1( "  data set '" << dset_name << "' opened." << endl );

  // Retrieve dimension attributes
  int ndims;
  H5LTget_dataset_ndims(file, dset_name.c_str(), &ndims );
  if ( ndims != 2 ) {
    ERROR( "light field data set '" << dset_name << "' is not two-dimensional." << endl );
    return NULL;
  }
  hsize_t dims[4];
  H5LTget_dataset_info(file, dset_name.c_str(), dims, NULL,NULL);
  W = dims[1];
  H = dims[0];
  TRACE1( "  data set '" << dset_name << "'  dim " << W << " x " << H << endl );

  // create buffer for light field data
  float *data = new float[ W*H ];
  H5LTread_dataset_float( file, dset_name.c_str(), data );
  H5Dclose( dset );
  return data;
}




/*
 * Operator function.
 */
herr_t dataset_info(hid_t loc_id, const char *name, void *datasets )
{
  H5G_stat_t statbuf;
  
  /*
   * Get type of the object and display its name and type.
   * The name of the object is passed to this function by 
   * the Library. Some magic :-)
   */
  H5Gget_objinfo(loc_id, name, false, &statbuf);
  switch (statbuf.type) {
  case H5G_GROUP: 
    break;
  case H5G_DATASET: 
    TRACE3( name << " " );
    if ( datasets != NULL ) {
      set<string> *__datasets = reinterpret_cast< set<string>* > (datasets);
      (*__datasets).insert( name );
    }
    break;
  case H5G_TYPE: 
    break;
  default:
    break;
  }
  return 0;
}


bool hdf5_dataset_names( hid_t file, set<string> &datasets )
{
  TRACE3(" Data sets in the root group are: ");
  H5Giterate(file, "/", NULL, dataset_info, &datasets);
  TRACE3( endl );
  return 0;
}
