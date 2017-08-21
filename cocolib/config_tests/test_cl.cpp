#include <stdio.h>
#include <stdlib.h>
#include <cl.h>

int main()
{
  const cl_uint max_platforms = 50;
  const int STRLEN = 500;

  cl_uint nplatforms = 5;
  cl_platform_id platform_ids[max_platforms];

  if ( CL_SUCCESS != clGetPlatformIDs( max_platforms, platform_ids, &nplatforms ) ) {
    printf( "CL: Failed to query platform IDs.\n" );
    printf( "CL: No ICD installed?\n" );
    return 255;
  }
  printf( "CL: %i platforms found.\n", nplatforms );

  
  // Write results
  for ( int j=0; j<nplatforms; j++ ) {
    char platform_vendor[STRLEN];
    if ( CL_SUCCESS !=
	 clGetPlatformInfo( platform_ids[j],
			    CL_PLATFORM_VENDOR,
			    STRLEN,
			    platform_vendor,
			    NULL )) {
      printf( "CL: could not obtain platform vendor for platform %i.\n", j );
      return 255;
    }

    char platform_name[STRLEN];
    if ( CL_SUCCESS !=
	 clGetPlatformInfo( platform_ids[j],
			    CL_PLATFORM_NAME,
			    STRLEN,
			    platform_name,
			    NULL )) {
      printf( "CL: could not obtain platform name for platform %i.\n", j );
      return 255;
    }


    // Platform info
    printf( "OpenCL platform #%i:\n  ID                  : %p\n  Name                : %s\n  Vendor              : %s\n",
	    j, platform_ids[j], platform_name, platform_vendor );
	    

    // Query devices
    const cl_uint max_devices = 50;
    cl_uint ndevices;
    cl_device_id devices[max_devices];

    if ( CL_SUCCESS != clGetDeviceIDs( platform_ids[j], CL_DEVICE_TYPE_ALL, max_devices, devices, &ndevices) ) {
      printf( "CL: could not list device IDs for platform %i.\n", j );
      return 255;
    }
    printf( "  Available devices   : %i\n\n", ndevices );

    for ( int i=0; i<ndevices; i++) {
      char device_name[STRLEN];
      if ( CL_SUCCESS != clGetDeviceInfo( devices[i], CL_DEVICE_NAME, STRLEN, device_name, NULL )) {
	printf( "Could not obtain device name for device %i.\n", i );
	return 255;
      }

      char device_version[STRLEN];
      if ( CL_SUCCESS != clGetDeviceInfo( devices[i], CL_DEVICE_VERSION, STRLEN, device_version, NULL )) {
	printf( "Could not obtain device version for device %i.\n", i );
	return 255;
      }

      printf( "    OpenCL device #%i:\n      ID         : %p\n      Name       : %s\n      Supports   : %s\n\n",
	      j, devices[j], device_name, device_version );
    }

    printf( "\n" );
  }

  return 0;
}
