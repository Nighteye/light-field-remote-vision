#ifndef SUPER_RESOLUTION_H
#define SUPER_RESOLUTION_H

#include <cocolib/cocolib/common/gsl_image.h>

class Config_data;

// Compare current and previous solutions to evaluate the number of moving pixels
bool sr_compare_images( const coco::gsl_image *A, const coco::gsl_image *B, double MAX_DIFF_THRESHOLD, int *nS, int *nM );

// Main entry function: compute compute superresolved novel view from unstructured lf
void sr_synthesize_view( Config_data *data, int frame = -1 );

#endif /* #ifndef SUPER_RESOLUTION_H */
