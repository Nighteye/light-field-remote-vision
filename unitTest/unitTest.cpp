#include <vector>
#include <iostream>
#include <string>

#define cimg_display 0
#define cimg_use_tiff
#define cimg_use_png
#include "CImg.h"

void loadPFM(std::vector<float>& output, std::string name)
{
    cimg_library::CImg<float> image(name.c_str());

    image.resize(image.width(), image.height(), 1, 1);
    output.resize(image.size());

    for(uint y = 0 ; y < (uint)image.height() ; ++y)
    {
        for(uint x = 0 ; x < (uint)image.width() ; ++x)
        {
            const uint i = y*(uint)image.width() + x;
            output[i] = *(((float*)(&image(0,0,0,0))) + i);
        }
    }
}

int main( int argc, char **argv )
{
    std::vector<float> img1, img2;
    loadPFM(img1, std::string(argv[1]));
    loadPFM(img2, std::string(argv[2]));

    bool equal = true;
    uint i = 0;
  
    while(i < img1.size() && equal) {
	
        equal = (img1[i] == img2[i]);
++i;
    }

    if(equal) {
        std::cout << argv[1] << " and " << argv[2] << " are equal." << std::endl;
    } else {
        std::cout << argv[1] << " and " << argv[2] << " are different." << std::endl;
    }

    return 0;
}
