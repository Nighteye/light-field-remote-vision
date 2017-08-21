#include "texture.h"
#include "assert.h"

#include <iostream>
#include <vector>
#include <fstream>

#define cimg_display 0
#define cimg_use_tiff
#define cimg_use_png
#include "../CImg.h"

Texture::Texture() : _index(0), _id(0), _imageFile(""), _width(0), _height(0), _format(0), _type(0), _internalFormat(0), _emptyTexture(false) {

}

Texture::Texture(std::string imageFile) : _index(0), _id(0), _imageFile(imageFile), _width(0), _height(0), _format(0), _type(0), _internalFormat(0), _emptyTexture(false) {

}

Texture::Texture(uint index, uint width, uint height, GLenum format, GLenum type, GLenum internalFormat, bool emptyTexture) : _index(index), _id(0), _imageFile(""), _width(width),
    _height(height), _format(format), _type(type), _internalFormat(internalFormat), _emptyTexture(emptyTexture) {

}

Texture::Texture(Texture const &textureToCopy) {

    _imageFile = textureToCopy._imageFile;

    _width = textureToCopy._width;
    _height = textureToCopy._height;
    _format = textureToCopy._format;
    _type = textureToCopy._type;
    _internalFormat = textureToCopy._internalFormat;
    _emptyTexture = textureToCopy._emptyTexture;

    // If the texture is empty, we call loadEmptyTexture()

    if(_emptyTexture && glIsTexture(textureToCopy._id) == GL_TRUE)
        loadEmptyTexture();

    // otherwise we call default load()

    else if(glIsTexture(textureToCopy._id) == GL_TRUE)
        load();
}

Texture::~Texture() {

    checkGLErrors();

    glDeleteTextures(1, &_id);

    checkGLErrors();
}

Texture& Texture::operator=(Texture const &textureToCopy) {

    _imageFile = textureToCopy._imageFile;

    _width = textureToCopy._width;
    _height = textureToCopy._height;
    _format = textureToCopy._format;
    _type = textureToCopy._type;
    _internalFormat = textureToCopy._internalFormat;
    _emptyTexture = textureToCopy._emptyTexture;

    // If the texture is empty, we call loadEmptyTexture()

    if(_emptyTexture && glIsTexture(textureToCopy._id) == GL_TRUE)
        loadEmptyTexture();

    // otherwise we call default load()

    else if(glIsTexture(textureToCopy._id) == GL_TRUE)
        load();

    return *this;
}

bool Texture::load(bool flip) {

    SDL_Surface *imageSDL = IMG_Load(_imageFile.c_str());

    if(imageSDL == 0) {

        std::cout << "Erreur : " << SDL_GetError() << std::endl;
        return false;
    }

    SDL_Surface *image;

    if(flip) {

        image = reversePixels(imageSDL);
        SDL_FreeSurface(imageSDL);

    } else {

        image = imageSDL;
    }

    // Destroy possible old texture
    if(glIsTexture(_id) == GL_TRUE)
        glDeleteTextures(1, &_id);

    // Generate ID
    glGenTextures(1, &_id);

    // Configure texture
    glBindTexture(GL_TEXTURE_RECTANGLE, _id);

    // default type
    if( _type == 0 ) {

        _type = GL_UNSIGNED_BYTE;
    }

    if( image->format->BytesPerPixel == 3 ) {

        _internalFormat = GL_RGB;

        if( image->format->Rmask == 0xff )
            _format = GL_RGB;

        else
            _format = GL_BGR;

    } else if ( image->format->BytesPerPixel == 4 ) {

        _internalFormat = GL_RGBA;

        if(image->format->Rmask == 0xff)
            _format = GL_RGBA;

        else
            _format = GL_BGRA;

    } else {

        std::cout << "Error, unknown internal format." << std::endl;
        SDL_FreeSurface(image);

        return false;
    }

    // Copy pixels
    glTexImage2D(GL_TEXTURE_RECTANGLE, 0, _internalFormat, image->w, image->h, 0, _format, _type, image->pixels);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // Apply filters
    glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_RECTANGLE, 0);

    SDL_FreeSurface(image);
    return true;
}

void Texture::loadEmptyTexture() {

    if(glIsTexture(_id) == GL_TRUE) {

        glDeleteTextures(1, &_id);
    }

    // ID generation
    glGenTextures(1, &_id);

    glBindTexture(GL_TEXTURE_RECTANGLE, _id);

    // Define texture properties
    glTexImage2D(GL_TEXTURE_RECTANGLE, 0, _internalFormat, _width, _height, 0, _format, _type, 0);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // Apply filters
    glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_RECTANGLE, 0);
}

void flipTexture(float *sourceImage, float *reversedImage, uint width, uint height) {

    // width and height are LOW resolution dimensions

    assert(sourceImage != 0);
    assert(reversedImage != 0);
    assert(width != 0);
    assert(height != 0);

    // Flip pixels
    for(uint i = 0 ; i < height ; ++i) {
        for(uint j = 0 ; j < width ; ++j) {
            reversedImage[width * (height - 1 - i) + j] = sourceImage[width * i + j];
        }
    }
}

void resizeTexture(float *sourceImage, float *resizedImage, uint width, uint height, uint srFactor) {

    // width and height are HIGH resolution dimensions

    for(uint i = 0 ; i < height ; ++i) {
        for(uint j = 0 ; j < width ; ++j) {
            resizedImage[width * i + j] = sourceImage[(width/srFactor) * (i/srFactor) + (j/srFactor)];
        }
    }
}

void Texture::loadFromData( const std::vector< std::vector<float> > &data, uint srFactor, float stdValue, bool flip ) {

    if(glIsTexture(_id) == GL_TRUE) {

        glDeleteTextures(1, &_id);
    }

    uint channels = data.size();
    float *resizedImage = new float[channels*_width*_height];
    float *reversedImage = new float[_width*_height/(srFactor*srFactor)];
    float *textureData = new float[channels*_width*_height];

    for(uint i = 0 ; i < channels ; ++i ) {

        // copy low resolution data in (larger) buffer, input data is RRRGGGBBBAAA...
        memcpy(&textureData[i*_height*_width/(srFactor*srFactor)], data[i].data(), _height/(srFactor*srFactor)*_width*sizeof(float));

        if(flip) {

            // flip image cause openGL texture origin is lower-left corner
            flipTexture(&textureData[i*_height*_width/(srFactor*srFactor)], reversedImage, _width/srFactor, _height/srFactor);
            // upsample source image (no interpolation)
            resizeTexture(reversedImage, &resizedImage[i*_height*_width], _width, _height, srFactor);

        } else {

            // upsample source image (no interpolation)
            resizeTexture(&textureData[i*_height*_width/(srFactor*srFactor)], &resizedImage[i*_height*_width], _width, _height, srFactor);
        }
    }

    // change from RRRGGGBBBAAA... to RBGARBGARBGA...
    for(uint i = 0 ; i < _width*_height ; ++i ) {

        bool set2std = true; // test if pixel == 0 for all channels
        for(uint j = 0 ; j < channels ; ++j) {

            set2std = set2std && (resizedImage[j*_width*_height + i] == 0.0f);

            if(!set2std) {

                break;
            }
        }

        if(set2std) {

            textureData[i*channels + 0] = stdValue;

            for(uint j = 1 ; j < channels ; ++j) {

                textureData[i*channels + j] = 0;
            }

        } else {

            for(uint j = 0 ; j < channels ; ++j) {

                memcpy(&textureData[i*channels + j], &resizedImage[j*_width*_height + i], sizeof(float));
            }
        }
    }

    // ID generation
    glGenTextures(1, &_id);

    glBindTexture(GL_TEXTURE_RECTANGLE, _id);

    // Define texture properties
    glTexImage2D(GL_TEXTURE_RECTANGLE, 0, _internalFormat, _width, _height, 0, _format, _type, (GLvoid*) textureData);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // Apply filters
    glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_RECTANGLE, 0);

    delete[] reversedImage;
    delete[] resizedImage;
    delete[] textureData;
    reversedImage = 0;
    resizedImage = 0;
    textureData = 0;
}

void Texture::loadFromData( const float* const data ) {

    if(glIsTexture(_id) == GL_TRUE) {

        glDeleteTextures(1, &_id);
    }

    // if 4 channels for example, format is supposed to be RGBARGBARGBA...

    // ID generation
    glGenTextures(1, &_id);

    glBindTexture(GL_TEXTURE_RECTANGLE, _id);

    // Define texture properties
    glTexImage2D(GL_TEXTURE_RECTANGLE, 0, _internalFormat, _width, _height, 0, _format, _type, (GLvoid*) data);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // Apply filters
    glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_RECTANGLE, 0);
}

SDL_Surface* Texture::reversePixels(SDL_Surface *sourceImage) const {

    // Copy of the source image without the pixels
    SDL_Surface *reversedImage = SDL_CreateRGBSurface(0, sourceImage->w, sourceImage->h, sourceImage->format->BitsPerPixel, sourceImage->format->Rmask,
                                                      sourceImage->format->Gmask, sourceImage->format->Bmask, sourceImage->format->Amask);

    // Temporary buffers
    unsigned char* pixelsSources = (unsigned char*) sourceImage->pixels;
    unsigned char* pixelsInverses = (unsigned char*) reversedImage->pixels;

    // Flip pixels
    for(int i = 0; i < sourceImage->h; i++) {
        for(int j = 0; j < sourceImage->w * sourceImage->format->BytesPerPixel; j++) {
            pixelsInverses[(sourceImage->w * sourceImage->format->BytesPerPixel * (sourceImage->h - 1 - i)) + j]
                    = pixelsSources[(sourceImage->w * sourceImage->format->BytesPerPixel * i) + j];
        }
    }

    return reversedImage;
}

void Texture::setImageFile(const std::string &imageFile) {

    _imageFile = imageFile;
}

inline bool endianness() {

    const int x = 1;
    return ((unsigned char *)&x)[0] ? false : true;
}

void flip_buffer( std::vector<float> &ptr, int w, int h, int depth ) {

    for (int i=0; i< h/2; ++i) {

        std::swap_ranges(ptr.begin() + i*w*depth, ptr.begin()+ i*w*depth + w*depth, ptr.end() - w*depth - i*w*depth);
    }
}

void Texture::saveRGBAIntTexture( int render_w, int render_h, int depth, const std::string &name, bool flip ) {

    checkGLErrors();

    assert(depth == 3);
    assert(_type == GL_UNSIGNED_BYTE);

    cimg_library::CImg<unsigned char> image;

    if (depth == 4) {

        image.resize(depth, render_w, render_h, 1);

    } else {

        image.resize(render_w, render_h, 1, depth);
    }

    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glBindTexture(GL_TEXTURE_RECTANGLE, _id);

    if (depth == 4) {

        glGetTexImage(GL_TEXTURE_RECTANGLE, 0, GL_RGBA, _type, &image(0,0,0,0));
        image.permute_axes("yzcx");

    } else {

        if (depth>0) {

            glGetTexImage(GL_TEXTURE_RECTANGLE, 0, GL_RED, _type, &image(0,0,0,0));
        }
        if (depth>1 ) {

            glGetTexImage(GL_TEXTURE_RECTANGLE, 0, GL_GREEN, _type, &image(0,0,0,1));
        }
        if (depth>2) {

            glGetTexImage(GL_TEXTURE_RECTANGLE, 0, GL_BLUE, _type, &image(0,0,0,2));
        }
    }

    glBindTexture(GL_TEXTURE_RECTANGLE, 0);

    if(flip) {
        image.mirror('y');
    }
    try {
        image.save(name.c_str());
    } catch (cimg_library::CImgIOException) {
        printf("Exception COUGHT: file not saved\n");
    }

    checkGLErrors();

    //    checkGLErrors();
    //  int depth = 3;
    //  assert(_format == GL_RGB);
    //  assert(_type == GL_UNSIGNED_BYTE);

    //	std::vector<unsigned char> ptr(render_w*render_h*depth);

    //  glPixelStorei(GL_PACK_ALIGNMENT, 1);

    //	glBindTexture(GL_TEXTURE_RECTANGLE, textureId);
    //	glGetTexImage(GL_TEXTURE_RECTANGLE, 0, GL_RGB, GL_UNSIGNED_BYTE, &ptr[0]);

    //	checkGLErrors();

    //	glBindTexture(GL_TEXTURE_RECTANGLE, 0);

    //  if (reverse) {
    //    reverse_buffer(ptr, render_w, render_h, depth);
    //  }

    //	WriteImage(name.c_str(), ptr, render_w, render_h, depth);
    //	checkGLErrors();
}

void Texture::saveRGBAFloatTexture( int render_w, int render_h, int depth, const std::string &name, bool flip ) {

    checkGLErrors();

    cimg_library::CImg<float> image;

    if (depth == 4) {

        image.resize(depth, render_w, render_h, 1);

    } else {

        image.resize(render_w, render_h, 1, depth);
    }

    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glBindTexture(GL_TEXTURE_RECTANGLE, _id);

    if (depth == 4) {

        glGetTexImage(GL_TEXTURE_RECTANGLE, 0, GL_RGBA, GL_FLOAT, &image(0,0,0,0));
        image.permute_axes("yzcx");

    } else {

        if (depth>0) {

            glGetTexImage(GL_TEXTURE_RECTANGLE, 0, GL_RED, GL_FLOAT, &image(0,0,0,0));
        }
        if (depth>1 ) {

            glGetTexImage(GL_TEXTURE_RECTANGLE, 0, GL_GREEN, GL_FLOAT, &image(0,0,0,1));
        }
        if (depth>2) {

            glGetTexImage(GL_TEXTURE_RECTANGLE, 0, GL_BLUE, GL_FLOAT, &image(0,0,0,2));
        }
    }

    glBindTexture(GL_TEXTURE_RECTANGLE, 0);

    if(flip) {
        image.mirror('y');
    }
    try {
        image.save(name.c_str());
    } catch (cimg_library::CImgIOException) {
        printf("Exception COUGHT: file not saved\n");
    }

    checkGLErrors();
}

GLuint Texture::getID() const {

    return _id;
}

GLenum Texture::getFormat() const {

    return _format;
}

GLenum Texture::getInternalFormat() const {

    return _internalFormat;
}

GLenum Texture::getType() const {

    return _type;
}

