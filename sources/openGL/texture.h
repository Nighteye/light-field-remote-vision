#ifndef TEXTURE_H
#define TEXTURE_H

#include <GL/glew.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL.h>
#include <string>
#include <vector>

#include "../optical_flow/CvUtil.h"

#define checkGLErrors() {\
    GLenum error = glGetError(); \
    if(error != GL_NO_ERROR) { \
    std::cout << "GL_ERROR :" << __FILE__ << " "<< __LINE__ << " " << error << std::endl; \
    } \
    }

void computePerPixelCorresp(std::string imageName, std::string outdir, std::string flowAlg);

void flipTexture(float *sourceImage, float *reversedImage, uint width, uint height);
void resizeTexture(float *sourceImage, float *resizedImage, uint width, uint height, uint srFactor);

class Texture {

public:

    uint _index; // index in map vector

    Texture();
    Texture(std::string imageFile);
    Texture(uint index, uint width, uint height, GLenum format, GLenum type, GLenum internalFormat, bool emptyTexture);
    Texture(Texture const &textureACopier);
    ~Texture();

    Texture& operator=(Texture const &textureToCopy);
    bool load(bool flip = false);
    void loadEmptyTexture();

    // texture is set to stdValue if pixel intensisty equals 0.0
    void loadFromData( const std::vector< std::vector<float> > &data, uint srFactor, float stdValue = 0.0f, bool flip = false );

    // load texture from an array
    void loadFromData( const float* const data );

    SDL_Surface* reversePixels(SDL_Surface *sourceImage) const;
    void setImageFile(const std::string &imageFile);
    void saveRGBAIntTexture( int render_w, int render_h, int depth, const std::string &name, bool flip = false );
    void saveRGBAFloatTexture( int render_w, int render_h, int depth, const std::string &name, bool flip = false );

    // getters
    GLuint getID() const;
    GLenum getFormat() const;
    GLenum getInternalFormat() const;
    GLenum getType() const;

private:

    GLuint _id;
    std::string _imageFile;

    uint _width;
    uint _height;
    GLenum _format;
    GLenum _type;
    GLenum _internalFormat;
    bool _emptyTexture;
};

#endif /* #ifndef TEXTURE_H */
