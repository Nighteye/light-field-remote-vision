#ifndef PINHOLECAMERA_H
#define PINHOLECAMERA_H

// Includes GLM
#include <glm/glm.hpp>

#define checkGLErrors() {\
    GLenum error = glGetError(); \
    if(error != GL_NO_ERROR) { \
    std::cout << "GL_ERROR :" << __FILE__ << " "<< __LINE__ << " " << error << std::endl; \
    } \
    }

// Pinhole camera P = K[R|t], t = -RC
struct PinholeCamera {

    PinholeCamera( // first constructor: given the decomposition K, R annd t
                   const glm::mat3 & K = glm::mat3(1.0),
                   const glm::mat3 & R = glm::mat3(1.0),
                   const glm::vec3 & t = glm::vec3(0.0),
                   uint W = 0,
                   uint H = 0);

    PinholeCamera(const PinholeCamera &pinholeCamera);

    PinholeCamera( const glm::mat4x3 & P );

    void display();
    // compute intrisic parameter for downscale views
    void scaleCamera(uint scale);
    glm::mat4 l2w_Camera() const;
    void setAsRenderCam( glm::mat4 &renderMatrix );

    /// Projection matrix P = K[R|t]
    glm::mat4x3 _P;

    /// Intrinsic parameter (Focal, principal point)
    glm::mat3 _K;

    /// Extrinsic Rotation
    glm::mat3 _R;

    /// Extrinsic translation
    glm::vec3 _t;

    uint _W, _H;

    /// Camera center
    glm::vec3 _C;

    // pad transformation
    void turn_render_cam_X(double angle);
    void turn_render_cam_Y(double angle);
    void turn_render_cam_Z(double angle);
    void translate_render_cam(const glm::vec3 &t);
};

#endif // #ifndef PINHOLECAMERA_H

