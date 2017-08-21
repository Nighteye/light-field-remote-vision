#include "pinholeCamera.h"
#include <GL/glew.h>
#include <iostream>

PinholeCamera::PinholeCamera( // first constructor: given the decomposition K, R annd t
                              const glm::mat3 & K,
                              const glm::mat3 & R,
                              const glm::vec3 & t,
                              uint W, uint H)
    : _K(K), _R(R), _t(t), _W(W), _H(H) {

    _C = -glm::transpose(R) * t;
    _P[0] = R[0]; _P[1] = R[1]; _P[2] = R[2]; _P[3] = t;
    _P = K * _P;
}

PinholeCamera::PinholeCamera( const glm::mat4x3 & P )
    : _P(P) { // second constructor: given the projection matrix P

    // Decompose using the RQ decomposition HZ A4.1.1 pag.579.
    _K[0][0] = _P[0][0]; _K[1][0] = _P[1][0]; _K[2][0] = _P[2][0];
    _K[0][1] = _P[0][1]; _K[1][1] = _P[1][1]; _K[2][1] = _P[2][1];
    _K[0][2] = _P[0][2]; _K[1][2] = _P[1][2]; _K[2][2] = _P[2][2];

    glm::mat3 Q(1.0);

    // Set K(2,1) to zero.
    if (_K[1][2] != 0) {
        float c = -_K[2][2];
        float s = _K[1][2];
        float l = sqrt(c * c + s * s);
        c /= l; s /= l;
        glm::mat3 Qx(glm::vec3(1, 0, 0), glm::vec3(0, c, s), glm::vec3(0, -s, c));
        _K = _K * Qx;
        Q = glm::transpose(Qx) * Q;
    }
    // Set K(2,0) to zero.
    if (_K[0][2] != 0) {
        float c = _K[2][2];
        float s = _K[0][2];
        double l = sqrt(c * c + s * s);
        c /= l; s /= l;
        glm::mat3 Qy(glm::vec3(c, 0, -s), glm::vec3(0, 1, 0), glm::vec3(s, 0, c));
        _K = _K * Qy;
        Q = glm::transpose(Qy) * Q;
    }
    // Set K(1,0) to zero.
    if (_K[0][1] != 0) {
        double c = -_K[1][1];
        double s = _K[0][1];
        double l = sqrt(c * c + s * s);
        c /= l; s /= l;
        glm::mat3 Qz(glm::vec3(c, s, 0), glm::vec3(-s, c, 0), glm::vec3(0, 0, 1));
        _K = _K * Qz;
        Q = glm::transpose(Qz) * Q;
    }

    _R = Q;

    //Mat3 H = P.block(0, 0, 3, 3);
    // RQ decomposition
    //Eigen::HouseholderQR<Mat3> qr(H);
    //Mat3 K = qr.matrixQR().triangularView<Eigen::Upper>();
    //Mat3 R = qr.householderQ();

    // Ensure that the diagonal is positive and R determinant == 1.
    if (_K[2][2] < 0) {
        _K = -_K;
        _R = -_R;
    }
    if (_K[1][1] < 0) {
        glm::mat3 S(glm::vec3(1, 0, 0), glm::vec3(0, -1, 0), glm::vec3(0, 0, 1));
        _K = _K * S;
        _R = S * _R;
    }
    if (_K[0][0] < 0) {
        glm::mat3 S(glm::vec3(-1, 0, 0), glm::vec3(0, 1, 0), glm::vec3(0, 0, 1));
        _K = _K * S;
        _R = S * _R;
    }

    // Compute translation.
    _t = glm::inverse(_K) * _P[3];

    if( glm::determinant(_R) < 0 ) {
        _R = -_R;
        _t = -_t;
    }

    // scale K so that K(2,2) = 1
    _K = _K / _K[2][2];

    _C = -glm::transpose(_R) * _t;
}

PinholeCamera::PinholeCamera(const PinholeCamera &pinholeCamera) {

    _W = pinholeCamera._W;
    _H = pinholeCamera._H;

    _P = pinholeCamera._P;
    _K = pinholeCamera._K;
    _R = pinholeCamera._R;
    _t = pinholeCamera._t;
    _C = pinholeCamera._C;
}

glm::mat4 PinholeCamera::l2w_Camera() const {

    //World to Local
    /// given rotation matrix R and translation vector t,
    /// column-major matrix m is equal to:
    /// [ R11 R12 R13 t.x ]
    /// | R21 R22 R23 t.y |
    /// | R31 R32 R33 t.z |
    /// [ 0.0 0.0 0.0 1.0 ]

    //Local to World => Coordinates of the camera in the 3d space
    glm::mat4 modelview(1.0);

    modelview[0][0] = _R[0][0]; modelview[1][0] = _R[1][0]; modelview[2][0] = _R[2][0]; modelview[3][0] = _t[0];
    modelview[0][1] = _R[0][1]; modelview[1][1] = _R[1][1]; modelview[2][1] = _R[2][1]; modelview[3][1] = _t[1];
    modelview[0][2] = _R[0][2]; modelview[1][2] = _R[1][2]; modelview[2][2] = _R[2][2]; modelview[3][2] = _t[2];

    return modelview;
}

// set the render matrix given the current pinhole camera
void PinholeCamera::setAsRenderCam( glm::mat4 &renderMatrix ) {

    checkGLErrors();

    GLfloat zNear = 1e-2;
    GLfloat zFar = 1e5;

    double focal_x = _K[0][0];
    double focal_y = _K[1][1];

    GLfloat fW = _W/focal_x * zNear /2;
    GLfloat fH = _H/focal_y * zNear /2;

    double pp_x = _K[2][0] - _W/2.;
    double pp_y = _K[2][1] - _H/2.;

    GLfloat pp_offset_x = pp_x/focal_x * zNear;
    GLfloat pp_offset_y = pp_y/focal_y * zNear;

    //glFrustum( -fW-pp_offset_x, fW-pp_offset_x, fH-pp_offset_y, -fH-pp_offset_y, zNear, zFar );
    // https://www.opengl.org/sdk/docs/man2/xhtml/glFrustum.xml
    double A = -pp_offset_x/ fW; //(fW-pp_offset_x) + (-fW-pp_offset_x) / ((fW-pp_offset_x)-(-fW-pp_offset_x));
    double B =  pp_offset_y/ fH;//(-fH-pp_offset_y + (fH-pp_offset_y)) / (-fH-pp_offset_y - (fH-pp_offset_y));
    double C = - (zFar + zNear) / (zFar-zNear);
    double D = -2* zFar * zNear / (zFar-zNear);

    glm::mat4 projection(1.0);

    projection[0][0] =  zNear / fW;
    projection[1][1] = -zNear / fH;

    projection[2][0] = A;
    projection[2][1] = B;
    projection[2][2] = C;
    projection[2][3] = -1;
    projection[3][2] = D;

    projection[3][3] = 0;

    // reverse the z of the camera. document cameras point to (0,0,1) whil opengl cameras point to (0,0,-1)
    //glMultMatrixf((GLfloat*)m_z_invert);
    glm::mat4 z_invert(1.0);
    z_invert[2][2] = -1.0;
    projection = projection * z_invert;

    // apply render camera transformation
    glm::mat4 modelview = l2w_Camera();

    //    renderMatrix = projection * modelview;
    //    renderMatrix = glm::transpose(renderMatrix);

    renderMatrix = projection * modelview;

    checkGLErrors();
}

void PinholeCamera::display() {

    std::cout << "P:" << std::endl
              << _P[0][0] << " " << _P[1][0] << " " << _P[2][0] << " " << _P[3][0] << std::endl
                                                                                   << _P[0][1] << " " << _P[1][1] << " " << _P[2][1] << " " << _P[3][1] << std::endl
                                                                                                                                                        << _P[0][2] << " " << _P[1][2] << " " << _P[2][2] << " " << _P[3][2] << std::endl;
    std::cout << "K:" << std::endl
              << _K[0][0] << " " << _K[1][0] << " " << _K[2][0] << std::endl
                                                                << _K[0][1] << " " << _K[1][1] << " " << _K[2][1] << std::endl
                                                                                                                  << _K[0][2] << " " << _K[1][2] << " " << _K[2][2] << std::endl;
    std::cout << "R:" << std::endl
              << _R[0][0] << " " << _R[1][0] << " " << _R[2][0] << std::endl
                                                                << _R[0][1] << " " << _R[1][1] << " " << _R[2][1] << std::endl
                                                                                                                  << _R[0][2] << " " << _R[1][2] << " " << _R[2][2] << std::endl;
    std::cout << "t:" << std::endl
              << _t[0] << " " << _t[1] << " " << _t[2] << std::endl;
    std::cout << "C:" << std::endl
              << _C[0] << " " << _C[1] << " " << _C[2] << std::endl;
}

void PinholeCamera::scaleCamera(uint scale) {

    uint ratio = (uint)pow(2.0, (double)scale);

    _K[0][0] /= ratio;

    _K[1][1] /= ratio;
    _K[2][2] = 1.0;
    _K[2][0] /= ratio;
    _K[2][1] /= ratio;

    _P[0] = _R[0]; _P[1] = _R[1]; _P[2] = _R[2]; _P[3] = _t;
    _P = _K * _P;

    _W /= ratio;
    _H /= ratio;
}

void PinholeCamera::turn_render_cam_X(double angle) {

    glm::mat3 RX;

    RX[0][0] = 1; RX[1][0] = 0; RX[2][0] = 0;
    RX[0][1] = 0; RX[1][1] = cos(angle); RX[2][1] = -sin(angle);
    RX[0][2] = 0; RX[1][2] = sin(angle); RX[2][2] = cos(angle);

    _R =  RX * _R;
    _t = -_R * _C;
}

void PinholeCamera::turn_render_cam_Y(double angle) {

    glm::mat3 RY;

    RY[0][0] = cos(angle); RY[1][0] = 0; RY[2][0] = sin(angle);
    RY[0][1] = 0; RY[1][1] = 1; RY[2][1] = 0;
    RY[0][2] = -sin(angle); RY[1][2] = 0; RY[2][2] = cos(angle);

    _R =  RY * _R;
    _t = -_R * _C;
}

void PinholeCamera::turn_render_cam_Z(double angle) {

    glm::mat3 RZ;

    RZ[0][0] = cos(angle); RZ[1][0] = -sin(angle); RZ[2][0] = 0;
    RZ[0][1] = sin(angle); RZ[1][1] = cos(angle); RZ[2][1] = 0;
    RZ[0][2] = 0; RZ[1][2] = 0; RZ[2][2] = 1;

    _R =  RZ * _R;
    _t = -_R * _C;
}

void PinholeCamera::translate_render_cam(const glm::vec3 &t) {

    _t += t;
    _C = -glm::transpose(_R) * _t;
}
