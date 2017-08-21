#version 410

in vec2 textureCoord;

uniform sampler2DRect vkDepthMap;

uniform int W;
uniform int H;

uniform mat3 u_K;
uniform mat3 u_R;
uniform vec3 u_t;
uniform mat3 vk_K;
uniform mat3 vk_R;
uniform vec3 vk_t;

out vec4 tauPartial;

// compute tau warps and deformation weights
// -2: point's not reconstructed
// -1: backward visibility = false. Means the point in vk is not visible from u

void main() {

    float sigma = 10.0;

    float vk_depth = texture(vkDepthMap, textureCoord).x;

    // compute coordinates on target view
    vec3 xk = vec3(gl_FragCoord.xy, 1.0);
    vec3 X = transpose(vk_R) * (inverse(vk_K) * vk_depth * xk - vk_t);
    vec3 xuh = u_K * (u_R * X + u_t); // homogeneous coordinates
    vec2 xu = vec2(0.0, 0.0); // euclidian coordinates
    if(xuh.z != 0) {
        xu = vec2(xuh.x/xuh.z, xuh.y/xuh.z);
    }

    // Jacobian matrix of the euclidean normalization
    mat3x2 Je;
    Je[0][0] = 1; Je[1][0] = 0; Je[2][0] = -xu.x;
    Je[0][1] = 0; Je[1][1] = 1; Je[2][1] = -xu.y;
    Je /= xuh.z;

    vec2 res = Je * u_K * u_R * transpose(vk_R) * inverse(vk_K) * xk;

    // (sigmaZ, dtaux/dz, dtauy/dz)
    tauPartial = vec4(sigma, res, 1.0);
}
