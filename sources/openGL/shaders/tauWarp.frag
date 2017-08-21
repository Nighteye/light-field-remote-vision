#version 410

in vec2 textureCoord;

uniform sampler2DRect uDepthMap;
uniform sampler2DRect vkDepthMap;

uniform int W;
uniform int H;

uniform mat3 u_K;
uniform mat3 u_R;
uniform vec3 u_t;
uniform mat3 vk_K;
uniform mat3 vk_R;
uniform vec3 vk_t;
uniform float invalidDepth;

out vec4 tauWarp;

// compute tau warps and deformation weights
// -2: point's not reconstructed
// -1: backward visibility = false. Means the point in vk is not visible from u

void main() {

    float vk_depth = texture(vkDepthMap, textureCoord.xy).x;

    // compute coordinates on target view
    vec3 xk = vec3(gl_FragCoord.xy, 1.0);
    vec3 X = transpose(vk_R) * ((inverse(vk_K) * (vk_depth * xk)) - vk_t);
    vec3 xuh = u_K * ((u_R * X) + u_t); // homogeneous coordinates
    vec2 xu = vec2(0.0, 0.0); // euclidian coordinates
    if(xuh.z != 0) {
        xu = vec2(xuh.x/xuh.z, xuh.y/xuh.z);
    }

    // deformation weights
    float dzdx = texture(vkDepthMap, gl_FragCoord.xy).y;
    float dzdy = texture(vkDepthMap, gl_FragCoord.xy).z;
    mat3x2 Je;
    Je[0][0] = 1; Je[1][0] = 0; Je[2][0] = -xu.x;
    Je[0][1] = 0; Je[1][1] = 1; Je[2][1] = -xu.y;
    Je /= xuh.z;
    mat2x3 temp;
    temp[0][0] = xk.x * dzdx + vk_depth; temp[1][0] = xk.x * dzdy;
    temp[0][1] = xk.y * dzdx;            temp[1][1] = xk.y * dzdy + vk_depth;
    temp[0][2] = dzdx;                   temp[1][2] = dzdy;
    mat2 dtaudx = Je * u_K * u_R * transpose(vk_R) * inverse(vk_K) * temp;
    float deformWeight = 1.0/abs(dtaudx[0][0] * dtaudx[1][1] - dtaudx[0][1] * dtaudx[1][0]);
    float maxWeight = 30.0;
    if(deformWeight >= maxWeight) {
        deformWeight = maxWeight;
    }

    // check backward visibility using target depth map
    float u_depth = texture(uDepthMap, xu).x;
    float epsilon = 0.10; // ???

    if(xu.x <= 0 || W <= xu.x ||
       xu.y <= 0 || H <= xu.y ||
       vk_depth >= invalidDepth/2) { // check forward visibility (outside field of view of u)

        tauWarp = vec4( -2.0, -2.0, 0.0, 1.0 );

    } else if(xuh.z > u_depth + epsilon) { // check backward visibility (occlusions)

        tauWarp = vec4( -1.0, -1.0, 0.0, 1.0 );

    } else {

        tauWarp = vec4( xu.x, xu.y, deformWeight, 1.0 );
    }
}
