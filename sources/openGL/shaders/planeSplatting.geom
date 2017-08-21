#version 410

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

in float dzdx[];
in float dzdy[];
in float z[];
in vec4 gcolor[];

uniform mat4 renderMatrix;
uniform mat3 vk_K;
uniform mat3 vk_R;
uniform vec3 vk_t;
uniform mat3 u_R;
uniform vec3 u_t;
uniform float depthFocal;

out vec4 fcolor;

void main() {

    fcolor = gcolor[0];
    // test splatted pixel origin
    // fcolor = vec4(gl_in[0].gl_Position.x, gl_in[0].gl_Position.y, 0.0, 1.0);

    // corner offset
    float Dx = 0.5, Dy = 0.5;

    // quad center
    float x = gl_in[0].gl_Position.x;
    float y = gl_in[0].gl_Position.y;

    vec3 ray; // ray exiting the input camera, at each pixel corner (homogeneous image point in WC)
    vec3 rayu; // same ray as seen from u

    // depth from input pov, at each pixel corner
    float z;

    // quad corner vertex
    vec3 X;

    vec3 C = - transpose(vk_R) * vk_t; // center of input camera
    vec3 Cu = u_R * C + u_t; // center of vk as seen from u

    // Bottom-Left
    ray = transpose(vk_R)*inverse(vk_K)*vec3(x - Dx, y - Dy, 1.0);
    rayu = u_R * ray;
    z = (depthFocal - Cu.z)/rayu.z;
    X = z * ray + C;
    gl_Position = renderMatrix * vec4(X, 1.0);
    gl_Position.y = -gl_Position.y;
    EmitVertex();

    // Top-Left
    ray = transpose(vk_R)*inverse(vk_K)*vec3(x - Dx, y + Dy, 1.0);
    rayu = u_R * ray;
    z = (depthFocal - Cu.z)/rayu.z;
    X = z * ray + C;
    gl_Position = renderMatrix * vec4(X, 1.0);
    gl_Position.y = -gl_Position.y;
    EmitVertex();

    // Bottom-Right
    ray = transpose(vk_R)*inverse(vk_K)*vec3(x + Dx, y - Dy, 1.0);
    rayu = u_R * ray;
    z = (depthFocal - Cu.z)/rayu.z;
    X = z * ray + C;
    gl_Position = renderMatrix * vec4(X, 1.0);
    gl_Position.y = -gl_Position.y;
    EmitVertex();

    // Top-Right
    ray = transpose(vk_R)*inverse(vk_K)*vec3(x + Dx, y + Dy, 1.0);
    rayu = u_R * ray;
    z = (depthFocal - Cu.z)/rayu.z;
    X = z * ray + C;
    gl_Position = renderMatrix * vec4(X, 1.0);
    gl_Position.y = -gl_Position.y;
    EmitVertex();

    EndPrimitive();
}
