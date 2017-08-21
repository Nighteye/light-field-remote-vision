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
uniform float invalidDepth;

out vec4 fcolor;

void main() {

    fcolor = gcolor[0];
    // test splatted pixel origin
    // fcolor = vec4(gl_in[0].gl_Position.x, gl_in[0].gl_Position.y, 0.0, 1.0);

    // don't create a quad if depth is invalid
    if(z[0] >= invalidDepth/2) {
        return;
    }

    // corner offset
//    float Dx = 0.5, Dy = 0.5;
    float Dx = 1.0, Dy = 1.0;

    // quad center
    float x = gl_in[0].gl_Position.x;
    float y = gl_in[0].gl_Position.y;

    // quad corner vertex
    vec3 X;

    // Bottom-Left
    X = transpose(vk_R) * (inverse(vk_K) * (z[0] - Dx*dzdx[0] - Dy*dzdy[0]) * vec3(x - Dx, y - Dy, 1.0) - vk_t);
    gl_Position = renderMatrix * vec4(X, 1.0);
    gl_Position.y = -gl_Position.y;
    EmitVertex();

    // Top-Left
    X = transpose(vk_R) * (inverse(vk_K) * (z[0] - Dx*dzdx[0] + Dy*dzdy[0]) * vec3(x - Dx, y + Dy, 1.0) - vk_t);
    gl_Position = renderMatrix * vec4(X, 1.0);
    gl_Position.y = -gl_Position.y;
    EmitVertex();

    // Bottom-Right
    X = transpose(vk_R) * (inverse(vk_K) * (z[0] + Dx*dzdx[0] - Dy*dzdy[0]) * vec3(x + Dx, y - Dy, 1.0) - vk_t);
    gl_Position = renderMatrix * vec4(X, 1.0);
    gl_Position.y = -gl_Position.y;
    EmitVertex();

    // Top-Right
    X = transpose(vk_R) * (inverse(vk_K) * (z[0] + Dx*dzdx[0] + Dy*dzdy[0]) * vec3(x + Dx, y + Dy, 1.0) - vk_t);
    gl_Position = renderMatrix * vec4(X, 1.0);
    gl_Position.y = -gl_Position.y;
    EmitVertex();

    EndPrimitive();
}
