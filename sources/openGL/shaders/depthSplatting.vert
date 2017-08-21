#version 410

in vec2 in_pixel;

uniform sampler2DRect vkDepthMap;

out float dzdx;
out float dzdy;
out float z;

void main() {

    z = texture(vkDepthMap, in_pixel).x;
    // normal orientation
    dzdx = texture(vkDepthMap, in_pixel).y;
    dzdy = texture(vkDepthMap, in_pixel).z;

    gl_Position = vec4(in_pixel, 0.0, 1.0);
}
