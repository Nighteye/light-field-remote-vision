#version 410

in vec2 in_pixel;

uniform sampler2DRect vkDepthMap;
uniform sampler2DRect vkImage;

uniform int ratio;

out float dzdx;
out float dzdy;
out float z;
out vec4 gcolor;

void main() {

    z = texture(vkDepthMap, in_pixel).x;

    // normal orientation
    dzdx = texture(vkDepthMap, in_pixel).y * ratio;
    dzdy = texture(vkDepthMap, in_pixel).z * ratio;

    gcolor = texture(vkImage, in_pixel);

    gl_Position = vec4(in_pixel, 0.0, 1.0);
}
