#version 410

in vec2 in_pixel;

uniform sampler2DRect vkDepthMap;
uniform sampler2DRect vkImage;
uniform sampler2DRect vkMask;

uniform int ratioDepth;
uniform int ratioImage;

out float dzdx;
out float dzdy;
out float z;
out vec4 gcolor;

void main() {

    z = texture(vkDepthMap, in_pixel).x;

    // normal orientation
    dzdx = texture(vkDepthMap, in_pixel).y * ratioDepth;
    dzdy = texture(vkDepthMap, in_pixel).z * ratioDepth;

    if(texture(vkMask, in_pixel/ratioImage).x <= 0.001) {
        gcolor = vec4(0.0, 0.0, 0.0, 0.0);
    } else {
        gcolor = texture(vkImage, in_pixel/ratioImage);
    }

    gl_Position = vec4(in_pixel, 0.0, 1.0);
}
