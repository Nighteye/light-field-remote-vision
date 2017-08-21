#version 410

in vec2 textureCoord;

uniform sampler2DRect tauk;
uniform sampler2DRect sourceImage;

uniform int W;
uniform int H;

out vec4 warpedImage;

// compute tau warps and deformation weights
// -2: point's not reconstructed
// -1: backward visibility = false. Means the point in vk is not visible from u

void main() {

    // no need to flip warp
    vec2 xu = texture(tauk, textureCoord).xy;

    if( xu.x < 0 || W < xu.x ||
        xu.y < 0 || H < xu.y ) {

        warpedImage = vec4(0, 0, 0, 1.0);

    } else {

        warpedImage = vec4(texture(sourceImage, xu).xyz, 1.0);
    }
}
