#version 410

in vec2 textureCoord;

uniform sampler2DRect accuDepthMap;
uniform sampler2DRect inDepthMap;
uniform float scale;
uniform float invalidDepth;

out vec3 outDepthMap;

void main() {

    vec3 accuDepth = texture(accuDepthMap, textureCoord).xyz;
    // don't forget to scale the normals
    vec3 inDepth = vec3(texture(inDepthMap, textureCoord).x, texture(inDepthMap, textureCoord).y/scale, texture(inDepthMap, textureCoord).z/scale);

    if(accuDepth.x > invalidDepth-1) { // no depth yet, we set it to new-scale depth

        outDepthMap = inDepth;

    } else { // depth at better scale, we don't touch the pixel

        outDepthMap = accuDepth;
    }
}
