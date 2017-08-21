#version 410

in vec2 textureCoord;

uniform sampler2DRect vkDepthMap;
uniform mat3 vk_K_inv;

out vec3 orthoZ;

void main() {

    float radialZ = texture(vkDepthMap, textureCoord).x;
    float radialDZDX = texture(vkDepthMap, textureCoord).y;
    float radialDZDY = texture(vkDepthMap, textureCoord).z;


    vec3 unProjectedX = vk_K_inv * vec3(textureCoord, 1.0);
    float d = length(unProjectedX);

    orthoZ = vec3(radialZ/d,
                  radialDZDX/d-radialZ*dot(unProjectedX, vk_K_inv[0])/pow(d, 3.0),
                  radialDZDY/d-radialZ*dot(unProjectedX, vk_K_inv[1])/pow(d, 3.0));
}
