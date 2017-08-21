#version 410

in vec2 textureCoord;

uniform sampler2DRect myTexture;

out vec4 outColor;

void main() {

    vec4 color = texture(myTexture, textureCoord);
    float weights = texture(myTexture, textureCoord).w;

    if(weights == 0) {
        outColor = vec4(0.0, 0.0, 0.0, 0.0);
    } else {
        outColor = color/weights;
    }
}
