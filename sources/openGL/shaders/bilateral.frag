#version 410

in vec2 textureCoord;

uniform sampler2DRect myTexture;
uniform sampler2DRect sparseDepthMap;

uniform int horizontal;

out vec4 depth;

float addContribution( vec2 texCoord, float sigmaIntensity, float sigmaSpatial, float Yp,
                      sampler2DRect myTexture, sampler2DRect sparseDepthMap ) {

    float Ym, F, G;

    Ym = 0.2126 * texture( myTexture, texCoord ).x + 0.7152 * texture( myTexture, texCoord ).y + 0.0722 * texture( myTexture, texCoord ).z;
    G = exp( - (Ym - Yp)*(Ym - Yp) / (2*sigmaIntensity*sigmaIntensity) );
    F = exp( - dot(texCoord-textureCoord, texCoord-textureCoord) / (2*sigmaSpatial*sigmaSpatial) );
    return F * G;
}

void main() {

    vec3 sum = vec3(0, 0, 0);
    float norm = 0.0;
    float weight = 0.0;
    float z = 0.0;
    float dzdx = 0.0;
    float dzdy = 0.0;
    float sigmaIntensity = 0.001;
    float sigmaSpatial = 1.0;
    int kernelSize = 20;

    // current texture coordinates
    vec2 texCoord = textureCoord;

    // luminance of current pixel
    float Yp = 0.2126 * texture( myTexture, texCoord ).x + 0.7152 * texture( myTexture, texCoord ).y + 0.0722 * texture( myTexture, texCoord ).z;

    z = texture( sparseDepthMap, texCoord ).x;
    dzdx = texture( sparseDepthMap, texCoord ).y;
    dzdy = texture( sparseDepthMap, texCoord ).z;
    if(z == 0) {
        weight = 0;
    } else {
        weight = addContribution( texCoord, sigmaIntensity, sigmaSpatial, Yp, myTexture, sparseDepthMap );
    }
    sum += weight * vec3(z, dzdx, dzdy);

    norm += weight;

    for( int i = 1 ; i <= kernelSize-2 ; ++i ) {

        if(horizontal == 1) { // blur in x (horizontal)
            texCoord = vec2(textureCoord.x - float(i), textureCoord.y);
        } else { // blur in y (vertical)
            texCoord = vec2(textureCoord.x, textureCoord.y - float(i));
        }

        z = texture( sparseDepthMap, texCoord ).x;
        dzdx = texture( sparseDepthMap, texCoord ).y;
        dzdy = texture( sparseDepthMap, texCoord ).z;
        if(z == 0) {
            weight = 0;
        } else {
            weight = addContribution( texCoord, sigmaIntensity, sigmaSpatial, Yp, myTexture, sparseDepthMap );
        }
        sum += weight * vec3(z, dzdx, dzdy);
        norm += weight;

        if(horizontal == 1) { // blur in x (horizontal)
            texCoord = vec2(textureCoord.x + float(i), textureCoord.y);
        } else { // blur in y (vertical)
            texCoord = vec2(textureCoord.x, textureCoord.y + float(i));
        }

        z = texture( sparseDepthMap, texCoord ).x;
        dzdx = texture( sparseDepthMap, texCoord ).y;
        dzdy = texture( sparseDepthMap, texCoord ).z;
        if(z == 0) {
            weight = 0;
        } else {
            weight = addContribution( texCoord, sigmaIntensity, sigmaSpatial, Yp, myTexture, sparseDepthMap );
        }
        sum += weight * vec3(z, dzdx, dzdy);
        norm += weight;
    }

    if( norm == 0.0 ) {
        depth = vec4( 0.0, 0.0, 0.0, 1.0 );
    } else {
        depth = vec4( sum / norm, 1.0 );
    }
}
