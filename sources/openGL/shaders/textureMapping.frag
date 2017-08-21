#version 410

in vec3 vi_coordinate;

uniform sampler2DRect myTexture;

uniform float vi_width;
uniform float vi_height;

out vec4 color;

void main() {

    // Projective correction
    vec2 vi_coords = vi_coordinate.xy/vi_coordinate.z;
    float vi_depth = vi_coordinate.z;
    // reverse pixels
    vec2 vi_texture_coords = vec2(vi_coords.x, vi_coords.y);

    // IN IMAGE TEST: if point projects out of the texturing image, it is not visible: discard
    if ( vi_coords.x < 0 || vi_width < vi_coords.x ||
         vi_coords.y < 0 || vi_height < vi_coords.y ) {
        // be carefull not to discard, otherwise the z-buffering for the fragment is not done
        color = vec4(0,0,0,0);
        return;
    }

    // TEST if point is behind camera vi_depth <0
    if (vi_depth < 0) {
        // be carefull not to discard, otherwise the z-buffering for the fragment is not done
        color = vec4( 0.0, 0.0, 0.0, 0.0 );
        return;
    }

    // if using distorded coordinates, use the distorded image
    color = vec4( texture(myTexture, vi_texture_coords).xyz, 1.0 );
}
