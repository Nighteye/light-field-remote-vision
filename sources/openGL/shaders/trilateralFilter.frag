#version 410

in vec3 currentPos;
in vec3 fnormal;

uniform sampler2DRect sourceTex;
uniform mat3 vi_Kinv;
uniform mat3 vi_R;
uniform vec3 vi_C;

out vec4 out_depthTex;

void main() {

    mat3 Cdd_inv = 10.0f*mat3(1.0);
    vec3 N = fnormal;

    vec3 xm = vec3(gl_FragCoord.xy, 1.0);
    vec3 xp = inverse(vi_Kinv) * vi_R * (currentPos - vi_C);
    float z = xp.z;
    xp /= z;

    // in world coordinates
    vec3 r = transpose(vi_R) * vi_Kinv * xm;
    vec3 Xr = vi_C + (dot(N, currentPos - vi_C) / dot(N, r)) * r;

    float F = exp( - dot(Xr - currentPos, Cdd_inv*(Xr - currentPos)) );

    // intensity sigma is 1
    float sigma = 1.0;

    // luminance of vertex
    float Yp = 0.2126 * texture( sourceTex, xp.xy ).x + 0.7152 * texture( sourceTex, xp.xy ).y + 0.0722 * texture( sourceTex, xp.xy ).z;
    float Ym = 0.2126 * texture( sourceTex, xm.xy ).x + 0.7152 * texture( sourceTex, xm.xy ).y + 0.0722 * texture( sourceTex, xm.xy ).z;
    float G = exp( - (Ym - Yp)*(Ym - Yp) / (2*sigma*sigma) );
    float H = exp( - 5*(z-2.5) );

    // in cam coordinates
    Xr = vi_R * (Xr - vi_C);

    // no contribution when depth is negative, and normal cullface
    if(Xr.z <= 0.0
    || dot(N, Xr) > 0.0) {
        out_depthTex = vec4(0.0, 0.0, 0.0, 0.0);
    } else {
        out_depthTex = vec4(Xr.z, 0.0, 0.0, F*G*H);
    }
}
