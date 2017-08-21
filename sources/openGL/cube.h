#ifndef CUBE_H
#define CUBE_H

#include <string>
#include <GL/glew.h>

// Includes GLM
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader.h"
#include "texture.h"

// Macro utile au VBO

#ifndef BUFFER_OFFSET

#define BUFFER_OFFSET(offset) ((char*)NULL + (offset))

#endif

class FrameBuffer;

class Cube {

public:

    Cube(float taille, std::string const vertexShader, std::string const fragmentShader);
    ~Cube();

    void charger();
    void afficher(glm::mat4 &projection, glm::mat4 &modelview);
    void updateVBO(void *donnees, int tailleBytes, int decalage);
    void initTextureShader( Shader *shader );
    void renderInFB( const glm::mat4 &renderView, FrameBuffer* FBO, Texture* inputView );

    // fill depthMap texture with z-buffer rendering
    void renderDepth( FrameBuffer* FBO, Texture* depthMap, Shader* depthFromMeshShader,
                      const glm::mat4 &renderView, const glm::mat3 &vk_K, const glm::mat3 &vk_R, const glm::vec3 &vk_t,
                      const float offset );

protected:

    Shader m_shader;
    float m_vertices[108];
    float m_couleurs[108];

    GLuint m_vboID;
    int m_tailleVerticesBytes;
    int m_tailleCouleursBytes;
    GLuint m_vaoID;
};

class Crate : public Cube {

public:

    Crate(float taille, std::string const vertexShader, std::string const fragmentShader, std::string const texture);
    ~Crate();

    void load();
    void display(glm::mat4 &projection, glm::mat4 &modelview);


private:

    Texture m_texture;
    float m_coordTexture[72];
    int m_tailleCoordTextureBytes;
};

#endif /* #ifndef CUBE_H */
