#include "cube.h"
#include "frameBuffer.h"

#include "iostream"

// Permet d'éviter la ré-écriture du namespace glm::

using namespace glm;

// Constructeur et Destructeur

Cube::Cube(float taille, std::string const vertexShader, std::string const fragmentShader) : m_shader(vertexShader, fragmentShader), m_vboID(0),
    m_tailleVerticesBytes(108 * sizeof(float)),
    m_tailleCouleursBytes(108 * sizeof(float)), m_vaoID(0)
{
    initTextureShader( &m_shader );

    // Division de la taille

    taille /= 2;


    // Vertices temporaires

    float verticesTmp[] = {-taille, -taille, -taille,   taille, -taille, -taille,   taille, taille, -taille,     // Face 1
                           -taille, -taille, -taille,   -taille, taille, -taille,   taille, taille, -taille,     // Face 1

                           taille, -taille, taille,   taille, -taille, -taille,   taille, taille, -taille,       // Face 2
                           taille, -taille, taille,   taille, taille, taille,   taille, taille, -taille,         // Face 2

                           -taille, -taille, taille,   taille, -taille, taille,   taille, -taille, -taille,      // Face 3
                           -taille, -taille, taille,   -taille, -taille, -taille,   taille, -taille, -taille,    // Face 3

                           -taille, -taille, taille,   taille, -taille, taille,   taille, taille, taille,        // Face 4
                           -taille, -taille, taille,   -taille, taille, taille,   taille, taille, taille,        // Face 4

                           -taille, -taille, -taille,   -taille, -taille, taille,   -taille, taille, taille,     // Face 5
                           -taille, -taille, -taille,   -taille, taille, -taille,   -taille, taille, taille,     // Face 5

                           -taille, taille, taille,   taille, taille, taille,   taille, taille, -taille,         // Face 6
                           -taille, taille, taille,   -taille, taille, -taille,   taille, taille, -taille};      // Face 6


    // Couleurs temporaires

    float couleursTmp[] = {1.0, 0.0, 0.0,   1.0, 0.0, 0.0,   1.0, 0.0, 0.0,           // Face 1
                           1.0, 0.0, 0.0,   1.0, 0.0, 0.0,   1.0, 0.0, 0.0,           // Face 1

                           0.0, 1.0, 0.0,   0.0, 1.0, 0.0,   0.0, 1.0, 0.0,           // Face 2
                           0.0, 1.0, 0.0,   0.0, 1.0, 0.0,   0.0, 1.0, 0.0,           // Face 2

                           0.0, 0.0, 1.0,   0.0, 0.0, 1.0,   0.0, 0.0, 1.0,           // Face 3
                           0.0, 0.0, 1.0,   0.0, 0.0, 1.0,   0.0, 0.0, 1.0,           // Face 3

                           1.0, 0.0, 0.0,   1.0, 0.0, 0.0,   1.0, 0.0, 0.0,           // Face 4
                           1.0, 0.0, 0.0,   1.0, 0.0, 0.0,   1.0, 0.0, 0.0,           // Face 4

                           0.0, 1.0, 0.0,   0.0, 1.0, 0.0,   0.0, 1.0, 0.0,           // Face 5
                           0.0, 1.0, 0.0,   0.0, 1.0, 0.0,   0.0, 1.0, 0.0,           // Face 5

                           0.0, 0.0, 1.0,   0.0, 0.0, 1.0,   0.0, 0.0, 1.0,           // Face 6
                           0.0, 0.0, 1.0,   0.0, 0.0, 1.0,   0.0, 0.0, 1.0};          // Face 6


    // Copie des valeurs dans les tableaux finaux

    for(int i(0); i < 108; i++)
    {
        m_vertices[i] = verticesTmp[i];
        m_couleurs[i] = couleursTmp[i];
    }
}

void Cube::initTextureShader( Shader *shader ) {

    assert( shader->add() );

    // Bind shader inputs
    shader->bindAttribLocation( 0, "in_Vertex" );
    shader->bindAttribLocation( 1, "in_TexCoord0" );

    assert( shader->link() );
}


Cube::~Cube()
{
    // Destruction du VBO

    glDeleteBuffers(1, &m_vboID);


    // Destruction du VAO

    glDeleteVertexArrays(1, &m_vaoID);
}


// Méthodes

void Cube::charger() {

    //-----------------------VBO-----------------------------

    // Destruction of possible old VBO
    if(glIsBuffer(m_vboID) == GL_TRUE)
        glDeleteBuffers(1, &m_vboID);

    // VBO ID generation
    glGenBuffers(1, &m_vboID);

    // Binding VBO
    glBindBuffer(GL_ARRAY_BUFFER, m_vboID);

    // VRAM allocation
    glBufferData(GL_ARRAY_BUFFER, m_tailleVerticesBytes + m_tailleCouleursBytes, 0, GL_STATIC_DRAW);

    // Data transfer
    glBufferSubData(GL_ARRAY_BUFFER, 0, m_tailleVerticesBytes, m_vertices);
    glBufferSubData(GL_ARRAY_BUFFER, m_tailleVerticesBytes, m_tailleCouleursBytes, m_couleurs);

    // Unbinding VBO
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //-----------------------VAO-----------------------------

    // Destruction of possible old VAO
    if(glIsVertexArray(m_vaoID) == GL_TRUE)
        glDeleteVertexArrays(1, &m_vaoID);

    // VAO ID generation
    glGenVertexArrays(1, &m_vaoID);

    // Binding VAO
    glBindVertexArray(m_vaoID);

    // Binding VBO
    glBindBuffer(GL_ARRAY_BUFFER, m_vboID);

    // Access to vertices in VRAM
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
    glEnableVertexAttribArray(0);

    // Access to colors in VRAM
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(m_tailleVerticesBytes));
    glEnableVertexAttribArray(1);

    // Unbinding VBO
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Unbinding VAO
    glBindVertexArray(0);
}


void Cube::afficher(glm::mat4 &projection, glm::mat4 &modelview)
{
    // Activation du shader

    glUseProgram(m_shader.getProgramID());


    // Verrouillage du VAO

    glBindVertexArray(m_vaoID);


    // Envoi des matrices

    m_shader.setUniformMat4("modelviewProjection", projection * modelview);


    // Rendu

    glDrawArrays(GL_TRIANGLES, 0, 36);


    // Déverrouillage du VAO

    glBindVertexArray(0);


    // Désactivation du shader

    glUseProgram(0);
}


void Cube::updateVBO(void *donnees, int tailleBytes, int decalage)
{
    // Verrouillage du VBO

    glBindBuffer(GL_ARRAY_BUFFER, m_vboID);


    // Récupération de l'adresse du VBO

    void *adresseVBO = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);


    // Si l'adresse retournée est nulle alors on arrête le transfert

    if(adresseVBO == NULL)
    {
        std::cout << "Erreur au niveau de la récupération du VBO" << std::endl;
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        return;
    }


    // Mise à jour des données

    memcpy((char*)adresseVBO + decalage, donnees, tailleBytes);


    // Annulation du pointeur

    glUnmapBuffer(GL_ARRAY_BUFFER);
    adresseVBO = 0;


    // Déverrouillage du VBO

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Cube::renderInFB( const glm::mat4 &renderView, FrameBuffer* FBO, Texture* inputTex ) {

    glBindFramebuffer( GL_FRAMEBUFFER, FBO->getID() );

    uint texIdx = GL_COLOR_ATTACHMENT0;
    glFramebufferTexture2D( GL_FRAMEBUFFER, texIdx, GL_TEXTURE_RECTANGLE, inputTex->getID(), 0 );
    glDrawBuffer(texIdx); // image texture

    glUseProgram(m_shader.getProgramID());

    glBindVertexArray(m_vaoID);

    m_shader.setUniformMat4("modelviewProjection", renderView);

    glDrawArrays(GL_TRIANGLES, 0, 36);

    glBindVertexArray(0);

    glUseProgram(0);

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );
}

void Cube::renderDepth( FrameBuffer* FBO, Texture* depthMap, Shader* depthFromMeshShader,
                        const glm::mat4 &renderView, const glm::mat3 &vk_K, const glm::mat3 &vk_R, const glm::vec3 &vk_t,
                        const float offset ) {

    glBindFramebuffer( GL_FRAMEBUFFER, FBO->getID() );

    uint depthIdx = GL_COLOR_ATTACHMENT0;
    glFramebufferTexture2D( GL_FRAMEBUFFER, depthIdx, GL_TEXTURE_RECTANGLE, depthMap->getID(), 0 );
    glDrawBuffer(depthIdx); // depth map

    glUseProgram(depthFromMeshShader->getProgramID());

    glBindVertexArray(m_vaoID);

    depthFromMeshShader->setUniformMat4("renderMatrix", renderView);
    depthFromMeshShader->setUniformMat3("vk_K", vk_K);
    depthFromMeshShader->setUniformMat3("vk_R", vk_R);
    depthFromMeshShader->setUniform3fv("vk_t", vk_t);
    depthFromMeshShader->setUniformf("offset", offset);

    glDrawArrays(GL_TRIANGLES, 0, 36);

    glBindVertexArray(0);

    glUseProgram(0);

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );
}

// Class Crate

Crate::Crate(float taille, std::string const vertexShader,std::string const fragmentShader, std::string const texture) : Cube(taille, vertexShader, fragmentShader),
    m_texture(texture),
    m_tailleCoordTextureBytes(72 * sizeof(float)) {

    m_texture.load();

    float coordTextureTmp[] = {0, 0,   1, 0,   1, 1,     // Face 1
                               0, 0,   0, 1,   1, 1,     // Face 1

                               0, 0,   1, 0,   1, 1,     // Face 2
                               0, 0,   0, 1,   1, 1,     // Face 2

                               0, 0,   1, 0,   1, 1,     // Face 3
                               0, 0,   0, 1,   1, 1,     // Face 3

                               0, 0,   1, 0,   1, 1,     // Face 4
                               0, 0,   0, 1,   1, 1,     // Face 4

                               0, 0,   1, 0,   1, 1,     // Face 5
                               0, 0,   0, 1,   1, 1,     // Face 5

                               0, 0,   1, 0,   1, 1,     // Face 6
                               0, 0,   0, 1,   1, 1};    // Face 6

    for(int i (0); i < 72; i++)
        m_coordTexture[i] = coordTextureTmp[i];
}

Crate::~Crate() {

}

void Crate::load() {

    // VBO config

    if(glIsBuffer(m_vboID) == GL_TRUE)
        glDeleteBuffers(1, &m_vboID);

    glGenBuffers(1, &m_vboID);

    glBindBuffer(GL_ARRAY_BUFFER, m_vboID);

    glBufferData(GL_ARRAY_BUFFER, m_tailleVerticesBytes + m_tailleCoordTextureBytes, 0, GL_STATIC_DRAW);

    glBufferSubData(GL_ARRAY_BUFFER, 0, m_tailleVerticesBytes, m_vertices);
    glBufferSubData(GL_ARRAY_BUFFER, m_tailleVerticesBytes, m_tailleCoordTextureBytes, m_coordTexture);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // VAO config

    if(glIsVertexArray(m_vaoID) == GL_TRUE)
        glDeleteVertexArrays(1, &m_vaoID);

    glGenVertexArrays(1, &m_vaoID);

    glBindVertexArray(m_vaoID);

    glBindBuffer(GL_ARRAY_BUFFER, m_vboID);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(m_tailleVerticesBytes));
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(0);
}

void Crate::display(glm::mat4 &projection, glm::mat4 &modelview) {

    glUseProgram(m_shader.getProgramID());

    glBindVertexArray(m_vaoID);

    m_shader.setUniformMat4("modelviewProjection", projection * modelview);

    glBindTexture(GL_TEXTURE_2D, m_texture.getID());

    glDrawArrays(GL_TRIANGLES, 0, 36);

    glBindTexture(GL_TEXTURE_2D, 0);

    glBindVertexArray(0);

    glUseProgram(0);
}
