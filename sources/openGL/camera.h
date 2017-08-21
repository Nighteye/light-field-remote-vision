#ifndef DEF_CAMERA
#define DEF_CAMERA


// Includes GLM

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>


// Autres includes

#include "input.h"


// Classe

class Camera
{
    public:

    Camera();
    Camera(glm::vec3 position, glm::vec3 pointCible, glm::vec3 axeVertical, float sensibilite, float vitesse);
    ~Camera();

    void orienter(int xRel, int yRel);
    void deplacer(Input const &input);
    void lookAt(glm::mat4 &modelview);

    void setPointcible();
    void setPosition(glm::vec3 position);

    float getSensibilite() const;
    float getVitesse() const;

    void setSensibilite(float sensibilite);
    void setVitesse(float vitesse);


    private:

    float m_phi;
    float m_theta;
    glm::vec3 m_orientation;

    glm::vec3 m_axeVertical;
    glm::vec3 m_deplacementLateral;

    glm::vec3 m_position;
    glm::vec3 m_pointCible;

    float m_sensibilite;
    float m_vitesse;
};

#endif
