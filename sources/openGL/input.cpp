#include "input.h"

// Constructor and Destructor

Input::Input() : m_x(0), m_y(0), m_xRel(0), m_yRel(0), m_end(false) {

    if(SDL_NumJoysticks() > 0) {

        SDL_JoystickEventState(SDL_ENABLE);

        _inJoystick = (InputJoystick*) malloc(sizeof(InputJoystick));

        _inJoystick->_joystick = SDL_JoystickOpen(0);
        _inJoystick->_numero = 0;
        _inJoystick->_boutons = (char*) malloc(SDL_JoystickNumButtons(_inJoystick->_joystick) * sizeof(char));
        _inJoystick->_axes = (int*) malloc(SDL_JoystickNumAxes(_inJoystick->_joystick) * sizeof(int));
        _inJoystick->_chapeaux = (int*) malloc(SDL_JoystickNumHats(_inJoystick->_joystick) * sizeof(int));

        for(int j = 0 ; j < SDL_JoystickNumButtons(_inJoystick->_joystick) ; ++j) {
            _inJoystick->_boutons[j] = 0;
        }

        for(int j = 0 ; j < SDL_JoystickNumAxes(_inJoystick->_joystick) ; ++j) {
            _inJoystick->_axes[j] = 0;
        }

        for(int j = 0 ; j < SDL_JoystickNumHats(_inJoystick->_joystick) ; ++j) {
            _inJoystick->_chapeaux[j] = 0;
        }
    }

    else {

        _inJoystick = 0;
    }

    // Initialisation du tableau m_keys[]

    for(int i(0); i < SDL_NUM_SCANCODES; i++) {

        m_keys[i] = false;
    }

    // Initialisation du tableau m_button[]

    for(int i(0); i < 8; i++) {

        m_button[i] = false;
    }
}

Input::~Input() {

    if(_inJoystick != 0) {

        SDL_JoystickEventState(SDL_DISABLE);

        _inJoystick->_numero = 0;
        free(_inJoystick->_boutons);
        free(_inJoystick->_axes);
        free(_inJoystick->_chapeaux);
        SDL_JoystickClose(_inJoystick->_joystick);

        free(_inJoystick);
        _inJoystick = 0;
    }
}

// Methods

void Input::updateEvents() {

    // Pour éviter des mouvements fictifs de la souris, on réinitialise les coordonnées relatives
    m_xRel = 0;
    m_yRel = 0;

    // Boucle d'évènements
    while(SDL_PollEvent(&m_evenements)) {

        // Switch sur le type d'évènement

        switch(m_evenements.type) {

        // Cas d'une touche enfoncée

        case SDL_KEYDOWN:
            m_keys[m_evenements.key.keysym.scancode] = true;
            break;

            // Cas d'une touche relâchée

        case SDL_KEYUP:
            m_keys[m_evenements.key.keysym.scancode] = false;
            break;

            // Cas de pression sur un bouton de la souris

        case SDL_MOUSEBUTTONDOWN:

            m_button[m_evenements.button.button] = true;

            break;

            // Cas du relâchement d'un bouton de la souris

        case SDL_MOUSEBUTTONUP:

            m_button[m_evenements.button.button] = false;

            break;

            // Cas d'un mouvement de souris

        case SDL_MOUSEMOTION:

            m_x = m_evenements.motion.x;
            m_y = m_evenements.motion.y;

            m_xRel = m_evenements.motion.xrel;
            m_yRel = m_evenements.motion.yrel;

            break;

            // Cas de la fermeture de la fenêtre

        case SDL_WINDOWEVENT:

            if(m_evenements.window.event == SDL_WINDOWEVENT_CLOSE) {

                m_end = true;
            }

            break;

        default:
            break;
        }

        if(_inJoystick != 0) {

            switch(m_evenements.type) {

            case SDL_JOYBUTTONDOWN:
                _inJoystick->_boutons[m_evenements.jbutton.button] = 1;
                break;

            case SDL_JOYBUTTONUP:
                _inJoystick->_boutons[m_evenements.jbutton.button] = 0;
                break;

            case SDL_JOYAXISMOTION:
                _inJoystick->_axes[m_evenements.jaxis.axis] = m_evenements.jaxis.value;
                break;

            case SDL_JOYHATMOTION:
                _inJoystick->_chapeaux[m_evenements.jhat.hat] = m_evenements.jhat.value;
                break;

            default:
                break;
            }
        }
    }
}


bool Input::end() const {

    return m_end;
}


void Input::showCursor(bool response) const {

    if(response) {

        SDL_ShowCursor(SDL_ENABLE);

    } else {

        SDL_ShowCursor(SDL_DISABLE);
    }
}


void Input::catchCursor(bool response) const {

    if(response) {

        SDL_SetRelativeMouseMode(SDL_TRUE);

    } else {

        SDL_SetRelativeMouseMode(SDL_FALSE);
    }
}



// Getters

bool Input::getKey(const SDL_Scancode key) const {

    return m_keys[key];
}


bool Input::getButton(const Uint8 button) const {

    return m_button[button];
}


bool Input::moveMouse() const {

    if(m_xRel == 0 && m_yRel == 0) {

        return false;

    } else {

        return true;
    }
}


// Getters concernant la position du curseur

int Input::getX() const {

    return m_x;
}

int Input::getY() const {

    return m_y;
}

int Input::getXRel() const {

    return m_xRel;
}

int Input::getYRel() const {

    return m_yRel;
}
