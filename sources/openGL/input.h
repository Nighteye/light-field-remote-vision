#ifndef INPUT_H
#define INPUT_H

#include <SDL2/SDL.h>

/* structure qui gère un joystick */
typedef struct InputJoystick InputJoystick;
struct InputJoystick
{
    SDL_Joystick *_joystick;
    char *_boutons;
    int *_axes;
    int *_chapeaux;
    int _numero;
};

class Input {

    public:

    Input();
    ~Input();

    void updateEvents();
    bool end() const;
    void showCursor(bool response) const;
    void catchCursor(bool response) const;

    bool getKey(const SDL_Scancode key) const;
    bool getButton(const Uint8 button) const;
    bool moveMouse() const;

    int getX() const;
    int getY() const;

    int getXRel() const;
    int getYRel() const;

    private:

    SDL_Event m_evenements;
    bool m_keys[SDL_NUM_SCANCODES];
    bool m_button[8];

    int m_x;
    int m_y;
    int m_xRel;
    int m_yRel;

    // joystick
    InputJoystick *_inJoystick;

    bool m_end;
};

#endif /* #ifndef INPUT_H */

