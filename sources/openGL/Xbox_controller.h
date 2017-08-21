#ifndef _XBOX_CONTROLLER_H_
#define _XBOX_CONTROLLER_H_



#ifdef __APPLE__
enum XBOX_AXES{
  JOY_LEFT_X = 0,
  JOY_LEFT_Y,
  JOY_RIGHT_X,
  JOY_RIGHT_Y,
  LT_BUTTON,
  RT_BUTTON,
  NB_AXES
};
enum XBOX_BUTTONS{
  PAD_UP,
  PAD_DOWN,
  PAD_LEFT,
  PAD_RIGHT,
  START_BUTTON,
  BACK_BUTTON,
  LEFT_JOY,
  RIGHT_JOY,
  LB_BUTTON,
  RB_BUTTON,
  XBOX_BUTTON,
  A_BUTTON,
  B_BUTTON,
  X_BUTTON,
  Y_BUTTON,
  NB_BUTTON
};
#else // NO MACOS
enum XBOX_AXES{
  JOY_LEFT_X = 0,
  JOY_LEFT_Y,
  LT_BUTTON,
  JOY_RIGHT_X,
  JOY_RIGHT_Y,
  RT_BUTTON,
  PAD_X,
  PAD_Y,
  NB_AXES
};
enum XBOX_BUTTONS{
  A_BUTTON = 0,
  B_BUTTON,
  X_BUTTON,
  Y_BUTTON,
  LB_BUTTON,
  RB_BUTTON,
  BACK_BUTTON,
  START_BUTTON,
  NB_BUTTON
};
#endif

const float PAD_deadzone = 0.2; // in pad units [0,1]
const float PAD_trans_speed = 0.2; // in world units
const float PAD_rotation_speed = 0.005; // in radians
const float PAD_focal_increment = 200.; // in pixels


float pad_normalize(float value) {
	if (value >0 ) {
		return (value - PAD_deadzone) * 1. / (1.-PAD_deadzone);
	}
	return (value + PAD_deadzone) * 1. / (1.-PAD_deadzone);
}


float getJoyLeftX(float *axes, unsigned char* buttons) {  return axes[JOY_LEFT_X];}
float getJoyLeftY(float *axes, unsigned char* buttons) {  return axes[JOY_LEFT_Y];}
float getJoyRightX(float *axes, unsigned char* buttons) { return axes[JOY_RIGHT_X];}
float getJoyRightY(float *axes, unsigned char* buttons) { return axes[JOY_RIGHT_Y];}
float getJoyLeftTop(float *axes, unsigned char* buttons) { return axes[LT_BUTTON];}
float getJoyRightTop(float *axes, unsigned char* buttons) { return axes[RT_BUTTON];}

bool button_RB_Pressed(float *axes, unsigned char* buttons) {return buttons[RB_BUTTON];}
bool button_LB_Pressed(float *axes, unsigned char* buttons) {return buttons[LB_BUTTON];}

bool button_A_Pressed(float *axes, unsigned char* buttons) {return buttons[A_BUTTON];}
bool button_B_Pressed(float *axes, unsigned char* buttons) {return buttons[B_BUTTON];}
bool button_X_Pressed(float *axes, unsigned char* buttons) {return buttons[X_BUTTON];}
bool button_Y_Pressed(float *axes, unsigned char* buttons) {return buttons[Y_BUTTON];}

bool button_BACK_Pressed(float *axes, unsigned char* buttons) {return buttons[BACK_BUTTON];}
bool button_START_Pressed(float *axes, unsigned char* buttons) {return buttons[START_BUTTON];}

bool button_XBOX_Pressed(float *axes, unsigned char* buttons) {
#ifdef __APPLE__
  return buttons[XBOX_BUTTON];
#else
  return 0;
#endif
}

bool button_PAD_LEFT_Pressed(float *axes, unsigned char* buttons) {
#ifdef __APPLE__
  return buttons[PAD_LEFT];
#else
  return (axes[PAD_X] > 0);
#endif
}

bool button_PAD_RIGHT_Pressed(float *axes, unsigned char* buttons) {
#ifdef __APPLE__
  return buttons[PAD_RIGHT];
#else
  return (axes[PAD_X] < 0);
#endif
}

bool button_PAD_UP_Pressed(float *axes, unsigned char* buttons) {
#ifdef __APPLE__
  return buttons[PAD_UP];
#else
  return (axes[PAD_Y] > 0);
#endif
}

bool button_PAD_DOWN_Pressed(float *axes, unsigned char* buttons) {
#ifdef __APPLE__
  return buttons[PAD_DOWN];
#else
  return (axes[PAD_Y] < 0);
#endif
}
  

/*void xbox_threshold();

class joystickData {
	public:
		const int nbAxes=8;
		const int nbButtons = 11;

		float m_axes[8];
		char m_buttons[11];
};

class XboxGamepad {
		float deadzoneX;
		float deadzoneY;

		XboxGamepad() : deadzoneX(0.1), deadzoneY(0.1) {}

		addmeasure();
};

float getAxeValue*/



#endif
