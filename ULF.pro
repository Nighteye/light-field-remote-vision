TEMPLATE	= app
CONFIG		+= console qt debug c++11
CONFIG      	-= app_bundle

HEADERS		= sources/*.h\
		  sources/*.cuh\
		  sources/openGL/*.h\
                  sources/optical_flow/*.h\

HEADERS		-= sources/openGL/map.h\

SOURCES		= sources/*.cpp\
                  sources/openGL/*.cpp\
                  sources/optical_flow/*.cpp\

SOURCES		-= sources/openGL/map.cpp\

SOURCES		-= sources/conj_grad_solv.cpp\
                 
CUDA_SOURCES    = sources/*.cu\
                  sources/openGL/*.cu\

CUDA_SOURCES    -= sources/conj_grad_solv.cu\

OTHER_FILES     = sources/openGL/shaders/*.frag\
                  sources/openGL/shaders/*.vert\
                  sources/openGL/shaders/*.geom\

DISTFILES =

TARGET		= ULF

########################################################################
#  OpenEXR
########################################################################
CONFIG += openexr
openexr {
INCLUDEPATH += /usr/include/OpenEXR
LIBS += -lIlmImf -lIex -lImath -lHalf
}

########################################################################
#  Eigen
########################################################################
INCLUDEPATH += /usr/include/eigen3

########################################################################
#  ceres
########################################################################
INCLUDEPATH += /usr/local/include/ceres
LIBS += -lceres -lgomp -lcholmod -lblas -llapack

########################################################################
#  GLFW (not sure if necessary)
########################################################################
INCLUDEPATH += /usr/include/GLFW
LIBS += -lglfw -lGL -lGLEW -lpng -ltiff

########################################################################
#  SDL2 (to catch events and manage timers)
########################################################################
INCLUDEPATH += /usr/include/SDL2
LIBS += -lSDL2_image -lSDL2

########################################################################
#  OPENCV (optical flow)
########################################################################

INCLUDEPATH += /usr/local/include/opencv2
LIBS += -lopencv_core -lopencv_imgproc -lopencv_imgcodecs 

########################################################################
#  GLOG (optical flow)
########################################################################

INCLUDEPATH += /usr/include/glog
LIBS += -lglog

########################################################################
#  Adress Sanitizer (to debug)
########################################################################

#QMAKE_LFLAGS += -fsanitize=address
#QMAKE_CXXFLAGS += -fsanitize=address

########################################################################
#  PLY
########################################################################

INCLUDEPATH += -I../rply
LIBS += -L../rply -lrply

########################################################################
#  cocolib
########################################################################

QMAKE_CXXFLAGS += -I./cocolib/cocolib
QMAKE_CXXFLAGS += -std=c++11
LIBS += -L./cocolib/cocolib -lcocolib -lgsl -lgslcblas -lhdf5 -lhdf5_hl -lz -lcudart -lX11

include("./cocolib/extra_qmake_flags.pri")
include("./cocolib/extra_nvcc_flags.pri")
include("./cocolib/extra_libs.pri")
include("./cocolib/ann_lib.pri")
include("./cocolib/nvcc_link.pri")

OBJECTS_DIR = ./obj

QMAKE_CXXFLAGS_NOC11 = $${QMAKE_CXXFLAGS}
QMAKE_CXXFLAGS_NOC11 -= -std=c++11

########################################################################
#  CUDA
########################################################################

cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.obj
cuda.commands = nvcc -c $$NVFLAGS -Xcompiler $$join(QMAKE_CXXFLAGS_NOC11,",") $$join(INCLUDEPATH,'" -I "','-I "','"') ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}

cuda.dependcy_type = TYPE_C

cuda.input = CUDA_SOURCES
QMAKE_EXTRA_COMPILERS += cuda
########################################################################

