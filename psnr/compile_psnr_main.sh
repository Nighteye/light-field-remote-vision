g++ -c -pipe -I../cocolib/cocolib -pipe -g -Wall -O99 -g -D_REENTRANT -Wall -W -DQT_OPENGL_LIB -DQT_GUI_LIB -DQT_CORE_LIB -DQT_SHARED -I/usr/lib64/qt4/mkspecs/linux-g++ -I. -I/usr/include/QtCore -I/usr/include/QtGui -I/usr/include/QtOpenGL -I/usr/include -I/usr/X11R6/include -I. -o psnr_main.o psnr_main.cpp

nvcc -o psnr psnr_main.o -L/usr/lib64 -L/usr/X11R6/lib -L../cocolib/cocolib -lcocolib -lgsl -lgslcblas -lhdf5 -lhdf5_hl -lz -lANN -lQtOpenGL -lQtGui -lQtCore -lGL -lpthread 
