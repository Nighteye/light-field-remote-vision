#!/bin/bash
# Create new cocolib defs and lib config file
cp cocolib/defs.h.in cocolib/defs.h

rm -f extra_libs.pri
touch extra_libs.pri

rm -f ann_lib.pri
touch ann_lib.pri

rm -f extra_nvcc_flags.pri
touch extra_nvcc_flags.pri

rm -f extra_qmake_flags.pri
touch extra_qmake_flags.pri

rm -r nvcc_link.pri
touch nvcc_link.pri


# Check for nVidias CUDA compiler
echo "Testing for 'nvcc' compiler ..."
if test "$(which nvcc)" == ""
then
  echo
  echo "*** FAILED TO LOCATE CUDA COMPILER ***"
  echo
  echo "The CUDA compiler 'nvcc' is not in the current path."
  echo "To compile cocolib and the examples, download and install the"
  echo "CUDA toolkit version 4.2 or higher.
  echo ""
  echo "On many apt-based distributions, it is available as a package"
  echo "'nvidia-cuda-toolkit' or similar.
  echo ""
  echo" Otherwise, you may obtain it from the CUDA zone on"
  echo "www.nvidia.com".
  echo
  echo "After manual installation, make sure your PATH variable includes the"
  echo "install path to the 'nvcc' executable (default /usr/local/cuda/bin)."
  echo
  exit
else
  echo "  nVidia CUDA compiler detected ... " `which nvcc`
fi

source local_flags.inc

# Check for supported toolkit version
# OUTDATED TEST, SHOULD ALWAYS WORK NOW
# SINCE IT'S PART OF THE PACKAGE DATABASE
#
#nvcc_test=$(nvcc --version | grep "release 4.2")
#if [ -z "$nvcc_test" ]; then
#    echo "WARNING: CUDA toolkit is not version 4.2."
#    echo "         If you have a lower version number,"
#    echo "         you have to make sure to use gcc version 4.4"
#    echo "         or older."
#else
#    echo "  CUDA toolkit version 4.2 detected."
#fi

# Check if minimal example compiles
echo "Testing minimal example compilation ..."
nvcc -o ./config_tests/test_compute_capability ./config_tests/test_compute_capability.cu -lcudart
if [ $? -ne 0 ]; then
  echo "  ERROR: compiler returned an error."
  exit
else 
  echo "  compiled successfully."
fi

# Check compute capability of the device
rm -f extra_nvcc_flags.pri
touch extra_nvcc_flags.pri

./config_tests/test_compute_capability
COMPUTE_CAPABILITY=$?
if [ $COMPUTE_CAPABILITY -lt 13 ]; then
    echo "WARNING: Your GPU has compute capability less than 1.3."
    echo "         This is not completely supported by cocolib - some algorithms may fail."
else
    echo "  GPU compute capability at least 1.3, which is good."
    echo "#define COMPUTE_API_BUFFER_TYPE float*" >> cocolib/defs.h
    echo "#define COMPUTE_API_CUDA_CAPABILITY "$COMPUTE_CAPABILITY >> cocolib/defs.h
#
# Vanilla GPU architecture
#
#    echo "NVFLAGS += --gpu-architecture sm_13" >> extra_nvcc_flags.inc

#
# Use the following instead to optimize GPU architecture - this will invalidate the
# reference results, since exact numeric results will now depend on compiler optimization
# strategy.
#
  echo "NVFLAGS += --gpu-architecture sm_"$COMPUTE_CAPABILITY" "$NVCC_EXTRA_LOCAL_FLAGS >> extra_nvcc_flags.pri
  echo "Configuring nvcc for compute capability "$COMPUTE_CAPABILITY" and extra local flags "$NVCC_EXTRA_LOCAL_FLAGS"."
#
fi

# Create .pri file with custom local configuration for .pro files
echo "QMAKE_CXXFLAGS += "$GCC_EXTRA_LOCAL_FLAGS" -I"$LOCAL_INCLUDE_PATH" -I"$LOCAL_CUDA_INCLUDE_PATH   >> extra_qmake_flags.pri
#echo "QMAKE_MACOSX_DEPLOYMENT_TARGET = 10.7" >> extra_qmake_flags.pri

echo "LIBS += "$GCC_EXTRA_LOCAL_LINK_FLAGS" -L"$LOCAL_LIB_PATH" -L"$LOCAL_CUDA_LIB_PATH >> extra_libs.pri
#echo "QMAKE_LINK = nvcc"  >> nvcc_link.pri
#

# Check for GSL library
echo "Testing for GSL library ..."
nvcc -o ./config_tests/test_gsl ./config_tests/test_gsl.cu -lgsl -lgslcblas -lcudart -I$LOCAL_INCLUDE_PATH -L$LOCAL_LIB_PATH $NVCC_EXTRA_LOCAL_FLAGS $NVCC_EXTRA_LOCAL_LINK_FLAGS
if [ $? -ne 0 ]; then
  echo "  ERROR: compiler returned an error."
  echo "         GSL library might not be installed."
  echo "         On Ubuntu systems, this is the package libgsl0-dev."
  exit
else 
  echo "  compiled successfully."
fi
./config_tests/test_gsl
if [ $? -ne 0 ]; then
    echo "WARNING: GSL example returned an error code."
    echo "         Something might be wrong with the installation."
else
    echo "  run successfully."
fi


# Check for ANN library
echo "Testing for ANN library ..."
nvcc -o ./config_tests/test_ann ./config_tests/test_ann.cu -lANN -I$LOCAL_INCLUDE_PATH -L$LOCAL_LIB_PATH $NVCC_EXTRA_LOCAL_FLAGS $NVCC_EXTRA_LOCAL_LINK_FLAGS
if [ $? -ne 0 ]; then
  echo "  failed to compile ANN library example."
  echo "  the library is probably not installed,"
  echo "  cocolib will compile without ANN support."
  echo "//#define LIB_ANN" >> cocolib/defs.h
else 
  echo "  compiled successfully, library seems to be installed."
  echo "#define LIB_ANN" >> cocolib/defs.h
  echo "LIBS += -lANN" >> ann_lib.pri
fi
echo "#endif" >> cocolib/defs.h



# QT4 development tools
echo "Testing for qt4 development tools ..."
QMAKE="qmake-qt4 $QMAKE_OPTIONS"
qt4_test=$(${QMAKE} --version | grep "version 4.")
if [ -z "$qt4_test" ]; then
  # try with plain qmake
  echo "qmake-qt4 failed: Testing with qmake ..."
  QMAKE="qmake $QMAKE_OPTIONS"
  qt4_test=$(${QMAKE} --version | grep "version 4.")
  if [ -z "$qt4_test" ]; then
    echo "  ERROR: failed to detect Qt4."
    echo "         Make sure you have libqt4-dev installed."
    exit
  else
    echo "  success."
  fi
else 
  echo "  success."
fi

# creating CUDA project files
echo "Generating CUDA project files ..."
cp ./cocolib.pro.cuda ./cocolib.pro

sed '1 i\
# AUTO-GENERATED BY CONFIGURE, DO NOT EDIT\
' cocolib.pro > cocolib.pro.bak
mv cocolib.pro.bak cocolib.pro

cp ./cocolib/cocolib.pro.cuda ./cocolib/cocolib.pro
sed '1 i\
# AUTO-GENERATED BY CONFIGURE, DO NOT EDIT\
' ./cocolib/cocolib.pro > ./cocolib/cocolib.pro.bak
mv ./cocolib/cocolib.pro.bak ./cocolib/cocolib.pro

# run qmake to create Makefile
echo "Running 'qmake-qt4' for cocolib ..."
${QMAKE} cocolib.pro
if [ $? -ne 0 ]; then
  echo "  ERROR: qmake-qt4 returned an error."
  exit
else 
  echo "  success."
fi

# clean up
echo "Cleaning up previous builds ..."
make --quiet clean

# done.
echo "Ready, run 'make' to build cocolib."
