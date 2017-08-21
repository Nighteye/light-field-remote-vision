#!/bin/bash

cd cocolib
./configure-cuda.sh
make -j8
cd ..

# QT4 development tools
QMAKE="qmake-qt4"

# run qmake to create Makefile
echo "Running 'qmake-qt4' for ULF ..."
${QMAKE} ULF.pro
if [ $? -ne 0 ]; then
  echo "  ERROR: qmake-qt4 returned an error."
  exit
else 
  echo "  success."
fi

# clean up and build
echo "Cleaning up previous builds ..." ; make --quiet clean && make -j8
