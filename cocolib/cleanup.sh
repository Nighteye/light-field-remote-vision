#!/bin/bash

# tests compiled in configure script
rm -rf config_tests/test_ann
rm -rf config_tests/test_compute_capability
rm -rf config_tests/test_cudpp
rm -rf config_tests/test_gsl

# tools
cd tools/parse_lfa_results 
make distclean
rm -rf Makefile
rm -rf build
rm -rf parse_lfa_results.xcodeproj
rm -rf obj
cd ../../

cd tools/reference_compare
make distclean
rm -rf Makefile
rm -rf build
rm -rf reference_compare.xcodeproj
rm -rf obj
cd ../../

# coco library
cd cocolib
make distclean
rm -rf Makefile
rm -rf build
rm -rf cocolib.xcodeproj
rm -rf obj
cd ..

# lightfield suite
cd lightfields
make distclean
rm -rf Makefile
rm -rf build
rm -rf lightfields.xcodeproj
rm -rf lightfields
rm -rf obj
cd ..

# examples 
rm -rf examples/out/tv
rm -rf examples/out/vtv
rm -rf examples/out/multilabel

cd examples/3rdparty/GCO3
make distclean
rm -rf Makefile
rm -rf build
rm -rf gco.xcodeproj
rm -rf obj
cd ../../..

cd examples/3rdparty/TRW_S
make distclean
rm -rf Makefile
rm -rf build
rm -rf trws.xcodeproj
rm -rf obj
cd ../../..

cd examples
make distclean
rm -rf Makefile
rm -rf build
rm -rf coco_ip.xcodeproj
rm -rf obj
cd ..

cd examples++
make distclean
rm -rf Makefile
rm -rf build
rm -rf coco.xcodeproj
rm -rf obj
cd ..

# base Makefile
make distclean
rm -rf Makefile
rm -rf build
rm -rf cocolib.xcodeproj
rm -rf obj

echo "Done cleaning"

