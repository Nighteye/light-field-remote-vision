
# Define your environement variables and options here
# include and library paths
LOCAL_INCLUDE_PATH=/usr/local/include
LOCAL_LIB_PATH=/usr/local/lib

# CUDA include and library paths
LOCAL_CUDA_INCLUDE_PATH=/usr/local/cuda/include
LOCAL_CUDA_LIB_PATH=/usr/local/cuda/lib64

# NVCC extra options (used for cuda files compilation)
#NVCC_EXTRA_LOCAL_FLAGS="-g -m64 -O3"
NVCC_EXTRA_LOCAL_FLAGS="-g -m64 -O99"
NVCC_EXTRA_LOCAL_LINK_FLAGS=""

# GCC extra options (used for .cpp files compilation)
#GCC_EXTRA_LOCAL_FLAGS="-g -pipe -Wall -fPIC -O99"
GCC_EXTRA_LOCAL_FLAGS="-g -pipe -Wall -Wno-unused-function -O99"
GCC_EXTRA_LOCAL_LINK_FLAGS=""

# OpenCL
LOCAL_OPENCL_INCLUDE_PATH=/System/Library/Frameworks/OpenCL.framework/Versions/A/Headers
OPENCL_LIB="-framework OpenCL"
#OPENCL_LIB=-lOpenCL

# use the -spec option ONLY if you have a specific configuration
#QMAKE_OPTIONS="-spec macx-xcode"
