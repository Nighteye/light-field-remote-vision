# INSTALL NVIDIA DRIVER
#----------------------------------------------------------------------------------------------------
# if nouveau is installed, remove it (pain in the ass), cause it's not CUDA compatible
# see post https://ask.fedoraproject.org/en/question/66187/how-to-disable-nouveau-and-install-nvidia/

# STEP 1: put nouveau on the blacklist
sudo touch /etc/modprobe.d/blacklist.conf
sudo echo "blacklist nouveau" >> /etc/modprobe.d/blacklist.conf

# STEP 2: change grub options so it doesn't automatically load nouveau when booting
# add the line if it doesn't already exist 
sudo sed -i 's/GRUB_CMDLINE_LINUX="/GRUB_CMDLINE_LINUX="nouveau.blacklist=1 /g' /etc/default/grub
# configure grub
sudo yum update grub2
sudo grub2-mkconfig -o /boot/grub2/grub
sudo grub2-mkconfig -o /boot/efi/EFI/fedora/grub

# STEP 3: remove nouveau
sudo yum remove xorg-x11-drv-nouveau

# STEP 4: rename the nouveau initramfs and create a new one
sudo mv /boot/initramfs-$(uname -r).img /boot/initramfs-$(uname -r)-nouveau.img
sudo dracut /boot/initramfs-$(uname -r).img $(uname -r) 

# STEP 5 (final step): runlevel
# runlevel is deprecated. to change target level definitively to graphic mode 
sudo systemctl set-default graphical.target
# to see current target mode
systemctl get-default

#----------------------------------------------------------------------------------------------------
# now you can finally install the driver

# error message if X is running when installing the driver. to disable graphical mode do
sudo init 4

# make sure kernel source tree is not missing, do
sudo yum install kernel-devel kernel-headers
# you must make sure it matches the kernel version. since the lattest version of source us installed by yum, I recommend you update the kernel
sudo yum update kernel-core
# to check the version of your kernel, type
uname -r

# then install the commercial nVidia driver (in my case, fedora22 64bits for a GeForce 460m)
sudo chmod u+x NVIDIA-Linux-x86_64-367.57.run
sudo ./NVIDIA-Linux-x86_64-367.57.run

# to enable graphical mode back, do
sudo init 5

# CUDA INSTALLATION
#----------------------------------------------------------------------------------------------------
# you'll need some space on /tmp, create a symbolic link to other directory if necessary
mkdir /var/tmp
sudo mv /tmp/* /var/tmp/
sudo umount -l /tmp
sudo ln -s /var/tmp /tmp
# it is recommended to install libXmu
sudo yum install libXmu-devel
# to enable more recent versions of gcc, comment line 115 of /usr/local/cuda/include/host_config.h

#----------------------------------------------------------------------------------------------------
# Configure CUDA (nvcc compiler), I recommend CUDA toolkit 7.5
# download the file cuda_6.5.14_linux_64.run and run the command lines
sudo chmod u+x cuda_7.5.18_linux.run
sudo ./cuda_7.5.18_linux.run --override
# override to force because gcc 4.9 and up are not supported, driver already installed

#----------------------------------------------------------------------------------------------------

# OPENCV INSTALLATION
#----------------------------------------------------------------------------------------------------
# some opencv2 dependencies
sudo yum install cmake eigen3-devel

# opencv2 installation (I hope you don't have gcc 6)
cd ~
git clone https://github.com/Itseez/opencv.git
cd opencv
git checkout tags/3.1.0	
cmake -DWITH_IPP=OFF
make -j8
sudo make install

# if CUDA version >= 8, change 
# if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)
# to 
# if !defined (HAVE_CUDA) || defined (CUDA_DISABLER) || (CUDART_VERSION >= 8000)
# in graphcuts.cpp

#----------------------------------------------------------------------------------------------------

# CERES INSTALLATION
#----------------------------------------------------------------------------------------------------
# CeresSolver dependencies
sudo yum install eigen3-devel cmake suitesparse-devel blas-devel lapack-devel atlas-devel

# CeresSolver installation
cd ~
git clone https://ceres-solver.googlesource.com/ceres-solver
mkdir ceres-solver-bin
cd ceres-solver-bin
cmake ../ceres-solver
make -j8
sudo make install

#----------------------------------------------------------------------------------------------------

# MY SOFT INSTALLATION (cocolib_unstructured and ULF)
#----------------------------------------------------------------------------------------------------

# some dependencies

# gcc-c++ compiler 
sudo yum install gcc-c++
# glsl-like matrix format
sudo yum install glm-devel
# for openGL
sudo yum install glew-devel glfw-devel
# reads and writes tiff images for Cimg.h
sudo yum install libtiff-devel
# read images
sudo yum install SDL2-devel SDL2_image-devel
# pixflow dependencies
sudo yum install gflags-devel glog-devel
# other dependencies (most for cocolib)
sudo yum install gsl-devel ann-devel zlib-devel hdf5-devel OpenEXR-devel rply-devel suitesparse-devel blas-devel lapack-devel

#----------------------------------------------------------------------------------------------------
# QT file parsing
# Qt creator config: failed to load help plugin, install following libraries:
sudo yum install gstreamer gstreamer-plugins-base
# to enbale .pro parsing 
sudo yum install qt-devel

#----------------------------------------------------------------------------------------------------
# to configure, parse and compile the project
cd ~/light-field-remote-vision
./configure-project.sh

#----------------------------------------------------------------------------------------------------

# don't forget to edit .bashrc to add binary and library paths for cuda and others
# other you'll get an error message at runtime
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib















