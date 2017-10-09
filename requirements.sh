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
sudo dnf update grub2
sudo grub2-mkconfig -o /boot/grub2/grub
sudo grub2-mkconfig -o /boot/efi/EFI/fedora/grub

# STEP 3: remove nouveau
sudo dnf remove xorg-x11-drv-nouveau

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
sudo dnf install kernel-devel kernel-headers
# you must make sure it matches the kernel version. since the lattest version of source us installed by dnf, I recommend you update the kernel
sudo dnf update kernel-core
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
sudo ln -s ~/tmp /var/tmp
# it is recommended to install libXmu
sudo dnf install libXmu-devel
# to enable more recent versions of gcc, comment line 115 of /usr/local/cuda/include/host_config.h

#----------------------------------------------------------------------------------------------------
# Configure CUDA (nvcc compiler), I recommend CUDA toolkit 7.5
# download the file cuda_6.5.14_linux_64.run and run the command lines
sudo chmod u+x cuda_6.5.14_linux_64.run
sudo ./cuda_7.5.18_linux.run --override
# override to force because gcc 4.9 and up are not supported, driver already installed

#----------------------------------------------------------------------------------------------------

# OPENCV INSTALLATION
#----------------------------------------------------------------------------------------------------
# some opencv2 dependencies
sudo dnf install cmake eigen3-devel

# opencv2 installation (I hope you don't have gcc 6)
cd ~
git clone https://github.com/Itseez/opencv.git
cd opencv
git checkout tags/3.1.0	
cmake -DWITH_IPP=OFF
make -j8
sudo make install

#----------------------------------------------------------------------------------------------------

# CERES INSTALLATION
#----------------------------------------------------------------------------------------------------
# CeresSolver dependencies
sudo dnf install eigen3-devel cmake suitesparse-devel blas-devel lapack-devel atlas-devel

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
sudo dnf install gcc-c++
# glsl-like matrix format
sudo dnf install glm-devel
# for openGL
sudo dnf install glew-devel glfw-devel
# reads and writes tiff images for Cimg.h
sudo dnf install libtiff-devel
# read images
sudo dnf install SDL2-devel SDL2_image-devel
# pixflow dependencies
sudo dnf install gflags-devel glog-devel
# other dependencies (most for cocolib)
sudo dnf install gsl-devel ann-devel zlib-devel hdf5-devel OpenEXR-devel rply-devel

#----------------------------------------------------------------------------------------------------
# QT file parsing
# Qt creator config: failed to load help plugin, install following libraries:
sudo dnf install gstreamer gstreamer-plugins-base
# to enbale .pro parsing 
sudo dnf install qt-devel

#----------------------------------------------------------------------------------------------------
# to configure cocolib and compile project
cd ~/lfremotevision/cocolib_unstructured
./configure-cuda.sh
make -j8
	
#----------------------------------------------------------------------------------------------------
# to parse the .pro file and compile project
cd ~/lfremotevision/ULF
qmake-qt4 ULF.pro -r CONFIG-=debug_and_release CONFIG+=release
make -j8

#----------------------------------------------------------------------------------------------------

# don't forget to edit .bashrc to add binary and library paths for cuda and others
# other you'll get an error message at runtime
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib















