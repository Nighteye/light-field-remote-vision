#!/bin/bash
echo "*************************************************"
echo "**                                             **"
echo "** This script will install the newest nVidia  **"
echo "** driver on Ubuntu-based Linux distributions  **"
echo "** by adding the ppa repository 'x-updates'    **"
echo "** Tested on Mint too, it works.               **"
echo "**                                             **"
echo "** Do NOT use the script if you have an        **"
echo "** Ubuntu older than 10.04 or some other       **"
echo "** flavour of Linux.                           **"
echo "**                                             **"
echo "** Also do not use the script if you installed **"
echo "** the nVidia driver manually, or it might     **"
echo "** mess things up.                             **"
echo "**                                             **"
echo "** The script needs 'sudo' rights to continue. **"
echo "**                                             **"
echo "*************************************************"

sudo add-apt-repository ppa:ubuntu-x-swat/x-updates
sudo apt-get update
sudo apt-get install nvidia-current nvidia-cuda-dev nvidia-cuda-toolkit
