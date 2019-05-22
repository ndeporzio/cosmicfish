#!/bin/bash

#Stop execution on error
set -o errexit 

#Collect filepath to install CLASS at
echo "Your version of gcc is: "
gcc --version
echo "Look up!"
echo "Is your gcc version at least 4.9? If not, edit the 'CC = ' line"
echo "in the MakeFile to point to such a version. Cancel execution"
echo "then re-run."

echo "Enter absolute path to project directory..."
read -p "Project Path: " path
echo "Installing CLASS at the following location: "
echo "$path/lib/"
mkdir "$path/lib/"
cd $path/lib/

#Install CLASS
git clone git@github.com:lesgourg/class_public.git
mv class_public class
cd class 
make clean
make -j class

#Distribute path to CLASS
CLASS_DIR=$PWD
export $CLASS_DIR

#Exit
echo "CLASS installation complete."
