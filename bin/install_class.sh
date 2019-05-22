#!/bin/bash

#Stop execution on error
set -o errexit 

#Collect filepath to install CLASS at
echo "Enter absolute path to project directory..."
read -p "Project Path: " path
echo "Installing CLASS at the following location: "
echo "$path/lib/"
cd $path/lib/

#Install CLASS
git clone git@github.com:lesgourg/class_public.git
mv class_public class
cd class 
make clean
make class

#Distribute path to CLASS
CLASS_DIR=$PWD
export $CLASS_DIR

#Exit
echo "CLASS installation complete."
