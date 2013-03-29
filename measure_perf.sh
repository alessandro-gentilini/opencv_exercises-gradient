#!/bin/sh
dest=$(date +%Y%m%d_%H%M%S)
mkdir $dest
cp sobel.cpp $dest
cp sobel.h $dest
cp CMakeLists.txt $dest
./sobel A.bmp blurred_scene.bmp