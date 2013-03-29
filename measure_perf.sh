#!/bin/sh
dest=$(date +%Y%m%d_%H%M%S)
mkdir $dest
cp sobel.cpp $dest
cp sobel.h $dest
cp CMakeLists.txt $dest
./sobel A.bmp blurred_scene.bmp
cp rt*.csv $dest
diff ./baseline/rt_0.csv $dest/rt_0.csv
if [ $? -ne 0 ]
	then
		echo Failed rt_0 comparison
		exit 1
fi
