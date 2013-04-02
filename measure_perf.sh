#!/bin/sh
dest=$(date +%Y%m%d_%H%M%S)
mkdir $dest
cp sobel.cpp $dest
cp sobel.h $dest
cp CMakeLists.txt $dest

./sobel test_model.bmp test_scene.bmp

cp rt*.csv $dest
diff ./baseline/rt_0.csv $dest/rt_0.csv
if [ $? -ne 0 ]
	then
		echo Failed rt_0 comparison.
		exit 1
fi

cp result.csv $dest
diff ./baseline/result.csv $dest/result.csv
if [ $? -ne 0 ]
	then
		echo Failed results comparison.
		exit 1
fi