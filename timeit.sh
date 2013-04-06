#!/bin/bash
fmt=%U,%S,%E,%P
log=log.txt
echo $fmt > $log 
for i in {1..10}
do
	/usr/bin/time -a -o $log -f "$fmt" ./sobel wheel.jpg lateral.jpg
done	