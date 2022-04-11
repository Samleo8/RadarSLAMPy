#!/bin/bash

OUTPUT_NAME=$1

if [ -z $OUTPUT_NAME ]; then
    echo "USAGE: ./mp4-from-folder.sh <folder-name>"
    exit
fi

ffmpeg -y -start_number 1 -framerate 30 -i $OUTPUT_NAME/%0004d.jpg -loop -1 -profile:v high -crf 28 -pix_fmt yuv420p outputs/$OUTPUT_NAME/result.mp4