#!/bin/bash

OUTPUT_NAME=${1%/}

if [ -z $OUTPUT_NAME ]; then
    echo "USAGE: ./mp4-from-folder.sh <folder-name> [frame-rate]"
    exit
fi

FRAME_RATE=${2:-10}

ffmpeg -y -start_number 1 -framerate $FRAME_RATE -i $OUTPUT_NAME/%0004d.jpg -loop -1 -profile:v high -crf 28 -pix_fmt yuv420p $OUTPUT_NAME.mp4