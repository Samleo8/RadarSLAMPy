#!/bin/bash

OUTPUT_NAME=${1%/}

if [ -z $OUTPUT_NAME ]; then
    echo "USAGE: ./gif-from-folder.sh <subfolder-output-name> [start_number] [frame-rate]"
    exit
fi

START_NUM=${2:-1}
FRAME_RATE=${3:-60}

ffmpeg -y -start_number $START_NUM -framerate $FRAME_RATE -i $OUTPUT_NAME/%0004d.jpg -loop -1 $OUTPUT_NAME.gif
