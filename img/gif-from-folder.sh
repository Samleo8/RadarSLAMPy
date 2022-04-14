#!/bin/bash

OUTPUT_NAME=${1%/}

if [ -z $OUTPUT_NAME ]; then
    echo "USAGE: ./gif-from-folder.sh <subfolder-output-name> [frame-rate]"
    exit
fi

# RESULTS=${2:-results}
FRAME_RATE=${2:-10}

ffmpeg -y -start_number 1 -framerate $FRAME_RATE -i $OUTPUT_NAME/%0004d.jpg -loop -1 $OUTPUT_NAME.gif