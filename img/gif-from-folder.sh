#!/bin/bash

OUTPUT_NAME=$1

if [ -z $OUTPUT_NAME ]; then
    echo "USAGE: ./gif-from-folder.sh <subfolder-output-name> [results-folder]"
    exit
fi

# RESULTS=${2:-results}

ffmpeg -y -start_number 1 -framerate 30 -i $OUTPUT_NAME/%0004d.jpg -loop -1 outputs/$OUTPUT_NAME/result.gif