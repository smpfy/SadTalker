#!/bin/sh

docker build . -t sadtalker && docker run -it --rm -p 7860:7860 --gpus all -v $(pwd)/results:/SadTalker/results/ -t sadtalker python3 app_sadtalker.py
