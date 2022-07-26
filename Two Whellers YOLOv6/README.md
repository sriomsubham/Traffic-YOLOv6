# Two Wheeler YOLOv6
## Introduction

India has one of the largest two wheeler population in the world. Such that at minimum every 8 ou of 10 household has one. Indian traffic is mainly
dominated by two wheeler/bike/scooty.

Recognising these bikes and the rider is a challenge specific to India and India has many defined traffic rules around bikes. This project is made 
in order to help monitor bike traffic. 

It can recognise 
    1.Bike,
    2.Their number pate,
    3.Rider wearing helmet or not,
from two differnet vedio/camera feed.

## Steps to use this project :-

1. Train YOLOv6 model on the custom dataset. The classes of the data set is in data/clsses.yaml file. Follow the README1.md for steps or
    the original YOLOv6 repo.
2. main11.py : set the path for two input vedios (line 8 and 9) in for inferencing and run it. output vedio will be saved with name 'Output_vedio.avi'. 
    In the output vedio, processed vedios will be displayed side by side along with FPS.
    
Sample output vedio :-    
    
 file:///home/ubuntu/Downloads/Output_vedio1.avi


## Extra :-

Inference.py uses CPU for objected detection. It can be modified to use GPU, follow the comments in inference.py and inference.ipynb for using GPU.


## Use Cases :-

1. Bike Traffic monitoring in/out of a housing society and office coumpounds.
2. Assit traffic police in bike Traffic monitoring on roads. 
