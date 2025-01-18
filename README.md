# Birds-detection-tool-which-is-based-on-YOLOv10
Here is a birds detection tool which is based on YOLOv10. This idea of tool refer to [Optimized SmallWaterbird Detection Method Using
Surveillance Videos Based on YOLOv7](https://www.mdpi.com/2076-2615/13/12/1929#:~:text=This%20study%20describes%20an%20improved,attention%20regions%20and%20monitor%20waterbirds.).

<img src="https://www.mdpi.com/animals/animals-13-01929/article_deploy/html/images/animals-13-01929-g002.png">

In this time, I don't only present this tool, but also compare the performances of different models which are produced by different pretrained file of YOLOv10.

## How to begin
for yolov10, you could follow [yolov10](https://github.com/THU-MIG/yolov10/tree/main). 

### photo dataset
I chose this [photo dataset of birds](https://universe.roboflow.com/yolo-9evjx/birds-wijmc/dataset/2/download) with some added birds photos for training, validation and testing of my birds detection tool.

In this dataset, there are not only photos of static birds, but also photos of flying birds in high speed and some of flying birds in photo is tiny. Those reason is very helpful for model to detect flying tiny stuff in high speed.

This dataset is also with label files, so I don't need to do so many preprocessing.  

Those are why I chose this dataset for this project.

### Pretrained-weight
In this project, I will run model with yolov10s yolov10m yolov10b yolov10l yolov10x, please download pretrained-weight files on yolov10 website.

### Computer hardware
My CPU is intel-i7-11th, GPU is NVIDIA GeForce RTX 3060, Size of memory is 40GB.

