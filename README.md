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
In this project, I will run model with yolov10s, yolov10m, yolov10b, yolov10l and yolov10x, please download pretrained-weight files on yolov10 website.

### Computer hardware
My CPU is intel-i7-11th, GPU is NVIDIA GeForce RTX 3060, Size of memory is 40GB.

## dataset
In this dataset, there are 18264 photos in training data, 938 photos in val data, 940 photos in testing data.

You have to put your train datas(including label files), val datas and test datas in a same direction.

## Process
### Preprocessing
Change names of photos and files -> Ensure all photos have corresponding labels

### training condition
In this section, I trained twice for every pretrained files.

The parameters of training are epochs = 250, batch = 32, imgsz = 160 and max_det = 100 in first time, other parameters are default.

The parameters of training are epochs = 250, batch = 32, imgsz = 160, max_det = 100, lr0 = 0.001, cos_lr = True, optimizer= 'AdamW' and box = 15 in second time.

AdamW is a suitable optimizer for small size dataset and tiny stuff detecting, it is a good choice for this project.

Cosine decay smoothly decrease learning rate, it could reduce oscillation during the convergence process for performance of learning stablizing.

In YOLOv10, box is the weight of the box loss component in the loss function, it affects the emphasis on accurately predicting the bounding box coordinates.
In this project, most objects(birds) are tiny and move in high speed, it means the emphasis for location of bounding box is more important than other loss components. That is why I raise the weight of the box loss component.

I combined those parameters above for model training, and have the best performance of those models so far. I also tried other parameters in yolov10s, but those condition wasn't as good as this combination.
If you tested another condition is better than mine, Please comment below, thank you.

## Results
### The results of training
In this section, I will compare performances of every models, and then discuss some feature of those models.

<img src="https://github.com/aegon1994/Birds-detection-tool-which-is-based-on-YOLOv10/blob/main/dataimage.png?raw=true">
In this table, originx is the first condition in yolov10x, opadcos_debox15x is the second condition in yolov10x. For example, origins is the first condition in yolov10s, opadcos_debox15s is the second condition in yolov10s.

