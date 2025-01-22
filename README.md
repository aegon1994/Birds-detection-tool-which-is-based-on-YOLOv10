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

This is the table which is presents the comprehensive performance of all models
<img src="https://github.com/aegon1994/Birds-detection-tool-which-is-based-on-YOLOv10/blob/main/dataimage.png?raw=true">
In this table, originx is the first condition in yolov10x, opadcos_debox15x is the second condition in yolov10x. For example, origins is the first condition in yolov10s, opadcos_debox15s is the second condition in yolov10s.

You will see in most comparison, the comprehensive performance of second condition of training is better than first except the model in yolov10s. I don't know the specific reason which causes this result, but I assume yolov10s is too simple to detect the stuff which is tiny and move in high speed.

Another fact is the most complex or the biggest model isn't the best model, opadcos_debox15m or opadcos_debox15b is less complex than models in yolov10l or yolov10x, but their comprehensive performance is better than models in yolov10l or yolov10x. It means If you wanted use this project, opadcos_debox15m or opadcos_debox15b would be good choices.

We will see a common of those model, that is the recall of all models is low and confidence is high at precision is 1(or precison of every models is high in high confidence). We should observe the relationship between recall and precison in subsequent results and discuss the reason.

These are normalized confusion matrixes of all models. In this part, I present features by confusion matrix.
<img src="https://github.com/aegon1994/Birds-detection-tool-which-is-based-on-YOLOv10/blob/main/sorigin/confusion_matrix_normalized.png?raw=true">
<img src="https://github.com/aegon1994/Birds-detection-tool-which-is-based-on-YOLOv10/blob/main/scos_debox15/confusion_matrix_normalized.png?raw=true">
<img src="https://github.com/aegon1994/Birds-detection-tool-which-is-based-on-YOLOv10/blob/main/morigin/confusion_matrix_normalized.png?raw=true">
<img src="https://github.com/aegon1994/Birds-detection-tool-which-is-based-on-YOLOv10/blob/main/mcos_debox15/confusion_matrix_normalized.png?raw=true">
<img src="https://github.com/aegon1994/Birds-detection-tool-which-is-based-on-YOLOv10/blob/main/borigin/confusion_matrix_normalized.png?raw=true">
<img src="https://github.com/aegon1994/Birds-detection-tool-which-is-based-on-YOLOv10/blob/main/bcos_debox15/confusion_matrix_normalized.png?raw=true">
<img src="https://github.com/aegon1994/Birds-detection-tool-which-is-based-on-YOLOv10/blob/main/lorigin/confusion_matrix_normalized.png?raw=true">
<img src="https://github.com/aegon1994/Birds-detection-tool-which-is-based-on-YOLOv10/blob/main/lcos_debox15/confusion_matrix_normalized.png?raw=true">
<img src="https://github.com/aegon1994/Birds-detection-tool-which-is-based-on-YOLOv10/blob/main/xorigin/confusion_matrix_normalized.png?raw=true">
<img src="https://github.com/aegon1994/Birds-detection-tool-which-is-based-on-YOLOv10/blob/main/xcos_debox15/confusion_matrix_normalized.png?raw=true">
At first, the order from above to below is confusion matrix of first condition in yolov10s -> second condition in yolov10s -> first condition in yolov10m -> second condition in yolov10m and so on, it is basically the same as the order of last table, and the rest of oreder of graph will be same.

In those graph above, We could see their common by confusion matrix, there are so many False Positive(FP) and False Negative(FN) cases. It means most bird results are actually background, and most background results are bird. The possibiliy of model which recognize the correct result is low and recognization is low too.

The Precision-Recall curve: In these graph, Precision is on y-dimension, Recall is on x-dimension, the blue line is the PR curve line.
<img src="https://github.com/aegon1994/Birds-detection-tool-which-is-based-on-YOLOv10/blob/main/sorigin/PR_curve.png?raw=true">
<img src="https://github.com/aegon1994/Birds-detection-tool-which-is-based-on-YOLOv10/blob/main/scos_debox15/PR_curve.png?raw=true">
<img src="https://github.com/aegon1994/Birds-detection-tool-which-is-based-on-YOLOv10/blob/main/morigin/PR_curve.png?raw=true">
<img src="https://github.com/aegon1994/Birds-detection-tool-which-is-based-on-YOLOv10/blob/main/mcos_debox15/PR_curve.png?raw=true">
<img src="https://github.com/aegon1994/Birds-detection-tool-which-is-based-on-YOLOv10/blob/main/borigin/PR_curve.png?raw=true">
<img src="https://github.com/aegon1994/Birds-detection-tool-which-is-based-on-YOLOv10/blob/main/bcos_debox15/PR_curve.png?raw=true">
<img src="https://github.com/aegon1994/Birds-detection-tool-which-is-based-on-YOLOv10/blob/main/lorigin/PR_curve.png?raw=true">
<img src="https://github.com/aegon1994/Birds-detection-tool-which-is-based-on-YOLOv10/blob/main/lcos_debox15/PR_curve.png?raw=true">
<img src="https://github.com/aegon1994/Birds-detection-tool-which-is-based-on-YOLOv10/blob/main/xorigin/PR_curve.png?raw=true">
<img src="https://github.com/aegon1994/Birds-detection-tool-which-is-based-on-YOLOv10/blob/main/xcos_debox15/PR_curve.png?raw=true">

In first table, We the recall of all models is low and confidence is high at precision is 1. We want to know the real relationship between recall and precison in all models. In those graph above, We will see a common of all model, that is precision is high when recall is low. When recall is over a threshold, precision would fall down immediately. It means you can only choose high precision but low recall, or high recall or high precision.

It also represents when models recognized more correct results of birds in all correct results, it actually recognized cautiously. When models recognized more correct results of birds in results of birds, the correct results of birds is little in in all correct results.

Here is some results of prediction for testing photos.
<img src="https://github.com/aegon1994/Birds-detection-tool-which-is-based-on-YOLOv10/blob/main/1.jpg?raw=true">
<img src="https://github.com/aegon1994/Birds-detection-tool-which-is-based-on-YOLOv10/blob/main/12.jpg?raw=true">
<img src="https://github.com/aegon1994/Birds-detection-tool-which-is-based-on-YOLOv10/blob/main/132.jpg?raw=true">
<img src="https://github.com/aegon1994/Birds-detection-tool-which-is-based-on-YOLOv10/blob/main/53.jpg?raw=true">
<img src="hhttps://github.com/aegon1994/Birds-detection-tool-which-is-based-on-YOLOv10/blob/main/86.jpg?raw=true">

## Discussion

From the first table, we know the best model for recognition of tiny objects which move in high speed is not the most complex one, it means We have to choose the most suitable pretrained file like yolov10m or yolov10b. We also know in recognition of tiny objects which move in high speed in YOLOv10, box loss function raising and cosine decay for learn rating decreasing is a better way to train your model. The smaller pretrained files like yolov10s is too simple to train your model for birds detecting.

From confusion matrixes of all models, we realize recognization of our models actually is not good enough to birds detecting. The color of top right and left area is deeper than others, it means our models confuse birds and background. I thought the reason is our models can't actually recognize difference between tiny birds and other tiny stuff like leaves.

From graph of Precision-Recall curve, we find precision would fall down when recall is over a threshold in our models. It means We could only choose high rate of correct results of birds in high False Positive or in high True Negative. 

Like we thought above, I thought our models is not good at tiny stuff detecting in high speed so far. I thought the limitations of mine is I don't have that kind of hardwave to train my model. In "Optimized SmallWaterbird Detection Method Using Surveillance Videos Based on YOLOv7", they trained their model by 8 RTX2080ti GPU. I have only one RTX3060 GPU. The diversity of my dataset is not enough to train my model even though I added other photos of birds.

There are also some ways to improve the results of my models without a lot of money like amplify the size of my photos, or tried other optimizers in future work.

## Reference 
Lei, J.; Gao, S.; Rasool, M.A.; Fan, R.; Jia, Y.; Lei, G. Optimized Small Waterbird Detection Method Using Surveillance Videos Based on YOLOv7. Animals 2023, 13, 1929. https://doi.org/10.3390/ani13121929

@article{wang2024yolov10,
  title={YOLOv10: Real-Time End-to-End Object Detection},
  author={Wang, Ao and Chen, Hui and Liu, Lihao and Chen, Kai and Lin, Zijia and Han, Jungong and Ding, Guiguang},
  journal={arXiv preprint arXiv:2405.14458},
  year={2024}
}

Photo dataset of birds: [https://universe.roboflow.com/yolo-9evjx/birds-wijmc/dataset/2/download](https://universe.roboflow.com/yolo-9evjx/birds-wijmc/dataset/2/download)
