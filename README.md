# Project Overview

This project showcases the experimentation of the U-Net algorithm through the implementation of semantic segmentation using the Cityscapes dataset. The model has been trained to fulfill the requirements of a self-driving car.

* Project's title: **SEMANTIC SEGMENTATION OF URBAN SCENE USING U-NET ARCHITECTURE ON CITYSCAPES DATASET**
* Authors: Tri Wahyu Prabowo
  
📝 Note: This project is part of the Computer Vision Bootcamp by PT Teknologi Artifisial Indonesia (IndonesiaAI).

## ★ Introduction

### Background

In Autonomous Driving Vehicles, the vehicle receives pixel-wise sensor data from RGB cameras, point-wise depth information from the cameras, and sensor data as input. The computer present inside the Autonomous Driving vehicle processes the input data and provides the desired output, such as steering angle, torque, and brake.

For the vehicle to make accurate decisions, the computer inside it must have a complete awareness of its surroundings and understand each pixel in the driving scene. Semantic Segmentation is the task of assigning a class label (such as Car, Road, Pedestrian, or Sky) to each pixel in the given image.

A Semantic Segmentation algorithm that performs better will significantly contribute to the advancement of the Autonomous Driving field.


### Project objective & scope

After studying specific deep learning frameworks, my objective is to apply this knowledge in a demonstration with two main goals:

1. **Experience the complete Deep Learning Development Cycle with the CityScapes dataset:**
   This process involves data collection, partitioning, preprocessing, exploration, model building, and concludes with evaluating model performance.

3. **Successfully develop a Segmentation model (U-Net) on the CityScapes dataset:**
   I aim to create one segmentation model and hope that the insights gained during its development will benefit autonomous driving companies such as Tesla, Waymo, and Baidu Apollo. These models should contribute to improving the precision of identifying and detecting targets in urban environments.
  
## ★ Getting Started

### 1. Data collection & Preparation

### 🔍 Dataset : [Cityscapes Dataset](https://www.cityscapes-dataset.com/)

<img align="center" src="https://github.com/triwahyupra/project3_semantic_segmentation/blob/9b3c8bd2352fcf89a5039bede78012428b3d352e/assets/cityscapesLogo.png" alt="LOGO" width="800" height="180">
The Cityscapes dataset, stands as a crucial resource for computer vision research, particularly in autonomous driving and urban scene analysis. Comprising diverse high-resolution images captured from the driver's perspective in various cities, the dataset provides a realistic representation of complex urban environments. Notably, it includes pixel-level annotations for semantic segmentation, categorizing each pixel into classes like road, sidewalk, car, and pedestrian. This detailed annotation facilitates the training and evaluation of deep learning models, making Cityscapes an invaluable asset for advancing algorithms and systems designed to enhance the understanding and navigation of intelligent vehicles in urban settings.

<img align="center" src="https://github.com/triwahyupra/project3_semantic_segmentation/blob/aaf3f9b5c69b42aa6e44abbddc1228d292aaf3bc/assets/data_example.png" alt="data" width="650" height="400">

Filter 20 classes instead using all classes for training:  
```
ignore_index=255
valid_trainIds_classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,-1, ignore_index] #trainIds
class_names = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', \
               'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', \
               'train', 'motorcycle', 'bicycle', 'unlabelled']
num_classes = 20
```  
Preprocess data with augmentation (crop, resize, and normalize pixel values from range [0-255] to range [0-1] in images)

<img align="center" src="https://github.com/triwahyupra/project3_semantic_segmentation/blob/aaf3f9b5c69b42aa6e44abbddc1228d292aaf3bc/assets/cityscapes_normalized.png" alt="normalizedimg" width="600" height="300">

### 2. Model Development

### U-Net 

U-Net, initially designed for biomedical image segmentation, is a groundbreaking deep learning architecture recognized for its distinctive U-shaped structure. Comprising a contracting path for context extraction and an expanding path for precise localization, U-Net employs skip connections to integrate fine-grained details. Widely adopted beyond its original medical imaging scope, U-Net's versatility and effectiveness make it a popular choice for diverse applications such as autonomous driving and general computer vision tasks.

<p align="center"> <img src="https://github.com/triwahyupra/project3_semantic_segmentation/blob/b1e3470daa335af4fe8b2e71c7882315e336fbb0/assets/unet.png" height=384 /> </p>

Initializing Hyperparameters :
```
IMG_SIZE = 256
BATCH_SIZE = 32
BUFFER_SIZE = 300
EPOCHS = 50
auto = tf.data.experimental.AUTOTUNE
num_classes = 20
```
```
optimizer='adam'
loss='sparse_categorical_crossentropy'
metrics='accuracy', "mean_iou"
```

### 3. Training & Optimization

In this phase, training and optimization of the U-Net model are conducted. The training process involves presenting the Cityscapes dataset to the U-Net architecture, where the model learns patterns and feature representations necessary for semantic segmentation tasks.

#### * Evaluasi Hasil Training

After the training process, the model is evaluated using a validation dataset to measure the accuracy of its predictions. Evaluation metrics used accuracy and meanIOU, providing a comprehensive overview of the model's performance in semantic segmentation tasks. 

#### * Train and Validation Graph

Here is the graph of training and validation results :
<img align="center" src="https://github.com/triwahyupra/project3_semantic_segmentation/blob/b1e3470daa335af4fe8b2e71c7882315e336fbb0/assets/train_results_graph.png" alt="graph" width="1000" height="250">

Here is the table of training and validation results :
train_loss | train_accuracy | mean_iou | val_mean_iou | val_loss | val_accuracy
:-----:|:----------:|:------:|:------------:|:----------:|:------:
0.1370 | 0.9554 | 0.7235 | 0.3306  | 0.6598 | 0.8531   


## ★ The Results

### Predict on the Dataset

Here are the prediction results for the images in the Cityscapes dataset:

<img align="center" src="https://github.com/triwahyupra/project3_semantic_segmentation/blob/6728ddd1f544f5f431d0207a71cf6d24fe210dc4/assets/unet_predict_test.png" alt="test_predict" width="800" height="600">

### Inference on Video

Here is the result of the inference using the video :

<img align="center" src="https://github.com/triwahyupra/project3_semantic_segmentation/blob/6728ddd1f544f5f431d0207a71cf6d24fe210dc4/assets/UNet.gif" alt="video" width="800" height="500">


## Explore the Code!

Notebook utama yang digunakan untuk pelatihan model dapat diakses pada link berikut :

[U-Net Training on Cityscapes Dataset](https://github.com/triwahyupra/project3_semantic_segmentation/blob/af88309aac1a4d76d6be73a0663941da9716e8cd/triwahyu_cityscape_segmentation.ipynb)

[Inference U-Net on Video](https://github.com/triwahyupra/project3_semantic_segmentation/blob/b77b335e8a3654c225a3ffebe39405097bdd3059/cityscapes_inference_on_video.ipynb)

## 📧 Contact

If you have any questions, feedback, or would like to contribute to this project, feel free to contact the following:

- Nama: [ Tri Wahyu Prabowo ]
- Email: [ triwahyu@reefgen.io ]
- LinkedIn: [ [triwahyupra](https://www.linkedin.com/in/triwahyupra) ]

hank you for your support and contributions!

