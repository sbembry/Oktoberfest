# Oktoberfest

In this project, we use the [Oktoberfest](https://github.com/a1302z/OktoberfestFoodDataset) dataset for object detection. The dataset is a series of images from multiple video screens of people checking out food at Oktoberfest. There are 1110 images and 2696 annotations in the dataset. Please see the readme of the dataset for more information.

We fit object detection models for [Faster RCNN](https://arxiv.org/pdf/1506.01497.pdf) and [CenterNet](https://arxiv.org/pdf/1904.07850.pdf).

# Data

We split the training data into training and validation sets to evaluate the model. We save the test dataset for inference. Since the images are a series of images from video screens, some of the images are related to each other. As a result, we cannot split the data randomly because images from the same video share information. Instead, we split the data by video. From there, we halve the size of the image (they come very large) and normalize it.

## Faster RCNN

### About

Faster R-CNN is an improvement on the earlier Fast R-CNN. It proposes regions in a much more efficient way by utilizing a Regional Proposal Network (RPN) to generate proposals instead of the more expensive selective search.

The main stages invovled in Faster R-CNN is as follows:

1. Initial convolutional layers to extract feature maps of the image to be shared by the RPN and Fast R-CNN Networks.
2. Regional Proposal Network generates region proposals and extractss a fixed length feature vector for each region using a ROI Pooling layer.
3. Feature vectors are then classified using the Fast R-CNN architecture. Class scores and bounding boxes are returned.

The RPN network is the key part of the architecture that differnetaties it from the previous Fast R-CNN. The RPN works as follows:

1. Anchor points are set up throughout the feature map and the RPN slides a window over each anchor location. 
2. For each anchor point K amount of proposals are generated. K is equal to the number of scales and aspect ratios that are chosen. So for 4 different scales and 4 different scales a total of 16 proposals would be generated for each anchor point.
3. Then the anchors are assinged a label representing whether the region is an object or not. This is done by choosing anchors with highest intersection-over-union with a ground truth box or anchors with IoU >= 0.7.

<b> Advantages </b>

1. Much Faster
2. Sharing of convolutional layers allows for a unified network where the RPN and Fast R-CNN are trained at once.
3. Addition of anchors that allow layer sharing.

</b> Disadvantages </b>

1. Since all samples are taken from a single image they may be correlated and take time for the network to converge

### Training

The training that I went through as well as the results on the test set are in [this notebook](code/faster_rcnn/faster_rcnn.ipynb). Much of the code is from the torchvision package and can be found in torchvision/references/detection. Some changes to the code had to be made in those files to work with our data. 

### Results

The model that was trained achieved a COCO-style mAP of .425 on the validation set after 7 epochs. And on the test set achieved a recall score of 0.851.

If I had more time I would have focused on flushing out the post-processing and interpreting the results in a better cleaner way.

## CenterNet

### About

CenterNet is a newer single shot object detection model. The idea around it is to predict the center of an object, then predict the box surrounding it. It is unique compared to other object detection methods because it uses keypoints and no anchor boxes. The model consists of a backbone, an upsampling stage, and a head to get the final outputs. The model has 3 outputs:

1. A CxH/4xW/4 tensor, where C is the number of classes. Each value represents the probability that one of the given pixels in its area (since the output is downsampled, each value corresponds to multiple pixels) is a center for an object of that class.
2. A 2xH/4xW/4 tensor, predicting the offset of the center onto the full image
3. A 2xH/4xW/4 tensor, predicting the width and height of the box

<b> Advantages </b>

1. No Anchor boxes
2. Single shot
3. In general good performance

<b> Disadvantages </b>

1. Has only 1 possible box per center point. If there are multiple objects with the same center point, this will be an issue.
2. Unlike other models, it doesn't classify each pixel (or box) between each other (softmax between categories), it classifies each pixel as a center for each class (sigmoid for each class). Although you can simply choose the class with the highest score, I think this is less intuitive than softmax.

### Training

In this repo, I provide my model class as well as the training pipeline I used. Please refer to [this notebook](code/centernet/train_example.ipynb) on how to train a model. The code is primarily from [Centernet-better](https://github.com/FateScript/CenterNet-better), with small changes to better fit our data.

My final model was run through 40 epochs with a final loss of .4 on the validation data. I was having issues with exploding gradients training the whole network initially, so I decided to freeze the backbone and the upsampling portions of the model for the first couple epochs. Afterwards, the model was run updating the backbone and upsample with learning rate 1/9 and 1/6 the size of my head learning rate, respectively. My head learning rate started as .001, and I decreased it on a 4 epoch plateau of validation loss.

### Inference

This notebook has examples on how to make predictions on images. However, I did not have a lot of time to do the significant amount of post-processing that is needed for object detection models. If I were to continue to work on this project, I would focus on working on this post-processing. Some things that I would do in order of importance are:

1. More properly choose bounding boxes without "cheating" by looking at the ground truth.
2. Currently, my model can predict multiple classes for the same bounding box. My processing right now does not account for this. I need to choose the class with the highest score when this occurs.
3. After I finish 1 and 2, I can work more on analyzing my results because I will have a streamlined prediction pipeline. Using this, I can calculate metrics (IOU, PR) that are comparable across networks.
