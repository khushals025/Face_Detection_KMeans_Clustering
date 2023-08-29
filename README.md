# Face Detection and K-Means Clustering 


## Table of Contents
- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Face Detection](#face-detection)
- [K-Means Clustering Algorithm](#k-means-clustering-algorithm)
- [Silhouette Analysis](#silhouette-analysis)
- [Clusters](#clusters)
- [Results](#results)
- [Future Scope](#future-scope)

## 1. Introduction
This repository explains the process of face detection using haar-cascade classifier and then applying clustering algorithm to form clusters, also exploring different distance metrics and employing Silhouette Analysis for optimal clustering.

## 3. Dependencies
Following libraries were used to cary out this project 

- Face Detection and k-mean from scratch

```bash
import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy.spatial.distance import cosine
```

- K-Means Library

```bash
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from PIL import Image
```

### 4. Face Detection

<p align="left"><img src="https://miro.medium.com/v2/resize:fit:1156/format:webp/1*XX8WqHo0lyrgZfTTRQ3ESQ.jpeg" alt="haarcascade" width="500"/>&nbsp;&nbsp;<img src="https://lh4.googleusercontent.com/XCtD8cPcylJbGX7HBGgmONbC-JivQWdqKhvc7kimMd7YsHz7yDTQENv37DPKI2xlA6Wph_JGLrMjnOm0HDqNfo-_I6ybCUu_8eD9jSemc3lwUPuiIvDzDC3msktpBEk-QmOygt_MUzWF1WBf" alt="aws" width="500" /> 

#### 4.1 Haar Features Cascade Method 

- Basically for face detection, the classifier looks for the most relevant features on the face such as eyes, nose, lips, forehead, eyebrows because we know that although people have different looks, these features are in the similar positions on the face.
- A cascade of classifier consists of multiple stages of filters. The Haar features are grouped into these different stages of classifiers.
- input image in Gray-Scale is given to the classifier, it compares the Haar Features from the xml file and applies it to the input image. If it passes through all the stages(cascade or layers) of haar feature comparison, then it’s a face, else not.
- For the ideal case, this difference between the average of black and white pixel values is 1. Therefore, for the real image, the closer this difference to 1, the more likely we have found a Haar feature.

```bash
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
```



<div align="center">
  <img src= "https://miro.medium.com/v2/resize:fit:1400/format:webp/1*GD9ZE_j8WpRPewsZfTUKww.jpeg" alt = "Alt Image" width = "700">
</div>


#### 4.2 Integral Image

- Haar-like features involve rectangular areas with specific pixel value differences, which are calculated by subtracting the sum of pixel values in one region from the sum of pixel values in another region.
- This process involves multiple additions and subtractions. If these calculations were performed directly on the original image for every possible position and scale, it would be computationally expensive and slow.
-  The integral image enables quick computation of the sum of pixel values in any rectangular region with just four lookups, regardless of the region's size.

<div align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/1*HhO9vGKpbx9p8x7uS49v-g.png" alt="Image Alt" width = "700">
</div>




#### 4.3 Scale Factor and Sliding Window
Since the classifier model was trained with the fixed face size images which can be seen in the xml file. It can detect faces with the same size which was used during the training.

what if our input image has faces smaller or bigger than what was used in training?

- Downsize or scale the image according to training
- Scaling factor of 1.3 is appropriate to solve this issue



<p align="center">&nbsp;<img src="https://waltpeter.github.io/open-cv-basic/Files/pyramid.png" alt = "Image Alt" width = "300'/>


 

- A sliding window is a rectangular region that shifts around the whole image(pixel-by-pixel) at each scale. Each time the window shifts, the window region is applied to the classifier and detects whether that region has Haar features of a face.

<p align="left"><img src="https://miro.medium.com/v2/resize:fit:1200/format:webp/1*2AEkrXCUSpKkYQxjg8lugQ.jpeg" alt="Image Alt" width= "450"/>&nbsp;&nbsp;<img src="https://miro.medium.com/v2/resize:fit:1400/1*pOZ9-EqqqZAn0B3uUOOrRw.gif" alt="Image Alt" width="450"/>

#### 4.4 Minimum Neighbours

-Since the object detection works in the combination of the image pyramid (multi-scaling) and sliding window, we get multiple true outputs for a single region of the face. These true outputs are the window region which satisfies the Haar features (could be actual face area or a non-face area taken into consideration).
- minNeighbor is the threshold value for the number of true outputs required to detect a face.
- with trial and error 5 was best suited for this task.

```bash
for i in img_path:
        img = cv2.imread(i)
        image_array = np.array(img, "uint8")
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
```


#### 4.5 Bounding Box 

- After detecting faces created bounding box over the face
- here, (x,y,w,h) are x-coordinate, y-coordinate of botom left corner of the box. h is height and w is width of the bounding box.
- Cropped the faces and saved to a folder path
- resized to (224X224)

```bash
for (x_, y_, w, h) in faces:
            #creating bounding box over the face detected 
            face_detect = cv2.rectangle(img, (x_, y_), (x_ + w, y_ + h), (255, 0, 255), 2)
            roi_gray = gray[y_:y_ + h, x_:x_ + w]
            roi_color = img[y_:y_ + h, x_:x_ + w]
            # resize all images to 224x244
            resized_image = cv2.resize(roi_gray, size)
            resized_img.append(resized_image)
            # save cropped image
            filename = os.path.splitext(os.path.basename(i))[0]
            save_path = os.path.join("/Users/khushal/Desktop/Spring 2023/CVIP/cropped_images", filename + '_cropped.jpg')
            cv2.imwrite(save_path, resized_image)
```
  
### 5. K-Means Clustering

- After Detecting faces the main task is to cluster images with respect to the person.
- To use images we need to convert them into numpy arrays (Flatten image matrix [224X224]).

```bash 
#Flattening Images 
flatten_img = []
for i in x:
    flatten_img.append(i.flatten())

#print(type(flatten_img))
img_arry = np.array(flatten_img)



```
- K-Means clustering is an unsupervised learning algorithm which aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest centroid. The algorithm aims to minimize the squared Euclidean distances or Manhattan distance between the observation and the centroid of cluster to which it belongs.
- But It is sensitive to outliers.
- Since, mean is also sensitive to outliers we will use median


Median absolute deviation is a robust way to identify outliers.
The median absolute deviation formula is:

<div align="center">
  <img src="https://w2.influxdata.com/wp-content/uploads/MAD-n-01.svg" alt="Image Alt" width="300">
</div>

where, 
- m is the median of a dataset; and
- Xi is the dataset in question.

If the value is greater than our threshold, then we have an anomalous point. 

<div align="center">
  <img src="https://w2.influxdata.com/wp-content/uploads/more-than-threshold-01.svg" alt="Image Alt" width="300">
</div>

  
```bash
#removing outliers 
median_abs_deviation = np.median(np.abs(img_arry - np.median(img_arry)), axis=0)
outliers = np.abs(img_arry - np.median(img_arry))/median_abs_deviation < 4
img_array = img_arry[~outliers] #---> image array except for outliers
print(len(img_arry))
print(img_arry.shape)

```

- Note: tilde (~) symbol is used as a logical NOT operator. It's used to negate a Boolean array, meaning it flips the values of True to False and False to True.

#### K-Means Algorithm 

The way kmeans algorithm works is as follows:
- Specify number of clusters K.
- Initialize centroids by first shuffling the dataset and then randomly selecting K data points for the centroids without replacement.
- Keep iterating until there is no change to the centroids. i.e assignment of data points to clusters isn’t changing.
- Compute the sum of the squared distance between data points and all centroids.
- Assign each data point to the closest cluster (centroid).
- Compute the centroids for the clusters by taking the average of the all data points that belong to each cluster.

### 6. Silhouette Analysis

### 7. Clusters

### 8. Results

### 9. Future Scope
