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



#### 4.2 Scale Factor and Sliding Window
Since the classifier model was trained with the fixed face size images which can be seen in the xml file. It can detect faces with the same size which was used during the training.

what if our input image has faces smaller or bigger than what was used in training?

- Downsize or scale the image according to training
- Scaling factor of 1.3 is appropriate to solve this issue



<p align="center">&nbsp;<img src="https://waltpeter.github.io/open-cv-basic/Files/pyramid.png" alt = "Image Alt" width = "300'/>


 
- A sliding window is a rectangular region that shifts around the whole image(pixel-by-pixel) at each scale. Each time the window shifts, the window region is applied to the classifier and detects whether that region has Haar features of a face.



<p align="left"><img src="https://miro.medium.com/v2/resize:fit:1200/format:webp/1*2AEkrXCUSpKkYQxjg8lugQ.jpeg" alt="Image Alt" width= "450"/>&nbsp;&nbsp;<img src="https://miro.medium.com/v2/resize:fit:1400/1*pOZ9-EqqqZAn0B3uUOOrRw.gif" alt="Image Alt" width="450"/>

#### 4.3 Minimum Neighbours

-Since the object detection works in the combination of the image pyramid (multi-scaling) and sliding window, we get multiple true outputs for a single region of the face. These true outputs are the window region which satisfies the Haar features (could be actual face area or a non-face area taken into consideration).
- minNeighbor is the threshold value for the number of true outputs required to detect a face.

```bash
for i in img_path:
        img = cv2.imread(i)
        image_array = np.array(img, "uint8")
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
```

#### 4.2 Integral Image
#### 4.3 Bounding Box 

  
### 5. K-Means Clustering

### 6. Silhouette Analysis

### 7. Clusters

### 8. Results

### 9. Future Scope
