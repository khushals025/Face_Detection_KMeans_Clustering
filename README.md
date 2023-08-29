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

#### 4.1 Haar Features

- Basically for face detection, the classifier looks for the most relevant features on the face such as eyes, nose, lips, forehead, eyebrows because we know that although people have different looks, these features are in the similar positions on the face.
  
- input image is given to the classifier, it compares the Haar Features from the xml file and applies it to the input image. If it passes through all the stages of haar feature comparison, then it’s a face, else not.
- For the ideal case, this difference between the average of black and white pixel values is 1. Therefore, for the real image, the closer this difference to 1, the more likely we have found a Haar feature.

<div align="center">
  <img src= "https://miro.medium.com/v2/resize:fit:1400/format:webp/1*GD9ZE_j8WpRPewsZfTUKww.jpeg" alt = "Alt Image" width = "700">
</div>


#### 4.2 Integral Image
#### 4.3 Bounding Box 

  
### 5. K-Means Clustering

### 6. Silhouette Analysis

### 7. Clusters

### 8. Results

### 9. Future Scope
