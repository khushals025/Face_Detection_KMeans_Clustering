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

## 2. Dependencies
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

