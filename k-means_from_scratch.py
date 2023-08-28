import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def preprocess(images, size=(50, 50)):
    preprocessed = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, size)
        preprocessed.append(resized)
    return preprocessed

def kmeans(data, k, max_iterations=100, tolerance=1e-4):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    prev_centroids = centroids.copy()
    distances = np.zeros((data.shape[0], k))

    for _ in range(max_iterations):
        for i, c in enumerate(centroids):
            distances[:, i] = np.linalg.norm(data - c, axis=1)
        
        labels = np.argmin(distances, axis=1)
        for i in range(k):
            if len(data[labels == i]) > 0:
                centroids[i] = np.mean(data[labels == i], axis=0)

        if np.linalg.norm(centroids - prev_centroids) < tolerance:
            break

        prev_centroids = centroids.copy()

    return centroids, labels

def visualize_clusters(images, labels, k):
    for i in range(k):
        cluster_images = [img for img, label in zip(images, labels) if label == i]
        n = len(cluster_images)
        ncols = 5
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3 * nrows))
        axes = axes.ravel()
        
        for j, img in enumerate(cluster_images):
            axes[j].imshow(img, cmap='gray')
            axes[j].axis('off')
        
        for j in range(n, nrows * ncols):
            axes[j].axis('off')

        plt.suptitle(f"Cluster {i + 1}")
        plt.show()

folder = 'path/to/your/facial_images'
images = load_images(folder)
preprocessed = preprocess(images)
vectors = [img.flatten() for img in preprocessed]
data = np.array(vectors)

k = 3
centroids, labels = kmeans(data, k)

visualize_clusters(preprocessed, labels, k)
