import face_recognition
import cv2
import os
import glob
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from PIL import Image
input_directory = "path/to/extracted_faces"
output_directory = "path/to/output_clusters"
face_encodings = []
image_paths = []

# Iterate through the face images in the input directory
for face_image_path in glob.glob(os.path.join(input_directory, "*.jpg")):
    # Load the face image
    face_image = face_recognition.load_image_file(face_image_path)
    
    # Extract the feature vector for the face
    face_encoding = face_recognition.face_encodings(face_image)
    
    if len(face_encoding) > 0:
        # Add the feature vector and image path to the respective lists
        face_encodings.append(face_encoding[0])
        image_paths.append(face_image_path)
# Normalize the feature vectors
scaler = StandardScaler()
normalized_face_encodings = scaler.fit_transform(face_encodings)

# Choose the number of clusters (K)
num_clusters = 5

# Apply K-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(normalized_face_encodings)
# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Iterate through the face images and their cluster assignments
for face_image_path, cluster_label in zip(image_paths, kmeans.labels_):
    # Create a directory for the cluster if it doesn't exist
    cluster_directory = os.path.join(output_directory, f"cluster_{cluster_label}")
    if not os.path.exists(cluster_directory):
        os.makedirs(cluster_directory)

    # Save the face image to the cluster directory
    img = Image.open(face_image_path)
    img.save(os.path.join(cluster_directory, os.path.basename(face_image_path)))
