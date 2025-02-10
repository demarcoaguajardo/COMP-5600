# imageCompression.py

# Importing Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plot
from sklearn.cluster import KMeans

# ---------- Load image ----------
img = cv2.imread('test_image.png')
# Coverts BGR image to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width, channels = np.shape(img)

# Reshape image to a 2D array of pixels
pixels = img.reshape(-1, 3)

# ---------- Perform K-Means ----------

# Get number of clusters as input from user
k = int(input("\nEnter the number of clusters (k): "))

# K-Means
kmeans = KMeans(n_clusters=k, random_state=88)
kmeans.fit(pixels)
clusters = kmeans.predict(pixels)
centroids = kmeans.cluster_centers_

# ---------- Reconstruct the image ----------

# Reconstruct the image using the centroids and clusters
def reconstructImage(centroids, clusters, height, width):
    compressedPixels = centroids[clusters].astype(np.uint8)
    return compressedPixels.reshape(height, width, 3)

compressedImg = reconstructImage(centroids, clusters, height, width)

# Iterate each pixel in the image and assign closest color to each pixel
for i in range(width):
    for j in range(height):
        pixel = img[j, i] # Read the pixel at location (i, j)
        clusterIndex = clusters[j * width + i]
        newValue = centroids[clusterIndex] 
        img[j][i] = newValue # Assign the new value to the pixel

# ---------- Display the images ----------
plot.imshow(img)
plot.title("Compressed Image with k={}".format(k))
plot.show()

# Save the compressed image
compressedImgBGR = cv2.cvtColor(compressedImg, cv2.COLOR_RGB2BGR) # Convert RGB to BGR
cv2.imwrite('compressed_test_image.png', compressedImgBGR)
