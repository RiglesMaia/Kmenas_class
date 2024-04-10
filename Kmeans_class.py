#@R.Maia
#Kmeans
###################################################

import tkinter as tk
import cv2
import os
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from skimage.measure import label, regionprops

# Function to get values from the GUI
def get_values():
    # Function called when the "OK" button is clicked
    global path_tif, name, n_clusters, optimal_k, desired_class, n_blur
    path_tif = str(entry_path.get())
    name = str(entry_name.get())
    n_clusters = int(entry_n_clusters.get())
    optimal_k = int(entry_optimal_k.get())
    desired_class = int(entry_desired_class.get())
    n_blur = int(entry_blur.get())
    root.destroy()

# Create the main window
root = tk.Tk()
root.title("Parameter Configuration")
root.geometry("400x500")  # Adjusting the window size

# Change font size
font = ("Helvetica", 15)

# Add input fields and labels
label_path = tk.Label(root, font=font, text="Address:")
label_path.pack()
entry_path = tk.Entry(root, font=font)
entry_path.pack()

label_name = tk.Label(root, font=font, text="Image ID:")
label_name.pack()
entry_name = tk.Entry(root, font=font)
entry_name.pack()

label_n_clusters = tk.Label(root, font=font, text="Maximum number of clusters:")
label_n_clusters.pack()
entry_n_clusters = tk.Entry(root, font=font)
entry_n_clusters.pack()

label_optimal_k = tk.Label(root, font=font, text="Number of clusters:")
label_optimal_k.pack()
entry_optimal_k = tk.Entry(root, font=font)
entry_optimal_k.pack()

label_desired_class = tk.Label(root, font=font, text="Desired class:")
label_desired_class.pack()
entry_desired_class = tk.Entry(root, font=font)
entry_desired_class.pack()

label_blur = tk.Label(root, font=font, text="Blur:")
label_blur.pack()
entry_blur = tk.Entry(root, font=font)
entry_blur.pack()

# Add OK button
button_ok = tk.Button(root, text="OK", font=font, command=get_values)
button_ok.pack()

# Start the GUI loop
root.mainloop()

print("************************************Starting K-means classification process************************************")

# Load the image
image = imread(os.path.join(path_tif, name))

# Apply blur using OpenCV
kernel_size = n_blur
blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Reshape to a 2D array
X = blurred_image.reshape(-1, 3)

# List to store inertia values
inertia_values = []

# Test different k values
for k in range(1, n_clusters):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

# Plot the elbow method
plt.plot(range(1, n_clusters), inertia_values, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Find the index of the smallest inertia value
optimal_k_index = optimal_k

# Optimal number of clusters chosen
optimal_k = optimal_k_index 

# Apply KMeans algorithm with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
kmeans.fit(X)

# Segment the image
segment_image = kmeans.labels_.reshape(image.shape[:2])

# Create a colormap to associate cluster indices with specific colors
color_map = plt.cm.get_cmap('viridis', optimal_k)

# Display the segmented image with legend
plt.imshow(segment_image, cmap=color_map)
plt.colorbar(ticks=range(1, optimal_k), label='Cluster')
plt.title('Segmented Image with ({} clusters)'.format(optimal_k))
plt.show()

# Extract the specific class
specific_class = kmeans.labels_.reshape(image.shape[:2]) == desired_class

# Label connected regions
labeled_regions = label(specific_class)

# Find region properties
props = regionprops(labeled_regions)

# Find the largest region
largest_region = max(props, key=lambda x: x.area)

# Get bounding box coordinates of the largest region
minr, minc, maxr, maxc = largest_region.bbox

# Crop the image around the largest region
cropped_image = specific_class[minr:maxr, minc:maxc]

# Display the original image
plt.imshow(specific_class)
plt.title('Original Image')
plt.show()

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Subplot 1: Cropped image
axs[0].imshow(cropped_image)
axs[0].set_title('Specific Class')

# Subplot 2: Original image
axs[1].imshow(image)
axs[1].set_title('Original Image')

# Adjust layout to avoid overlapping
plt.tight_layout()

# Display the figure
plt.show() 

num_pixels = largest_region.area
print("....................Finished....................")
print(f"Number of pixels: {num_pixels}")
