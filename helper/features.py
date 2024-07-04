
# Import the necessary packages
from imutils import paths
import numpy as np
import cv2

def quantify_image(image, bins=(4, 6, 3)):
	
	# Compute a 3D color histogram over the image and normalize it
	hist = cv2.calcHist([image], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
	hist = cv2.normalize(hist, hist).flatten()
	
	# Inside quantify_image function in features.py
	print(f"Extracted features shape: {hist.shape}")
	print(f"Extracted features: {hist}")

	# Return the histogram
	return hist

def load_dataset(datasetPath, bins):
	# Grab the paths to all images in our dataset directory, then
	# Initialize our lists of images
	imagePaths = list(paths.list_images(datasetPath))
	data = []
	
	# Loop over the image paths
	for imagePath in imagePaths:
		# Load the image and convert it to the HSV color space
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		# Quantify the image and update the data list
		features = quantify_image(image, bins)
		data.append(features)
		
	# Return our data list as a NumPy array
	return np.array(data)