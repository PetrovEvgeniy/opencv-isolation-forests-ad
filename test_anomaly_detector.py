# Import the necessary packages
from helper.features import quantify_image
import argparse
import pickle
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained anomaly detection model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# Load the anomaly detection model
print("[INFO] Loading anomaly detection model...")
model = pickle.loads(open(args["model"], "rb").read())
print("[INFO] Model details:", model)

# Load the input image, convert it to the HSV color space, and
# Quantify the image in the *same manner* as we did during training
image = cv2.imread(args["image"])
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
features = quantify_image(hsv, bins=(3, 3, 3))


# Use the anomaly detector model and extracted features to determine
# If the example image is an anomaly or not
preds = model.predict([features])[0]
print(f"Extracted features shape: {features.shape}")
print("Extracted features:", features)
print(f"Prediction result: {preds}")

label = "anomaly" if preds == -1 else "normal"
color = (0, 0, 255) if preds == -1 else (0, 255, 0)

# Draw the predicted label text on the original image
cv2.putText(image, label, (10,  25), cv2.FONT_HERSHEY_SIMPLEX,
	0.7, color, 2)

# Display the image
cv2.imshow("Output", image)
cv2.waitKey(0)