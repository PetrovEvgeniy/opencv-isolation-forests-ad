# Import the necessary packages
from helper.features import load_dataset
from sklearn.ensemble import IsolationForest
import argparse
import pickle

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to dataset of images")
ap.add_argument("-m", "--model", required=True,
	help="path to output anomaly detection model")
args = vars(ap.parse_args())

# Load and quantify our image dataset
print("[INFO] Preparing dataset...")
data = load_dataset(args["dataset"], bins=(3, 3, 3))
print(f"Number of images in dataset: {len(data)}")

# Train the anomaly detection model
print("[INFO] Fitting anomaly detection model...")
model = IsolationForest(n_estimators=100, contamination=0.01,
	random_state=42)
model.fit(data)

# Serialize the anomaly detection model to disk
f = open(args["model"], "wb")
f.write(pickle.dumps(model))
f.close()