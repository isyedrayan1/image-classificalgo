import cv2
import os
import numpy as np
from joblib import load
def preprocess_image(image_path, image_size=(64, 64)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unreadable.")
    image = cv2.resize(image, image_size)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.flatten() / 255.0
model = load('knn_model.joblib')
test_image_path = os.path.join("data", "validation-h2h","horses", "horse1-122.png") #use any image from data/validation-h2h folder or any image from internet to
feature = preprocess_image(test_image_path).reshape(1, -1)
prediction = model.predict(feature)
print("Predicted Label:", prediction[0])