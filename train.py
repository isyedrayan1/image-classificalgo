import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump


dataset_path = os.path.join("data", "train-h2h")
image_size = (64, 64)

labels = []
features = []

for label_name in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label_name)
    if not os.path.isdir(label_path):
        continue
    for filename in os.listdir(label_path):
        image_path = os.path.join(label_path, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue
        image = cv2.resize(image, image_size)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features.append(gray_image.flatten() / 255.0)
        labels.append(label_name)
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
#save the model
dump(model, 'knn_model.joblib')
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))