#  Horses vs Humans Image Classifier (KNN)

This is a basic machine learning project that uses **K-Nearest Neighbors (KNN)** to classify images as either **horse** or **human**.  
It uses **OpenCV (cv2)** to process images and **joblib** to save and load the trained model.



## 📦 Imports Used

- `cv2` – for reading, resizing, and converting images to grayscale  
- `numpy` – for handling image arrays  
- `joblib` – to save (`dump`) and load (`load`) the trained model  
- `sklearn` – for training the KNN model and evaluating accuracy



##  How to Run This Project

### 1. Clone the Repository
```bash
git clone https://github.com/isyedrayan1/image-classificalgo.git
cd image-classificalgo
````

### 2. Install Dependencies

```bash
pip install opencv-python scikit-learn joblib numpy
```

### 3. Delete Old Model (Optional)

```bash
del knn_model.joblib  # or delete manually from the folder
```

### 4. Train the Model

Make sure your training images are inside `data/train/` in folders like `horses/` and `humans/`.

Dowmload them from this link https://laurencemoroney.com/datasets.html and export them in your repective folder. Then run the train.py

```bash
python train.py
```

This will automatically create `knn_model.joblib` in the project folder.

### 5. Test the Model

Put your test image inside `data/test/` and update the filename in `test.py`. as it is in the dataset or just use any image from training datat or internet.

```bash
python test.py
```

You’ll see the predicted label printed in the terminal.

---
