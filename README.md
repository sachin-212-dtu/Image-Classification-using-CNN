# 🐱🐶 Cat vs Dog Image Classification using CNN

A **Convolutional Neural Network (CNN)** based image classification project that classifies images into two categories: **Cats** and **Dogs**.

The model was built using **TensorFlow/Keras** and trained on the Kaggle **Dogs vs Cats** dataset. The CNN achieved approximately **97% training accuracy (97.14%)** after 10 epochs. The best validation accuracy recorded during training was **79.76%**.

---

## 📌 Project Overview

Image classification is one of the fundamental applications of Computer Vision. In this project, a CNN model is trained to automatically identify whether an input image contains a **cat** or a **dog**.

The complete pipeline includes:

* Dataset downloading and extraction
* Image loading using TensorFlow/Keras
* Image resizing to `256 × 256`
* Pixel normalization
* CNN architecture design
* Batch Normalization
* Max Pooling
* Dropout for regularization
* Model training for 10 epochs
* Training and validation performance visualization

The dataset contains **25,000 images belonging to two classes**, with **20,000 images used for training** and **5,000 images used for validation/testing**.

---

## 🚀 Results

| Metric                   |              Result |
| ------------------------ | ------------------: |
| Classes                  |                   2 |
| Training Images          |              20,000 |
| Validation/Test Images   |               5,000 |
| Image Size               |           256 × 256 |
| Epochs                   |                  10 |
| Best Training Accuracy   |          **97.14%** |
| Best Validation Accuracy |          **79.76%** |
| Optimizer                |                Adam |
| Loss Function            | Binary Crossentropy |

The model reached **97.14% training accuracy** by the 10th epoch. The highest validation accuracy during the recorded training run was **79.76% at Epoch 9**.

---

## 🧠 CNN Architecture

The model uses a sequential CNN architecture consisting of three convolutional blocks followed by fully connected layers.

### Architecture

```text
Input Image
   │
   ▼
Conv2D (32 filters, 3×3)
   │
Batch Normalization
   │
MaxPooling2D
   │
   ▼
Conv2D (64 filters, 3×3)
   │
Batch Normalization
   │
MaxPooling2D
   │
   ▼
Conv2D (128 filters, 3×3)
   │
Batch Normalization
   │
MaxPooling2D
   │
   ▼
Flatten
   │
Dense (128, ReLU)
   │
Dropout (0.1)
   │
Dense (64, ReLU)
   │
Dropout (0.1)
   │
Dense (1, Sigmoid)
   │
   ▼
Cat / Dog Prediction
```

The CNN contains **14,848,193 total parameters**, with **14,847,745 trainable parameters**.

---

## 🛠️ Technologies Used

* **Python**
* **TensorFlow**
* **Keras**
* **NumPy**
* **Matplotlib**
* **CNN**
* **Google Colab**
* **Kaggle Dataset**

---

## 📂 Dataset

The project uses the **Dogs vs Cats** dataset from Kaggle.

### Dataset

**Dogs vs Cats — Kaggle**

https://www.kaggle.com/datasets/salader/dogsvscats

The dataset was downloaded directly using the Kaggle API in the notebook.

---

## ⚙️ Data Preprocessing

The images are loaded using Keras' `image_dataset_from_directory()`.

The images are:

* Resized to **256 × 256 pixels**
* Loaded in batches of **32**
* Labels are automatically inferred from directory names
* Pixel values are normalized from the range **0–255 to 0–1**

```python
train_ds = keras.utils.image_dataset_from_directory(
    directory='/content/train',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256,256)
)
```

The normalization process converts image pixel values using:

```python
image = tf.cast(image / 255, tf.float32)
```

## This preprocessing pipeline is implemented in the notebook.

## 🏗️ Model Configuration

The model is compiled using:

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

Training was performed for **10 epochs** using the training dataset and validation dataset.

---

## 📈 Training Performance

### Accuracy

Add the generated accuracy graph from the notebook to your repository:

```markdown
![Training and Validation Accuracy](images/accuracy.png)
```

The notebook plots both training and validation accuracy across epochs.

### Loss

Add the loss graph from the notebook:

```markdown
![Training and Validation Loss](images/loss.png)
```

The notebook also plots training and validation loss across epochs.

---

## 📊 Training Results by Epoch

| Epoch | Training Accuracy | Validation Accuracy |
| ----: | ----------------: | ------------------: |
|     1 |            58.10% |              69.12% |
|     2 |            71.32% |              74.46% |
|     3 |            77.43% |              77.20% |
|     4 |            81.24% |              75.50% |
|     5 |            84.67% |              77.68% |
|     6 |            88.74% |              77.70% |
|     7 |            92.15% |              79.54% |
|     8 |            94.95% |              73.88% |
|     9 |            96.39% |          **79.76%** |
|    10 |        **97.14%** |              78.40% |

The training log from the notebook reports these values for all 10 epochs.

---

## 🖼️ Sample Images

You can add sample Cat/Dog images to the repository and display them here:

```markdown
<p align="center">
  <img src="images/cat_sample.jpg" width="300">
  <img src="images/dog_sample.jpg" width="300">
</p>
```

### Example

![Cat Sample](images/cat_sample.jpg)
![Dog Sample](images/dog_sample.jpg)

> Place your sample images inside the `images/` folder of the repository.

---

## 📁 Project Structure

```text
Cat-Dog-Image-Classification/
│
├── Image_classification.ipynb
├── README.md
│
├── images/
│   ├── accuracy.png
│   ├── loss.png
│   ├── cat_sample.jpg
│   └── dog_sample.jpg
│
└── requirements.txt
```

---

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

Install the required libraries:

```bash
pip install tensorflow numpy matplotlib
```

---

## ▶️ How to Run

### 1. Open the Notebook

Open:

```text
Image_classification.ipynb
```

You can run it using **Google Colab** or a local Jupyter environment.

### 2. Download the Dataset

Download the dataset from:

```text
https://www.kaggle.com/datasets/salader/dogsvscats
```

### 3. Prepare the Dataset

The notebook expects the extracted dataset to contain:

```text
train/
test/
```

with the two classes:

```text
cats/
dogs/
```

### 4. Train the Model

Run the notebook cells sequentially to:

1. Load the dataset
2. Preprocess images
3. Build the CNN
4. Compile the model
5. Train for 10 epochs
6. Visualize accuracy and loss

---

## 🔍 Key Concepts Demonstrated

This project demonstrates practical implementation of:

* Image Classification
* Convolutional Neural Networks
* Convolution Layers
* ReLU Activation
* Max Pooling
* Batch Normalization
* Flattening
* Fully Connected Layers
* Dropout
* Sigmoid Activation
* Binary Classification
* Binary Crossentropy Loss
* Adam Optimizer
* Image Normalization
* Model Training
* Validation
* Performance Visualization

---

## 📌 Future Improvements

Possible improvements for the project include:

* Data augmentation
* Transfer learning using pretrained models
* Increasing validation accuracy
* Hyperparameter tuning
* Early stopping
* Learning-rate scheduling
* Confusion matrix
* Precision, recall and F1-score
* Deployment using Flask/FastAPI or Streamlit

---

## 🎯 Project Highlights

* Built a **CNN-based binary image classification model**
* Trained on **20,000 images**
* Evaluated on **5,000 images**
* Used **TensorFlow/Keras**
* Implemented **Batch Normalization and Dropout**
* Achieved **97.14% training accuracy**
* Visualized training and validation performance

---

## 👨‍💻 Author

**Sachin Kumar**

GitHub: [Add your GitHub profile link here](https://github.com/)

LinkedIn: [Add your LinkedIn profile link here](https://www.linkedin.com/)

---

## ⭐ If You Found This Project Useful

If you found this project helpful, consider giving the repository a ⭐ on GitHub!
