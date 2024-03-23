# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
To develop a Convolutional Deep Neural Network (CNN) for digit classification using the MNIST dataset. The goal is to create a model capable of accurately classifying handwritten digits (0-9) from the provided images or scanned images. 
<br>
<br>
The MNIST dataset consists of 60,000 training images and 10,000 testing images, each of which is a grayscale image of size 28x28 pixels. MNIST has been extensively used for training and testing various machine learning models, particularly for digit recognition tasks.

The problem statement outlined above specifically addresses the task of digit classification using the MNIST dataset. It involves developing a Convolutional Neural Network (CNN) model to accurately recognize handwritten digits from the provided images. 

## Neural Network Model

![Screenshot 2024-03-23 204838](https://github.com/Jenishajustin/mnist-classification/assets/119405070/78f67bf8-d364-4372-89d0-0b1412e0ab01)


## DESIGN STEPS

### STEP 1:
Import the required tensorflow and preprocessing libraries.

### STEP 2:
Load the MNIST dataset and split into training and test datasets.
### STEP 3:
Scale the actual training dataset and the corresponding test dataset between 0 and 1.
### STEP 4:
Apply One-hot Encoding.
### STEP 5:
Build a CNN model.
### STEP 6:
Compile and fit the model.

### STEP 7:
Evaluate loss and accuracy with training and test datasets.
### STEP 8:
Evaluate and report model performance.
### STEP 9:
Predict for the single output.
## PROGRAM

### Name: J.JENISHA
### Register Number: 212222230056
```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
from PIL import Image

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape
single_image= X_train[20000]
single_image.shape
plt.imshow(single_image,cmap='gray')
y_train.shape
X_train.min()
X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
X_train_scaled.min()
X_train_scaled.max()

y_train[0]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
type(y_train_onehot)
y_train_onehot.shape
single_image = X_train[0]
plt.imshow(single_image,cmap='gray')
y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(5,5),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(20,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')
model.fit(X_train_scaled ,y_train_onehot, epochs=5,batch_size=70, validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)
metrics.head()
print("Name : J.JENISHA\nReg. No : 212222230056")
metrics[['accuracy','val_accuracy']].plot()
print("Name : J.JENISHA\nReg. No : 212222230056")
metrics[['loss','val_loss']].plot()
plt.title("Training Loss vs Validation Loss")
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)  
print("Name : J.JENISHA\nReg. No : 212222230056\n\nCONFUSION MATRIX\n")
print(confusion_matrix(y_test,x_test_predictions))
print("Name : J.JENISHA\nReg. No : 212222230056\n\nCLASSIFICATION REPORT\n")
print(classification_report(y_test,x_test_predictions))

# Prediction for a Single Input 
img = image.load_img('7_f.jpg')
type(img)
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
print("Name : J.JENISHA\nReg. No : 212222230056")
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![Screenshot 2024-03-23 201354](https://github.com/Jenishajustin/mnist-classification/assets/119405070/a993d1c3-646f-478b-bbc4-3f60f8636e2c)

![Screenshot 2024-03-23 201323](https://github.com/Jenishajustin/mnist-classification/assets/119405070/7ec6783a-8d4d-4886-a756-2f6c7a6784ed)


### Classification Report
![Screenshot 2024-03-23 201255](https://github.com/Jenishajustin/mnist-classification/assets/119405070/6d9ba54b-5337-41da-9252-dc3c2b5216dc)



### Confusion Matrix
![Screenshot 2024-03-23 201226](https://github.com/Jenishajustin/mnist-classification/assets/119405070/82ab1d50-6b88-40a4-b15e-fbfc217e7d8f)



### New Sample Data Prediction
<img src="https://github.com/Jenishajustin/mnist-classification/assets/119405070/4a8aea9a-f0f8-485b-800b-f46fe147cba6" height=400 width=400>

![Screenshot 2024-03-23 201156](https://github.com/Jenishajustin/mnist-classification/assets/119405070/cf7ff2a8-b468-46f2-a63c-7203b9841c7a)
![Screenshot 2024-03-23 201829](https://github.com/Jenishajustin/mnist-classification/assets/119405070/f84f8181-2531-44cc-9077-78af284c28e4)

## RESULT
A convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed successfully.
