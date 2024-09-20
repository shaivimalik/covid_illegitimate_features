::: {.cell .markdown}

# Exploring ConvNet Activations

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaivimalik/covid_illegitimate_features/blob/main/notebooks/Exploring_ConvNet_Activations.ipynb)

:::

::: {.cell .markdown}
## Introduction

In this notebook, we will continue our discussion of data leakage and its impact on model performance by training a convolutional neural network (CNN) to distinguish between husky dogs and wolves [2]. We will use small datasets of 100 images (50 of each class). This classification task will help us understand how a model can learn illegitimate features. 

We will follow two approaches:

- Train without Data Leakage: In this case, the dataset will have husky images with grass backgrounds and wolf images with snow backgrounds. We will train and evaluate our CNN on this dataset and report the accuracy and confusion matrix obtained on the test set.

- Train without Data Leakage: In this case, images of both classes have a white background. We will train and evaluate our CNN on this dataset and report the accuracy and confusion matrix obtained on the test set.

Model Architecture (for both approaches):

- 3 convolutional layers with ReLU activation, each followed by a max-pooling layer
- Flatten layer
- 2 fully connected layers
    - First layer with ReLU activation
    - Second layer with softmax activation

Each model will be trained for 10 epochs using the Adam optimizer, with a learning rate of 0.001 and a batch size of 4 images.

:::

::: {.cell .code}
```python
# Uncomment the following lines if running on Google Colab
#!git clone https://github.com/shaivimalik/covid_illegitimate_features.git
#!pip install -r covid_illegitimate_features/requirements.txt
#%cd covid_illegitimate_features/notebooks
```
:::

::: {.cell .markdown}
## Data Leakage

Data leakage occurs when a model learns to recognise patterns or relationships between the features and target variable during training that don't exist in the real-world data. Since these patterns won’t be present in the real-world data about which the claims are made, models with data leakage errors fail to generalise to unseen data [1]. Data leakage includes errors such as:

- **No test set:** If the model is trained and tested on the same data, it will perform exceptionally well on the test set, but it will fail on unseen data.

- **Temporal leakage:** This occurs when data from the future is used to train a model created to predict future events.

- **Duplicates in datasets:** If there are duplicate data points in both the training and test sets, the model can memorize these instances, leading to inflated performance metrics.

- **Pre-processing on training and test set:** If pre-processing is performed on the entire dataset, information about the test set may leak into the training set. 

- **Model uses features that are not legitimate:** If the model has access to features that should not be legitimately available for use. For example, when information about the target variable is incorporated into the features used for training.

Data leakage leads to overly optimistic estimates of model performance. It is also identified as the major cause behind the reproducibility crisis in ML-Based science [3].

In this notebook, we will discover the consequences of Model uses features that are not legitimate on model performance.

:::

::: {.cell .markdown}
## Load the Dataset with Different Backgrounds for Each Class

In this section, we will load our dataset and split it into training, validation and test sets. The training set is used to train the model, the validation is used to find optimal hyperameter values (learning rate, epochs, batch size) and the test set is used to evaluate the classifier. 

We will divide the dataset into a 70-10-20 split: 70% of the images will be used for training, 10% of the images will be used for validation and 20% will be used for testing. In this notebook, we will not perform any hyperparameter optimization, but feel free to experiment with the hyperparameters to find optimal values for each model using the validation set.

We start by importing the required libraries.

:::

::: {.cell .code}
```python
import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
keras.utils.set_random_seed(27)
```
:::

::: {.cell .markdown}

In this cell, we load our dataset using `image_dataset_from_directory` function from Keras. The images are resized to (256,256) pixels and batches of 4 images are created. The label associated with each image is one-hot encoded.

:::


::: {.cell .code}
```python
# Define image size and batch size
image_size = (256,256)
batch_size = 4

# Load training and validation sets from directory
train_ds_leak, val_ds_leak= keras.utils.image_dataset_from_directory(
    '../different_backgrounds/train', 
    label_mode="categorical", 
    image_size=image_size, 
    batch_size=batch_size,
    seed=27,
    validation_split=0.125,
    subset='both'
)
```
:::

::: {.cell .code}
```python
# Load test set from directory
test_ds_leak= keras.utils.image_dataset_from_directory(
    '../different_backgrounds/test', 
    label_mode="categorical", 
    image_size=image_size, 
    batch_size=batch_size,
    seed=27,
    shuffle=False
)
```
:::

::: {.cell .markdown}
## Training and Evaluating Model with Data Leakage

In this section, we will train and evaluate our model. We will use the `Sequential` class from Keras to create the model. The model will consist of 3 convolutional layers, each followed by a max pooling layer. These will be followed by a flatten layer and two fully connected (dense) layers. All convolutional layers and the first dense layer will use ReLU activation. The final dense layer will use softmax activation, which is equivalent to sigmoid when doing binary classification.

:::

::: {.cell .code}
```python
num_classes = 2

# Create model
model_leak = keras.Sequential()

# Add input layer
model_leak.add(keras.Input(shape=image_size + (3,)))

# Add rescaling layer to normalize pixel values
model_leak.add(layers.Rescaling(scale=1./255))

# Add convolutional and pooling layers
model_leak.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding="valid", activation='relu', use_bias=True))
model_leak.add(layers.MaxPooling2D(pool_size=(2, 2),padding="valid"))
model_leak.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="valid", activation='relu', use_bias=True))
model_leak.add(layers.MaxPooling2D(pool_size=(2, 2),padding="valid"))
model_leak.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="valid", activation='relu', use_bias=True))
model_leak.add(layers.MaxPooling2D(pool_size=(2, 2),padding="valid"))

# Flatten the output and add dense layers
model_leak.add(layers.Flatten())
model_leak.add(layers.Dense(64, activation='relu'))
model_leak.add(layers.Dense(num_classes, activation="softmax"))

model_leak.summary()
```
:::

::: {.cell .markdown}

In this cell, we compile and train our model. We use `categorical_crossentropy` loss and train our model on the training set for 10 epochs with `adam` optimizer.

:::

::: {.cell .code}
```python
epochs = 10

# Compile the model
model_leak.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
history_leak = model_leak.fit(train_ds_leak, batch_size=batch_size, epochs=epochs, validation_data=val_ds_leak)
```
:::

::: {.cell .markdown}

Next, we'll plot the accuracy and loss for both the training and validation sets across epochs. These plots are useful for identifying signs of overfitting.

:::

::: {.cell .code}
```python
# summarize history for accuracy
plt.plot(history_leak.history['accuracy'])
plt.plot(history_leak.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history_leak.history['loss'])
plt.plot(history_leak.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
```
:::

::: {.cell .markdown}

Now, we evaluate our model on the test set, compute the accuracy and display the confusion matrix.

:::

::: {.cell .code}
```python
# Evaluate the model on test data
score_leak = model_leak.evaluate(test_ds_leak)
print("Test loss:", score_leak[0])
print("Test accuracy:", score_leak[1])
```
:::

::: {.cell .code}
```python
# Predict classes for the test dataset using the 'model_leak'
y_pred_leak = model_leak.predict(test_ds_leak)

# Convert predicted probabilities to class labels (0 or 1)
y_pred_leak = np.argmax(y_pred_leak, axis=1)

# Extract true labels from the test dataset
y_true_leak = np.concatenate([np.argmax(label, axis=1) for _, label in test_ds_leak], axis=0)

# Create a confusion matrix comparing true labels to predicted labels
conf_mat_leak = confusion_matrix(y_true_leak, y_pred_leak)

# Display the confusion matrix as a heatmap
ConfusionMatrixDisplay(conf_mat_leak, display_labels=['husky', 'wolf']).plot(cmap='Blues')
```
:::

::: {.cell .markdown}

Let's find the pixels responsible for an image being classified as a husky dog or wolf using GradCAM [4]. You may learn more about saliency maps and feature visualisation [here](https://harvard-iacs.github.io/2021-CS109B/lectures/lecture17/).

:::

::: {.cell .code}
```python
from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam

# Define image titles for visualization
image_titles = ['husky', 'wolf']

# Create lists of file paths for husky and wolf images
husky_files = np.array(['../different_backgrounds/test/husky/'+x for x in os.listdir('../different_backgrounds/test/husky')])
wolf_files = np.array(['../different_backgrounds/test/wolf/'+x for x in os.listdir('../different_backgrounds/test/wolf')])

# Load random images for each class and convert them to a Numpy array
husky = keras.utils.load_img(np.random.choice(husky_files), target_size=image_size)
wolf = keras.utils.load_img(np.random.choice(wolf_files), target_size=image_size)
images = np.asarray([np.array(husky), np.array(wolf)])
X = np.array([keras.utils.img_to_array(img) for img in images])

# Render the original images
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
for i, title in enumerate(image_titles):
    ax[i].set_title(title, fontsize=16)
    ax[i].imshow(images[i])
    ax[i].axis('off')
plt.tight_layout()
plt.show()

# Define a function to modify the model for GradCAM
def model_modifier_function(cloned_model):
    cloned_model.layers[-1].activation = keras.activations.linear

# Define a score function for GradCAM
def score_function(output):
    return (output[0,0], output[1,1])

# Create Gradcam object
gradcam = Gradcam(model_leak, model_modifier=model_modifier_function, clone=True)

# Generate heatmap with GradCAM
cam = gradcam(score_function, X)

# Render the images with GradCAM heatmaps overlaid
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
for i, title in enumerate(image_titles):
    heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
    ax[i].set_title(title, fontsize=16)
    ax[i].imshow(images[i])
    ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
    ax[i].axis('off')
plt.tight_layout()
plt.show()
```
:::

::: {.cell .markdown}

The heatmaps reveal that background pixels are primarily responsible for the model's classification decisions. Instead of learning to distinguish between wolves and huskies, the model has learned to differentiate between grass and snow backgrounds. We can further validate this observation by evaluating the classifier on a dataset where the backgrounds are swapped: huskies with snow backgrounds and wolves with green backgrounds. The next cell loads this background swapped dataset using `image_dataset_from_directory` and computes the accuracy obtained by the model.

:::

::: {.cell .code}
```python
# Load and display wolf image with green background
keras.utils.load_img("../background_swap/wolf/"+np.random.choice(os.listdir("../background_swap/wolf")), target_size=(256,256))
```
:::

::: {.cell .code}
```python
# Load and display husky image with snow background
keras.utils.load_img("../background_swap/husky/"+np.random.choice(os.listdir("../background_swap/husky")), target_size=(256,256))
```
:::

::: {.cell .code}
```python
# Evaluate the model on background_swap dataset
background_swap = keras.utils.image_dataset_from_directory(
    '../background_swap', 
    label_mode="categorical", 
    image_size=image_size, 
    batch_size=batch_size,
    shuffle=False,
    seed=27
)
score_swap = model_leak.evaluate(background_swap)
print("Test loss:", score_swap[0])
print("Test accuracy:", score_swap[1])
```
:::

::: {.cell .markdown}

The model's performance on this dataset is extremely poor. In the next cell, we display the confusion matrix obtained on this dataset. 

:::

::: {.cell .code}
```python
# Predict classes for the background-swapped dataset using the 'model_leak'
y_pred_swap = model_leak.predict(background_swap)

# Convert predicted probabilities to class labels (0 or 1)
y_pred_swap = np.argmax(y_pred_swap, axis=1)

# Extract true labels from the background-swapped dataset
y_true_swap = np.concatenate([np.argmax(label, axis=1) for _, label in background_swap], axis=0)

# Create a confusion matrix comparing true labels to predicted labels
conf_mat_swap = confusion_matrix(y_true_swap, y_pred_swap)

# Display the confusion matrix as a heatmap
ConfusionMatrixDisplay(conf_mat_swap, display_labels=['husky', 'wolf']).plot(cmap='Blues')
```
:::

::: {.cell .markdown}

From the confusion matrix, we can conclude that the model has misclassified wolves as huskies and vice versa. This indicates that the model has learned features that recognize the snow and grass backgrounds.

:::

::: {.cell .markdown}
## Load the Dataset with White Backgrounds Across All Classes

In this section, we will load a dataset consisting of images with a white background. The dataset consists of 100 images, split into 70 for training, 10 for validation, and 20 for testing. `image_dataset_from_directory` from Keras is used to load the dataset with `batch_size` of 4 and `image_size` of (256,256).

:::

::: {.cell .code}
```python
# Define image size and batch size
image_size = (256,256)
batch_size = 4

# Load training and validation sets from directory
train_ds, val_ds = keras.utils.image_dataset_from_directory(
    '../same_backgrounds/train', 
    label_mode="categorical", 
    image_size=image_size, 
    batch_size=batch_size,
    seed=27,
    validation_split=0.125,
    subset='both'
)
```
:::

::: {.cell .code}
```python
# Load test set from directory
test_ds= keras.utils.image_dataset_from_directory(
    '../same_backgrounds/test', 
    label_mode="categorical", 
    image_size=image_size, 
    batch_size=batch_size,
    seed=17,
    shuffle=False
)
```
:::

::: {.cell .markdown}
## Training and Evaluating model without data leakage

In this section, we will train and evaluate a new model, having no data leakage errors. We'll create a model with the same architecture as before, using the `Sequential` class from Keras.

:::

::: {.cell .code}
```python
num_classes = 2

# Create the model
model = keras.Sequential()

# Add input layer
model.add(keras.Input(shape=image_size + (3,)))

# Add rescaling layer to normalize pixel values
model.add(layers.Rescaling(scale=1./255))

# Add convolutional and pooling layers
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding="valid", activation='relu', use_bias=True))
model.add(layers.MaxPooling2D(pool_size=(2, 2),padding="valid"))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="valid", activation='relu', use_bias=True))
model.add(layers.MaxPooling2D(pool_size=(2, 2),padding="valid"))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="valid", activation='relu', use_bias=True))
model.add(layers.MaxPooling2D(pool_size=(2, 2),padding="valid"))

# Flatten the output and add dense layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_classes, activation="softmax"))

model.summary()
```
:::

::: {.cell .markdown}

We compile and train the model using a learning rate of 0.001, the `adam` optimizer, and `categorical_crossentropy` loss for 10 epochs.

_Use different hyperparameter values and see how it affects performance on validation set._

:::

::: {.cell .code}
```python
epochs = 10

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
history = model.fit(train_ds, batch_size=batch_size, epochs=epochs, validation_data=val_ds)
```
:::

::: {.cell .markdown}

We plot loss and accuracy on the test and validation sets against the number of epochs.

:::

::: {.cell .code}
```python
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
```
:::

::: {.cell .markdown}

We compute the model's accuracy on the test set and display the confusion matrix.

:::

::: {.cell .code}
```python
# Evaluate the model on test data
score = model.evaluate(test_ds)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
```
:::

::: {.cell .code}
```python
# Predict classes for the test dataset using the 'model'
y_pred = model.predict(test_ds)

# Convert predicted probabilities to class labels (0 or 1)
y_pred = np.argmax(y_pred, axis=1)

# Extract true labels from the test dataset
y_true = np.concatenate([np.argmax(label, axis=1) for _, label in test_ds], axis=0)

# Create a confusion matrix comparing true labels to predicted labels
conf_mat = confusion_matrix(y_true, y_pred)

# Display the confusion matrix as a heatmap
ConfusionMatrixDisplay(conf_mat, display_labels=['husky', 'wolf']).plot(cmap='Blues')
```
:::

::: {.cell .markdown}

Next, we visualise which pixels are responsible for image I being classified as an image of class C using GradCAM [4]. 

:::

::: {.cell .code}
```python
from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam

# Define image titles for visualization
image_titles = ['husky', 'wolf']

# Create lists of file paths for husky and wolf images
husky_files = np.array(['../same_backgrounds/test/husky/'+x for x in os.listdir('../same_backgrounds/test/husky')])
wolf_files = np.array(['../same_backgrounds/test/wolf/'+x for x in os.listdir('../same_backgrounds/test/wolf')])

# Load random images for each class and convert them to a Numpy array
husky = keras.utils.load_img(np.random.choice(husky_files), target_size=image_size)
wolf = keras.utils.load_img(np.random.choice(wolf_files), target_size=image_size)
images = np.asarray([np.array(husky), np.array(wolf)])
X = np.array([keras.utils.img_to_array(img) for img in images])

# Render the original images
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
for i, title in enumerate(image_titles):
    ax[i].set_title(title, fontsize=16)
    ax[i].imshow(images[i])
    ax[i].axis('off')
plt.tight_layout()
plt.show()

# Define a function to modify the model for GradCAM
def model_modifier_function(cloned_model):
    cloned_model.layers[-1].activation = keras.activations.linear

# Define a score function for GradCAM
def score_function(output):
    return (output[0,0], output[1,1])

# Create Gradcam object
gradcam = Gradcam(model, model_modifier=model_modifier_function, clone=True)

# Generate heatmap with GradCAM
cam = gradcam(score_function, X)

# Render the images with GradCAM heatmaps overlaid
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
for i, title in enumerate(image_titles):
    heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
    ax[i].set_title(title, fontsize=16)
    ax[i].imshow(images[i])
    ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
    ax[i].axis('off')
plt.tight_layout()
plt.show()
```
:::

::: {.cell .markdown}
## Discussion

| Metric        | With Data Leakage | Without Data Leakage |
|:-------------:|:-----------------:|:--------------------:|
| Accuracy      | 90.0              | 70.0                 | 

In the case with data leakage, we achieved an accuracy of 90%, while in the case without data leakage, we achieved an accuracy of 70%. This indicates that data leakage led to overly optimistic measures of the model's performance.

The model should not have access to any information about the target variable, nor should it be allowed to learn features that are not legitimate. Determining which features are legitimate or illegitimate requires domain expertise. However, it is always good practice to examine the weights learned by a machine learning model. This can be done using saliency maps for complex models or by simply inspecting the array of weights for simpler machine learning models.

:::

::: {.cell .markdown}
## References

[1]: Kapoor S, Narayanan A. Leakage and the reproducibility crisis in machine-learning-based science. Patterns (N Y). 2023 Aug 4;4(9):100804. doi: 10.1016/j.patter.2023.100804. PMID: 37720327; PMCID: PMC10499856.

[2]: Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. 2016. "Why Should I Trust You?": Explaining the Predictions of Any Classifier. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16). Association for Computing Machinery, New York, NY, USA, 1135–1144. https://doi.org/10.1145/2939672.2939778

[3]: Nisbet, R., Elder, J., and Miner, G. Handbook of Statistical Analysis and Data Mining Applications. Elsevier, 2009. ISBN 978-0-12-374765-5.

[4]: R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh and D. Batra, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," 2017 IEEE International Conference on Computer Vision (ICCV), Venice, Italy, 2017, pp. 618-626, doi: 10.1109/ICCV.2017.74. 

:::