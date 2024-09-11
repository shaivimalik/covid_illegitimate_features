::: {.cell .markdown}

# Exploring ConvNet Activations

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaivimalik/covid_illegitimate_features/blob/main/notebooks/Exploring_ConvNet_Activations.ipynb)

:::

::: {.cell .code}
```python
# Uncomment the following lines if running on Google Colab
#!git clone https://github.com/shaivimalik/covid_illegitimate_features.git
#!pip install -r covid_illegitimate_features/requirements.txt
#%cd covid_illegitimate_features/notebooks
```
:::

::: {.cell .code}
```python
import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers
from tensorflow import data as tf_data
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
keras.utils.set_random_seed(27)
```
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

::: {.cell .code}
```python
epochs = 10

# Compile the model
model_leak.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
history_leak = model_leak.fit(train_ds_leak, batch_size=batch_size, epochs=epochs, validation_data=val_ds_leak)
```
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
y_pred_leak = model_leak.predict(test_ds_leak)
y_pred_leak = np.argmax(y_pred_leak, axis=1)
y_true_leak = np.concatenate([np.argmax(label, axis=1) for _, label in test_ds_leak], axis=0)
conf_mat_leak = confusion_matrix(y_true_leak,y_pred_leak)
ConfusionMatrixDisplay(conf_mat_leak,display_labels=['husky','wolf']).plot(cmap='Blues')
```
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

::: {.cell .code}
```python
# Evaluate the model on test data
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

::: {.cell .code}
```python
y_pred_swap = model_leak.predict(background_swap)
y_pred_swap = np.argmax(y_pred_swap, axis=1)
y_true_swap = np.concatenate([np.argmax(label, axis=1) for _, label in background_swap], axis=0)
conf_mat_swap = confusion_matrix(y_true_swap,y_pred_swap)
ConfusionMatrixDisplay(conf_mat_swap,display_labels=['husky','wolf']).plot(cmap='Blues')
```
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

::: {.cell .code}
```python
epochs = 10

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(train_ds, batch_size=batch_size, epochs=epochs, validation_data=val_ds)
```
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
y_pred = model.predict(test_ds)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.concatenate([np.argmax(label, axis=1) for _, label in test_ds], axis=0)
conf_mat = confusion_matrix(y_true,y_pred)
ConfusionMatrixDisplay(conf_mat,display_labels=['husky','wolf']).plot(cmap='Blues')
```
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