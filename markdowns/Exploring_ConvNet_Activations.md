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
```
:::

::: {.cell .code}
```python
# Define image size and batch size
image_size = (256,256)
batch_size = 4

# Load the dataset from directory
dataset_leak = keras.utils.image_dataset_from_directory(
    '../different_backgrounds', 
    label_mode="categorical", 
    image_size=image_size, 
    batch_size=batch_size
)

# Split the dataset into train, validation, and test sets (70-10-20)
train_ds_leak = dataset_leak.take(tf_data.experimental.cardinality(dataset_leak).numpy()*0.7)
remaining_ds_leak = dataset_leak.skip(tf_data.experimental.cardinality(dataset_leak).numpy()*0.7)
val_ds_leak = remaining_ds_leak.take(tf_data.experimental.cardinality(dataset_leak).numpy()*0.1)
test_ds_leak = remaining_ds_leak.skip(tf_data.experimental.cardinality(dataset_leak).numpy()*0.1)
```
:::

::: {.cell .code}
```python
# Define data augmentation layers
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]
# Function to apply data augmentation
def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

# Apply data augmentation to training dataset
train_ds_leak = train_ds_leak.map(lambda img, label: (data_augmentation(img), label),num_parallel_calls=tf_data.AUTOTUNE)
# Prefetch test and validation datasets for performance
test_ds_leak = test_ds_leak.prefetch(tf_data.AUTOTUNE)
val_ds_leak = val_ds_leak.prefetch(tf_data.AUTOTUNE)
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
score = model_leak.evaluate(test_ds_leak)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
```
:::

::: {.cell .code}
```python
from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam

# Define image titles for visualization
image_titles = ['husky', 'wolf']

# Create lists of file paths for husky and wolf images
husky_files = np.array(['../different_backgrounds/husky/'+x for x in os.listdir('../different_backgrounds/husky')])
wolf_files = np.array(['../different_backgrounds/wolf/'+x for x in os.listdir('../different_backgrounds/wolf')])

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
    batch_size=batch_size
)
score = model_leak.evaluate(background_swap)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
```
:::

::: {.cell .code}
```python
# Define image size and batch size
image_size = (256,256)
batch_size = 4

# Load the dataset from directory
dataset = keras.utils.image_dataset_from_directory(
    '../same_backgrounds', 
    label_mode="categorical", 
    image_size=image_size, 
    batch_size=batch_size
)

# Split the dataset into train, validation, and test sets (70-10-20)
train_ds = dataset.take(tf_data.experimental.cardinality(dataset).numpy()*0.7)
remaining_ds = dataset.skip(tf_data.experimental.cardinality(dataset).numpy()*0.7)
val_ds = remaining_ds.take(tf_data.experimental.cardinality(dataset).numpy()*0.1)
test_ds = remaining_ds.skip(tf_data.experimental.cardinality(dataset).numpy()*0.1)
```
:::

::: {.cell .code}
```python
# Define data augmentation layers
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

# Apply data augmentation to training dataset
train_ds = train_ds.map(lambda img, label: (data_augmentation(img), label),num_parallel_calls=tf_data.AUTOTUNE)
# Prefetch test and validation datasets for performance
test_ds = test_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)
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
from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam

# Define image titles for visualization
image_titles = ['husky', 'wolf']

# Create lists of file paths for husky and wolf images
husky_files = np.array(['../same_backgrounds/husky/'+x for x in os.listdir('../same_backgrounds/husky')])
wolf_files = np.array(['../same_backgrounds/wolf/'+x for x in os.listdir('../same_backgrounds/wolf')])

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