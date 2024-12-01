::: {.cell .markdown}

# Reproducing "Identification of COVID-19 samples from chest X-Ray images using deep learning: A comparison of transfer learning approaches" without Data Leakage

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaivimalik/covid_illegitimate_features/blob/main/notebooks/Correcting_Original_Result.ipynb)

:::

::: {.cell .markdown}
## Introduction

In this notebook, we will reproduce the results published in **Identification of COVID-19 samples from chest X-Ray images using deep learning: A comparison of transfer learning approaches** [1] without data leakage. This study aims to recognize the chest X-ray images of COVID-19 cases from normal and pneumonia cases. 

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
## Retrieve the datasets

We will use two datasets:

- **COVID-19 Image Data Collection** [2] is a public open dataset of chest X-ray and CT images of patients which are positive or suspected of COVID-19 or other viral and bacterial pneumonias (MERS, SARS, and ARDS.). The images in this dataset were extracted from public databases, such as Radiopaedia.org, the Italian Society of Medical and Interventional Radiology, and Figure1.com, through manual collection and web scraping. The database is regularly updating with new cases.

- **ChestX-ray8** [3] dataset comprises of 108,948 frontal-view X-ray images (collected from the year of 1992 to 2015) of 32,717 unique patients with the text-mined eight common disease labels.

The code cell below will download the datasets. Then, we will create TensorFlow Dataset objects and visualize chest X-ray images.

:::

::: {.cell .code}
```python
!wget -O images_01.tar.gz https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz
!wget -O images_02.tar.gz https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz
!wget -O images_03.tar.gz https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz
!wget -O images_04.tar.gz https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz
!wget -O images_05.tar.gz https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz
!wget -O images_06.tar.gz https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz
!wget -O images_07.tar.gz https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz
!wget -O images_08.tar.gz https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz
!wget -O images_09.tar.gz https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz
!wget -O images_10.tar.gz https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz
!wget -O images_11.tar.gz https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz
!wget -O images_12.tar.gz https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz
```
:::

::: {.cell .code}
```python
!mkdir chest_xray

!gunzip images_01.tar.gz
!gunzip images_02.tar.gz
!gunzip images_03.tar.gz
!gunzip images_04.tar.gz
!gunzip images_05.tar.gz
!gunzip images_06.tar.gz
!gunzip images_07.tar.gz
!gunzip images_08.tar.gz
!gunzip images_09.tar.gz
!gunzip images_10.tar.gz
!gunzip images_11.tar.gz
!gunzip images_12.tar.gz

!tar -xvf images_01.tar -C chest_xray
!tar -xvf images_02.tar -C chest_xray
!tar -xvf images_03.tar -C chest_xray
!tar -xvf images_04.tar -C chest_xray
!tar -xvf images_05.tar -C chest_xray
!tar -xvf images_06.tar -C chest_xray
!tar -xvf images_07.tar -C chest_xray
!tar -xvf images_08.tar -C chest_xray
!tar -xvf images_09.tar -C chest_xray
!tar -xvf images_10.tar -C chest_xray
!tar -xvf images_11.tar -C chest_xray
!tar -xvf images_12.tar -C chest_xray
```
:::

::: {.cell .code}
```python
!git clone https://github.com/ieee8023/covid-chestxray-dataset.git
```
:::

::: {.cell .markdown}

We start by importing the required libraries.

:::

::: {.cell .code}
```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from keras_cv import layers as layers_cv
# Set random seeds for reproducibility
np.random.seed(20)
tf.random.set_seed(20)
```
:::

::: {.cell .markdown}

The **COVID-19 Image Data Collection** and **ChestX-ray8** contain chest X-ray images of various lung diseases, so we need to filter and identify the COVID-19 images. 

Here, we extract the file paths of X-ray images from COVID-19 cases and remove duplicates.

:::

::: {.cell .code}
```python
covid_files = os.listdir('covid-chestxray-dataset/images')
df = pd.read_csv('covid-chestxray-dataset/metadata.csv')
df = df[df['finding']=='Pneumonia/Viral/COVID-19']
df = df[df['modality']=='X-ray']
df = df[df['view']!='L']
df.drop_duplicates(subset='patientid', keep='first', inplace=True)
covid_files = df['filename'].to_list()
covid_paths = np.random.choice(covid_files, size=260)
covid_paths = ['covid-chestxray-dataset/images/' + i for i in covid_paths]
```
:::

::: {.cell .markdown} 

We extract the file paths of X-ray images for both Pneumonia and normal cases and remove any duplicates.

:::

::: {.cell .code}
```python
df = pd.read_csv('../Data_Entry_2017_v2020.csv')
normal_paths = df[df['Finding Labels']=='No Finding'].sample(300)
normal_paths = normal_paths['Image Index'].to_list()
normal_paths = ['chest_xray/images/' + i for i in normal_paths]
pneumonia_paths = df[df['Finding Labels']=='Pneumonia'].sample(300)
pneumonia_paths = pneumonia_paths['Image Index'].to_list()
pneumonia_paths = ['chest_xray/images/' + i for i in pneumonia_paths]
```
:::

::: {.cell .code}
```python
print("Number of COVID-19 samples:", len(covid_paths))
print("Number of Normal samples:", len(normal_paths))
print("Number of Pneumonia samples:", len(pneumonia_paths))
```
:::

::: {.cell .markdown}

Now, we use the `from_tensor_slices` method to create `tf.data.Dataset` objects from the lists of paths.

:::

::: {.cell .code}
```python
covid_ds = tf.data.Dataset.from_tensor_slices(covid_paths)
normal_ds = tf.data.Dataset.from_tensor_slices(normal_paths)
pneumonia_ds = tf.data.Dataset.from_tensor_slices(pneumonia_paths)
```
:::

::: {.cell .markdown}

Next, we assign labels to each image and use the `process_path` function to load and resize them to 224x224 pixels.

:::

::: {.cell .code}
```python
def process_path(file_path, label):
  # Load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  # Convert the compressed string to a 3D uint8 tensor
  img = tf.io.decode_jpeg(img, channels=3)
  # Resize the image to the desired size
  img = tf.image.resize(img, [224, 224])
  return img, label

labels = {"covid-19":0, "normal":1, "pneumonia":2}
covid_ds = covid_ds.map(lambda x: process_path(x,labels['covid-19']))
normal_ds = normal_ds.map(lambda x: process_path(x,labels['normal']))
pneumonia_ds = pneumonia_ds.map(lambda x: process_path(x,labels['pneumonia']))
```
:::

::: {.cell .markdown}

Finally, we visualize the chest X-ray images of each class.

:::

::: {.cell .code}
```python
keras.utils.load_img(covid_paths[5], color_mode='grayscale', target_size=(224,224))
```
:::

::: {.cell .code}
```python
keras.utils.load_img(normal_paths[5], color_mode='grayscale', target_size=(224,224))
```
:::

::: {.cell .code}
```python
keras.utils.load_img(pneumonia_paths[5], color_mode='grayscale', target_size=(224,224))
```
:::

::: {.cell .markdown}
## Train and evaluate the convolutional neural network(VGG19) via Transfer Learning

In this section, we will train and evaluate our Convolutional Neural Network following the methodology outlined in the paper:

- Split the datasets into training, test, and validation sets.
- Apply the data augmentation strategy.
- Load the VGG19 model to extract features from the images.
- Add two dense layers with ReLU activation.
- Add a final dense layer with softmax activation.
- Train the model for 50 epochs with a learning rate of 0.001 using the RMSprop optimizer.
- Use ReduceLROnPlateau to adjust the learning rate when validation loss has stopped improving.
- Report the model's accuracy, confusion matrix, and class-wise precision, recall, and F1-score.

:::

::: {.cell .markdown}

We start by splitting `covid_ds`, `normal_ds` and `pneumonia_ds` according to the statistics given in the paper. We then concatenate these splits to form training, test, and validation sets for model training and evaluation. `keras.applications.vgg19.preprocess_input` method is applied to preprocess the images, ensuring they are in the correct format required by the VGG19 model.

:::

::: {.cell .code}
```python
# Splitting normal patients data acc. to stats given in paper
normal_ds_train = normal_ds.take(200)
normal_remaining = normal_ds.skip(200)
normal_ds_val = normal_remaining.take(50)
normal_ds_test = normal_remaining.skip(50)
# Splitting pneumonia patients data acc. to stats given in paper
pneumonia_ds_train = pneumonia_ds.take(200)
pneumonia_remaining = pneumonia_ds.skip(200)
pneumonia_ds_val = pneumonia_remaining.take(50)
pneumonia_ds_test = pneumonia_remaining.skip(50)
# Splitting covid patients data acc. to stats given in paper
covid_ds_train = covid_ds.take(180)
covid_remaining = covid_ds.skip(180)
covid_ds_val = covid_remaining.take(40)
covid_ds_test = covid_remaining.skip(40)
```
:::

::: {.cell .code}
```python
# Combine datasets
train_ds = (covid_ds_train.concatenate(normal_ds_train).concatenate(pneumonia_ds_train))
validation_ds = (covid_ds_val.concatenate(normal_ds_val).concatenate(pneumonia_ds_val))
test_ds = (covid_ds_test.concatenate(normal_ds_test).concatenate(pneumonia_ds_test))
# Preprocess the images in each dataset using VGG19 preprocess_input
train_ds = train_ds.map(lambda x, y: (keras.applications.vgg19.preprocess_input(x), y))
validation_ds = validation_ds.map(lambda x, y: (keras.applications.vgg19.preprocess_input(x), y))
test_ds = test_ds.map(lambda x, y: (keras.applications.vgg19.preprocess_input(x), y))
```
:::

::: {.cell .markdown}

In this cell, we create `RandomRotation` , `RandomTranslation`, `RandomShear`, `RandomZoom` and `RandomFlip` data augmentation layers and apply them on the training set.

:::

::: {.cell .code}
```python
# Define data augmentation layers
augmentation_layers = [
    layers.RandomRotation(0.2), # Randomly rotate images by up to 20 degrees
    layers.RandomTranslation(0.1, 0.1), # Randomly translate images by up to 10% in x and y directions
    layers_cv.RandomShear(x_factor=0.1, y_factor=0.1), # Randomly shear images by up to 10% in x and y directions
    layers.RandomZoom(0.2), # Randomly zoom images by up to 20%
    layers.RandomFlip("horizontal_and_vertical") # Randomly flip images horizontally and vertically
]

def data_augmentation(x):
    for layer in augmentation_layers:
        x = layer(x)
    return x

# Apply data augmentation to training set
train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))
```
:::

::: {.cell .markdown}

This code cell batches the datasets (`train_ds`, `validation_ds`, and `test_ds`) into batches of 32 samples, uses prefetching to improve performance, and caches the datasets in memory for faster subsequent access.

:::

::: {.cell .code}
```python
batch_size = 32
# Configure datasets for performance
train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()
validation_ds = validation_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()
test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()
```
:::

::: {.cell .markdown}

Next, we initialize the VGG19 model with weights pretrained on the ImageNet dataset. By setting `include_top=False`, we exclude the final classification layer of the VGG19 model. We set the `trainable` attribute of the VGG19 layers to `False`. Then, we add a `Flatten` layer to convert the VGG19 output into a one-dimensional vector. We follow this with two `Dense` layers with ReLU activation, having 1024 and 512 neurons respectively. Finally, the output is fed into a `Dense` layer with a softmax activation function.

:::

::: {.cell .code}
```python
# Create a base model using VGG19 pre-trained on ImageNet
base_model = keras.applications.VGG19(
    weights="imagenet",  # Load weights pre-trained on ImageNet
    input_shape=[224, 224, 3],  # Specify input shape
    include_top=False,  # Do not include the ImageNet classifier at the top
)

# Freeze the base model to prevent its weights from being updated during training
base_model.trainable = False

# Flatten the output of VGG19
x = layers.Flatten()(base_model.output)

# Add Dense layer with 1024 neurons and ReLU activation
x = layers.Dense(1024, activation='relu')(x)

# Add Dense layer with 512 neurons and ReLU activation
x = layers.Dense(512, activation='relu')(x)

# Add final Dense layer with softmax activation for 3-class prediction
predictions = layers.Dense(3, activation='softmax')(x)

# Create the full model
model = keras.Model(inputs=base_model.input, outputs=predictions)

# Display model summary, showing which layers are trainable
model.summary(show_trainable=True)
```
:::

::: {.cell .markdown}

We train the model on the training set using the RMSprop optimizer for 50 epochs. We use `sparse_categorical_crossentropy` as our loss function since the labels are encoded as integers. We apply `ReduceLROnPlateau` to reduce the learning rate by a factor of 0.3 when the validation loss plateaus.

:::

::: {.cell .code}
```python
# Compile the model
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=[keras.metrics.SparseCategoricalAccuracy()])

# Define learning rate reduction callback
reduce_lr = keras.callbacks.ReduceLROnPlateau(factor=0.3)

# Set number of training epochs
epochs = 50

# Train the model
history = model.fit(train_ds, epochs=epochs, validation_data=validation_ds, callbacks=[reduce_lr])
```
:::

::: {.cell .markdown}

We plot the training accuracy (`sparse_categorical_accuracy`) and validation accuracy (`val_sparse_categorical_accuracy`) against the number of epochs.

:::

::: {.cell .code}
```python
# summarize history for accuracy
plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```
:::

::: {.cell .markdown}

Here, we evaluate the model on the test set and report the accuracy.

:::

::: {.cell .code}
```python
# Evaluate model on test set
loss, accuracy = model.evaluate(test_ds)
print('Test loss :', loss)
print('Test accuracy :', accuracy)
```
:::

::: {.cell .markdown}

The following code cells display the true and predicted labels on the test set and generate a confusion matrix.

:::

::: {.cell .code}
```python
# Make predictions on test set
y_pred = model.predict(test_ds)
# Convert predicted probabilities to class labels by taking the index of the highest probability
y_pred = np.argmax(y_pred, axis=1)
print("Predictions:",y_pred)
# Extract true labels from the test dataset
y_true = tf.concat([label for _, label in test_ds], axis=0).numpy()
print("True labels:", y_true)
```
:::

::: {.cell .code}
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Create confusion matrix
conf_mat = confusion_matrix(y_true,y_pred)
# Display confusion matrix with labels
ConfusionMatrixDisplay(conf_mat,display_labels=labels.keys()).plot(cmap='Blues')
```
:::

::: {.cell .markdown}

Finally, we report the class-wise precision, recall and f1-score of the model's performance on the test set.

:::

::: {.cell .code}
```python
# Generate classification report
report = classification_report(y_true, y_pred, output_dict=True)

# Print metrics for each class
for key in labels.keys():
  print("class:", key)
  print("Precision:",report[str(labels[key])]['precision'])
  print("Recall:",report[str(labels[key])]['recall'])
  print("F1-score:",report[str(labels[key])]['f1-score'])
  print()
```
:::

::: {.cell .markdown}

Let's save the model for future inference tasks.

:::

::: {.cell .code}
```python
model.save('correct_covid.keras')
```
:::

::: {.cell .markdown}

Next, we use GradCAM [4] to identify the pixels responsible for an image being classified as normal, pneumonia, or COVID-19.

:::

::: {.cell .code}
```python
from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam

# Image titles for each class
image_titles = ['Covid', 'Normal', 'Pneumonia']

index = np.random.randint(-40, 0)

# Load images and Convert them to a Numpy array
covid = keras.utils.load_img(covid_paths[index], target_size=(224, 224))
normal = keras.utils.load_img(normal_paths[index], target_size=(224, 224))
pneumonia = keras.utils.load_img(pneumonia_paths[index], target_size=(224, 224))
images = np.asarray([np.array(covid), np.array(normal), np.array(pneumonia)])

X = np.array([tf.keras.utils.img_to_array(img) for img in images])

# Rendering
f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i, title in enumerate(image_titles):
    ax[i].set_title(title, fontsize=16)
    ax[i].imshow(images[i])
    ax[i].axis('off')
plt.tight_layout()
plt.show()

# Function to modify the model for GradCAM
def model_modifier_function(cloned_model): 
  cloned_model.layers[-1].activation = tf.keras.activations.linear

# Score function for GradCAM
def score_function(output): return (output[0][0], output[1][1], output[2][2])

# Create Gradcam object
gradcam = Gradcam(model, model_modifier=model_modifier_function, clone=True)

# Generate heatmap with GradCAM
cam = gradcam(score_function, X)

# Rendering images with GradCAM heatmaps
f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
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

| Metric        | Original | Reproduced | Reproduced without Data Leakage |
|:-------------:|:--------:|:----------:|:-------------------------------:|
| Accuracy      | 89.3     | 92.14      | 51.43                           |

In this notebook, we successfully reproduced the findings of **"Identification of COVID-19 Samples from Chest X-Ray Images Using Deep Learning: A Comparison of Transfer Learning Approaches"** without data leakage. The model's accuracy dropped by 40%, confirming that it had previously relied on illegitimate features to distinguish between COVID-19, pneumonia, and normal patients when using the Chest X-Ray Images (Pneumonia) dataset from Kaggle. When working with medical data, it is crucial to consult a professional and ensure that no information is inadvertently leaked to the model—information that would not be available during real-world deployment.

:::

::: {.cell .markdown}
## References

[1]: Rahaman, Md Mamunur et al. “Identification of COVID-19 samples from chest X-Ray images using deep learning: A comparison of transfer learning approaches.” Journal of X-ray science and technology vol. 28,5 (2020): 821-839. doi:10.3233/XST-200715

[2]: COVID-19 Image Data Collection: Prospective Predictions Are the Future Joseph Paul Cohen and Paul Morrison and Lan Dao and Karsten Roth and Tim Q Duong and Marzyeh Ghassemi arXiv:2006.11988, 2020

[3]: X. Wang, et al., "ChestX-Ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases," in 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, USA, 2017 pp. 3462-3471. doi: 10.1109/CVPR.2017.369

[4]: R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh and D. Batra, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," 2017 IEEE International Conference on Computer Vision (ICCV), Venice, Italy, 2017, pp. 618-626, doi: 10.1109/ICCV.2017.74. 

:::