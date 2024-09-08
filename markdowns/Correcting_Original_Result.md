::: {.cell .markdown}

# Reproducing "Identification of COVID-19 samples from chest X-Ray images using deep learning: A comparison of transfer learning approaches" without Data Leakage

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaivimalik/covid_illegitimate_features/blob/main/notebooks/Correcting_Original_Result.ipynb)

:::

::: {.cell .markdown}
## Introduction

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

::: {.cell .code}
```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import data as tf_data
import keras
from keras import layers
from keras_cv import layers as layers_cv
from keras.models import Model
# Set random seeds for reproducibility
np.random.seed(20)
tf.random.set_seed(20)
```
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

::: {.cell .code}
```python
covid_ds = tf.data.Dataset.from_tensor_slices(covid_paths)
normal_ds = tf.data.Dataset.from_tensor_slices(normal_paths)
pneumonia_ds = tf.data.Dataset.from_tensor_slices(pneumonia_paths)
```
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
train_ds = (covid_ds_train.concatenate(normal_ds_train).concatenate(pneumonia_ds_train))
validation_ds = (covid_ds_val.concatenate(normal_ds_val).concatenate(pneumonia_ds_val))
test_ds = (covid_ds_test.concatenate(normal_ds_test).concatenate(pneumonia_ds_test))

train_ds = train_ds.map(lambda x, y: (keras.applications.vgg19.preprocess_input(x), y))
validation_ds = validation_ds.map(lambda x, y: (keras.applications.vgg19.preprocess_input(x), y))
test_ds = test_ds.map(lambda x, y: (keras.applications.vgg19.preprocess_input(x), y))
```
:::

::: {.cell .code}
```python
augmentation_layers = [
    layers.RandomRotation(0.2),
    layers.RandomTranslation(0.1, 0.1),
    layers_cv.RandomShear(x_factor=0.1, y_factor=0.1),
    layers.RandomZoom(0.2),
    layers.RandomFlip("horizontal_and_vertical")
]

def data_augmentation(x):
    for layer in augmentation_layers:
        x = layer(x)
    return x

train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))
```
:::

::: {.cell .code}
```python
batch_size = 32

train_ds = train_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()
validation_ds = validation_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()
test_ds = test_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()
```
:::

::: {.cell .code}
```python
base_model = keras.applications.VGG19(include_top=False, input_shape =[224, 224, 3], weights="imagenet")

base_model.trainable = False

x = layers.Flatten()(base_model.output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dense(512, activation='relu')(x)

predictions = layers.Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.summary(show_trainable=True)
```
:::

::: {.cell .code}
```python
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=[keras.metrics.SparseCategoricalAccuracy()])

reduce_lr = keras.callbacks.ReduceLROnPlateau(factor=0.3)

epochs = 50

history = model.fit(train_ds, epochs=epochs, validation_data=validation_ds, callbacks=[reduce_lr])
```
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

::: {.cell .code}
```python
# Evaluate model on test set
loss, accuracy = model.evaluate(test_ds)
print('Test loss :', loss)
print('Test accuracy :', accuracy)
```
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

::: {.cell .code}
```python
model.save('correct_covid.keras')
```
:::

::: {.cell .markdown}
## Discussion 

| Metric        | Original | Reproduced | Reproduced without Data Leakage |
|:-------------:|:--------:|:----------:|:-------------------------------:|
| Accuracy      | 89.3     | 92.14      | 51.43                           |

:::