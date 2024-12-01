## Reproducing "Identification of COVID-19 samples from chest X-Ray images using deep learning: A comparison of transfer learning approaches"

In this series of notebooks we reproduce a result published in 
> Identification of COVID-19 samples from chest X-Ray images using deep learning: A comparison of transfer learning approaches [1]

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaivimalik/covid_illegitimate_features/blob/main/notebooks/Reproducing_Original_Result.ipynb) Reproducing “Identification of COVID-19 samples from chest X-Ray images using deep learning: A comparison of transfer learning approaches”

In Reproducing_Original_Result, we reproduce the results obtained using the VGG19 model and achieve an accuracy of 92% on the test set. However, as noted in [2], a significant demographic inconsistency exists: normal and pneumonia chest X-ray images are from pediatric patients, while COVID-19 chest X-ray images are from adults. This allows the model to achieve high accuracy by learning features that are not clinically relevant.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaivimalik/covid_illegitimate_features/blob/main/notebooks/Exploring_ConvNet_Activations.ipynb) Exploring ConvNet Activations

In Exploring_ConvNet_Activations, we explore how a model can learn illegitimate features using a small dataset of wolf and husky images. The model achieves 90% accuracy, but we reveal that this performance is due to a data leakage issue: all wolf images have snow backgrounds, while husky images have grass backgrounds. This enables the model to simply distinguish between white (snow) and green (grass) backgrounds to make predictions. To prove this, we test the model on a new dataset where the backgrounds are swapped (dogs with snow, wolves with grass). The model's accuracy drops to 0%, confirming it was indeed using background cues for classification. We provide GradCAM heatmaps to visualize pixel attributions, further illustrating the model's focus on background rather than animal features. Then, we train a new model on a dataset where both wolf and husky images have white backgrounds and achieve an accuracy of 70%. This shows that the accuracy obtained earlier was an overly optimistic measure due to data leakage.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaivimalik/covid_illegitimate_features/blob/main/notebooks/Correcting_Original_Result.ipynb) Reproducing “Identification of COVID-19 samples from chest X-Ray images using deep learning: A comparison of transfer learning approaches”

In Correcting_Original_Result, we reproduce the results obtained using the VGG19 model, but with a key change: we use datasets containing adult chest X-ray images. This time, the model achieves an accuracy of 51%, a 40% drop from the earlier results, confirming that the metrics reported in the paper were overly optimistic due to data leakage, where the model learned illegitimate features.

---

### Running the project

#### Google Colab

Click on the "Open in Colab" buttons above to run the notebooks in Google Colab.

#### Local Machine

1. Clone the repository:
   ```
   $ git clone https://github.com/shaivimalik/medicine_preprocessing-on-entire-dataset.git
   $ cd medicine_preprocessing-on-entire-dataset
   ```

2. Install the required dependencies:
   ```
   $ pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook:
   ```
   $ jupyter notebook
   ```

#### Chameleon

You can run these notebooks on [Chameleon](https://chameleoncloud.org/) using the Chameleon Jupyter environment.

---

### Acknowledgements

This project was part of the 2024 Summer of Reproducibility organized by the [UC Santa Cruz Open Source Program Office](https://ucsc-ospo.github.io/).

* Contributor: [Shaivi Malik](https://github.com/shaivimalik)
* Mentors: [Fraida Fund](https://github.com/ffund), [Mohamed Saeed](https://github.com/mohammed183)

---

### References

[1]: Rahaman, Md Mamunur et al. “Identification of COVID-19 samples from chest X-Ray images using deep learning: A comparison of transfer learning approaches.” Journal of X-ray science and technology vol. 28,5 (2020): 821-839. doi:10.3233/XST-200715

[2]: Roberts, M., Driggs, D., Thorpe, M. et al. Common pitfalls and recommendations for using machine learning to detect and prognosticate for COVID-19 using chest radiographs and CT scans. Nat Mach Intell 3, 199–217 (2021). https://doi.org/10.1038/s42256-021-00307-0