# NNProjectCurtiPescetelli
The code provides a Colab reimplementation of the MaskTune method presented in the paper ["MaskTune: Mitigating Spurious Correlations by Forcing to Explore"](https://arxiv.org/abs/2210.00055). 

# Installations and Requirements
The following installations are required for running the code:
 
```bash
!pip install torchvision
 ```
This is a package that provides access to popular datasets, model architectures, and image transformation utilities for computer vision tasks in PyTorch. It simplifies the process of working with image-based neural networks, making it easier to handle datasets and preprocessing.

```bash
!pip install timm
 ```
This is a deep learning library that provides a large collection of pre-trained models for image classification, segmentation, and other computer vision tasks. It includes state-of-the-art models, such as ResNet variants, which can be easily integrated and fine-tuned for custom tasks. In our case, we use this library in order to create a pre-trained ResNet50 model.

```bash
!pip install git+https://github.com/jacobgil/pytorch-grad-cam.git
 ```
This package enables Gradient-weighted Class Activation Mapping (Grad-CAM) in PyTorch models. Grad-CAM is a technique used to visualize which parts of an input image contribute the most to a model's decision, making it useful for understanding and interpreting the behavior of neural networks. In particular, as the authors of the MaskTune paper did, we use it to implement the XGradCAM technique. 

```bash
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```
The mounting of Google Drive is necessary to run the experiments directly from the code. Indeed, some datasets, files, intermediate results and checkpoints are saved here by the code.

# Datasets
1) For the `MNIST` and `CIFAR10` experiments, the datasets are downloaded directly from the code, being them provided by the `torchvision` package.
2) For the `CELEBA` experiments, the dataset can be downloaded from [here](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset?resource=download-directory). In order to run the code directly from the notebook, it is necessary to upload the downloaded dataset on Google Drive.
