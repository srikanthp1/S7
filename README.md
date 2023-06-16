# Image classification

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/), 
[PyTorch](https://pytorch.org/), 
[torchvision](https://github.com/pytorch/vision) 0.8, 
Uses [matplotlib](https://matplotlib.org/)  for ploting accuracy and losses.

## Info

 * we are training our model on MNIST dataset. 
 * we are using a custom convolutional neural network (CNN) architectures. 
 * Implemented in pytorch 

## Getting Started

To install **PyTorch**, see installation instructions on the [PyTorch website](https://pytorch.org/).

The instructions to install PyTorch should also detail how to install **torchvision** but can also be installed via:

``` bash
pip install torchvision
```


## Usage

```bash
git clone https://github.com/srikanthp1/era5.git
```
* utils.py has util functions
* model.py has models 
* run cell by cell to download, visualize data and train model

## Demo 

[**CustomNetwork**](https://github.com/srikanthp1/era5/blob/main/S5.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/srikanthp1/era5/blob/main/S5.ipynb)

* we have the following:
    * load datasets 
    * augment data
    * define a custom CNN
    * train a model
    * view the outputs of our model
    * visualize the model's representations
    * view the loss and accuracy of the model. 

* **transforms** for trainset and testset are in .ipynb. 
* you will also find **dataloaders** in __.ipynb__**. 
* **train** and **test** functions are written in __model.py__**.
* model is written in __model.py__**.
* dataset is downloaded in __S5.ipynb__ file as we may want to try new datasets. 
* transforms if needed to be added or modified refer __utils.py__**.
* visualization of dataset is in __S5.ipynb__**  
* __graphs__** for loss and accuracy is added after training and testing is done


## Model details

```python
model = Net().to(device)
summary(model, input_size=(1, 28, 28))
```

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             288
              ReLU-2           [-1, 32, 26, 26]               0
            Conv2d-3           [-1, 64, 24, 24]          18,432
              ReLU-4           [-1, 64, 24, 24]               0
            Conv2d-5          [-1, 128, 22, 22]          73,728
              ReLU-6          [-1, 128, 22, 22]               0
            Conv2d-7           [-1, 32, 22, 22]           4,096
         MaxPool2d-8           [-1, 32, 11, 11]               0
            Conv2d-9             [-1, 64, 9, 9]          18,432
             ReLU-10             [-1, 64, 9, 9]               0
           Conv2d-11            [-1, 128, 7, 7]          73,728
             ReLU-12            [-1, 128, 7, 7]               0
           Conv2d-13             [-1, 10, 7, 7]           1,280
           Conv2d-14             [-1, 10, 1, 1]           4,900
================================================================
Total params: 194,884
Trainable params: 194,884
Non-trainable params: 0
----------------------------------------------------------------
```

* above model is for testing out best accuracy architecture and ended up at this. 
* still needs to be regularized and BN can benefit it and also transforms, lr schedulers


```python
model = Net().to(device)
summary(model, input_size=(1, 28, 28))
```

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              80
              ReLU-2            [-1, 8, 28, 28]               0
       BatchNorm2d-3            [-1, 8, 28, 28]              16
           Dropout-4            [-1, 8, 28, 28]               0
            Conv2d-5            [-1, 8, 28, 28]             584
              ReLU-6            [-1, 8, 28, 28]               0
       BatchNorm2d-7            [-1, 8, 28, 28]              16
           Dropout-8            [-1, 8, 28, 28]               0
            Conv2d-9           [-1, 16, 28, 28]           1,168
             ReLU-10           [-1, 16, 28, 28]               0
      BatchNorm2d-11           [-1, 16, 28, 28]              32
          Dropout-12           [-1, 16, 28, 28]               0
           Conv2d-13           [-1, 10, 28, 28]             170
        MaxPool2d-14           [-1, 10, 14, 14]               0
           Conv2d-15           [-1, 12, 14, 14]           1,092
             ReLU-16           [-1, 12, 14, 14]               0
      BatchNorm2d-17           [-1, 12, 14, 14]              24
          Dropout-18           [-1, 12, 14, 14]               0
           Conv2d-19           [-1, 12, 14, 14]           1,308
             ReLU-20           [-1, 12, 14, 14]               0
      BatchNorm2d-21           [-1, 12, 14, 14]              24
          Dropout-22           [-1, 12, 14, 14]               0
           Conv2d-23           [-1, 10, 14, 14]             130
        MaxPool2d-24             [-1, 10, 7, 7]               0
           Conv2d-25             [-1, 14, 7, 7]           1,274
             ReLU-26             [-1, 14, 7, 7]               0
      BatchNorm2d-27             [-1, 14, 7, 7]              28
           Conv2d-28             [-1, 14, 7, 7]           1,778
             ReLU-29             [-1, 14, 7, 7]               0
      BatchNorm2d-30             [-1, 14, 7, 7]              28
           Conv2d-31             [-1, 10, 7, 7]             150
        AvgPool2d-32             [-1, 10, 1, 1]               0
================================================================
Total params: 7,902
Trainable params: 7,902
Non-trainable params: 0
----------------------------------------------------------------
```
* above model has changed as less parameters will likely have less layers and should be used efficiently
* 2-M-A-2-M-A-2-A-gap(idea) -> 2 convs, maxpooling, 1*1 agregator etc til global 2d pool
* thinking aggregator should come after maxpool as maxpool on sparse info channels is better than dense
* adding transforms and lr will make it better because after multiple trails this has the most stable expected result. 



```python
model = Net().to(device)
summary(model, input_size=(1, 28, 28))
```

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              72
              ReLU-2            [-1, 8, 26, 26]               0
       BatchNorm2d-3            [-1, 8, 26, 26]              16
           Dropout-4            [-1, 8, 26, 26]               0
            Conv2d-5           [-1, 12, 24, 24]             864
              ReLU-6           [-1, 12, 24, 24]               0
       BatchNorm2d-7           [-1, 12, 24, 24]              24
           Dropout-8           [-1, 12, 24, 24]               0
         MaxPool2d-9           [-1, 12, 12, 12]               0
           Conv2d-10            [-1, 8, 12, 12]              96
           Conv2d-11           [-1, 12, 10, 10]             864
             ReLU-12           [-1, 12, 10, 10]               0
      BatchNorm2d-13           [-1, 12, 10, 10]              24
          Dropout-14           [-1, 12, 10, 10]               0
           Conv2d-15             [-1, 16, 8, 8]           1,728
             ReLU-16             [-1, 16, 8, 8]               0
      BatchNorm2d-17             [-1, 16, 8, 8]              32
          Dropout-18             [-1, 16, 8, 8]               0
        MaxPool2d-19             [-1, 16, 4, 4]               0
           Conv2d-20             [-1, 10, 4, 4]             160
           Conv2d-21             [-1, 14, 4, 4]           1,260
             ReLU-22             [-1, 14, 4, 4]               0
      BatchNorm2d-23             [-1, 14, 4, 4]              28
          Dropout-24             [-1, 14, 4, 4]               0
           Conv2d-25             [-1, 18, 4, 4]           2,268
             ReLU-26             [-1, 18, 4, 4]               0
      BatchNorm2d-27             [-1, 18, 4, 4]              36
          Dropout-28             [-1, 18, 4, 4]               0
        AvgPool2d-29             [-1, 18, 1, 1]               0
           Conv2d-30             [-1, 10, 1, 1]             180
================================================================
Total params: 7,652
Trainable params: 7,652
Non-trainable params: 0
----------------------------------------------------------------
```

* added transforms to our model along with onecyclelr which boosted the accuracy highly 
* though we see some underfitting, mostly it is due to dropout used for training. 