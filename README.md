# Traffic Signs Classifier Project

The second project of Udacity's Self Driving Car Nanodegree program requires developing a convolutional neural network to classify traffic signs from the [German Traffic Sign Database](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).  Submission performance acceptability criteria is a minimum accuracy of 0.93 on the validation dataset.  The data is provided by Udacity in training, validation, and testing splits of 34799, 4410, and 12630 32x32 RGB examples.

## Approach Overview

My approach is based on [Simonyan and Zisserman's VGGNet](https://arxiv.org/pdf/1409.1556.pdf).  Because VGGNet was designed for 227x227 input images and training on 1000 classes, I scale down the model by removing the final 3 convolutional layers and final pooling layer.  I make three additional changes:
 1. To the end of the second block of convolutional layers, I add a 1x1 convolution.  This adds additional representative power to compensate for the removal of the last block of convolutional layers.
 2. After the final convolutional layer, I pool through the spatial (height and width) dimensions.  This is based on the recommendation of Andrej Karpathy in Stanford [CS231 Lecture 7](https://www.youtube.com/watch?v=LxfUGhug-iQ&feature=youtu.be&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC&t=3665) to reduce the number of parameters of the network.
 3. I reduce the fully connected layer width from VGGNet's 4096 neurons to 256 to scale down with the reduced size of activation volume entering the first FC layer.  Like VGGNet, I use dropout regularization with a ration of 0.5 after each FC layer.

The final architecture is:

| Layer Type        | Activation Volume |
| ----------------- | ----------------- |
| conv3-64          | 32x32x64          |
| conv3-64          | 32x32x64          |
| max pool          | 16x16x64          |
| conv3-128         | 16x16x128         |
| conv3-128         | 16x16x128         |
| conv1-128         | 16x16x128         |
| max pool          | 8x8x128           |
| conv3-256         | 8x8x256           |
| conv3-256         | 8x8x256           |
| conv1-256         | 8x8x256           |
| max pool          | 4x4x256           |
| conv3-512         | 4x4x512           |
| conv3-512         | 4x4x512           |
| conv1-512         | 4x4x512           |
| avg pool          | 512               |
| FC-256            | 256               |
| FC-256            | 256               |
| FC-43             | 43                |

To augment the available training data, I train on a repeated shuffle of the training set with the following transormations:
 * Random cropping to 24x24 pixels
 * Randomly flipped left to right
 * Randomly adjusted contrast 

Training the final model was accomplished on an Amazon Web Service GPU instance in 300,000 steps (12.6 hours) using the following hyperparameters:
 * Adadelta Optimizer
 * Initial learning rate of 0.1
 * Training batch size of 128
 * L2 Regularization with scalar value of 0.0005 (consistent with VGGNet)

This network (and hyperparameter combination) provided sufficient representative power and reasonable training times.  Learning rate, convolutional layer count, and regularization all significantly affected the performance.

## Results Summary

A test set accuracy of 0.949 is achieved with the trained model.  Validation set accuracy is 0.956.  Validation set accuracy was continuing to improve when the max steps limit was reached but further training was not pursued due to schedule constraints.

The web images of German traffic signs were correctly predicted for three of the five images tested.  The first incorrect prediction was a mistaken speed limit reading confounding a '50' with an '80'.  The second incorrect prediction inexplicably mistook a 'Double Curve' sign with 'Children Crossing'.  Visualizations of these predictions can be found in `Traffic_Sign_Classifer.html` in this repository.

## Project (Repository Contents)

* `Traffic_Sign_Classifier.ipynb` - Jupyter notebook that demonstrates inference on the test dataset and on five example images
* `signs_input.py` - Using the TensorFlow DataSet API, produces repeating batches of training, validation, and test data
* `signs.py` - Defines operations to train the network.  Training monitoring is accomplished using Tensorboard.  The trained model is saved after final training step reached.
