# cnn-urban-sound-recognition
Convolutional Neural Network for Urban Sound Classification

Project developed by Stefano Carnà and Lorenzo Vitali in the context of Neural Network Exam at Sapienza - Università degli studi di Roma.

Dataset (Urban Sound 8K): https://serv.cusp.nyu.edu/projects/urbansounddataset/urbansound.html

Based on Keras with TensorFlow as backend (actually you can choose between TensorFlow and Theano).

Files overwiev:

- data_features.py is the python code that extract the MFCC, delta and delta-delta features frome the dataset. It saves the output matrix as files
- keras_cnn.py is the file in which the neural network is made and trained. It saves the trained net as files
- keras_trained.py is the file which load the trained net, take the data input pre-processed by data_features file and return the accuracy and the confusion matrix
- model.keras.h and model.keras.json represent our trained net.
