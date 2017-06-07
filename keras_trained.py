from __future__ import print_function

import matplotlib.pyplot as plt
import pickle
import itertools
import numpy as np

from keras.models import model_from_json
from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix

CLASSES = ["AC", "CH", "CP", "DB", "Dr", "EI", "GS", "Ja", "Si", "SM"]

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
## END plot_confusion_matrix


np.random.seed(1337)  # for reproducibility

'''
dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension (the depth) is at index 1,
in 'tf' mode is it at index 3. It defaults to the  image_dim_ordering value found in your
Keras config file at ~/.keras/keras.json. If you never set it, then it will be "tf".

print K.image_dim_ordering()
'''


# load json and create model
json_file = open('model.keras.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.keras.h5")


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
loaded_model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

print("Loaded model from disk")


## GET DATA TO WORK ON
print("Start loading data")

fd = open("data_x.pkl", 'r')
fd2 = open("data_y.pkl", 'r')
features = pickle.load(fd)
labels = pickle.load(fd2)

print("Data loaded")

p_train = 0.8

rnd_indices = np.random.rand(len(labels)) < p_train

X_train = features[rnd_indices]
Y_train = labels[rnd_indices]
X_test = features[~rnd_indices]
Y_test = labels[~rnd_indices]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


## FIX FOR KERAS
# Y_train = Y_train.reshape((-1, 1))
Y_test = Y_test.reshape((-1, 1))


## FIX FOR KERAS
# labels = labels.reshape((-1, 1))

## Prediction
predictions = loaded_model.predict_classes(X_test)

plot_confusion_matrix( cm=confusion_matrix(Y_test, predictions) , classes=CLASSES)

right = 0
total = 0

for idx in range(len(predictions)):
    if (predictions[idx] == Y_train[idx]):
        right += 1
    total += 1

## Decode and display prediction
print("Got Predictions: ", len(predictions))

print ("Result: ", str(right), " on ", str(total), " -> ", str( ( (right * 1.0) / total ) ))
