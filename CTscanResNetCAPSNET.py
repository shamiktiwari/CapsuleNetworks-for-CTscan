# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 00:13:39 2020

@author: shamik.tiwari
"""

from __future__ import print_function
from keras import backend as K
from keras import activations
from keras import utils
from keras.models import Model
from keras.layers import *
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras import regularizers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
disease_types=['COVID', 'non-COVID']
data_dir = 'E:\covidCTscan'
train_dir = os.path.join(data_dir)
train_data = []
for defects_id, sp in enumerate(disease_types):
    for file in os.listdir(os.path.join(train_dir, sp)):
        train_data.append(['{}\{}'.format(sp, file), defects_id, sp])
        
train = pd.DataFrame(train_data, columns=['File', 'DiseaseID','Disease Type'])
train.head()

SEED = 42
train = train.sample(frac=1, random_state=SEED) 
train.index = np.arange(len(train)) # Reset indices
train.head()

IMAGE_SIZE = 128
def read_image(filepath):
    return cv2.imread(os.path.join(data_dir, filepath)) 

def resize_image(image, image_size):
    return cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_AREA)

data = np.zeros((train.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))
for i, file in tqdm(enumerate(train['File'].values)):
    image = read_image(file)
    if image is not None:
        data[i] = resize_image(image, (IMAGE_SIZE, IMAGE_SIZE))

data = data / 255.
print('Train Shape: {}'.format(data.shape))


Y_train = train['DiseaseID'].values
Y_train = to_categorical(Y_train, num_classes=2)

BATCH_SIZE = 8

from sklearn.model_selection import train_test_split
X_train, X_1, y_train, y_1 = train_test_split(data, Y_train, test_size=0.3, random_state=27)


X_cv, X_test, y_cv, y_test = train_test_split(X_1, y_1, test_size=0.3, random_state=33)

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x

def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)

def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1 
    return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)
    
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.layers import Input
input_image = Input(shape=(128,128, 3))

base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=Input(shape=(128,128,3)))



output = Conv2D(256, kernel_size=(3,3), strides=(1, 1), activation='relu')(base_model.output)

x = Reshape((-1, 256))(output)
capsule = CapsuleLayer(num_capsule=2, dim_capsule=16, routings=4,name='CTscnaCaps')(x)

#capsule = CapsuleLayer(3, 16, 4, True)(x)
output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)
ResCapsNetmodel = Model(base_model.input, outputs=output)

ResCapsNetmodel.summary()

train_batch = 8
val_batch = 2

optimizer = Adam(lr=0.0015, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
ResCapsNetmodel.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='min', restore_best_weights=True)
checkpoint = ModelCheckpoint("weights.h5", 
                             monitor='val_loss', 
                             verbose=1, 
                             save_best_only=True, 
                             save_weights_only=False, 
                             mode='min')
callback_list = [checkpoint, early]
hist=ResCapsNetmodel.fit(x=X_train, y=y_train, batch_size=16, epochs=100, validation_data=(X_cv,y_cv),callbacks=callback_list)




#ResCapsNetmodel.compile(loss='binary_crossentropy', optimizer=SGD(lr=lr,decay=0.001, momentum=0.9), metrics=['accuracy'])

#hist=ResCapsNetmodel.fit(x=X_train, y=y_train,batch_size=32, epochs=100, validation_data=(X_cv,y_cv), callbacks=callback_list)
    
import matplotlib.pyplot as plt
plt.style.use('ggplot')
accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs = range(len(accuracy))
plt.figure(figsize=(12,12))
plt.rcParams.update({'font.size': 18})
plt.plot(epochs, accuracy, 'go-', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'ms-', label='Validation accuracy')
plt.title('Training and validation accuracy curves for ResCapsnet Model')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.figure()
plt.figure(figsize=(12,12))
plt.rcParams.update({'font.size': 18})
plt.plot(epochs, loss, 'bo-', label='Training loss')
plt.plot(epochs, val_loss, 'rs-', label='Validation loss')
plt.title('Training and validation loss curves for ResCapsNet Model')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.show()


predicted_classes = ResCapsNetmodel.predict(X_test)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
tar = np.argmax(np.round(y_test),axis=1)
#correct = np.where(predicted_classes==train_Y_one_hot)[0]
#print "Found %d correct labels" % len(correct)
   
#    incorrect = np.where(predicted_classes!=train_Y_one_hot)[0]
#print "Found %d incorrect labels" % len(incorrect)
from sklearn.metrics import classification_report,confusion_matrix
target_names = ["Class {}".format(i) for i in range(2)]
print(classification_report(tar, predicted_classes, labels=None, target_names=None))

from mlxtend.plotting import plot_confusion_matrix
#Creating a confusion matrix
mat = confusion_matrix(tar,predicted_classes)
fig, ax = plt.subplots(figsize=(10,10))
plt.rcParams.update({'font.size': 28})
plot_confusion_matrix(conf_mat=mat,cmap = "GnBu",colorbar=True, show_absolute=False,show_normed=True,figsize=(10,10),class_names=['Covid', 'Non-Covid'])
plt.show()

from sklearn.metrics import roc_curve,auc
from scipy import interp
from itertools import cycle
n_classes=2
lw=2
preds = ResCapsNetmodel.predict(X_test)
y_score = preds
target=y_test
fpr = dict()
tpr = dict()
roc_auc = dict()
fig, ax = plt.subplots(figsize=(20,20))
plt.rcParams.update({'font.size': 12})
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(target[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(target.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=3)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['b', 'g'])
classes = ['Covid', 'Non-Covid']
for (i, j, color) in zip(range(n_classes),classes, colors):
       plt.plot(fpr[i], tpr[i], color=color, lw=lw,label=j+'(area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

    
plt.plot([0, 1], [0, 1], 'k+', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for for Covid classification using ResCapsNet')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
