import os
import cv2
import numpy as np
import pandas as pd

from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split


DATA_PATH = '/home/asish/Desktop/DATASET'


broken_files = os.listdir(os.path.join(DATA_PATH, 'broken'))


X = []
y = []


br_path = os.path.join(DATA_PATH, 'broken')
for file in broken_files:
    X.append(cv2.resize(cv2.imread(br_path+"/"+file), (128, 128)))
    y.append(0)


unbroken_files = os.listdir(os.path.join(DATA_PATH, 'notbroken'))


un_path = os.path.join(DATA_PATH, 'notbroken')
for file in unbroken_files:
    X.append(cv2.resize(cv2.imread(un_path+"/"+file), (128, 128)))
    y.append(1)


X_train, X_val, Y_train, Y_val = train_test_split(
    X, y, test_size=0.2, random_state=1)


model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


train_generator = train_datagen.flow(np.array(X_train), Y_train, batch_size=16)
validation_generator = val_datagen.flow(np.array(X_val), Y_val, batch_size=16)


history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(X_train),
    epochs=10,
    validation_data=validation_generator,
    validation_steps=len(X_val)
)
