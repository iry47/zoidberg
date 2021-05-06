import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.utils import np_utils
from keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_path='/c/Users/donquijote/Desktop/IA/zoidberg_2020_15/project/dataset/train'
test_path='/c/Users/donquijote/Desktop/IA/zoidberg_2020_15/project/dataset/test'
val_path='/c/Users/donquijote/Desktop/IA/zoidberg_2020_15/project/dataset/val'

img_size=(256,256)
batch_size=64

datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator=datagen.flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    color_mode='grayscale'
)

test_generator=datagen.flow_from_directory(
    test_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    color_mode='grayscale'
)

val_generator=datagen.flow_from_directory(
    val_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    color_mode='grayscale'
)


# We are using convo2D and fully connected dense layers for our classification.

model=keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3,3), input_shape=(256,256,1), activation='relu',padding='same'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Conv2D(32, (3,3), activation='relu',padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Conv2D(128, (3,3), activation='relu',padding='same'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Conv2D(128, (3,3), activation='relu',padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1024, activation="relu"))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(512,activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1,activation="sigmoid"))

model.summary()

# Time based learning rate scheduling
epoch=50
learning_rate=0.01
decay_rate=learning_rate/epoch
momentum=0.8
sgd=SGD(lr=learning_rate,momentum=momentum,decay=decay_rate)
# Early stopping callback
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
# compiling the model 
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
model_history=model.fit(train_generator,epochs=epoch,validation_data=val_generator,shuffle=True, callbacks=[callback])
# plotting the varoius metrics
pd.DataFrame(model_history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()
# Testing the model
model.evaluate(test_generator)