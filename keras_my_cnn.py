from __future__ import print_function
import keras
from time import time
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import datetime
batch_size = 32
num_classes = 32
epochs = 200

input_shape = (300, 400, 3)


model = Sequential([
    Conv2D(32, (3, 3), input_shape=input_shape, padding='same',
           activation='relu'),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Dropout(0.2),
    Conv2D(64, (3, 3), activation='relu', padding='same', ),
    Conv2D(64, (3, 3), activation='relu', padding='same', ),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Dropout(0.2),
    Conv2D(128, (3, 3), activation='relu', padding='same', ),
    Conv2D(128, (3, 3), activation='relu', padding='same', ),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(num_classes, activation='softmax')
])
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=30
)
train_generator = train_datagen.flow_from_directory('leaves_data/train',
                                                    target_size=(300, 400),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')
test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=30
)
validation_generator = test_datagen.flow_from_directory('leaves_data/validate',
                                                        target_size=(300, 400),
                                                        class_mode='categorical')

model.compile(loss = "categorical_crossentropy", optimizer ='adam', metrics=["accuracy"])
model.summary()
model_name = str(datetime.datetime.now())+".h5"
model.fit_generator(
    train_generator,
    steps_per_epoch=1527//batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=200//batch_size,
)
model.save(model_name)