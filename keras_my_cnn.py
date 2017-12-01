from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K, Input
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
from keras.layers.normalization import BatchNormalization

batch_size = 32
num_classes = 32
epochs = 200
from keras.datasets import cifar10

input_shape = (400,300,3)

# model = Sequential()
# model.add(Conv2D(32,(3,3),input_shape=(400,300,3),data_format='channels_first',activation='relu',name='conv1_1'))
# model.add(ZeroPadding2D((1,1)))
# model.add(MaxPooling2D(pool_size=2,strides=2,padding='same'))
# model.add(ZeroPadding2D((1,1)))
#
# model.add(Conv2D(32,(3,3),activation='relu',name='conv2_1'))
# model.add(ZeroPadding2D((1,1)))
#
# model.add(MaxPooling2D(pool_size=2,strides=2,padding='same'))
# model.add(ZeroPadding2D((1,1)))
#
# model.add(Conv2D(64,(3,3),activation='relu',name='conv_1'))
# model.add(ZeroPadding2D((1,1)))
#
# model.add(MaxPooling2D(pool_size=2,strides=2,padding='same'))
# model.add(ZeroPadding2D((1,1)))
#
# #bottle neck
# model.add(Flatten())
# model.add(Dense(256,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(num_classes,activation='softmax'))
# # model.add(Activation('softmax'))
# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))



# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    horizontal_flip=True
)
train_generator = train_datagen.flow_from_directory('leaves_data/train',
                                                    target_size=(400,300),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

validation_generator = train_datagen.flow_from_directory('leaves_data/validate',
                                                        target_size=(400,300),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

model.summary()
# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit_generator(
    train_generator,
    steps_per_epoch=30,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=800 // batch_size
)
score = model.evaluate_generator(validation_generator)
print(score)
model.save('keras_model_final.h5')