from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Activation,Dense,Dropout,Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
from keras.layers.normalization import BatchNormalization

batch_size = 32
num_classes = 32
epochs = 200

img_rows,img_cols = 400,300



K.set_image_dim_ordering('th')

model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(3,400,300),data_format='channels_first'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2,strides=2,padding='same'))

model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2,strides=2,padding='same'))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2,strides=2,padding='same'))


model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])



train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory('leaves_data/train',
                                                    target_size=(400,300),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')
test_datagen = ImageDataGenerator()
validation_generator = test_datagen.flow_from_directory('leaves_data/validate',
                                                        target_size=(400,300),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=2000 // batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=800 // batch_size
)
model.save('first_keras.h5')