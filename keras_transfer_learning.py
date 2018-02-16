from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model,load_model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

img_width, img_height = 400, 300
train_data_dir = "split/train"
validation_data_dir = "split/test"
nb_train_samples = 1527
nb_validation_samples = 180
batch_size = 16
epochs = 50

# model = applications.InceptionV3(weights='imagenet',include_top=False)
#
#
# #Adding custom Layers
# x = model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(1024,activation='relu')(x)
# predictions = Dense(32,activation='softmax')(x)
#
# # creating the final model
# model_final = Model(input = model.input, output = predictions)
#
# for layer in model.layers : layer.trainable = False

# compile the model
#
#model_final.compile(loss = "categorical_crossentropy", optimizer = 'rmsprop', metrics=["accuracy"])

# Initiate the train and test generators with data Augumentation
train_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

test_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size,
class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
validation_data_dir,
target_size = (img_height, img_width),
class_mode = "categorical")


# model_name = "keras_cnn_tf_2.h5"
# model_final.fit_generator(
#     train_generator,
#     steps_per_epoch=1527//batch_size,
#     epochs=5,
#     validation_data=validation_generator,
#     validation_steps=180//batch_size,
# )
# model_final.save(model_name)
#


model = load_model('keras_cnn_tf_2.h5')

for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True

model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                       metrics=["accuracy"])

model_name = "keras_cnn_tf_tuned_3.h5"
model.fit_generator(
    train_generator,
    steps_per_epoch=1527//batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=180//batch_size,
)
model.save(model_name)