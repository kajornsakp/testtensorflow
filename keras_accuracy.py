import glob
from PIL import Image
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# model = load_model('keras_cnn_no_transfer_learning_split_shuffle.h5')
# model = load_model('keras_cnn_no_transfer_learning_split_shuffle_rotate_0.h5')
# model.summary()
# model = load_model('keras_lamina_1.h5')
# allArr = []
# for i in range(1,16):
#     arr = []
#     for file in glob.iglob('lamina/test/' + str(i) + '/*.jpg'):
#         img = Image.open(file)
#         imgarr = np.expand_dims(np.asarray(img), axis=0)
#         prediction = model.predict(imgarr)[0]
#         classId = np.argmax(prediction)
#         arr.append(int(classId))
#     arr.sort()
#     for a in arr:
#         print(a)
#     print("======================================================================")
#     allArr.append(arr)
# npArr = np.array(allArr)
# print(npArr)
# npArr.transpose()
# print(npArr)

# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=90,
#     horizontal_flip=True,
# )
# for i in range(1,17):
#     for file in glob.iglob('leaves_data/train/'+str(i)+'/*.jpg'):
#         img = Image.open(file)
#         arr = np.asarray(img)
#         arr = arr.reshape((1,) + arr.shape)
#         j = 0
#         for batch in train_datagen.flow(arr, batch_size=1,
#                                         save_to_dir='gen_rotated/'+str(i), save_prefix='temp', save_format='jpeg'):
#             j += 1
#             if j > 4:
#                 break

classes = ['aceriform',
'acicular',
'cordate',
'deltoid',
'elliptic',
'falcate',
'flabellate',
'lanceolate',
'linear',
'oblancceolate',
'oblong',
'obovate',
'ovate',
'spathulate',
'subulate',
'tulip']


model = load_model('keras_lamina_1.h5')
allArr = []
for i in classes:
    arr = []
    for file in glob.iglob('lamina/test/' + i + '/*.jpg'):
        img = Image.open(file)
        imgarr = np.expand_dims(np.asarray(img), axis=0)
        prediction = model.predict(imgarr)[0]
        classId = np.argmax(prediction)
        arr.append(int(classId))
    arr.sort()
    for a in arr:
        print(a)
    print("======================================================================")
    allArr.append(arr)
npArr = np.array(allArr)
print(npArr)