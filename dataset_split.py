import glob
import shutil
import numpy as np
from PIL import Image
import math

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
for ii in classes:
    arr = []
    for file in glob.iglob('lamina/full/' + ii  + '/*.jpg'):
        arr.append(file)
    nparr = np.array(arr)
    count = nparr.size
    eighty = int(math.floor(count*0.8))
    twenty = int(math.ceil(count*0.2))
    idx = np.hstack((np.ones(eighty), np.zeros(twenty)))
    np.random.shuffle(idx)
    train = nparr[idx == 1]
    test = nparr[idx == 0]
    #
    # print(train)
    # print(test)
    c = 0
    for i in train:
        print(i)
        shutil.copyfile(i,'./lamina/train/'+ii+'/'+ str(c) + '.jpg')
        c += 1
    x = 0
    for i in test:
        print(i)
        shutil.copyfile(i, './lamina/test/' + ii + '/' + str(x) + '.jpg')
        x += 1

            #
    # print(arr)