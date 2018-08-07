
import os
import numpy as np
from keras.layers import Dense, Activation,  BatchNormalization, Flatten, Conv2D
from keras.layers import  MaxPooling2D
from keras.preprocessing import image
from keras.models import Sequential
from PIL import Image


X_train_ori = np.zeros((1200,96,96,3))
X_test_ori  = np.zeros((100,96,96,3))
Y_train  = np.zeros((1200, 1))
Y_test  = np.zeros((100, 1))

train_nogi = './img/train/nogi'
train_akb = './img/train/akb/'
test_nogi = './img/test/nogi/'
test_akb = './img/test/akb/'

# nogi_train
for i, file in enumerate(os.listdir(train_nogi)):
    if file[-3:] == "jpg":
        img = image.load_img(os.path.join(train_nogi, file), target_size= (96, 96))
        arr = image.img_to_array(img)
        arr = np.expand_dims(img, axis=0)
        X_train_ori [i,] = arr
        Y_train [i,] = 0

# akb_train
for i, file in enumerate(os.listdir(train_akb)):
    if file[-3:] == "jpg":
        img = image.load_img(os.path.join(train_akb, file), target_size= (96, 96))
        img = image.img_to_array(img)
        arr = np.expand_dims(img, axis=0)
        X_train_ori [i + 600,] = arr
        Y_train[i + 600] = 1

# nogi _test
for i, file in enumerate(os.listdir(test_nogi)):
    if file[-3:] == "jpg":
        img = image.load_img(os.path.join(test_nogi, file), target_size= (96, 96))
        arr = image.img_to_array(img)
        #arr = np.expand_dims(img, axis=0)
        X_test_ori [i,] = arr
        Y_test[i,] = 0

# akb_test
for i, file in enumerate(os.listdir(test_akb)):
    if file[-3:] == "jpg":
        img = image.load_img(os.path.join(test_akb, file), target_size= (96, 96))
        arr = image.img_to_array(img)
        #arr = np.expand_dims(img, axis=0)
        X_test_ori [i + 50,] = arr
        Y_test[i + 50] = 1

X_train = X_train_ori.astype('float32')
X_test = X_test_ori.astype('float32')
X_train = X_train/255.
X_test = X_test/255.




# model
model = Sequential()
model.add(Conv2D(16,(7,7),input_shape=(96,96,3),name ='conv0'))
model.add(BatchNormalization(axis = 3, name = 'bn0'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), name='max_pool1'))

model.add(Conv2D(32,(5,5),name ='conv1'))
model.add(BatchNormalization(axis = 3, name = 'bn1'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), name='max_pool2'))

model.add(Flatten())
model.add(Dense(100, name='fc1'))
model.add(Activation('relu'))
model.add(Dense(1, name='fc2'))
model.add(Activation('sigmoid'))


# train
model.compile(optimizer="Adam", loss ="binary_crossentropy", metrics = ["accuracy"])
stack  = model.fit(x=X_train, y= Y_train, epochs=100, batch_size=128)

model.save('./model/model_20180805')