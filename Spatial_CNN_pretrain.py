# 在FER2013上对静态空间流CNN进行预训练
import numpy as np
import pandas as pd
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, PReLU, Conv2D, ZeroPadding2D, MaxPooling2D, AveragePooling2D, Flatten
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

# 准备FER2013数据
file = pd.read_csv('fer2013.csv')

data = file.values
FER_label = data[:, 0]
pixels = data[:, 1]

FER_data = np.zeros((pixels.shape[0], 48*48))
for ix in range(FER_data.shape[0]):
    p = pixels[ix].split(' ')
    for iy in range(FER_data.shape[1]):
        FER_data[ix, iy] = int(p[iy])

mean = np.mean(FER_data, axis=0)
std = np.std(FER_data, axis=0)

np.save("mean", mean)
np.save("std", std)

FER_data -= np.mean(FER_data, axis=0)
FER_data /= np.std(FER_data, axis=0)

CASME2_data = np.load("Spatial_data.npy")
CASME2_label = np.load("label.npy")

CASME2_data -= np.mean(FER_data, axis=0)
CASME2_data /= np.std(FER_data, axis=0)

X_train = FER_data.reshape((FER_data.shape[0], 48, 48, 1))
Y_train = np_utils.to_categorical(FER_label, num_classes=5)
X_validation = CASME2_data.reshape((CASME2_data.shape[0], 48, 48, 1))
Y_validation = np_utils.to_categorical(CASME2_label, num_classes=5)

# 搭建模型
inputs = Input(shape=(48, 48, 1), name='inputs')

conv1 = Conv2D(64, (5, 5), padding='valid')(inputs)
relu1 = PReLU(alpha_initializer='zeros')(conv1)
pad1 = ZeroPadding2D(padding=(2, 2))(relu1)
pool1 = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(pad1)

pad2 = ZeroPadding2D(padding=(1, 1))(pool1)
conv2 = Conv2D(64, (3, 3))(pad2)
relu2 = PReLU(alpha_initializer='zeros')(conv2)
pad3 = ZeroPadding2D(padding=(1, 1))(relu2)
conv3 = Conv2D(64, (3, 3))(pad3)
relu3 = PReLU(alpha_initializer='zeros')(conv3)
pool2 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(relu3)

pad4 = ZeroPadding2D(padding=(1, 1))(pool2)
conv4 = Conv2D(128, (3, 3))(pad4)
relu4 = PReLU(alpha_initializer='zeros')(conv4)
pad5 = ZeroPadding2D(padding=(1, 1))(relu4)
conv5 = Conv2D(128, (3, 3))(pad5)
relu5 = PReLU(alpha_initializer='zeros')(conv5)

pad6 = ZeroPadding2D(padding=(1, 1))(relu5)
pool3 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(pad6)

flatten = Flatten()(pool3)
dense1 = Dense(1024)(flatten)
relu6 = PReLU(alpha_initializer='zeros')(dense1)
dropout1 = Dropout(0.2)(relu6)
dense2 = Dense(1024)(dropout1)
relu7 = PReLU(alpha_initializer='zeros')(dense2)

dropout2 = Dropout(0.2)(relu7)
dense3 = Dense(5)(dropout2)
outputs = Activation('softmax')(dense3)

model = Model(inputs=inputs, outputs=outputs)

sgd = SGD(lr=0.1, decay=1e-4, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary()

# 使用数据增强，进行训练
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(X_train)

filepath = 'pre_Spatial_CNN_Model.hdf5'
checkpointer = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='auto', period=1000)

model.fit_generator(datagen.flow(X_train, Y_train, batch_size=512), nb_epoch=1000, steps_per_epoch=X_train.shape[0]/512,
                    validation_data=(X_validation, Y_validation), callbacks=[checkpointer])
