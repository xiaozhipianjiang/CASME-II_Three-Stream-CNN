# 训练三流CNN
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, PReLU, Conv2D, ZeroPadding2D, MaxPooling2D, AveragePooling2D, Flatten, concatenate
from keras.optimizers import SGD
from keras.utils import np_utils

# 准备数据
spatial_data = np.load("Spatial_data.npy")

mean = np.load("mean.npy")
std = np.load("std.npy")
spatial_data -= mean
spatial_data /= std

temporal_data = np.load("Temporal_data.npy")

stacked_data = np.load("Stacked_data.npy")

label = np.load("label.npy")
label = np_utils.to_categorical(label, num_classes=5)

# 留一主题交叉验证
i = 0
index_list = [0, 9, 22, 29, 34, 53, 58, 67, 70, 83, 96, 106, 118, 126, 130, 133, 137, 171, 174, 189, 200, 202, 204, 216, 224, 231, 247]
b = index_list[i]
c = index_list[i + 1]

spatial_train = np.concatenate((spatial_data[0:b, :], spatial_data[c:247, :]), axis=0)
spatial_validation = spatial_data[b:c, :]
spatial_train = spatial_train.reshape((spatial_train.shape[0], 48, 48, 1))
spatial_validation = spatial_validation.reshape((spatial_validation.shape[0], 48, 48, 1))

temporal_train = np.concatenate((temporal_data[0:b, :], temporal_data[c:247, :]), axis=0)
temporal_validation = temporal_data[b:c, :, :, :]

stacked_train = np.concatenate((stacked_data[0:b, :], stacked_data[c:247, :]), axis=0)
stacked_validation = stacked_data[b:c, :, :, :]

label_train = np.concatenate((label[0:b, :], label[c:247, :]), axis=0)
label_validation = label[b:c, :]

# 静态空间流,加载预训练权重
spatial_inputs = Input(shape=(48, 48, 1), name='spatial_inputs')

spatial_conv1 = Conv2D(64, (5, 5), padding='valid')(spatial_inputs)
spatial_relu1 = PReLU(alpha_initializer='zeros')(spatial_conv1)
spatial_pad1 = ZeroPadding2D(padding=(2, 2))(spatial_relu1)
spatial_pool1 = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(spatial_pad1)

spatial_pad2 = ZeroPadding2D(padding=(1, 1))(spatial_pool1)
spatial_conv2 = Conv2D(64, (3, 3))(spatial_pad2)
spatial_relu2 = PReLU(alpha_initializer='zeros')(spatial_conv2)
spatial_pad3 = ZeroPadding2D(padding=(1, 1))(spatial_relu2)
spatial_conv3 = Conv2D(64, (3, 3))(spatial_pad3)
spatial_relu3 = PReLU(alpha_initializer='zeros')(spatial_conv3)
spatial_pool2 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(spatial_relu3)

spatial_pad4 = ZeroPadding2D(padding=(1, 1))(spatial_pool2)
spatial_conv4 = Conv2D(128, (3, 3))(spatial_pad4)
spatial_relu4 = PReLU(alpha_initializer='zeros')(spatial_conv4)
spatial_pad5 = ZeroPadding2D(padding=(1, 1))(spatial_relu4)
spatial_conv5 = Conv2D(128, (3, 3))(spatial_pad5)
spatial_relu5 = PReLU(alpha_initializer='zeros')(spatial_conv5)

spatial_pad6 = ZeroPadding2D(padding=(1, 1))(spatial_relu5)
spatial_pool3 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(spatial_pad6)

spatial_flatten = Flatten()(spatial_pool3)
spatial_dense1 = Dense(1024)(spatial_flatten)
spatial_relu6 = PReLU(alpha_initializer='zeros')(spatial_dense1)
spatial_dropout1 = Dropout(0.2)(spatial_relu6)
spatial_dense2 = Dense(1024)(spatial_dropout1)
spatial_relu7 = PReLU(alpha_initializer='zeros')(spatial_dense2)

spatial_dropout2 = Dropout(0.2)(spatial_relu7)
spatial_dense3 = Dense(5)(spatial_dropout2)
spatial_outputs = Activation('softmax')(spatial_dense3)

model = Model(inputs=spatial_inputs, outputs=spatial_outputs)

model.load_weights("pre_Spatial_CNN_Model.hdf5")


# 动态时间流
temporal_inputs = Input(shape=(48, 48, 20), name='temporal_inputs')

temporal_conv1 = Conv2D(64, (5, 5), padding='valid')(temporal_inputs)
temporal_relu1 = PReLU(alpha_initializer='zeros')(temporal_conv1)
temporal_pad1 = ZeroPadding2D(padding=(2, 2))(temporal_relu1)
temporal_pool1 = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(temporal_pad1)

temporal_pad2 = ZeroPadding2D(padding=(1, 1))(temporal_pool1)
temporal_conv2 = Conv2D(64, (3, 3))(temporal_pad2)
temporal_relu2 = PReLU(alpha_initializer='zeros')(temporal_conv2)
temporal_pad3 = ZeroPadding2D(padding=(1, 1))(temporal_relu2)
temporal_conv3 = Conv2D(64, (3, 3))(temporal_pad3)
temporal_relu3 = PReLU(alpha_initializer='zeros')(temporal_conv3)
temporal_pool2 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(temporal_relu3)

temporal_pad4 = ZeroPadding2D(padding=(1, 1))(temporal_pool2)
temporal_conv4 = Conv2D(128, (3, 3))(temporal_pad4)
temporal_relu4 = PReLU(alpha_initializer='zeros')(temporal_conv4)
temporal_pad5 = ZeroPadding2D(padding=(1, 1))(temporal_relu4)
temporal_conv5 = Conv2D(128, (3, 3))(temporal_pad5)
temporal_relu5 = PReLU(alpha_initializer='zeros')(temporal_conv5)

temporal_pad6 = ZeroPadding2D(padding=(1, 1))(temporal_relu5)
temporal_pool3 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(temporal_pad6)

temporal_flatten = Flatten()(temporal_pool3)
temporal_dense1 = Dense(1024)(temporal_flatten)
temporal_relu6 = PReLU(alpha_initializer='zeros')(temporal_dense1)
temporal_dropout1 = Dropout(0.2)(temporal_relu6)
temporal_dense2 = Dense(1024)(temporal_dropout1)
temporal_relu7 = PReLU(alpha_initializer='zeros')(temporal_dense2)

temporal_dropout2 = Dropout(0.5)(temporal_relu7)
temporal_dense3 = Dense(5)(temporal_dropout2)
temporal_outputs = Activation('softmax')(temporal_dense3)

# 堆叠空间流
stacked_inputs = Input(shape=(48, 48, 9), name='stacked_inputs')

stacked_conv1 = Conv2D(64, (5, 5), padding='valid')(stacked_inputs)
stacked_relu1 = PReLU(alpha_initializer='zeros')(stacked_conv1)
stacked_pad1 = ZeroPadding2D(padding=(2, 2))(stacked_relu1)
stacked_pool1 = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(stacked_pad1)

stacked_pad2 = ZeroPadding2D(padding=(1, 1))(stacked_pool1)
stacked_conv2 = Conv2D(64, (3, 3))(stacked_pad2)
stacked_relu2 = PReLU(alpha_initializer='zeros')(stacked_conv2)
stacked_pad3 = ZeroPadding2D(padding=(1, 1))(stacked_relu2)
stacked_conv3 = Conv2D(64, (3, 3))(stacked_pad3)
stacked_relu3 = PReLU(alpha_initializer='zeros')(stacked_conv3)
stacked_pool2 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(stacked_relu3)

stacked_pad4 = ZeroPadding2D(padding=(1, 1))(stacked_pool2)
stacked_conv4 = Conv2D(128, (3, 3))(stacked_pad4)
stacked_relu4 = PReLU(alpha_initializer='zeros')(stacked_conv4)
stacked_pad5 = ZeroPadding2D(padding=(1, 1))(stacked_relu4)
stacked_conv5 = Conv2D(128, (3, 3))(stacked_pad5)
stacked_relu5 = PReLU(alpha_initializer='zeros')(stacked_conv5)

stacked_pad6 = ZeroPadding2D(padding=(1, 1))(stacked_relu5)
stacked_pool3 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(stacked_pad6)

stacked_flatten = Flatten()(stacked_pool3)
stacked_dense1 = Dense(1024)(stacked_flatten)
stacked_relu6 = PReLU(alpha_initializer='zeros')(stacked_dense1)
stacked_dropout1 = Dropout(0.2)(stacked_relu6)
stacked_dense2 = Dense(1024)(stacked_dropout1)
stacked_relu7 = PReLU(alpha_initializer='zeros')(stacked_dense2)

stacked_dropout2 = Dropout(0.5)(stacked_relu7)
stacked_dense3 = Dense(5)(stacked_dropout2)
stacked_outputs = Activation('softmax')(stacked_dense3)

# 构建三流CNN模型
relu = concatenate([spatial_relu7, temporal_relu7, stacked_relu7])
dropout = Dropout(0.8)(relu)
dense = Dense(5)(dropout)
outputs = Activation('softmax')(dense)

model = Model(inputs=[spatial_inputs, temporal_inputs, stacked_inputs], outputs=outputs)

sgd = SGD(lr=1e-3, decay=1e-5, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

filepath = 'Model/Three_Stream_CNNModel.{epoch:02d}-{val_acc:.4f}.hdf5'
checkpointer = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)

model.fit([spatial_train, temporal_train, stacked_train], label_train, batch_size=247, epochs=500,
          validation_data=([spatial_validation, temporal_validation, stacked_validation], label_validation), callbacks=[checkpointer])
