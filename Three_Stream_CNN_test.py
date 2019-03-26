# 测试三流CNN
import os
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, PReLU, Conv2D, ZeroPadding2D, MaxPooling2D, AveragePooling2D, Flatten, concatenate

# 静态空间流
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
stacked_dense3 = Dense(3)(stacked_dropout2)
stacked_outputs = Activation('softmax')(stacked_dense3)

# 构建三流CNN模型
relu = concatenate([spatial_relu7, temporal_relu7, stacked_relu7])
dropout = Dropout(0.8)(relu)
dense = Dense(5)(dropout)
outputs = Activation('softmax')(dense)

model = Model(inputs=[spatial_inputs, temporal_inputs, stacked_inputs], outputs=outputs)

# 准备验证数据
spatial_data = np.load("Spatial_data.npy")

mean = np.load("mean.npy")
std = np.load("std.npy")
spatial_data -= mean
spatial_data /= std

temporal_data = np.load("Temporal_data.npy")

stacked_data = np.load("Stacked_data.npy")

index_list = [0, 9, 22, 29, 34, 53, 58, 67, 70, 83, 96, 106, 118, 126, 130, 133, 137, 171, 174, 189, 200, 202, 204, 216, 224, 231, 247]
model_list = os.listdir("./Three_Stream_Model")
model_list.sort(key=lambda x: int(x[0:2]))
print(model_list)

for i in range(26):
    b = index_list[i]
    c = index_list[i + 1]

    spatial_validation = spatial_data[b:c, :]
    spatial_validation = spatial_validation.reshape((spatial_validation.shape[0], 48, 48, 1))

    temporal_validation = temporal_data[b:c, :, :, :]

    stacked_validation = stacked_data[b:c, :, :, :]

    model.load_weights("./Three_Stream_Model/" + model_list[i])
    result = model.predict([spatial_validation, temporal_validation, stacked_validation])
    for j in range(result.shape[0]):
        print(result[j, :].argmax())
