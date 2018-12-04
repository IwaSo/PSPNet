from __future__ import print_function
from math import ceil
from keras import layers
from tools import Patch_DataLoader
import tools as tl

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization, Activation, Input, Dropout, ZeroPadding2D, Lambda
from keras.layers.merge import Concatenate, Add
from keras.models import Model
from keras.optimizers import SGD
from keras.backend import tf as ktf
from keras.layers.core import Reshape
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint,TensorBoard,LearningRateScheduler

import tensorflow as tf
import os
import numpy as np
import scipy as sp
import cv2
import time
from PIL import Image

learning_rate = 1e-3  # Layer specific learning rate
# Weight decay not implemented


def BN(name=""):
    return BatchNormalization(momentum=0.95, name=name, epsilon=1e-5)


class Interp(layers.Layer):

    def __init__(self, new_size, **kwargs):
        self.new_size = new_size
        super(Interp, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Interp, self).build(input_shape)

    def call(self, inputs, **kwargs):
        new_height, new_width = self.new_size
        resized = ktf.image.resize_images(inputs, [new_height, new_width],
                                          align_corners=True)
        return resized

    def compute_output_shape(self, input_shape):
        return tuple([None, self.new_size[0], self.new_size[1], input_shape[3]])

    def get_config(self):
        config = super(Interp, self).get_config()
        config['new_size'] = self.new_size
        return config


# def Interp(x, shape):
#    new_height, new_width = shape
#    resized = ktf.image.resize_images(x, [new_height, new_width],
#                                      align_corners=True)
#    return resized


def residual_conv(prev, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False):
    lvl = str(lvl)
    sub_lvl = str(sub_lvl)
    names = ["conv" + lvl + "_" + sub_lvl + "_1x1_reduce",
             "conv" + lvl + "_" + sub_lvl + "_1x1_reduce_bn",
             "conv" + lvl + "_" + sub_lvl + "_3x3",
             "conv" + lvl + "_" + sub_lvl + "_3x3_bn",
             "conv" + lvl + "_" + sub_lvl + "_1x1_increase",
             "conv" + lvl + "_" + sub_lvl + "_1x1_increase_bn"]
    if modify_stride is False:
        prev = Conv2D(64 * level, (1, 1), strides=(1, 1), name=names[0],
                      use_bias=False)(prev)
    elif modify_stride is True:
        prev = Conv2D(64 * level, (1, 1), strides=(2, 2), name=names[0],
                      use_bias=False)(prev)

    prev = BN(name=names[1])(prev)
    prev = Activation('relu')(prev)

    prev = ZeroPadding2D(padding=(pad, pad))(prev)
    prev = Conv2D(64 * level, (3, 3), strides=(1, 1), dilation_rate=pad,
                  name=names[2], use_bias=False)(prev)

    prev = BN(name=names[3])(prev)
    prev = Activation('relu')(prev)
    prev = Conv2D(256 * level, (1, 1), strides=(1, 1), name=names[4],
                  use_bias=False)(prev)
    prev = BN(name=names[5])(prev)
    return prev


def short_convolution_branch(prev, level, lvl=1, sub_lvl=1, modify_stride=False):
    lvl = str(lvl)
    sub_lvl = str(sub_lvl)
    names = ["conv" + lvl + "_" + sub_lvl + "_1x1_proj",
             "conv" + lvl + "_" + sub_lvl + "_1x1_proj_bn"]

    if modify_stride is False:
        prev = Conv2D(256 * level, (1, 1), strides=(1, 1), name=names[0],
                      use_bias=False)(prev)
    elif modify_stride is True:
        prev = Conv2D(256 * level, (1, 1), strides=(2, 2), name=names[0],
                      use_bias=False)(prev)

    prev = BN(name=names[1])(prev)
    return prev


def empty_branch(prev):
    return prev


def residual_short(prev_layer, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False):
    prev_layer = Activation('relu')(prev_layer)
    block_1 = residual_conv(prev_layer, level,
                            pad=pad, lvl=lvl, sub_lvl=sub_lvl,
                            modify_stride=modify_stride)

    block_2 = short_convolution_branch(prev_layer, level,
                                       lvl=lvl, sub_lvl=sub_lvl,
                                       modify_stride=modify_stride)
    added = Add()([block_1, block_2])
    return added


def residual_empty(prev_layer, level, pad=1, lvl=1, sub_lvl=1):
    prev_layer = Activation('relu')(prev_layer)

    block_1 = residual_conv(prev_layer, level, pad=pad,
                            lvl=lvl, sub_lvl=sub_lvl)
    block_2 = empty_branch(prev_layer)
    added = Add()([block_1, block_2])
    return added


def ResNet(inp, layers):
    # Names for the first couple layers of model
    names = ["conv1_1_3x3_s2",
             "conv1_1_3x3_s2_bn",
             "conv1_2_3x3",
             "conv1_2_3x3_bn",
             "conv1_3_3x3",
             "conv1_3_3x3_bn"]

    # Short branch(only start of network)

    cnv1 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', name=names[0],
                  use_bias=False)(inp)  # "conv1_1_3x3_s2"
    bn1 = BN(name=names[1])(cnv1)  # "conv1_1_3x3_s2/bn"
    relu1 = Activation('relu')(bn1)  # "conv1_1_3x3_s2/relu"

    cnv1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name=names[2],
                  use_bias=False)(relu1)  # "conv1_2_3x3"
    bn1 = BN(name=names[3])(cnv1)  # "conv1_2_3x3/bn"
    relu1 = Activation('relu')(bn1)  # "conv1_2_3x3/relu"

    cnv1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name=names[4],
                  use_bias=False)(relu1)  # "conv1_3_3x3"
    bn1 = BN(name=names[5])(cnv1)  # "conv1_3_3x3/bn"
    relu1 = Activation('relu')(bn1)  # "conv1_3_3x3/relu"

    res = MaxPooling2D(pool_size=(3, 3), padding='same',
                       strides=(2, 2))(relu1)  # "pool1_3x3_s2"

    # ---Residual layers(body of network)

    """
    Modify_stride --Used only once in first 3_1 convolutions block.
    changes stride of first convolution from 1 -> 2
    """

    # 2_1- 2_3
    res = residual_short(res, 1, pad=1, lvl=2, sub_lvl=1)
    for i in range(2):
        res = residual_empty(res, 1, pad=1, lvl=2, sub_lvl=i + 2)

    # 3_1 - 3_3
    res = residual_short(res, 2, pad=1, lvl=3, sub_lvl=1, modify_stride=True)
    for i in range(3):
        res = residual_empty(res, 2, pad=1, lvl=3, sub_lvl=i + 2)
    if layers is 50:
        # 4_1 - 4_6
        res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)
        for i in range(5):
            res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)
    elif layers is 101:
        # 4_1 - 4_23
        res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)
        for i in range(22):
            res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)
    else:
        print("This ResNet is not implemented")

    # 5_1 - 5_3
    res = residual_short(res, 8, pad=4, lvl=5, sub_lvl=1)
    for i in range(2):
        res = residual_empty(res, 8, pad=4, lvl=5, sub_lvl=i + 2)

    res = Activation('relu')(res)
    return res


def interp_block(prev_layer, level, feature_map_shape, input_shape):
    if input_shape == (473, 473):
        kernel_strides_map = {1: 60,
                              2: 30,
                              3: 20,
                              6: 10}
    elif input_shape == (713, 713):
        kernel_strides_map = {1: 90,
                              2: 45,
                              3: 30,
                              6: 15}
    elif input_shape == (160, 160):
        kernel_strides_map = {1: 20,
                              2: 12,
                              3: 8,
                              6: 4}
    elif input_shape == (240, 240):
        kernel_strides_map = {1: 32,
                              2: 16,
                              3: 12,
                              6: 48}
    else:
        print("Pooling parameters for input shape ",
              input_shape, " are not defined.")
        exit(1)

    names = [
        "conv5_3_pool" + str(level) + "_conv",
        "conv5_3_pool" + str(level) + "_conv_bn"
    ]
    kernel = (kernel_strides_map[level], kernel_strides_map[level])
    strides = (kernel_strides_map[level], kernel_strides_map[level])
    prev_layer = AveragePooling2D(kernel, strides=strides)(prev_layer)
    prev_layer = Conv2D(512, (1, 1), strides=(1, 1), name=names[0],
                        use_bias=False)(prev_layer)
    prev_layer = BN(name=names[1])(prev_layer)
    prev_layer = Activation('relu')(prev_layer)
    # prev_layer = Lambda(Interp, arguments={
    #                    'shape': feature_map_shape})(prev_layer)
    prev_layer = Interp(feature_map_shape)(prev_layer)
    return prev_layer


def build_pyramid_pooling_module(res, input_shape):
    """Build the Pyramid Pooling Module."""
    # ---PSPNet concat layers with Interpolation
    feature_map_size = tuple(int(ceil(input_dim / 8.0))
                             for input_dim in input_shape)
    print("PSP module will interpolate to a final feature map size of %s" %
          (feature_map_size, ))

    interp_block1 = interp_block(res, 1, feature_map_size, input_shape)
    interp_block2 = interp_block(res, 2, feature_map_size, input_shape)
    interp_block3 = interp_block(res, 3, feature_map_size, input_shape)
    interp_block6 = interp_block(res, 6, feature_map_size, input_shape)

    # concat all these layers. resulted
    # shape=(1,feature_map_size_x,feature_map_size_y,4096)
    res = Concatenate()([res,
                         interp_block6,
                         interp_block3,
                         interp_block2,
                         interp_block1])
    return res


def build_pspnet(nb_classes, resnet_layers, input_shape, activation):
    """Build PSPNet."""
    print("Building a PSPNet based on ResNet %i expecting inputs of shape %s predicting %i classes" % (
        resnet_layers, input_shape, nb_classes))

    inp = Input((input_shape[0], input_shape[1], 3))
    res = ResNet(inp, layers=resnet_layers)
    psp = build_pyramid_pooling_module(res, input_shape)

    x = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv5_4",
               use_bias=False)(psp)
    x = BN(name="conv5_4_bn")(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(nb_classes, (1, 1), strides=(1, 1), name="conv6")(x)
    # x = Lambda(Interp, arguments={'shape': (
    #    input_shape[0], input_shape[1])})(x)
    x = Interp([input_shape[0], input_shape[1]])(x)
    x = Activation('softmax')(x)

    model = Model(inputs=inp, outputs=x)

    # Solver
    sgd = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    for d_num in [4]:
        #parameters +++++++++++++++++
        epochs = 200
        batch_size = 16
        size = 160
        n_class = 3
        l2_reg = 0
        width = 1600
        hight = 1200
        activation='softmax'
        nb_classes=3
        resnet_layers=101
        in_size=160
        input_shape=(160,160)
        method="classification"
        resolution=None
        dataset = "ips"
        border_weight=None
        s=45
        num=0
        #++++++++++++++++++++++++++++
        print("training dataset_%d"%d_num)
        path = "/home/sora/project"
        lists = os.listdir(path + "/data/ips/dataset")
        os.chdir(path + "/data/ips/dataset")
        f = open("test_data%d.txt" %(d_num))
        lines = f.readlines()
        f.close()

        #改行を取り除く
        for q in range(len(lines)):
            lines[q] = lines[q].strip()

        os.chdir(path + "/ips/PSPnet2/X_crop/160/%d"%d_num)
        #os.chdir(path + "/PSPnet2/X/%d"%d_num)
        X_train = np.load("X_train.npy")
        X_test = np.load("X_test.npy")

        os.chdir(path + "/ips/PSPnet2/Y_crop/160/%d"%d_num)
        #os.chdir(path + "/PSPnet2/Y/%d"%d_num)
        Y_train = np.load("Y_train.npy")
        Y_test = np.load("Y_test.npy")



        model = build_pspnet(nb_classes, resnet_layers, input_shape, activation)
        model.summary()
        os.chdir(path + "/ips/PSPnet2/weights/%d"%d_num)
        #es_cb = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
        #model.load_weights('model100.h5')
        cp_cb = ModelCheckpoint(filepath='model{epoch:03d}.h5', verbose=1, save_best_only=True)
        dirc = path + "/ips/PSPnet2/log"
        tb_cb = TensorBoard(log_dir=dirc)
        #lrs_cb = LearningRateScheduler(lambda ep: float(1e-3 / 3 ** (ep * 4 // epochs)))
        history = model.fit(
            X_train,
            Y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_test, Y_test),
            callbacks=[tb_cb,cp_cb],
            shuffle=True
        )
        os.chdir(path + "/ips/PSPnet2/weights/%d"%d_num)
        model.save_weights('model%d.h5'%epochs)
        X_train = []
        y_train = []
        y_test = []


        Yp = model.predict(X_test)
        X_test=[]

        os.chdir(path + "/ips/PSPnet2/data/ips/dataset")
        #testdata
        f = open("test_data%d.txt" %(d_num))
        lines = f.readlines()
        f.close()

        t1 = time.time()

        print("------Start Segmentation------")
        for X in range(0,len(lines)):
            slice = lines[X]
            name = slice[13:22]
            pre_label = np.zeros((hight,width,3))

            for y in range(0,hight-size+1,s):
                for x in range(0,width-size+1,s):
                    patch = np.ones((size,size,3))
                    patch[:,:,0] = patch[:,:,0]*Yp[num,:,:,0]
                    patch[:,:,1] = patch[:,:,1]*Yp[num,:,:,1]
                    patch[:,:,2] = patch[:,:,2]*Yp[num,:,:,2]

                    pre_label[y:y+size,x:x+size,:] += patch
                    num += 1

            for i in range(hight):
                for j in range(width):
                    if pre_label[i,j,0] < pre_label[i,j,1]:
                        pre_label[i,j,0] = 0
                    else:
                        pre_label[i,j,1] = 0

                    if pre_label[i,j,0] < pre_label[i,j,2]:
                        pre_label[i,j,0] = 0
                    else:
                        pre_label[i,j,2] = 0

                    if pre_label[i,j,1] < pre_label[i,j,2]:
                        pre_label[i,j,1] = 0
                    else:
                        pre_label[i,j,2] = 0


                    if pre_label[i,j,0] != 0:
                        pre_label[i,j,0] = 1
                    if pre_label[i,j,1] != 0:
                        pre_label[i,j,1] = 1
                    if pre_label[i,j,2] != 0:
                        pre_label[i,j,2] = 1
            os.chdir("/home/sora/project/ips/PSPnet2/result/%d"%d_num)
            sp.misc.imsave(name + 'PNG',pre_label[:,:,[2,1,0]])
            print("Finish %s"%name)
        # 処理後の時刻
        t2 = time.time()

            # 経過時間を表示
        elapsed_time = t2-t1
        print(f"経過時間：{elapsed_time}")
        patch=[]
        pre_label=[]
        Yp=[]
