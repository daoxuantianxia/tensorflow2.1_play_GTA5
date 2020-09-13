from PIL import ImageGrab
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,MaxPool2D,Dropout,Flatten,Dense,GlobalAveragePooling2D
import cv2 as cv
import time
from getkeys import key_check
from directkeys import PressKey,ReleaseKey,W,A,S,D #把键盘操作的包导入
#模型保存路径
model_save_path = './checkpoint/resNet_GTA5_01.ckpt'

#定义键盘操作
def turn_left():
    print('----左转弯----')
    PressKey(W) #按下键盘W
    PressKey(A) #按下键盘A
    ReleaseKey(S) #松开键盘S
    ReleaseKey(D) #松开键盘D
def turn_right():
    print('----右转弯----')
    PressKey(W)
    PressKey(D)
    ReleaseKey(S)
    ReleaseKey(A)
def straight():
    print('----直线行驶----')
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
def back_left():
    print('----左后转弯----')
    PressKey(S)
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
def back_right():
    print('----右后转弯----')
    PressKey(S)
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)
def back():
    print('----后退----')
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
def right():
    print('----右行----')
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(S)
def left():
    print('----左行----')
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(W)
    ReleaseKey(S)
#复现神经网络
class ResnetBlock(Model):

    def __init__(self, filters, strides=1, residual_path=False):
        super(ResnetBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b2 = BatchNormalization()

        # residual_path为True时，对输入进行下采样，即用1x1的卷积核做卷积操作，保证x能和F(x)维度相同，顺利相加
        if residual_path:
            self.down_c1 = Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False)
            self.down_b1 = BatchNormalization()

        self.a2 = Activation('relu')

    def call(self, inputs):
        residual = inputs  # residual等于输入值本身，即residual=x
        # 将输入通过卷积、BN层、激活层，计算F(x)
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)

        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)

        out = self.a2(y + residual)  # 最后输出的是两部分的和，即F(x)+x或F(x)+Wx,再过激活函数
        return out


class ResNet18(Model):

    def __init__(self, block_list, initial_filters=64):  # block_list表示每个block有几个卷积层
        super(ResNet18, self).__init__()
        self.num_blocks = len(block_list)  # 共有几个block
        self.block_list = block_list
        self.out_filters = initial_filters
        self.c1 = Conv2D(self.out_filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.blocks = tf.keras.models.Sequential()
        # 构建ResNet网络结构
        for block_id in range(len(block_list)):  # 第几个resnet block
            for layer_id in range(block_list[block_id]):  # 第几个卷积层

                if block_id != 0 and layer_id == 0:  # 对除第一个block以外的每个block的输入进行下采样
                    block = ResnetBlock(self.out_filters, strides=2, residual_path=True)
                else:
                    block = ResnetBlock(self.out_filters, residual_path=False)
                self.blocks.add(block)  # 将构建好的block加入resnet
            self.out_filters *= 2  # 下一个block的卷积核数是上一个block的2倍
        self.p1 = tf.keras.layers.GlobalAveragePooling2D()
        self.f1 = tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y


model = ResNet18([2, 2, 2, 2])
cp_callback =tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,save_weights_only=True,save_best_only=True)    #启用回调函数
model.load_weights(model_save_path)    #载入已训练好的模型



def ceshi():
    #倒计时
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    zhanting = False
    while True:
        if not zhanting:
            screen =ImageGrab.grab(bbox=(0,30,800,620)) #抓取屏幕
            screen =np.array(screen) #转成矩阵
            screen =cv.cvtColor(screen,cv.COLOR_BGR2RGB) #转成RGB图像
            screen =cv.resize(screen,(32,32)) #把图片转成32*32的图像
            screen =screen/255.0 #归一化
            screen =screen[tf.newaxis, ...] #增加一个维度
            predict =model.predict(screen) #把图片输入模型
            prd =tf.argmax(predict, axis=1) #选出概率最高的数值
            #print(prd[0])
            #prd=tf.print(prd) #打印模型预测的结果
            #print(type(prd))
            if prd == 0:
                turn_left()
            elif prd == 1:
                turn_right()
            elif prd == 2:
                back_left()
            elif prd == 3:
                back_right()
            elif prd== 4:
                straight()
            elif prd == 5:
                left()
            elif prd == 6:
                back()
            elif prd ==7:
                right()
            else:
                pass
        keys = key_check() #获取键盘输入
        if 'T' in keys: #定义暂停 T键
            if zhanting:
                zhanting =False
                time.sleep(1)
            else:
                zhanting=True
                ReleaseKey(W)
                ReleaseKey(A)
                ReleaseKey(D)
                time.sleep(1)

start =ceshi()



