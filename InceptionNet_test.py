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
model_save_path = './checkpoint/inceptionNet_GTA5_01.ckpt'

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
class ConvBNRelu(Model):
    def __init__(self, ch, kernelsz=3, strides=1, padding='same'):
        super(ConvBNRelu, self).__init__()
        self.model = tf.keras.models.Sequential([
            Conv2D(ch, kernelsz, strides=strides, padding=padding),
            BatchNormalization(),
            Activation('relu')
        ])

    def call(self, x):
        x = self.model(x, training=False) #在training=False时，BN通过整个训练集计算均值、方差去做批归一化，training=True时，通过当前batch的均值、方差去做批归一化。推理时 training=False效果好
        return x


class InceptionBlk(Model):
    def __init__(self, ch, strides=1):
        super(InceptionBlk, self).__init__()
        self.ch = ch
        self.strides = strides
        self.c1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
        self.c2_1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
        self.c2_2 = ConvBNRelu(ch, kernelsz=3, strides=1)
        self.c3_1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
        self.c3_2 = ConvBNRelu(ch, kernelsz=5, strides=1)
        self.p4_1 = MaxPool2D(3, strides=1, padding='same')
        self.c4_2 = ConvBNRelu(ch, kernelsz=1, strides=strides)

    def call(self, x):
        x1 = self.c1(x)
        x2_1 = self.c2_1(x)
        x2_2 = self.c2_2(x2_1)
        x3_1 = self.c3_1(x)
        x3_2 = self.c3_2(x3_1)
        x4_1 = self.p4_1(x)
        x4_2 = self.c4_2(x4_1)
        # concat along axis=channel
        x = tf.concat([x1, x2_2, x3_2, x4_2], axis=3)
        return x


class Inception10(Model):
    def __init__(self, num_blocks, num_classes, init_ch=16, **kwargs):
        super(Inception10, self).__init__(**kwargs)
        self.in_channels = init_ch
        self.out_channels = init_ch
        self.num_blocks = num_blocks
        self.init_ch = init_ch
        self.c1 = ConvBNRelu(init_ch)
        self.blocks = tf.keras.models.Sequential()
        for block_id in range(num_blocks):
            for layer_id in range(2):
                if layer_id == 0:
                    block = InceptionBlk(self.out_channels, strides=2)
                else:
                    block = InceptionBlk(self.out_channels, strides=1)
                self.blocks.add(block)
            # enlarger out_channels per block
            self.out_channels *= 2
        self.p1 = GlobalAveragePooling2D()
        self.f1 = Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y


model = Inception10(num_blocks=2, num_classes=10)
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



