from PIL import ImageGrab
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,MaxPool2D,Dropout,Flatten,Dense
import cv2 as cv
import time
from getkeys import key_check
from directkeys import PressKey,ReleaseKey,W,A,S,D #把键盘操作的包导入
#模型保存路径
model_save_path = './checkpoint/VGGNet_GTA5_01.ckpt'

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
class VGG16(Model):
    def __init__(self):
        super(VGG16, self).__init__()
        self.c1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')  # 卷积层1
        self.b1 = BatchNormalization()  # BN层1
        self.a1 = Activation('relu')  # 激活层1
        self.c2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', )
        self.b2 = BatchNormalization()  # BN层1
        self.a2 = Activation('relu')  # 激活层1
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d1 = Dropout(0.2)  # dropout层

        self.c3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.b3 = BatchNormalization()  # BN层1
        self.a3 = Activation('relu')  # 激活层1
        self.c4 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.b4 = BatchNormalization()  # BN层1
        self.a4 = Activation('relu')  # 激活层1
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d2 = Dropout(0.2)  # dropout层

        self.c5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b5 = BatchNormalization()  # BN层1
        self.a5 = Activation('relu')  # 激活层1
        self.c6 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b6 = BatchNormalization()  # BN层1
        self.a6 = Activation('relu')  # 激活层1
        self.c7 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b7 = BatchNormalization()
        self.a7 = Activation('relu')
        self.p3 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d3 = Dropout(0.2)

        self.c8 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b8 = BatchNormalization()  # BN层1
        self.a8 = Activation('relu')  # 激活层1
        self.c9 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b9 = BatchNormalization()  # BN层1
        self.a9 = Activation('relu')  # 激活层1
        self.c10 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b10 = BatchNormalization()
        self.a10 = Activation('relu')
        self.p4 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d4 = Dropout(0.2)

        self.c11 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b11 = BatchNormalization()  # BN层1
        self.a11 = Activation('relu')  # 激活层1
        self.c12 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b12 = BatchNormalization()  # BN层1
        self.a12 = Activation('relu')  # 激活层1
        self.c13 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b13 = BatchNormalization()
        self.a13 = Activation('relu')
        self.p5 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d5 = Dropout(0.2)

        self.flatten = Flatten()
        self.f1 = Dense(512, activation='relu')
        self.d6 = Dropout(0.2)
        self.f2 = Dense(512, activation='relu')
        self.d7 = Dropout(0.2)
        self.f3 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.c3(x)
        x = self.b3(x)
        x = self.a3(x)
        x = self.c4(x)
        x = self.b4(x)
        x = self.a4(x)
        x = self.p2(x)
        x = self.d2(x)

        x = self.c5(x)
        x = self.b5(x)
        x = self.a5(x)
        x = self.c6(x)
        x = self.b6(x)
        x = self.a6(x)
        x = self.c7(x)
        x = self.b7(x)
        x = self.a7(x)
        x = self.p3(x)
        x = self.d3(x)

        x = self.c8(x)
        x = self.b8(x)
        x = self.a8(x)
        x = self.c9(x)
        x = self.b9(x)
        x = self.a9(x)
        x = self.c10(x)
        x = self.b10(x)
        x = self.a10(x)
        x = self.p4(x)
        x = self.d4(x)

        x = self.c11(x)
        x = self.b11(x)
        x = self.a11(x)
        x = self.c12(x)
        x = self.b12(x)
        x = self.a12(x)
        x = self.c13(x)
        x = self.b13(x)
        x = self.a13(x)
        x = self.p5(x)
        x = self.d5(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d6(x)
        x = self.f2(x)
        x = self.d7(x)
        y = self.f3(x)
        return y


model =VGG16()
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



