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
model_save_path = './checkpoint/LeNet_GTA5_01.ckpt'

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
class LeNet5(Model):
    def __init__(self):
        super(LeNet5,self).__init__()
        self.c1 =Conv2D(filters=6,kernel_size=(5,5),activation='sigmoid')
        self.p1 =MaxPool2D(pool_size=(2,2),strides=2)

        self.c2 =Conv2D(filters=16,kernel_size=(5,5),activation='sigmoid')
        self.p2 =MaxPool2D(pool_size=(2,2),strides=2)

        self.flatten = Flatten()
        self.f1 =Dense(120,activation='sigmoid')
        self.f2= Dense(84, activation='sigmoid')
        self.f3 = Dense(10, activation='softmax')

    def call(self,x):
        x =self.c1(x)
        x = self.p1(x)
        x = self.c2(x)
        x = self.p2(x)
        x = self.flatten(x)
        x = self.f1(x)
        x = self.f2(x)
        y = self.f3(x)
        return y

model =LeNet5()
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



