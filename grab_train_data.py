from PIL import ImageGrab
import cv2 as cv
import numpy as np
import os
import time
from getkeys import key_check #获取键盘输入的包

#collect_number = 1
#file_name ='trian_data_{}.npy'.format(collect_number)
def keys_output(keys): #定义键盘输入和标签的关系
    output =None
    if 'W' in keys and 'A' in keys:
        output = 0   #wa =0 左转弯
    elif 'W' in keys and 'D' in keys:
        output = 1   #wd =1 右转弯
    elif 'S' in keys and 'A' in keys:
        output = 2   #sa =2 左倒车
    elif 'S' in keys and 'D' in keys:
        output = 3   #sd =3  右倒车
    elif 'W' in keys:
        output = 4   #w =4    直行
    elif 'A' in keys:
        output = 5   #a =5    左行
    elif 'S' in keys:
        output = 6    #s =6    刹车/后退
    elif 'D' in keys:
        output = 7    #d =7    右行
    else:
        output =8    #nokey =8 （不做任何操作）
    return output

def main():
    file_name = 'trian_data_00.npy'
    train_data=[]
    collect_number = 1
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    #last_time = time.time()
    print('开始采集数据集啦！！！！')
    paused = False
    while True:
        if not paused:
            #                                   1020,790
            screen =ImageGrab.grab(bbox=(0,30,800,620))
            screen =np.array(screen)    #x,y,  a,   b
            screen=cv.cvtColor(screen,cv.COLOR_BGR2RGB) #把图片转成RGB图像
            # cv.imshow('srcreen',screen)
            # if cv.waitKey(25) & 0xFF == ord('q'):
            #     cv.destroyAllWindows()
            #     break
            #screen=cv.resize(screen,(32,32)) #调整图片尺寸成32*32
            keys=key_check() #获取键盘输入
            print(keys)
            output=keys_output(keys)
            print(output)
            train_data.append([screen,output]) #将图片和标签存入列表
            #print('time pass:',time.time()-last_time)
            last_time =time.time()
            print(len(train_data))
            if len(train_data) == 1000 :
                np.save(file_name,train_data)
                print('已保存1000张样本')
                train_data =[]
                collect_number += 1
                file_name ='trian_data_0{}.npy'.format(collect_number)
                #collect_number += 1
        keys = key_check()
        #设置暂停键
        if 'T' in keys:
            if paused:
                paused = False
                print('已开始!')
                time.sleep(1)
            else:
                print('已暂停!')
                paused = True
                time.sleep(1)


start=main()