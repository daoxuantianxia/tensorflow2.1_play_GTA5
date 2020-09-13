import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import Model
from tensorflow import keras
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,MaxPool2D,Dropout,Flatten,Dense
np.set_printoptions(threshold=np.inf)

#########载入数据集
train =np.load('F:/re_train-data/train_data.npy',allow_pickle=True)
simple =np.load('F:/re_train-data/target.npy',allow_pickle=True)
x_train =train[:-5000]
y_train=simple[:-5000]
x_test =train[-5000:]
y_test=simple[-5000:]
x_train,x_test =x_train/255.0,x_test/255.0 #归一化

#建立Lenet神经网络
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
model.compile(optimizer='adam',#使用adam优化器
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),#使用SparseCategoricalCrossentropy损失函数
              metrics=['sparse_categorical_accuracy'])#准确率
#加入回调函数
checkpoint_savepath = './checkpoint/LeNet_GTA5_01.ckpt'
if os.path.exists(checkpoint_savepath + '.index'):
    print('______启用已经训练的模型______')
    model.load_weights(checkpoint_savepath)#加载已经训练的模型
#设置回调函数参数
cp_callback =tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_savepath,save_weights_only=True,save_best_only=True)
history =model.fit(x_train,y_train,batch_size=32,epochs=15,validation_data=(x_test,y_test),validation_freq=5,callbacks=[cp_callback])
model.summary()
#保存训练数据
file =open('./GTA5_weight.txt','w')
for v in model.trainable_variables:
    file.write(str(v.name)+'\n')
    file.write(str(v.shape)+'\n')
    file.write(str(v.numpy()) + '\n')
file.close()
#使用MAT显示图表
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
