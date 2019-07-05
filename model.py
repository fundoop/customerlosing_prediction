# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 10:02:18 2019

@author: fz
"""

import pandas as pd
from random import shuffle
from itertools import combinations
from time import mktime,strptime,time


from keras.models import Sequential #导入神经网络初始化函数
from keras.layers.core import Dense, Dropout #导入神经网络层函数、激活函数
from keras.layers import advanced_activations
from keras import backend as K
from sklearn.metrics import roc_curve #导入ROC曲线函数
import matplotlib.pyplot as plt
# Function to create model, required for KerasClassifier
def data_regular(data):
    mod_list=['机构编码','大于历史','降幅',
            '是否流失'
            ]
    for i in list(data.head()):
        if i not in mod_list:
            mad=max(data[i])
            mid=min(data[i])
            data[i]=data[i].apply(lambda x:(mad-x)/(mad-mid))
    return data

def data_imp(datafile):
    data=pd.read_excel(datafile)   
    return data
  
def roc_plot(y, yp):
    fpr, tpr, thresholds = roc_curve(y, yp, pos_label=1)
    plt.plot(fpr, tpr, linewidth=2, label = 'ROC of LM') #作出ROC曲线
    plt.xlabel('False Positive Rate') #坐标轴标签
    plt.ylabel('True Positive Rate') #坐标轴标签
    plt.ylim(0,1.05) #边界范围
    plt.xlim(0,1.05) #边界范围
    plt.legend(loc=4) #图例
    return plt #显示作图结果
def cm_plot(y, yp):
    from sklearn.metrics import confusion_matrix #导入混淆矩阵函数
    cm = confusion_matrix(y, yp) #混淆矩阵
    import matplotlib.pyplot as plt #导入作图库
    plt.matshow(cm, cmap=plt.cm.Greens) #画混淆矩阵图，配色风格使用cm.Greens，更多风格请参考官网。
    plt.colorbar() #颜色标签

    for x in range(len(cm)): #数据标签
        for y in range(len(cm)):
            plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='center')

    plt.ylabel('True label') #坐标轴标签
    plt.xlabel('Predicted label') #坐标轴标签
    return plt

def model(train,first,res):
    anal_dim=len(train[1,:])-2
    netfile = 'net.model' #构建的神经网络模型存储路径
#函数式
#    from keras.layers import Input
#    from keras.models import Model
#    inputs = Input(shape=(anal_dim,))
#    #所有的模型都是可调用的，就像层一样
#    # a layer instance is callable on a tensor, and returns a tensor
#    x = Dense(50, activation='relu')(inputs)
#    x = Dense(30, activation='relu')(x)
#    x = Dense(10, activation='relu')(x)
#    #得到输出的张量prediction
#    predictions = Dense(1, activation='linear')(x)
#    # This creates a model that includes
#    # the Input layer and three Dense layers
#    #用model生成模型
#    net = Model(inputs=inputs, outputs=predictions)
#    #编译模型，指定优化参数、损失函数、效用评估函数
#    net.compile(optimizer='adam',
#                  loss='mape',
#                  metrics=['accuracy'])    

    net = Sequential() #建立神经网络
    #kernel_initializer='uniform',
    net.add(Dense(input_dim = anal_dim, units = 20,activation=K.relu))
    #net.add(Dropout(0.1))
    net.add(Dense(input_dim = 20, units = 5,activation=K.softsign))
    #net.add(Dropout(0.1))
    net.add(Dense(input_dim = 5, units = 1,activation=K.sigmoid))
    from keras.callbacks import TensorBoard
    net.compile(loss = 'binary_crossentropy', optimizer = 'adam') #编译模型，使用adam方法求解
    
    net.fit(train[:,first:res], train[:,res], epochs=100, batch_size=64,callbacks=[TensorBoard(log_dir='./logs', batch_size=32, write_graph=True, write_grads=True, write_images=True)]) #训练模型，循环1000次,verbose=0
    net.save_weights(netfile) #保存模型
    return net

def result(dataa,datab):
    tmp=[0,0,0,0]
    for i in range(len(dataa)):
        if dataa[i]==1 and datab[i]==1:tmp[0]+=1
        elif dataa[i]==1 and datab[i]==0:tmp[1]+=1
        elif dataa[i]==0 and datab[i]==0:tmp[2]+=1
        elif dataa[i]==0 and datab[i]==1:tmp[3]+=1
    return tmp

def main(data,head):
    data = data[head].values
    shuffle(data)
    p = 0.8 #设置训练数据比例
    train = data[:int(len(data)*p),:]
    test = data[int(len(data)*p):,:]
    anal_dim=len(head)-1
    #构建LM神经网络模型
    from keras.models import Sequential #导入神经网络初始化函数
    from keras.layers.core import Dense, Activation #导入神经网络层函数、激活函数
    
    netfile = 'net.model' #构建的神经网络模型存储路径
    
    net = Sequential() #建立神经网络
    net.add(Dense(input_dim = anal_dim, units = 20)) #添加输入层（3节点）到隐藏层（10节点）的连接
    net.add(Activation('relu')) #隐藏层使用relu激活函数
    net.add(Dense(input_dim = 20, units = 1)) #添加隐藏层（10节点）到输出层（1节点）的连接
    net.add(Activation(advanced_activations.LeakyReLU(alpha=0.3))) #输出层使用sigmoid激活函数
    net.compile(loss = 'binary_crossentropy', optimizer = 'adam') #编译模型，使用adam方法求解
    
    net.fit(train[:,:anal_dim], train[:,anal_dim].astype(int), epochs=100, batch_size=32,verbose=0) #训练模型，循环1000次
    net.save_weights(netfile) #保存模型
    
    predict_result = net.predict_classes(test[:,:anal_dim]).reshape(len(test)) #预测结果变形
    print(head,'%.6f'%(sum(predict_result==test[:,anal_dim].astype(int))/len(predict_result)))
    '''这里要提醒的是，keras用predict给出预测概率，predict_classes才是给出预测类别，而且两者的预测结果都是n x 1维数组，而不是通常的 1 x n'''
    return [str(head)]+result(predict_result,test[:,anal_dim].astype(int))

def all_var():
    data = pd.read_excel('continuous20181227.xlsx')
    data = data_regular(data)
    head=[]
    for i in range(1):
        for j in combinations(list(data.head())[1:-1], i+1):
        	head.append(list(j)+['是否流失'])
    res=[['因子','11','10','00','01']]
    for i in head:
        res.append(main(data,i))
    tmp=pd.DataFrame(res[1:],columns=res[0])
    tmp.to_excel('result.xlsx')

if __name__ == '__main__':
    time1=time()
    #all_var()
    train_data = data_imp('continuous20181227.xlsx')
    head=train_data.head()
    train_data=data_regular(train_data).values
    shuffle(train_data) 
    p=0.8
    first=1
    res=-1
    train = train_data[:int(len(train_data)*p),:]
    test = train_data[int(len(train_data)*p):,:]

    net=model(train,first,res)
    predict_result = net.predict_classes(test[:,first:res]).reshape(len(test)) #预测结果变形
    print('%.6f'%(sum(predict_result==test[:,res].astype(int))/len(predict_result)))
    tmp=test.tolist()
    for i in range(len(tmp)):tmp[i].append(predict_result[i])
    tmp=pd.DataFrame(tmp,columns=list(head)+['结果'])
    tmp.to_excel('result.xlsx')
    cm_plot(predict_result,test[:,res].astype(int)).show() #显示混淆矩阵可视化结果
    roc_plot(net.predict_classes(test[:,first:res]).reshape(len(test)),test[:,res].astype(int)).show()
    print(time()-time1)
