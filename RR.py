import numpy as np
import math
import random

def Client(epsilon,data):
    p=(math.pow(math.e,epsilon))/(math.pow(math.e,epsilon)+1)
    #print(p)
    result=[]
    for i in range(len(data)):#扰动
        dice = random.random()
        if(data[i]==0):
            if(dice<p): #p的概率不变
                result.append(0)
            else: #1-p的概率翻转
                result.append(1)
        else:
            if(dice<p): #p的概率不变
                result.append(1)
            else: #1-p的概率翻转
                result.append(0)
    return result

def Server(epsilon,pdata):
    p=(math.pow(math.e,epsilon))/(math.pow(math.e,epsilon)+1)
    n0 = 0
    n1 = 0
    for i in pdata:
        if i==0:
            n0+=1
        else:
            n1+=1
    n=n0+n1
    pi_1=(p-1)/(2*p-1)+n1/((2*p-1)*n)
    pi_0=(p-1)/(2*p-1)+n0/((2*p-1)*n)
    print("1频率:",pi_1)
    print("0频率:",pi_0)
    return [pi_0,pi_1]

def MSE(datalist,FV,d):
    n=len(datalist)
    REALCOUNT=[0 for i in range(0,d)]
    for data in datalist:
        REALCOUNT[data]+=1
    print(REALCOUNT)
    print([n*i for i in FV])
    mse=0
    for i in range(0,d):
        mse+=(REALCOUNT[i]-FV[i]*n)**2
    mse/=d
    print("MSE=",mse)

    

epsilon = 5
data = np.concatenate(([0]*20000, [1]*9500))
perturbdata = Client(epsilon,data)
FV = Server(epsilon,perturbdata)
MSE(data,FV,2)




























