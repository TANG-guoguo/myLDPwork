from numpy import random
import math

def Client(epsilon,d,data):
    q = 1/(math.pow(math.e, epsilon) + 1)
    p = 0.5
    B=[0 for i in range(0,d)]  #对data进行一元编码
    B[data-1]=1
    pdata=[]
    for i in B:
        if i==1:
            dice1 = random.random()
            if dice1<p:
                pdata.append(1)
            else:
                pdata.append(0)
        else:
            dice2 = random.random()
            if dice2<q:
                pdata.append(1)
            else:
                pdata.append(0)
    return pdata

def Server(epsilon,d,pdatalist):
    q = 1 / (math.pow(math.e, epsilon) + 1)
    p = 0.5
    n = len(pdatalist)
    #print(p,q,n)
    C=[0 for i in range(0,d)]
    for usr in pdatalist:
        for j in range(0,d):
            if usr[j]==1:
                C[j]=C[j]+1
    #print(C)
    fv=[]
    for i in range(0,d):
        fv.append((C[i]/n-q)/(p-q))
    print("估计频率:",fv)
    return fv

def MSE(datalist,FV,d):
    n=len(datalist)
    REALCOUNT=[0 for i in range(0,d)]
    for data in datalist:
        REALCOUNT[data-1]+=1
    print("真实回答频数:",REALCOUNT)
    REALF=[i/n for i in REALCOUNT]
    print("真实回答频率:",REALF)
    print("估计频数:",[n*i for i in FV])
    mse=0
    for i in range(0,d):
        mse+=(REALF[i]-FV[i])**2
    mse/=d
    print("MSE=",mse)

x_true = random.zipf(a=1.1, size=100000)#（无0数据）
epsilon = 1# Privacy budget of 3
d = 1024 # For simplicity, we use a dataset with d possible data items
datalist = x_true[x_true<=d]#select size<d data upload to server(无0数据)
for i in range(0,200):
    print(datalist[i],end=' ')
print(" ")
print(len(datalist))
pdatalist=[]
for data in datalist:
    pdata = Client(epsilon, d, data)
    pdatalist.append(pdata)
FV = Server(epsilon,d,pdatalist)
MSE(datalist,FV,d)
