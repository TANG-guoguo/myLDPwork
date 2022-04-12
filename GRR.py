from numpy import random
import math

def Client(epsilon,d,data):
    p= (math.pow(math.e, epsilon))/(math.pow(math.e, epsilon) + d - 1)
    q= 1/(math.pow(math.e, epsilon) + d - 1)
    dice = random.random()
    if dice<p-q:
        return data
    else:
        return random.randint(0,d-1)


def Server(epsilon,d,pdatalist):
    p= (math.pow(math.e, epsilon))/(math.pow(math.e, epsilon) + d - 1)
    q= 1/(math.pow(math.e, epsilon) + d - 1)
    n=len(pdatalist)
    B=[0 for i in range(0,d)]
    for d in pdatalist:
        B[d]=B[d]+1
    #print(B)
    #print(math.pow(math.e, epsilon))
    print(p,q)
    fv=[]
    for b in B:
        fv.append(((b/n-q)-q)/(p-q))
    print(fv)#全部回答的频率统计
    return fv
    

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
d = 1024 #回答种类数
x_true = random.zipf(a=1.1, size=50000)
datalist = x_true[x_true<d]#select size<d data upload to server
print(len(datalist))
print(datalist)
pdatalist=[]
for i in datalist:
    pdata = Client(epsilon,d,i)
    pdatalist.append(pdata)
print(pdatalist)

FV = Server(epsilon,d,pdatalist)
MSE(datalist,FV,d)











