from numpy import random
import math
import xxhash

def Client(epsilon,data,i):
    g = int(math.pow(math.e, epsilon) + 1)
    p = math.pow(math.e, epsilon)/(math.pow(math.e, epsilon)+g-1)
    q = 1/(math.pow(math.e, epsilon)+g-1)
    #print(g)
    vi = (xxhash.xxh32(str(data), seed=i).intdigest() % g)
    dice = random.random()
    if dice < p - q:
        return vi
    else:
        return random.randint(0, g-1)

def Server(epsilon,pdatalist,d):
    g = int(math.pow(math.e, epsilon) + 1)
    p = math.pow(math.e, epsilon) / (math.pow(math.e, epsilon) + g - 1)
    q = 1 / (math.pow(math.e, epsilon) + g - 1)
    n = len(pdatalist)
    pp = p
    qq = 1/(math.pow(math.e, epsilon) + 1)
    C=[0 for i in range(0,d)]
    for i in range(0,n):
        for j in range(0,d):
            if pdatalist[i]==(xxhash.xxh32(str(j), seed=i).intdigest() % g):
                C[j]+=1
    fv=[]
    for c in C:
        fv.append((c/n-qq)/(pp-qq))
    print("fv=",fv) #频率
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



x_true = random.zipf(a=1.1, size=50000)
epsilon = 5# Privacy budget of 3
d = 1024 # For simplicity, we use a dataset with d possible data items
datalist = x_true[x_true<d]#select size<d data upload to server
print("n=",len(datalist))
print("g=",int(math.pow(math.e, epsilon) + 1))
pdatalist=[]
i=0
for data in datalist:
    pdata = Client(epsilon,data,i)
    i+=1
    pdatalist.append(pdata)
#print(pdatalist) #扰动后数据
FV=Server(epsilon,pdatalist,d)
MSE(datalist,FV,d)





