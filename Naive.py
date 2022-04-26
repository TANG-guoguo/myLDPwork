import numpy as np
import math
from numpy import random

def Client(epsilon,d,data):
    global pdatalist
    q = 1/(math.pow(math.e, epsilon) + 1)
    p = 0.5
    B=[0 for i in range(0,d)]  #对data进行一元编码
    B[data]=1
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
    for i in range(0,d):
        pdatalist[i]+=pdata[i]

def Server(epsilon,d,pdatalist):
    q = 1 / (math.pow(math.e, epsilon) + 1)
    p = 0.5
    n = 494740
    #print(p,q,n)
    C=pdatalist
    # for usr in pdatalist:
    #     for j in range(0,d):
    #         if usr[j]==1:
    #             C[j]=C[j]+1
    #print(C)
    fv=[]
    for i in range(0,d):
        fv.append((C[i]/n-q)/(p-q))
    print("估计频率:",fv)
    return fv




def  non_negativity(LST,k): #非负性处理
    while(True):
        positive_count = 0
        positive_num = 0
        flag=1
        for i in range(0, k):
            if LST[i] < 0:
                LST[i] = 0
                flag=0
            elif LST[i] > 0:
                positive_count += LST[i]
                positive_num += 1
        if flag==1:   #没有负频率
            print("非负性处理后：",LST)
            print(sum(LST))
            return LST
        #print("正值总和=", positive_count)
        x = positive_count - 1  # 总差值
        y = x / positive_num  # 平均差值
        for i in range(0, k):
            if LST[i] > 0:
                LST[i] -= y



def answer(qlist,fvlist,realfvlist):
    resultlist=[]
    realresultlist=[]
    for query in qlist:
        answertmp1 = sum(fvlist[query[0]:query[1]+1])
        answertmp2 = sum(realfvlist[query[0]:query[1]+1])
        resultlist.append(answertmp1)
        realresultlist.append(answertmp2)
    print("估计回答：",resultlist)
    print("真实回答：",realresultlist)
    #计算mse
    MSE=0
    for i in range(0,200):
        MSE+=(realresultlist[i]-resultlist[i])**2
    print("均方误差=",MSE/200)
    return MSE/200



def realfv(datalist,num,d):
    realcountlist=[0 for i in range(0,d)]
    for data in datalist:
        realcountlist[data]+=1
    realfvlist = [c/num for c in realcountlist]
    return realfvlist





def main_func(datalist, user_num, d, epsilon):

    time=0
    for data in datalist:
        Client(epsilon, d, data)
        time+=1
        if(time%10000==0):
            print(time)
    global pdatalist
    FV = Server(epsilon, d, pdatalist)
    #print(FV)
    NNFV=non_negativity(FV,len(FV))
    #print(NNFV)
    return NNFV

if __name__ == "__main__":
    repeat=1
    epsilon = 1  # Privacy budget
    d = 1024  # For simplicity, we use a dataset with d possible data items
    pdatalist = [0 for i in range(0,d)]
    #读取数据
    dataset = np.loadtxt("tangjiadataset.txt", int)
    print("用户数量：", len(dataset))  # 用户个数
    #print(dataset)
    datalist=[]
    for i in dataset:
        datalist.append(i-1)  ##############dataset1是i-1
    #datalist=datalist[:100000]
    print(datalist)
    print("用户数量：", len(datalist))  # 用户个数


    #读取查询
    query_interval_table = np.loadtxt("tjquery.txt", int)

    #计算真实分布频率
    realfvlist = realfv(datalist, len(datalist), d)
    print("真实分布",realfvlist)

    MSELIST=[]
    for i in range(0,repeat):
        LOWEST_NODE_FV = main_func(datalist, len(datalist), d, epsilon)
        MSE=answer(query_interval_table, LOWEST_NODE_FV, realfvlist)
        MSELIST.append(MSE)

    print("均方误差", MSELIST)
    print("最终均方误差",sum(MSELIST)/repeat)
