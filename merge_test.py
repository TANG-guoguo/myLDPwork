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

def merge_A(LST,k): #输入含k个节点的估计频率序列LST，输出对LST的一个划分
    print("k=",k)
    fa=max(LST)  #fa记录最大频率
    a=LST.index(fa)  #a记录最大频率节点的下标
    print("最大频率节点下标：",a)
    print("最大频率：", fa)
    FINALIST = []
    flaglist = [0 for i in range(0, k)]
    if a==0 :
        F = fa + LST[1]
        flaglist[0] = 1
        flaglist[1] = 1
    elif a==1 :
        F = fa + LST[0]
        flaglist[0] = 1
        flaglist[1] = 1
    elif a==k-1 :
        F = fa + LST[k-2]
        flaglist[k-1] = 1
        flaglist[k-2] = 1
    elif a==k-2 :
        F = fa + LST[k-1]
        flaglist[k-1] = 1
        flaglist[k-2] = 1
    else :
        if (fa + LST[a - 1]) >= (fa + LST[a + 1]):
            F = (fa + LST[a-1])
            flaglist[a] = 1
            flaglist[a-1] = 1
        else :
            F = (fa + LST[a+1])
            flaglist[a] = 1
            flaglist[a+1] = 1
    print("目标频率F=",F)
    l = round(1/F)
    eta = 1/l    #eta为矫正后的目标频率
    print("矫正后目标频率及节点个数：",eta,"  ",l)
    fcountlist=[]
    fcount=0
    p=[]
    for i in range(0,k):
        if(flaglist[i]==2):
            continue
        if(flaglist[i]==1): #说明该节点为最大节点已经被结合
            # 如果有前一个区间，收尾前一个区间
            if(i!=0):
                FINALIST.append(p)
                fcountlist.append(fcount)
                p = []
                fcount = 0
            #已结合节点入新区间
            FINALIST.append([i,i+1])
            fcountlist.append(LST[i] + LST[i+1])
            flaglist[i+1] = 2 #flag为2表示跳过本次循环
            continue

        if(i==0):   #边界处理
            p.append(0)
            flaglist[0] = 1
            p.append(1)
            flaglist[1] = 1
            fcount += LST[0] + LST[1]
            flaglist[1] = 2 #flag为2表示跳过本次循环

        elif(i==k-2):  #边界处理?????????直接加入？？？？？？
            p.append(k-2)
            flaglist[k-2] = 1
            p.append(k-1)
            flaglist[k-1] = 1
            fcount += LST[k-2] + LST[k-1]
            # 结束，收尾，退出
            FINALIST.append(p)
            fcountlist.append(fcount)
            break

        else:   #非边界,尝试加入
            tmp = LST[i] + fcount
            if(tmp >= eta): #当前区间会满,临界
                if(tmp-eta)<=(eta-fcount): #加上极差小则加上当前节点
                    p.append(i)
                    flaglist[i] = 1
                    fcount = tmp
                    #达标，收尾
                    FINALIST.append(p)
                    fcountlist.append(fcount)
                    p=[]
                    fcount=0
                else:#不加该节点，该节点进入下一区间
                    # 达标，收尾
                    FINALIST.append(p)
                    fcountlist.append(fcount)
                    p = []
                    fcount = 0
                    p.append(i)#当前节点进下一区间
                    flaglist[i] = 1
                    fcount += LST[i]
            else:#当前区间不满，直接加入
                p.append(i)
                flaglist[i] = 1
                fcount = tmp
    print("划分：",FINALIST)
    print("划分后的频率：",fcountlist)
    print("区间个数",len(fcountlist))













##################main

epsilon = 1# Privacy budget of 3
d = 1024 # For simplicity, we use a dataset with d possible data items
x_true = random.zipf(a=1.1, size=50000)#（无0数据）#############zipf分布
datalist = x_true[x_true<=d]#select size<d data upload to server(无0数据) #获取用户真实数据
print("用户数量：",len(datalist))  #用户个数
pdatalist=[]
for data in datalist:     #获取用户扰动后数据
    pdata = Client(epsilon, d, data)
    pdatalist.append(pdata)
FV = Server(epsilon,d,pdatalist)    #server返回聚合频率结果
#MSE(datalist,FV,d)
merge_A(FV,len(FV))








