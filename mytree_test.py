import treelib
import numpy as np
import math
from treelib import Tree, Node
import pandas as pd
import os
from func_module import freqoracle
from func_module import errormetric
from func_module import realfreq
from numpy import random

class Nodex(object):
    def __init__(self, frequency, ffrequency,divide_flag, count, interval):
        self.frequency = frequency
        self.frequency = ffrequency
        self.divide_flag = divide_flag
        self.count = count
        self.interval = interval




def get_ZIPF(a,size,d):
    x_true = random.zipf(a, size)  # （无0数据）#############zipf分布
    datalist = x_true[x_true <= d]  # select size<d data upload to server(无0数据) #获取用户真实数据????????????这句为什么能这么写？？？？？？？？？
    return datalist

def get_UNIFORM(size,d):
    datalist = []
    for i in range(0,size):
        datalist.append(random.randint(1,d))
    return datalist

def get_NORMAL(size,d):
    datalist=[]
    for i in range(0,size):
        x = int(random.normal(d/2,200))
        if x>=1 and x<=d :
            datalist.append(x)
    #print(datalist)
    return datalist

def get_LPLS(size,d):
    datalist=[]
    for i in range(0,size):
        x = int(random.laplace(d/2,500))
        if x>=1 and x<=d :
            datalist.append(x)
    #print(datalist)
    return datalist

def CUT_TO_INTERVAL(FBTree, CUT ,layer):
    INTERVAL=[]
    node_num = len(CUT)
    tmp=[]
    for i in range(0,node_num):
        name='L-'+str(layer)+'N-'+str(i)
        INTERVAL.append([FBTree[name].data.interval[0],FBTree[name].data.interval[1]])
    return INTERVAL


def tree_firstconstruction(FBTree,CUT,fvlist,layertag,start_size):
    node_num = len(CUT)
    for i in range(0,node_num):
        name='L-'+str(layertag)+'N-'+str(i)
        frequency = fvlist[i]
        left = CUT[i][0]
        right = CUT[i][-1]
        num_in_node = start_size
        FBTree.create_node(tag=name, identifier=name, parent='Root',data=Nodex(frequency,frequency, True, num_in_node, np.array([left, right])))
    #FBTree.show(key=False)
    #print("树总节点个数：",FBTree.size())


def tree_upconstruction(FBTree,CUT,fvlist,layertag):
    node_num = len(fvlist)
    for i in range(0,node_num):
        name = 'L-' + str(layertag) + 'N-' + str(i)
        frequency = fvlist[i]
        left_index = CUT[i][0]
        right_index = CUT[i][-1]
        lname='L-'+ str(layertag-1) + 'N-' + str(left_index)
        rname = 'L-' + str(layertag - 1) + 'N-' + str(right_index)
        left = FBTree[lname].data.interval[0]   #左边界
        right = FBTree[rname].data.interval[1]   #右边界
        num_in_node = len(CUT[i])
        FBTree.create_node(tag=name, identifier=name, parent='Root',data=Nodex(frequency,frequency, True, num_in_node, np.array([left, right])))
        for j in CUT[i]:  #为新节点安插子节点
            childname = 'L-'+ str(layertag-1) + 'N-' + str(j)
            FBTree.move_node(childname,name)



def Client(epsilon,CUT,CUT_num,data):
    q = 1/(math.pow(math.e, epsilon) + 1)
    p = 0.5
    B = []
    for i in range(0,CUT_num):    #对data进行一元编码
        if(data>=CUT[i][0] and data<=CUT[i][-1]):
            B.append(1)
        else:
            B.append(0)

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

def get_user_pdata(datalist,user_num,CUT,CUT_num,epsilon,num):
    print("本层用户个数=",num)
    now_user_num = user_num
    pdatalist=[]
    i=0
    while(i<num):
        randomindex = random.randint(0,now_user_num)
        data = datalist[randomindex]
        datalist.pop(randomindex)   #已经选过的数据从datalist里删除
        #np.delete(datalist,[randomindex])
        now_user_num-=1
        pdata = Client(epsilon,CUT,CUT_num,data)
        pdatalist.append(pdata)
        i+=1
    return pdatalist


def frequency_aggregation(epsilon,pdatalist,CUT_num):
    q = 1 / (math.pow(math.e, epsilon) + 1)
    p = 0.5
    n = len(pdatalist)
    #print(p,q,n)
    C=[0 for i in range(0,CUT_num)]
    for usr in pdatalist:
        for j in range(0,CUT_num):
            if usr[j]==1:
                C[j]=C[j]+1
    #print("频数统计结果",C)
    fv=[]
    for i in range(0,CUT_num):
        fv.append((C[i]/n-q)/(p-q))
    print("估计频率:",fv)
    print("估计频率之和",sum(fv))
    return fv


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
        if (fa + LST[a - 1]) <= (fa + LST[a + 1]):
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
            if(len(p)!=0):
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

        elif (i == k - 1):  # 边界处理?????????直接加入？？？？？？k-2自己跑了k-1跟上
            # 结束，收尾，退出
            FINALIST[-1].append(k-1)
            fcountlist[-1]+=LST[k - 1]
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
    assert FINALIST[-1][-1]==k-1
    return [FINALIST,fcountlist,len(fcountlist)]    #返回：划分、划分后频率、划分成的个数



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

def weighted_averaging(FBTree,NNFV,layer,user_scale_in_each_layer):   #对layer层节点进行加群平均处理，返回加权平均后频率
    lowlayer = layer-1
    node_num = len(NNFV)
    resultlist=[]
    varience_low = 4 * math.exp(epsilon) / (user_scale_in_each_layer[lowlayer] * (math.exp(epsilon) - 1) ** 2)   #下层方差
    varience_this = 4 * math.exp(epsilon) / (user_scale_in_each_layer[layer] * (math.exp(epsilon) - 1) ** 2)   #本层方差
    for i in range(0,node_num):
        name = 'L-' + str(layer) + 'N-' + str(i)
        fv = FBTree[name].data.frequency
        n = FBTree[name].data.count   #n为该节点所包含的子节点个数
        varience_child =n*varience_low    #子节点方差
        lambda1 = varience_child / (varience_this + varience_child)
        lambda2 = 1-lambda1
        newfv = lambda1*NNFV[i]+lambda2*fv
        resultlist.append(newfv)
        FBTree[name].data.frequency = newfv  #更新树中的节点频率
        FBTree[name].data.ffrequency = newfv  # 更新树中的节点频率


    return resultlist


def Mean_Consistency(FBTree):
    for node in FBTree.expand_tree(mode=Tree.WIDTH, sorting=False):
        vname = node
        if vname =='Root':
            continue
        else:   #非叶节点
            fv = FBTree[vname].data.frequency
            vpname = FBTree.parent(vname).tag    #当前节点v的父节点的名字
            vpB = FBTree[vpname].data.count
            fpv = FBTree[vpname].data.frequency
            fvsum = fv
            for u in FBTree.siblings(vname):
                fvsum += u.data.frequency
            #print(fvsum)
            newfv = fv + (fpv-fvsum)/vpB
            #print(newfv)
            FBTree[vname].data.ffrequency = newfv
    return

def answer(qlist,fvlist):
    resultlist=[]
    for query in qlist:
        answertmp=sum(fvlist[query[0]:query[1]+1])
        resultlist.append(answertmp)
    print(resultlist)






def main_func(datalist, user_num, d, epsilon, start_size):
    global tmp_h
    user_scale_in_each_layer=[]
    #对域进行初次分割
    CUTLIST = []
    NNFVLIST = []
    CUT = [[i,i+start_size-1] for i in range(0,d,start_size)]
    node_num = len(CUT) #本层节点个数

    #创建树
    FBTree = Tree()  #构建频率平衡树FBTree
    FBTree.create_node('Root', 'Root', data=Nodex(1,1, True, 1, np.array([0, d])))  #创建根节点

    #收集本层用户回答得到pdatalist
    pdatalist = get_user_pdata(datalist,user_num,CUT,node_num,epsilon,user_num//2)  #本层用户个数user_num//tmp_h
    user_scale_in_each_layer.append(len(pdatalist))  #记录该层用户数量
    #print(len(pdatalist))

    #聚合频率
    fv = frequency_aggregation(epsilon,pdatalist,node_num)
    NNFV = non_negativity(fv,node_num)
    NNFVLIST.append(NNFV)

    #构建初层树
    tree_firstconstruction(FBTree,CUT,NNFV,0,start_size)
    layertag=1
    #根据初层频率划分建上层树
    CUT, NNFV, node_num = merge_A(NNFV, node_num)
    CUTLIST.append(CUT)
    tree_upconstruction(FBTree, CUT, NNFV, layertag)
    while(len(CUT)>3):  #获得虚空树高
        CUT,NNFV,node_num = merge_A(NNFV,node_num)
        #tree_upconstruction(FBTree,CUT,NNFV,layertag)
        layertag+=1
        CUTLIST.append(CUT)
        NNFVLIST.append(NNFV)
    #FBTree.show(key=False)
    tmp_h = layertag   #更新树高(含根节点)   其实是虚空树高，上层理论上存在但其实没有变成节点
    assert tmp_h==len(CUTLIST)
    print("本次建树完毕，预计树高",tmp_h)

    #收集上层用户回答，调整树结构
    layer = 1
    FLAG=0
    while(True):
        print("正在对第",layer,"层回答…………")
        CUT = CUTLIST[layer-1]
        print('分割：', CUT)
        INTERVAL = CUT_TO_INTERVAL(FBTree,CUT,layer)
        print('分割转区间结果：',INTERVAL)

        # 收集本层用户回答得到pdatalist
        user_num=len(datalist)
        node_num=len(INTERVAL)
        print("剩余用户数量：",user_num)
        if(FLAG==1):
            user_num_thislayer = user_num
        else:
            user_num_thislayer = user_num// tmp_h
        pdatalist = get_user_pdata(datalist, user_num, INTERVAL, node_num, epsilon,user_num_thislayer)  # 本层用户个数user_num//tmp_h
        user_scale_in_each_layer.append(len(pdatalist))  #记录本层用户数量

        # 聚合频率
        fv = frequency_aggregation(epsilon, pdatalist, node_num)
        NNFV = non_negativity(fv, node_num)
        #加权平均处理
        NNFV = weighted_averaging(FBTree,NNFV,layer,user_scale_in_each_layer)
        print("加权后：",NNFV)

        if(FLAG==1):
            #print("CUTLIST=", CUTLIST)
            FBTree['Root'].data.count = len(NNFV)
            print("真的拜拜！")
            break
        #根据新频率构建上层树
        ######################
        CUT,NNFV,node_num = merge_A(NNFV,len(NNFV))
        CUTLIST = CUTLIST[0:layer]   #重构cutlist
        CUTLIST.append(CUT)
        tree_upconstruction(FBTree, CUT, NNFV, layer+1)
        layertag=layer+1
        if (len(CUT)<=3):   #不需要再向上合并
            print("CUTLIST=",CUTLIST)
            print("拜拜")
            FLAG=1

        while (len(CUT) > 3):  # 更新虚空树高
            CUT, NNFV, node_num = merge_A(NNFV, node_num)
            layertag += 1
            CUTLIST.append(CUT)
            NNFVLIST.append(NNFV)
        #FBTree.show(key=False)
        tmp_h = layertag  # 更新树高(含根节点)   其实是虚空树高，上层理论上存在但其实没有变成节点
        assert tmp_h == len(CUTLIST)
        print("本次建树完毕，预计树高", tmp_h)

        ######################
        layer+=1
    FBTree.show(key=False)

    #一致化处理
    print("开始一致化处理!")
    Mean_Consistency(FBTree)

    #拆到最底层
    LOWEST_NODE_FV=[]
    for i in range(0,d//start_size):
        name = 'L-' + str(0) + 'N-' + str(i)
        tmp=FBTree[name].data.ffrequency
        LOWEST_NODE_FV.append(tmp / 2)
        LOWEST_NODE_FV.append(tmp / 2)
    print("最终频率=",LOWEST_NODE_FV)
    print(len(LOWEST_NODE_FV))
    print("剩余用户",len(datalist))
    LOWEST_NODE_FV=non_negativity(LOWEST_NODE_FV, d)
    print("非负处理后最终频率=", LOWEST_NODE_FV)
    return LOWEST_NODE_FV













if __name__ == "__main__":
    epsilon = 1  # Privacy budget
    d = 1024  # For simplicity, we use a dataset with d possible data items
    start_size = 2    #初始划分粒度
    tmp_h = math.ceil(math.log2(d/start_size))    #当前预计树高
    #datalist = get_ZIPF(1.01,500000,d)    #用这个要改80行删除语句
    # datalist = get_UNIFORM(50000,d)
    #datalist = get_NORMAL(500000,d)     #datalist为真实数据集
    # datalist = get_LPLS(50000, d)
    #print("用户数量：", len(datalist))  # 用户个数

    # #存储datalist
    # with open("tangjiadataset.txt","w") as f:
    #     for tjdata in datalist:
    #         f.write(str(tjdata))
    #         f.write('\n')

    #读取数据
    dataset = np.loadtxt("tangjiadataset.txt", int)


    print("用户数量：", len(dataset))  # 用户个数
    print(dataset)
    datalist=[]
    for i in dataset:
        datalist.append(i)
    print(datalist)
    print("用户数量：", len(datalist))  # 用户个数


    #读取查询
    query_interval_table = np.loadtxt("tjquery.txt", int)





    LOWEST_NODE_FV = main_func(datalist,len(datalist),d,epsilon,start_size)
    answer(query_interval_table, LOWEST_NODE_FV)

















