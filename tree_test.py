from treelib import Tree, Node
import numpy as np

class Nodex(object):
    def __init__(self, frequency, divide_flag, count, interval):
        self.frequency = frequency
        self.divide_flag = divide_flag
        self.count = count
        self.interval = interval



domain_size = 1024
mytree = Tree()
mytree.create_node('Root', 'root', data=Nodex(1, True, 1, np.array([0, domain_size])))
j=0
for i in range(0,256):
    name = 'node-'+str(j*4)
    j+=1
    mytree.create_node(tag=name,identifier=name,parent='root',data=Nodex(1, True, 1, np.array([i*4, i*4+3])))
mytree.show(key=False)
mytree.create_node(tag='TEST',identifier='TEST',parent='root',data=Nodex(1, True, 1, np.array([5, 55])))
mytree.move_node('node-0','TEST')
mytree.show(key=False)
#print(mytree.all_nodes())
print(mytree.size())











