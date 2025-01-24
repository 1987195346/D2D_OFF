# -*- coding: UTF-8 -*-
import numpy as np

from gym.spaces import Space
from gym import logger

# OneHot是一个n维的独热编码空间
class OneHot(Space):
    '''
    An n-dimensional onehot space. 

    '''

    def __init__(self, n):
        self.n = n
        super(OneHot, self).__init__((self.n,), np.int8)
    #生成一个随机的独热编码向量
    def sample(self):
        vec = [0]*self.n   #创建一个长度为n的全零列表
        vec[np.random.randint(n)] = 1   #随机选择一个索引，将该位置的值设为1，从而生成独热编码的向量
        return vec
    #检查输入x是否为有效的独热编码向量
    def contains(self, x):
        if isinstance(x, list):
            x = np.array(x)  # Promote list to array for contains check
        return (sum(x == 0) == self.n-1 & sum(x == 1) == 1)  #检查x中是否恰好有n-1个0和1个1来判断其是否合法

    def __repr__(self):
        return "OneHot({})".fortensorboardmat(self.n)

    def __eq__(self, other):
        return isinstance(other, OneHot) and self.n == other.n
