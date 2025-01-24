# -*- coding: UTF-8 -*-
import numpy as np

from gym.spaces import Space
from gym import logger

#实现了一个 n 维的 和为 1 的空间
class SumOne(Space):
    '''
    An n-dimensional sum-1 space. 

    '''

    def __init__(self, n):
        self.n = n
        super(SumOne, self).__init__((self.n,), np.int8)
    #生成符合该空间性质的随机样本
    def sample(self):
        vec = np.random.rand(self.n)  #生成一个包含 n 个随机浮点数的向量
        vec = vec/sum(vec)            #将向量中的值进行归一化，使得所有值的总和为 1
        return vec
    #检查输入向量 x 是否在该空间内
    def contains(self, x):
        return (all(x >= 0) & sum(x) <= 1) #判断 x 中所有元素是否都大于等于 0 并且整个向量的和是否小于等于 1

    def __repr__(self):
        return "SumOne({})".format(self.n)

    def __eq__(self, other):
        return isinstance(other, SumOne) and self.n == other.n
