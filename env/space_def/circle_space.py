# -*- coding: UTF-8 -*-
import numpy as np

from gym.spaces import Space
from gym import logger


# 定义圆类，继承父类Space
class Circle(Space):
    def __init__(self, r):
        assert r >= 0
        self.r = r
        super(Circle, self).__init__((), np.float)

    #在圆内随机生成一个点
    def sample(self):
        x = self.r*2*(np.random.rand()-0.5)
        y = np.sqrt(self.r**2-x**2)*2*(np.random.rand()-0.5)
        return (x, y)
    #检查一个点是否在圆内
    def contains(self, x):
        return (x[0]**2+x[1]**2 <= self.r**2)
    #定义类的字符串表示形式
    def __repr__(self):
        return "Circle(%f)" % self.r
    #比较两个 Circle 对象是否相等
    def __eq__(self, other):
        return isinstance(other, Circle) and self.r == other.r

# 定义离散圆类，继承父类Space
class Discrete_Circle(Space):
    def __init__(self, r):
        assert r >= 0
        self.r = r
        super(Discrete_Circle, self).__init__((), np.float)

    #生成在离散圆内部的随机点
    def sample(self):
        x = np.random.randint(-self.r, self.r+1)
        y = np.random.randint(-np.sqrt(self.r**2-x**2),
                              np.sqrt(self.r**2-x**2)+1)
        return (x, y)

    def contains(self, x):
        return (x[0]**2+x[1]**2 <= self.r**2)

    def __repr__(self):
        return "Discrete_Circle(%f)" % self.r

    def __eq__(self, other):
        return isinstance(other, Discrete_Circle) and self.r == other.r
