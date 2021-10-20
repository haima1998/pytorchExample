##1.2 demo softmoid/relu/softmax shape

import numpy as np
from numpy import random as nr


X=np.random.rand(2,3)
def softmoid(x):
    return 1/(1+np.exp(-x))
def relu(x):
    return np.maximum(0,x)
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

print("X.shape:",X.shape)
print("softmoid(X).shape:",softmoid(X).shape)
print("relu(X).shape:",relu(X).shape)
print("softmax(X).shape:",softmax(X).shape)