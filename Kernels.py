import numpy as np

global kernel_d
global kernel_sig

def kernel(u, v):
    return np.dot(u, v)

def kernel1(u, v):
    return np.dot(u, v) + 1

def kerneld(u, v):
    return (np.dot(u, v) + 1)**kernel_d

def kernelg(u, v):
    return np.exp(-(np.linalg.norm(u - v)**2) / (2 * kernel_sig**2))

