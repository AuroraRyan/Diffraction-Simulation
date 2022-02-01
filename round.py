import scipy
from scipy.integrate import nquad
from joblib import Parallel, delayed #另外一种并行计算库
import multiprocessing
import time
from tqdm import tqdm
import multiprocessing as mp

from numpy import exp
from math import sqrt
import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
from matplotlib import cm

X0 = 0 #mm 中心位置
Y0 = 0 #mm 
W0 = 5 #mm 光斑半径
z0 = 10 #mm 焦平面附近观察点
f0 = 10 #mm
Lambda = 780E-6 #mm
K = 2*math.pi/Lambda
WZ = W0 * sqrt(1+(Lambda * f0/(math.pi * (W0**2)) )**2) #mm
E = 2.718281828459

#计算物镜尺寸

NA = 0.55
D = 2*f0/(sqrt(1/(NA**2)-1)) #mm
x0, x1 = -D/2, D/2

options={'limit':1000} #积分设置最大迭代次数

def complex_quadrature(func, options): #scipy直接积分会抛弃复数部分，分开积
    def real_func(r,th):
        return np.real(func(r,th))
    def imag_func(r,th):
        return np.imag(func(r,th))
    real_integral = nquad(real_func, [[0, D], [0,2*np.pi]], opts=[options,options])
    imag_integral = nquad(imag_func, [[0, D], [0,2*np.pi]], opts=[options,options])
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

# def complex_quadrature(func, options): #scipy直接积分会抛弃复数部分，分开积
#     real_integral = nquad(func, [[0, D], [0,2*np.pi]], opts=[options,options])
#     return (real_integral[0], real_integral[1:])



#* exp(-K*(sqrt(r**2+f0**2)-f0)*1j) * exp(K/2/z0*(r**2)*1j) 
#上面这个因子按理说应该在积分里的，但会造成不收敛
#(z0-f0)/(2*f0*z0)*(r**2)     *exp(-1j * K *( - 1/8*(r**4/(f0**4)) )) 


def processInput(ele):
    def f(r, th):
        return W0/WZ * exp(-1j * K *( - 1/8*(r**4/(f0**4)) )) * exp(-((r*np.cos(th)-X0)**2+(r*np.sin(th)-Y0)**2)/WZ) * exp(-K/z0*(+ele[0]*r*np.cos(th)+ele[1]*r*np.sin(th))*1j) * r #ele[0]是焦平面x，ele[1]是焦平面y
    result, err1, err2 = complex_quadrature(f,options) #上面的因子是taylor展开后化简后剩的第一项（由于z0,f0相接近，本来还有一项约去了，这里只能展开sqrt(1+n),否则计算报错，D/2小于f接近可以认为能收敛，精度更高可以增加3阶、4阶项）
    complex_factor = exp((K*z0 + K/2/z0*(ele[0]**2+ele[1]**2)) * 1j) / (1j * Lambda * z0) #积分外面的相位因子
    fullend = result*complex_factor #焦平面上某点完整结果
    # phase_arr.append(np.arctan(fullend.imag/fullend.real)) #相位
    return abs(fullend)**2



if __name__=='__main__':


    upl = 0.003 #外圈界限
    downl = -0.003

    node1 = -0.0015 #x方向内圈界限
    node2 = 0.0015 

    node3 = -0.0015 #y方向内圈界限
    node4 = 0.0015 

    size1=30 #内圈选点数
    size2=10 #外圈选点数
    size3=10
    size = size1 + size2 + size3

    x_arr1 = np.linspace(node1, node2, size1) #外疏内密
    x_arr2 = np.linspace(downl, node1, size2)
    x_arr3 = np.linspace(node2, upl, size3)
    x_arr = np.concatenate((x_arr2,x_arr1,x_arr3), axis=0)




    y_arr1 = np.linspace(node3, node4, size1)
    y_arr2 = np.linspace(downl, node3, size2)
    y_arr3 = np.linspace(node4, upl, size3)
    y_arr = np.concatenate((y_arr2,y_arr1,y_arr3), axis=0)

    xy = np.transpose([np.tile(y_arr, len(x_arr)),np.repeat(x_arr, len(y_arr))])



    result_arr = []
    phase_arr = []
    count = 0
    count_all = size ** 2
    
    num_cores = multiprocessing.cpu_count()
    print('计算核数量{0}').format(num_cores)

    with mp.Pool(num_cores) as p:
        result_arr = list(tqdm(p.imap(processInput, xy), total=len(xy)))
    result_arr = np.array(result_arr).reshape(size,size)
   
    print('next stage')

    # start = time.time()
    # result_arr = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in xy)
    # result_arr = np.array(result_arr).reshape(size,size)
    # end = time.time()
    # print(-start + end)





    #matplotlib 绘制格点图
    interp = 'nearest'


    fig, ax = plt.subplots()
    fig.suptitle('NonUniformImage class', fontsize='large')
    im = NonUniformImage(ax, interpolation=interp, extent=(downl, upl, downl, upl),
                        cmap=cm.Purples)
    im.set_data(x_arr, y_arr, result_arr)
    ax.images.append(im)
    ax.set_xlim(downl, upl)
    ax.set_ylim(downl, upl)
    ax.set_title(interp)
    plt.show()