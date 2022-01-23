import scipy
from scipy.integrate import quad_vec
from numpy import exp
from math import sqrt
import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
from matplotlib import cm

X0 = 0 #mm 中心位置
Y0 = 8 #mm 
W0 = 10 #mm
z0 = 10 #mm 观察点
f0 = 10 #mm
Lambda = 780E-6 #mm
K = 2*math.pi/Lambda
WZ = 10 * sqrt(1+(Lambda * f0/(math.pi * (W0**2)) )**2) #mm
print(WZ)
E = 2.718281828459

upl = 0.003 #外圈界限
downl = -0.003

node1 = -0.0015 #x方向内圈界限
node2 = 0.0015 

node3 = -0.0015 #y方向内圈界限
node4 = 0.0015 

size1=100 #内圈选点数
size2=40 #外圈选点数
size3=40 
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
print(xy)






#计算物镜尺寸

NA = 0.55
D = 2*f0/(sqrt(1/(NA**2)-1)) #mm
x0, x1 = -D/2, D/2

print(x0)

#x,y没有交叉，分开积，圆形透镜会有交叉项，速度更慢
def f1(x):
        return W0/WZ * exp(-((x-X0)**2)/WZ) * exp(-K/z0*(ele[0]*x)*1j) #ele[0]是焦平面x，ele[1]是焦平面y
def f2(y):
        return W0/WZ * exp(-((y-Y0)**2)/WZ) * exp(-K/z0*(ele[1]*y)*1j) #ele[0]是焦平面x，ele[1]是焦平面y

result_arr = []
for i,ele in enumerate(xy):
    result1, error1 = quad_vec(f1, x0, x1)
    result2, error2 = quad_vec(f2, x0, x1)

    complex_factor = exp((K*z0 + K/2/z0*(ele[0]**2+ele[1]**2)) * 1j) / (1j * Lambda * z0) #积分外面的相位因子
    ee = abs(result1*result2*complex_factor)**2
    # ee = math.log(abs(result1*result2*complex_factor))
    result_arr.append(ee)
result_arr = np.array(result_arr).reshape(size,size)
print(result_arr)


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
