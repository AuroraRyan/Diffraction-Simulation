from numpy import exp
from math import sqrt
import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
from matplotlib import cm

X0 = 0  # mm 中心位置
Y0 = 0  # mm
W0 = 10  # mm
z0 = 10  # mm 焦平面附近观察点
f0 = 10  # mm
Lambda = 780E-6  # mm
K = 2*math.pi/Lambda
WZ = 10 * sqrt(1+(Lambda * f0/(math.pi * (W0**2)))**2)  # mm
print(WZ)
E = 2.718281828459

upl = 0.003  # 外圈界限
downl = -0.003

node1 = -0.0015  # x方向内圈界限
node2 = 0.0015

node3 = -0.0015  # y方向内圈界限
node4 = 0.0015

size1 = 10  # 内圈选点数
size2 = 4  # 外圈选点数
size3 = 4
size = size1 + size2 + size3

x_arr1 = np.linspace(node1, node2, size1)  # 外疏内密
x_arr2 = np.linspace(downl, node1, size2)
x_arr3 = np.linspace(node2, upl, size3)
x_arr = np.concatenate((x_arr2, x_arr1, x_arr3), axis=0)


y_arr1 = np.linspace(node3, node4, size1)
y_arr2 = np.linspace(downl, node3, size2)
y_arr3 = np.linspace(node4, upl, size3)
y_arr = np.concatenate((y_arr2, y_arr1, y_arr3), axis=0)

xy = np.transpose([np.tile(y_arr, len(x_arr)), np.repeat(x_arr, len(y_arr))])
print(xy)

# 计算物镜尺寸

NA = 0.55
D = 2*f0/(sqrt(1/(NA**2)-1))  # mm
x0, x1 = -D/2, D/2

print('物镜', x0)


result_arr = []
phase_arr = []
for i, ele in enumerate(xy):  # 大循环，对焦平面上的每个点
    # 设置模拟点
    N1 = 1000
    R = np.random.uniform(low=0, high=D, size=N1)
    N2 = 100
    TH = np.random.uniform(low=0, high=2*np.pi, size=N2)
    random_data = np.transpose([np.tile(R, len(TH)), np.repeat(TH, len(R))])
    randow_sim = 0
    for item in random_data:  # 小循环，蒙特卡罗模拟
        sim_res = W0/WZ * exp(-((item[0]*np.cos(item[1])-X0)**2+(item[0]*np.sin(item[1])-Y0)**2)/WZ) * exp(-K*(sqrt(item[0]**2+f0**2)-f0)*1j) * exp(K/2/z0*(item[0]**2)*1j) * exp(-K/z0 *
                                                                                                                                                                                  (+ele[0]*item[0]*np.cos(item[1])+ele[1]*item[0]*np.sin(item[1]))*1j) * item[0]  # ele[0]是焦平面x，ele[1]是焦平面y
        randow_sim += sim_res
    integ = randow_sim/N1/N2*np.pi*(D/2**2)
    print(integ)
    complex_factor = exp(
        (K*z0 + K/2/z0*(ele[0]**2+ele[1]**2)) * 1j) / (1j * Lambda * z0)  # 积分外面的相位因子
    fullend = integ*complex_factor  # 焦平面上某点完整结果
    result_arr.append(abs(fullend)**2)  # 幅度的平方，光强
    phase_arr.append(np.arctan(fullend.imag/fullend.real))  # 相位
result_arr = np.array(result_arr).reshape(size, size)


# matplotlib 绘制格点图
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
