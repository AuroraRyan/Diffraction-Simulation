import scipy
from scipy import integrate
from numpy import exp
from math import sqrt
import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
from matplotlib import cm

X0 = 5 #mm
Y0 = 5 #mm
W0 = 10 #mm
f0 = 10 #mm
Lambda = 780E-6 #mm
K = 2*math.pi/Lambda
WZ = 10 * sqrt(1+(Lambda * f0/(math.pi * (W0**2)) )**2) #mm
E = 2.718281828459
size=20

x_arr = np.linspace(-0.003, 0.003, size)
y_arr = np.linspace(-0.003, 0.003, size)
xy = np.transpose([np.tile(y_arr, len(x_arr)),np.repeat(x_arr, len(y_arr))])
print(xy)

upl = 0.003
downl = -0.003


def complex_quadrature(func, **kwargs): #scipy直接积分会抛弃复数部分，分开积
    def real_func(x,y):
        return scipy.real(func(x,y))
    def imag_func(x,y):
        return scipy.imag(func(x,y))
    real_integral = integrate.dblquad(real_func, -float("inf"), float("inf"), lambda x : -float("inf"), lambda h : float("inf"), **kwargs)
    imag_integral = integrate.dblquad(imag_func, -float("inf"), float("inf"), lambda x : -float("inf"), lambda h : float("inf"), **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])


result_arr = []
for i,ele in enumerate(xy):
    def f(x, y):
        return W0/WZ * exp(-((x-X0)**2+(y-Y0)**2)/WZ) * exp(-K/f0*(ele[0]*x+ele[1]*y)*1j) #ele[0]是焦平面x，ele[1]是焦平面y
    result, err1, err2 = complex_quadrature(f)
    complex_factor = exp((K*f0 + K/2/f0*(ele[0]**2+ele[1]**2)) * 1j) / (1j * Lambda * f0) #积分外面的相位因子
    ee = math.log(abs(result*complex_factor))
    result_arr.append(ee)
result_arr = np.array(result_arr).reshape(size,size)
print(result_arr)


#matplotlib 绘制格点图
interp = 'nearest'


# z = np.sqrt(x[np.newaxis, :]**2 + y[:, np.newaxis]**2)

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
exit()
-(K/f * (x_end*x+y_end*y))
