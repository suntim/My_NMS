#_*_coding:utf-8_*_
#!/usr/bin/env python
'__author__' == 'Alex_XT'
import numpy as np
import matplotlib.pyplot as plt

def SoftNMS(ov,method=1,sigma=0.5,Nt=0.3):

    if method == 1:  # linear
        if ov > Nt:
            weight = 1 - ov
        else:
            weight = 1
    elif method == 2:  # gaussian
        weight = np.exp(-(ov * ov) / sigma)

    return weight

if __name__ == '__main__':
    ov = np.arange(1,100)
    ov = 1.0*ov/100

    NtNew = 0.1
    weight = [SoftNMS(i,method=1,sigma=0.5,Nt=NtNew) for i in ov]
    fig1 = plt.figure('fig1')
    plt.plot(ov,weight,'-',label='linear Nt={}'.format(NtNew))
    NtNew = 0.3
    weight = [SoftNMS(i, method=1, sigma=0.5, Nt=NtNew) for i in ov]
    plt.plot(ov, weight, '-+', label='linear Nt={}'.format(NtNew))
    NtNew = 0.5
    weight = [SoftNMS(i, method=1, sigma=0.5, Nt=NtNew) for i in ov]
    plt.plot(ov, weight, '-*', label='linear Nt={}'.format(NtNew))
    plt.xlabel('ov')
    plt.ylabel('weight')
    plt.legend(loc=1)

    sigmaNew = 0.1
    weight = [SoftNMS(i, method=2, sigma =sigmaNew, Nt=0.3) for i in ov]
    fig2 = plt.figure('fig2')
    plt.plot(ov, weight, '-', label='gaussian sigma={}'.format(sigmaNew))
    plt.xlabel('ov')
    plt.ylabel('weight')
    sigmaNew = 0.3
    weight = [SoftNMS(i, method=2, sigma=sigmaNew, Nt=0.3) for i in ov]
    plt.plot(ov, weight, '-+', label='gaussian {}'.format(sigmaNew))
    sigmaNew = 0.5
    weight = [SoftNMS(i, method=2, sigma=sigmaNew, Nt=0.3) for i in ov]
    plt.plot(ov, weight, '-*', label='gaussian {}'.format(sigmaNew))
    plt.legend(loc=1)

    plt.show()
