#_*_coding:utf-8_*_
#!/usr/bin/env python
'__author__' == 'Alex_XT'
import numpy as np
import matplotlib.pyplot as plt

def SoftNMS(ov,method=1,sigma=0.5,Nt=0.3,a=1.0):

    if method == 1:  # linear
        if ov > Nt:
            weight = (1 - ov)*a
        else:
            weight = 1
    elif method == 2:  # gaussian
        weight = np.exp(-(ov * ov) / sigma)

    elif method == 3:  # exp
        if ov > Nt:
            weight = np.exp(Nt-ov)
        else:
            weight = 1

    return weight

if __name__ == '__main__':
    ov = np.arange(1,100)
    ov = 1.0*ov/100

    fig1 = plt.figure('fig1')
    NtNew = 0.1
    aNew=0.5
    weight = [SoftNMS(i,method=1,sigma=0.5,Nt=NtNew,a=aNew) for i in ov]
    plt.plot(ov,weight,'-',label='linear Nt={},a={}'.format(NtNew,aNew))
    NtNew = 0.1
    aNew = 0.7
    weight = [SoftNMS(i, method=1, sigma=0.5, Nt=NtNew,a=aNew) for i in ov]
    plt.plot(ov, weight, '-+', label='linear Nt={},a={}'.format(NtNew,aNew))
    NtNew = 0.1
    aNew = 1.0
    weight = [SoftNMS(i, method=1, sigma=0.5, Nt=NtNew,a=aNew) for i in ov]
    plt.plot(ov, weight, '-^', label='linear Nt={},a={}'.format(NtNew,aNew))
    plt.xlabel('IoU')
    plt.ylabel('weight')
    plt.legend(loc=1)

    fig2 = plt.figure('fig2')
    sigmaNew = 0.1
    weight = [SoftNMS(i, method=2, sigma =sigmaNew, Nt=0.3) for i in ov]
    plt.plot(ov, weight, '-', label='gaussian sigma={}'.format(sigmaNew))
    plt.xlabel('IoU')
    plt.ylabel('weight')
    sigmaNew = 0.3
    weight = [SoftNMS(i, method=2, sigma=sigmaNew, Nt=0.3) for i in ov]
    plt.plot(ov, weight, '-+', label='gaussian sigma={}'.format(sigmaNew))
    sigmaNew = 0.5
    weight = [SoftNMS(i, method=2, sigma=sigmaNew, Nt=0.3) for i in ov]
    plt.plot(ov, weight, '-^', label='gaussian sigma={}'.format(sigmaNew))
    plt.legend(loc=1)

    fig3 = plt.figure('fig3')
    NtNew = 0.1
    weight = [SoftNMS(i, method=3, sigma=0.5, Nt=NtNew) for i in ov]
    plt.plot(ov, weight, '-', label='linear Nt={}'.format(NtNew))
    NtNew = 0.3
    weight = [SoftNMS(i, method=3, sigma=0.5, Nt=NtNew) for i in ov]
    plt.plot(ov, weight, '-+', label='linear Nt={}'.format(NtNew))
    NtNew = 0.5
    weight = [SoftNMS(i, method=3, sigma=0.5, Nt=NtNew) for i in ov]
    plt.plot(ov, weight, '-^', label='linear Nt={}'.format(NtNew))
    plt.xlabel('IoU')
    plt.ylabel('weight')
    plt.legend(loc=1)

    plt.show()
