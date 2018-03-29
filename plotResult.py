# -*- coding: UTF-8 -*-
#!/usr/bin/env python
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']            #SimHei是黑体的意思
import matplotlib.pyplot as plt
import numpy as np

line_threshold = ['0.0001','0.0005','0.001','0.003','0.005','0.01','0.1']
Nt = np.array([[0.7355, 0.7355, 0.7355, 0.7354, 0.7349, 0.7347, 0.7321],
[0.7350,0.7351,0.7351,0.7348,0.7348,0.7344,0.7319],
[0.7299,0.7299,0.7299,0.7299,0.7300,0.7295,0.7270],
[0.7135,0.7136,0.7134,0.7134,0.7135,0.7135,0.7108]])
Nt = Nt*100


sigma_threshold = ['0.0001','0.001','0.003','0.005','0.01','0.1']
Sigma = np.array([[0.7297,0.7296,0.7290,0.7290,0.7276,0.7208],
                  [0.7370,0.7369,0.7367,0.7364,0.7357,0.7335],
                  [0.7342,0.7341,0.7342,0.7343,0.7341,0.7312],
                  [0.7296,0.7296,0.7296,0.7295,0.7294,0.7269]])
Sigma = Sigma*100


fig1=plt.figure('fig1')
x = np.linspace(0,len(line_threshold)-1,len(line_threshold))
# print x
plt.plot(x,Nt[0],"-d",label='Nt=0.1')
plt.plot(x,Nt[1],"-^",label='Nt=0.3')
plt.plot(x,Nt[2],"-h",label='Nt=0.4')
plt.plot(x,Nt[3],"-p",label='Nt=0.5')
plt.xticks(x,line_threshold)
plt.xlabel(u'置信度阈值threshold',fontproperties='SimHei')
plt.ylabel('mAP%')
plt.legend(loc=5)
# plt.show()

fig2=plt.figure('fig2')
x = np.linspace(0,len(sigma_threshold)-1,len(sigma_threshold))
# print x
plt.plot(x,Sigma[0],"-d",label='Sigma=0.1')
plt.plot(x,Sigma[1],"-^",label='Sigma=0.3')
plt.plot(x,Sigma[2],"-h",label='Sigma=0.5')
plt.plot(x,Sigma[3],"-p",label='Sigma=0.7')
plt.xticks(x,sigma_threshold)
plt.xlabel(u'置信度阈值threshold',fontproperties='SimHei')
plt.ylabel('mAP%')
plt.legend(loc=3)
# plt.show()

fig3=plt.figure('fig3')
x = np.linspace(0,len(Nt[:,0])-1,len(Nt[:,0]))
# print Nt[:,0]
plt.plot(x,Nt[:,0],"-d",label=u'置信度阈值=0.0001')
plt.plot(x,Nt[:,1],"-^",label=u'置信度阈值=0.0005')
plt.plot(x,Nt[:,2],"-h",label=u'置信度阈值=0.001')
plt.xlabel('Nt',fontproperties='SimHei')
plt.ylabel('mAP%')
plt.xticks(x,['0.1','0.3','0.4','0.5'])
plt.legend(loc=3)

fig4=plt.figure('fig4')
x = np.linspace(0,len(Sigma[:,0])-1,len(Sigma[:,0]))
# print Nt[:,0]
plt.plot(x,Sigma[:,0],"-d",label=u'置信度阈值=0.0001')
plt.plot(x,Sigma[:,1],"-^",label=u'置信度阈值=0.001')
plt.plot(x,Sigma[:,2],"-h",label=u'置信度阈值=0.003')
plt.xlabel('Sigma',fontproperties='SimHei')
plt.ylabel('mAP%')
plt.xticks(x,['0.1','0.3','0.5','0.7'])
plt.legend(loc=1)
plt.show()
