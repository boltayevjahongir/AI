# y=2x+b
import math
import matplotlib.pyplot as plt
import numpy as np
#y=ax
x_s=[0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]
y_b=[5.1, 6.4, 6, 7.6, 7.2, 9.1, 8.8, 10.1, 9.2, 10.8]
y_pred=[]
y_pred_2=[]
w1=0
w2=0
alfa = 0.01

def forward(x):
    return x*w1+w2


def loss(x, y):
    y_pred = forward(x)
    return (y_pred-y)**2

def gradent1(x, y):
    return math.fabs((x*w1+w2-y) * x / len(x_s))

def gradent2(x, y):
    return math.fabs((x*w1+w2-y) / len(x_s))


MSE = 0
arrMSE = []
MSE_Min = 0.01
arrW = []
arrW2 = []
for epoch in range(200):
    for x_qiy, y_qiy in zip(x_s, y_b):
        grad1 = gradent1(x_qiy, y_qiy)
        grad2 = gradent2(x_qiy, y_qiy)
        los = loss(x_qiy, y_qiy)
        w1 = w1 + alfa * grad1
        w2 = w2 + alfa * grad2
        MSE += los
        print(f"gradent1: {grad1}. gradent2: {grad2} los: {los}. w1:{w1}. w2:{w2}")
    arrMSE.append(MSE/len(x_s))
    arrW.append(w1)
    arrW2.append(w2)
    print('')
    print(epoch, f" MSE:{MSE/len(x_s)} w1:{w1} w2:{w2}")
    if(MSE/len(x_s)<MSE_Min):
        break
    MSE = 0
    for yi in x_s:
        y_pred.append(yi * w1 +w2)
    y_pred_2.append(y_pred)
    y_pred=[]

print(y_pred_2)
for item in y_pred_2:
    plt.plot(x_s, item, color="blue")

plt.plot(x_s, y_b, color="red")
plt.show()
plt.plot(arrW, arrMSE, 'o')
plt.plot(arrW2, arrMSE, 'o')
plt.show()