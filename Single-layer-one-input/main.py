# y=ax
import math
import matplotlib.pyplot as plt
import numpy as np
#y=ax
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data.csv')
x_s = dataset.iloc[:, :-1].values
y_b = dataset.iloc[:, 1].values

y_pred=[]
y_pred_2=[]
w=0
alfa = 0.01

def forward(x):
    return x*w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred-y)**2

def gradent(x, y):
    return (x*w-y)*x/len(x_s)


MSE = 0
arrMSE = []
MSE_Min = 0
arrW = []
for epoch in range(1000):
    for x_qiy, y_qiy in zip(x_s, y_b):
        grad = gradent(x_qiy, y_qiy)
        los = loss(x_qiy, y_qiy)
        w = w - alfa * grad
        MSE += los
        # print(f"gradent: {grad}. los: {los}. w:{w}.")
    arrMSE.append(MSE/len(x_s))
    arrW.append(w)
    print('')
    print(epoch, f" MSE:{MSE/3} w:{w}")
    if(MSE/3<MSE_Min):
        min_mse = float(MSE / len(x_s))
        w = float(w)
        print(f"{epoch} epochda eng yaxshi natijaga erishdi.  \nMSE:{round(min_mse, 5)} \nw1:{round(w, 2)}\n\n")
        break
    MSE = 0
    for yi in x_s:
        y_pred.append(yi * w)
    y_pred_2.append(y_pred)
    y_pred=[]

# print(y_pred_2)
for item in y_pred_2:
    plt.plot(x_s, item, color="blue")

plt.plot(x_s, y_b, color="red")
plt.show()
plt.plot(arrW, arrMSE, 'o')
plt.show()