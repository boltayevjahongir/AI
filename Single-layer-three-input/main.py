# y=ax1+bx2+cx3
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data.csv')
x_1 = dataset.iloc[:, 0:1].values
x_2 = dataset.iloc[:, 1:2].values
x_3 = dataset.iloc[:, 2:3].values
y_b = dataset.iloc[:, 3:4].values

y_pred=[]
y_pred_2=[]
w1=0
w2=0
w3=0
alfa = 0.001

def forward(x1,x2, x3 ):
    return x1*w1+x2*w2+x3*w3


def loss(x1, x2, x3, y):
    y_pred = forward(x1, x2, x3)
    return (y_pred-y)**2

def gradent1(x1, x2, x3, y):
    return (x1*w1+x2*w2+x3*w3-y) * x1 / len(x_1)

def gradent2(x1, x2, x3, y):
    return (x1*w1+x2*w2+x3*w3-y) * x2 / len(x_1)

def gradent3(x1, x2, x3, y):
    return (x1*w1+x2*w2+x3*w3-y) * x3 / len(x_1)



MSE = 0
arrMSE = []
MSE_Min = 0.01
arrW = []
arrW2 = []
arrW3 = []
for epoch in range(400):
    for x_qiy_1, x_qiy_2, x_qiy_3, y_qiy in zip(x_1, x_2, x_3, y_b):
        grad1 = gradent1(x_qiy_1, x_qiy_2, x_qiy_3, y_qiy)
        grad2 = gradent2(x_qiy_1, x_qiy_2, x_qiy_3, y_qiy)
        grad3 = gradent3(x_qiy_1, x_qiy_2, x_qiy_3, y_qiy)
        los = loss(x_qiy_1, x_qiy_2, x_qiy_3, y_qiy)
        w1 = w1 - alfa * grad1
        w2 = w2 - alfa * grad2
        w3 = w3 - alfa * grad3
        MSE += los
    arrMSE.append(MSE/len(x_1))
    arrW.append(w1)
    arrW2.append(w2)
    arrW3.append(w3)
    print('')
    print(epoch, f" MSE:{MSE/len(x_1)} w1:{w1} w2:{w2} w3:{w3}")
    if(MSE/len(x_1)<MSE_Min):
        min_mse = float(MSE / len(x_1))
        w1=float(w1)
        w2=float(w2)
        w3=float(w3)
        print(f"{epoch} epochda eng yaxshi natijaga erishdi.  \nMSE:{round(min_mse, 5)} \nw1:{round(w1, 2)} \nw2:{round(w2, 2)}\nw3:{round(w3, 2)}\n\n")
        break

    MSE = 0
    for x_qiy_1, x_qiy_2, x_qiy_3 in zip(x_1, x_2, x_3):
        y_pred.append(x_qiy_1 * w1 +w2*x_qiy_2+w3*x_qiy_3)
    y_pred_2.append(y_pred)
    y_pred=[]
y_end = []
for x_qiy_1, x_qiy_2, x_qiy_3 in zip(x_1, x_2, x_3):
     y_end.append(x_qiy_1 * w1 + w2 * x_qiy_2 + w3 * x_qiy_3)

j=0
plt.plot(y_b, color="red")
plt.plot(y_end, color="blue")
plt.show()
plt.plot(arrW, arrMSE, 'o', color='red')
plt.plot(arrW2, arrMSE, 'o', color='blue')
plt.plot(arrW3, arrMSE, 'o', color='green')
plt.show()
