import math
import pandas as pd
import matplotlib.pylab as plt

data_combo = pd.read_excel('data.xlsx')
x1 = list(data_combo['x1'])
x2= list(data_combo['x2'])
y= list(data_combo['y'])

yArr = []
w1_1 = w1_2 = w1_3 = w1_4 = w1_5 = w1_6 = 0
w2_1 = w2_2 = w2_3 = 0
b1=b2=b3=b4=0
tetta = 0.01
def sigmoid(y):
    return 1/(1+math.exp(-y))


def ubdateWeight(w, d, x):
    return w + tetta * d * x


def ubdateBias(b, d):
    return b + tetta * d


for epoch in range(1000):
    loss = 0
    # print(w1_1)
    for x_1, x_2, y_act in zip(x1, x2, y):
        z1=x_1 * w1_1 + x_2 * w1_2+b1
        z2=x_1 * w1_3 + x_2 * w1_4+b2
        z3=x_1 * w1_5 + x_2 * w1_6+b3
        g1 = sigmoid(z1)
        g2 = sigmoid(z2)
        g3 = sigmoid(z3)

        z4 = g1 * w2_1 + g2 * w2_2 +g3*w2_3+b4
        g4 = (z4)
        y_pred=g4

        d4 = y_act - y_pred
        d3 = w2_3 * d4 * g3 * (1-g3)
        d2 = w2_2 * d4 * g2 * (1-g2)
        d1 = w2_1 * d4 * g1 * (1-g1)

        w1_1 = ubdateWeight(w1_1, d1, x_1)
        w1_2 = ubdateWeight(w1_2, d1, x_2)
        w1_3 = ubdateWeight(w1_3, d2, x_1)
        w1_4 = ubdateWeight(w1_4, d2, x_2)
        w1_5 = ubdateWeight(w1_5, d3, x_1)
        w1_6 = ubdateWeight(w1_6, d3, x_2)

        w2_1 = w2_1 + tetta * d4 * g1
        w2_2 = w2_2 + tetta * d4 * g2
        w2_3 = w2_3 + tetta * d4 * g3

        b1=ubdateBias(b1, d1)
        b2=ubdateBias(b2, d2)
        b3=ubdateBias(b3, d3)
        b4=ubdateBias(b4, d4)
        loss+=d4**2



# for drawing
for x_1, x_2 in zip(x1, x2 ):
        g1 = sigmoid(x_1 * w1_1 + x_2 * w1_2+b1)
        g2 = sigmoid(x_1 * w1_3 + x_2 * w1_4+b2)
        g3 = sigmoid(x_1 * w1_5 + x_2 * w1_6+b3)
        g4 = (g1 * w2_1 + g2 * w2_2 +g3*w2_3+b4)
        yArr.append(g4)

plt.plot(y, color='red')
plt.plot(yArr, color='blue')
plt.show()
