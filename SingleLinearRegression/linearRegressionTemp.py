# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import model_selection
from sklearn import linear_model

# y = mx + c
# F = 1.8 * C + 32

x = list(range(0, 30))  # C(Celsius)
# y = [1.8 * F + 32 for F in x]  # F(Fahrenheit)
y = [1.8 * F + 32 + random.randint(-3, 3) for F in x]  # F(Fahrenheit)
print(f'X:{x}')
print(f'Y:{y}')

plt.plot(x, y, '-*r')
# plt.show()

x = np.array(x).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)
print(f'X:{x}')
print(f'Y:{y}')

xTrain, xTest, yTrain, yTest = model_selection.train_test_split(x, y, test_size=0.2)
print(f'Shape:{xTrain.shape}')

model = linear_model.LinearRegression()
model.fit(xTrain, yTrain)
print(f'Coefficient:{model.coef_}')
print(f'Intercept:{model.intercept_}')

accuracy = model.score(xTest, yTest)
print(f'Accuracy:{round(accuracy * 100, 2)}')

x = x.reshape(1, -1)[0]
m = model.coef_[0][0]
c = model.intercept_[0]
y = [m * F + c for F in x]  # F(Fahrenheit)
plt.plot(x, y, '-*b')
plt.show()
