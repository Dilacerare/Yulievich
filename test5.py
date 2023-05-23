import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

np.random.seed(seed=451)

b1 = -100
w = np.arange(0.5, 1.6, 0.1)

# Генерация данных
x = np.random.uniform(low=150, high=200, size=10)
y = x - 100 + np.random.normal(loc=0, scale=3, size=10)

medAbsErr = [0] * len(w)
for j in range(len(w)):
    absErr = []
    for i in range(len(x)):
        absErr.append(abs(y[i] - w[j] * x[i] + 100))
    medAbsErr[j] = np.median(absErr)



# Построение графика
# plt.scatter(x, y)
plt.plot(w, medAbsErr, color='red')
plt.show()

