import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

np.random.seed(seed=451)

b1 = -100
w = np.arange(0.5, 1.6, 0.1)

print(w)

# Генерация данных
x = np.random.uniform(low=150, high=200, size=10)
y = x - 100 + np.random.normal(loc=0, scale=3, size=10)

sq_err = [0] * len(w)
for j in range(len(sq_err)):
    for i in range(len(x)):
        sq_err[j] += (y[i] - w[j] * x[i] + 100)**2
print(sq_err)

# Построение графика
# plt.scatter(x, y)
plt.plot(w, sq_err, color='red')
plt.show()

# y_pred = np.polyval(w, x)
#
# sse = np.sum((y - y_pred)**2)
#
# print(y_pred)

# # Обучение регрессии
# # regressor = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000)
# regressor = MLPRegressor(hidden_layer_sizes=(1,), activation="identity", solver="lbfgs")
# regressor.fit(x.reshape(-1, 1), y)
#
# # Построение графика
# plt.scatter(x, y)
# plt.plot(x, regressor.predict(x.reshape(-1, 1)), color='red')
# # plt.show()
#
# print(regressor.coefs_)
# print(regressor.intercepts_)
#
# b = []
# w = []
#
# for i in regressor.coefs_:
#     for j in i:
#         for k in j:
#             w.append(k)
#
# for i in regressor.intercepts_:
#     for j in i:
#         b.append(j)
#
#
# print(f"y = {b[0] * w[1] + b[1]:.3f} + {w[0] * w[1]:.3f} * x")