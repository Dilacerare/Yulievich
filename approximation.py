import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

n_points = 100

# reg = MLPRegressor(hidden_layer_sizes=(1, ), activation='relu', solver='lbfgs')
# reg = MLPRegressor(hidden_layer_sizes=(2, 2), activation='relu', solver='lbfgs')
# reg = MLPRegressor(hidden_layer_sizes=(9, ), activation='logistic', solver='lbfgs')
# reg = MLPRegressor(hidden_layer_sizes=(5, 2), activation='logistic', solver='lbfgs')
# reg = MLPRegressor(hidden_layer_sizes=(10, ), activation='tanh', solver='lbfgs')
reg = MLPRegressor(hidden_layer_sizes=(5, 2), activation='tanh', solver='lbfgs')

xs = np.linspace(0, 6.28, n_points + 1)
# ys = np.sin(xs)
ys = np.sin(xs) + np.random.normal(0, 0.1, 101)

reg.fit(xs.reshape(-1, 1), ys)
yy = reg.predict(xs.reshape(-1, 1))

plt.scatter(xs,ys)
plt.plot(xs, yy)
plt.show()


xx = np.linspace(-3, 10, 50)

# ys = np.sin(xs) + np.random.normal(0, 0.1, 50)

# reg.fit(xs.reshape(-1, 1), ys)
yy = reg.predict(xx.reshape(-1, 1))

plt.scatter(xs,ys)
plt.plot(xx, yy)
plt.show()
