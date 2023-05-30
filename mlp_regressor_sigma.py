import numpy as np
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor

reg = MLPRegressor(hidden_layer_sizes=(1, ), activation='identity', solver='lbfgs')

coef = []
inercept = []
for i in range(1000):
    x = np.random.uniform(150, 200, 10)
    y = x - 100 + np.random.normal(0, 3, 10)
    reg.fit(x.reshape(-1, 1), y)
    w0 = reg.coefs_[0][0][0]
    w1 = reg.coefs_[1][0][0]
    b0 = reg.intercepts_[0][0]
    b1 = reg.intercepts_[1][0]
    inercept += [b0 * w1 + b1]
    coef += [w0 * w1]

# plt.hist(coef, bins=30)
# plt.show()
#
# plt.hist(inercept, bins=30)
# plt.show()

plt.scatter(x, y)
xx = np.array([150, 200])

for i in range(1000):
    x = np.random.uniform(150, 200, 10)
    y = x - 100 + np.random.normal(0, 3, 10)
    reg.fit(x.reshape(-1, 1), y)
    # yy = inercept[i] + coef[i] * xx
    yy = reg.predict(xx.reshape(-1, 1))
    plt.plot(xx, yy, color="green")

plt.show()
