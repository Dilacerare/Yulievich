# 1000 раз сгенирировать датасет и получим оценки коэффициентов регрессии.
# Построим гистограмму распределения этих оценок

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

lin_mod = LinearRegression()

coef = []
inercept = []
for i in range(1001):
    x = np.random.uniform(150, 200, 10)
    y = x - 100 + np.random.normal(0, 3, 10)
    lin_mod.fit(x.reshape(-1, 1), y)
    coef += [lin_mod.coef_[0]]
    inercept += [lin_mod.intercept_]

# plt.hist(coef, bins=30)
# plt.show()

plt.hist(inercept, bins=30)
plt.show()
