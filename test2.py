from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(seed=451)

# данные иксы
x = np.random.uniform(low=150, high=200, size=10)
y = x - 100 + np.random.normal(loc=0, scale=3, size=10)

# создаем объект линейной регрессии
reg = LinearRegression().fit(x.reshape(-1, 1), y)

# предсказываем значения y для всех значений x
y_pred = reg.predict(x.reshape(-1, 1))

# строим график
plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
# plt.show()

print(reg.coef_)
print(reg.intercept_)


print(f"y = {reg.intercept_:.3f} + {reg.coef_[0]:.3f} * x")
