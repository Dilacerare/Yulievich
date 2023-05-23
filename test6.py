from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(seed=451)

# данные иксы
x = np.random.uniform(low=150, high=200, size=10)
y = x - 100 + np.random.normal(loc=0, scale=3, size=10)

x = np.append(x, 180)
x = np.append(x, 190)
x = np.append(x, 190)
print(x)

y = np.append(y, 50)
y = np.append(y, 60)
y = np.append(y, 55)
print(y)

ransac = RANSACRegressor(random_state=0)
ransac.fit(x.reshape(-1, 1), y)

# создаем объект линейной регрессии
# reg = LinearRegression().fit(x.reshape(-1, 1), y)

# предсказываем значения y для всех значений x
y_pred = ransac.predict(x.reshape(-1, 1))

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# строим график
plt.scatter(x[inlier_mask], y[inlier_mask], color="blue", label="Inlier")
plt.scatter(x[outlier_mask], y[outlier_mask], color="red", label="Outlier")
plt.plot(x, y_pred)
plt.legend()
plt.show()
