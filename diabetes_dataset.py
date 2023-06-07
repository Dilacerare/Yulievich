import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_diabetes
import pandas as pd
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

# Загрузка датасета Diabetes
diabetes = load_diabetes()

# Преобразование данных в DataFrame
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

# Добавление столбца с целевой переменной
df['target'] = diabetes.target

###########################################################################

# myBMI = 70/(1.85**2)
# print(f"myBMI = {myBMI}")

###########################################################################

url = 'https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt'
df2 = pd.read_csv(url, header=0, sep='\t')
# print(df2.head())
# print(df2)
#
# plt.scatter(df2.BMI, df2.Y)
# plt.show()
#
# plt.scatter(df2.S5, df2.Y)
# plt.show()
#
# plt.scatter(df2.S6, df.Y)
# plt.show()

###########################################################################

# # Создание матричной диаграммы рассеяния
# sns.pairplot(df, vars=diabetes.feature_names, hue='target')
#
# # Отображение графика
# plt.show()

#Вывод: судя по данным о пациентах с диабетом, индекс массы тела больше 25 является серьезным фактором диабета

###########################################################################

# # Вывод первых 5 строк датасета
# print(df.head())
#
# print(df.shape)
#
# # Вывод статистических характеристик датасета
# print(df.describe())

# print(type(diabetes))
# print(diabetes.data)
# print(diabetes.target.shape)
# print(diabetes.DESCR)
# print(diabetes.feature_names)

###########################################################################

# reg = MLPRegressor(hidden_layer_sizes=(1, ), activation='identity', solver='lbfgs')
# x = np.array(df2.BMI)
# y = np.array(df2.Y)
# reg.fit(x.reshape(-1, 1), y)
# yy = reg.predict(x.reshape(-1, 1))
# plt.scatter(df2.BMI, df2.Y)
# plt.plot(x, yy, color="green")
# plt.show()

###########################################################################

#MLP Regressor
# reg = MLPRegressor(hidden_layer_sizes=(1, ), alpha=1e-4, activation='identity', solver='lbfgs', verbose=10, tol=1e-4,
#                    learning_rate_init=.1, random_state=1)
# reg2 = MLPRegressor(hidden_layer_sizes=(1, ), alpha=1e-4, activation='identity', solver='lbfgs', verbose=10, tol=1e-4,
#                    learning_rate_init=.1, random_state=1)
# x = np.array(df2.BMI)
# y = np.array(df2.Y)
#
# reg.fit(x.reshape(-1, 1), y)
# yy = reg.predict(x.reshape(-1, 1))
#
# reg2.fit(y.reshape(-1, 1), x)
# xx = reg2.predict(y.reshape(-1, 1))
#
# plt.scatter(df2.BMI, df2.Y)
#
# plt.plot(x, yy, color="red")
# plt.plot(xx, y, color="green")
#
# plt.show()

###########################################################################

# r2_BMI = r2_score(df2.Y, yy)
# print(r2_BMI)

###########################################################################

# corr_matrix = df2.corr()
#
# print(corr_matrix)
#
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# plt.show()

###########################################################################

reg3 = MLPRegressor(hidden_layer_sizes=(1, ), alpha=1e-4, activation='identity', solver='lbfgs', verbose=10, tol=1e-4,
                   learning_rate_init=.1, random_state=1)
x = df2.drop(["Y"], axis=1)
reg3.fit(x, df2.Y)
yy = reg3.predict(x)
r2_mlp10 = r2_score(df2.Y, yy)
print(r2_mlp10)

plt.scatter(df2.Y, yy)
plt.show()
