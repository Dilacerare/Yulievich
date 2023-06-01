from matplotlib import pyplot as plt
from sklearn.datasets import load_diabetes
import pandas as pd
import seaborn as sns

# Загрузка датасета Diabetes
diabetes = load_diabetes()

# Преобразование данных в DataFrame
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

# Добавление столбца с целевой переменной
df['target'] = diabetes.target

myBMI = 70/(1.85**2)
print(f"myBMI = {myBMI}")

url = 'https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt'
df2 = pd.read_csv(url, header=0, sep='\t')
print(df2)

plt.scatter(df2.BMI, df2.Y)
plt.show()

plt.scatter(df2.S5, df2.Y)
plt.show()

plt.scatter(df2.S6, df.Y)
plt.show()

# Создание матричной диаграммы рассеяния
sns.pairplot(df, vars=diabetes.feature_names, hue='target')

# Отображение графика
plt.show()


#Вывод: судя по данным о пациентах с диабетом, индекс массы тела больше 25 является серьезным фактором диабета


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
