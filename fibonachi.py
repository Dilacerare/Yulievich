# Импортируем необходимые библиотеки и модули
import math
import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

# Определяем функцию, которую будем аппроксимировать
def coustomExp(x):
    return math.exp(x)/(10**14.5)

# Определяем функцию для вычисления чисел Фибоначчи
def F(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return F(n - 1) + F(n - 2)

# Задаем диапазон значений n и количество повторений для каждого n
nn = []
tt = []
const = 36
for n in range(20, const):
    for i in range(11):
        start_time = time.time()
        F(n)
        tt.append(time.time() - start_time)
        nn.append(n)

# temp = 1
# for i in range(len(tt)):#     if i % 7 == 0:
#         temp *= 10#     if tt[i] == 0:
#         tt[i] = 0.000005974 * temp

# Выполняем логарифмическую регрессию для определения зависимости времени выполнения от n
log_t = np.log10(np.array(tt))
ransac = LinearRegression()
ransac.fit(np.array(nn).reshape(-1, 1), np.array(log_t))

# Выводим полученные коэффициенты уравнения логарифмической регрессии
b = 100 ** ransac.coef_[0]
a = 100 ** ransac.intercept_
print(f"log10(t) = {a:.3e} * {b:.3f} ^ n")

# Строим графики зависимости времени выполнения от n и функции, которую аппроксимируем
plt.plot(nn, ransac.predict(np.array(nn).reshape(-1, 1)), color="red")
plt.scatter(nn, log_t)
plt.title("Линейная регрессия по МНК")
plt.xlabel("Номер числа фибрначи, n")
plt.ylabel("Длительность времени, t")
plt.show()
plt.plot(list(range(const)), list(map(coustomExp, range(const))))
plt.scatter(nn, tt)
plt.show()

# Вывод: время растёт быстрее чем прямая, а точнее по экспоненте

# Иследовали эксперемнтально длительность рассчета числа Фибоначи
# для заданного номера числа в диопазоне n = 2..35

# С увелечением номера n длительность рассчетов растёт экспоненциально

# С помощью линейной регресии получили зависимость времени рассчетов t
# от номера числа фибоначи n:

# Made by Dilaerare©.

# log10(t) = 2.234e-14 * 2.632 ^ n
