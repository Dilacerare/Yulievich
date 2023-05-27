import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression


def method_max(arr):
    start_time = time.time()
    maxValue = arr[-1]
    k = len(arr) - 1
    while (k >= 0):
        if (arr[k] > maxValue):
            maxValue = arr[k]
        k -= 1
    # print(f"Наибольшее число: {maxValue}")
    # print("--- %s seconds ---" % (time.time() - start_time))
    return time.time() - start_time

#################################################################
# n = []
# times = []
#
# for i in range(1, 9):
#     x = np.random.randint(0, 10000, size=10**i)
#     print(f"n = {10**i}")
#     n.append(10**i)
#     times.append(method_max(x))
#
# plt.scatter(n, times)
# plt.plot(n, times)
# plt.show()
#################################################################
# n1 = []
# times1 = []
# for i in range(1000):
#     start_time = time.time()
#     x = np.random.randint(0, 1000000, size=1000000)
#     temp = method_max(x)
#     if time.time() - start_time >= 0.017:
#         continue
#     times1.append(time.time() - start_time)
#     n1.append(i)
#
# plt.scatter(n1, times1, color="red")
# plt.plot(n1, times1)
# plt.show()
#
# plt.hist(times1, bins=200)
# plt.show()
#################################################################
# n3 = []
# times3 = []
#
# for i in range(1000000, 4000001, 100000):
#     for j in range(10):
#         start_time = time.time()
#         x = np.random.randint(0, 100000000, size=i)
#         temp = method_max(x)
#         # if time.time() - start_time >= 0.009:
#         #     continue
#         times3.append(time.time() - start_time)
#         n3.append(i)
#
# n3 = np.array(n3).reshape(-1, 1)
# times3 = np.array(times3)
#
# ransac = RANSACRegressor()
# ransac.fit(n3, times3)
#
# x = [1000000, 4000000]
# y = [ransac.estimator_.coef_[0] * x[0] + ransac.estimator_.intercept_.tolist(), ransac.estimator_.coef_[0] * x[1] + ransac.estimator_.intercept_.tolist() ]
#
# plt.scatter(n3, times3)
# plt.plot(x, y, "red")
# plt.title("Средниее время, данные перемешаны")
# plt.xlabel("Количество данных n")
# plt.ylabel("Время поиска максимума t сек.")
# plt.show()

#Выводы
#средния продолжительномть поиска максимума t при
#случайных перемешанных исходных данных растёт линейно
#при увелечении размера исходных данных n

########Анализ наихудшего случая#################################

#данные расположены по убыванию

nn = []
tt = []

for n in range(1000000, 4000001, 100000):
    for i in range(10):
        start_time = time.time()
        x = list(range(n, 0, -1))
        temp = method_max(x)
        # if time.time() - start_time >= 0.009:
        #     continue
        tt.append(time.time() - start_time)
        nn.append(n)

nn = np.array(nn).reshape(-1, 1)
tt = np.array(tt)

ransac = RANSACRegressor()
ransac.fit(nn, tt)

x = [1000000, 4000000]
y = [ransac.estimator_.coef_[0] * x[0] + ransac.estimator_.intercept_.tolist(), ransac.estimator_.coef_[0] * x[1] + ransac.estimator_.intercept_.tolist() ]

plt.scatter(nn, tt)
plt.plot(x, y, "red")
plt.title("Максимальное время, данные отсортированы по убыванию")
plt.xlabel("Количество данных n")
plt.ylabel("Время поиска максимума t сек.")
plt.show()

#Выводы
#и средния, и максимальная продолжительномть поиска максимума t при
#случайных перемешанных и отсортированных по убыванию исходных данных растёт линейно
#при увелечении размера исходных данных n