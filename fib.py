import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from functools import lru_cache


@lru_cache(maxsize=None)
def fibonacciRecursion(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)


def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for i in range(2, n+1):
            c = a + b
            a, b = b, c
        return c


print(fibonacciRecursion(1000))

df = pd.read_csv('fib.csv', header=0, encoding='windows-1251', sep=',', names=["n", "t"])

df['lgt'] = np.log10(df.t)
# print(f.head(6))

coeffs = np.polyfit(df.n, df.t, 1)

lin_mod = LinearRegression()
n = np.array(df.n).reshape(-1, 1)
lin_mod.fit(n, df.lgt)
log_t_pred = lin_mod.predict(n)



# plt.scatter(df.n, df.t, marker='^')

plt.scatter(df.n, df.lgt, marker='^')
plt.plot(n, log_t_pred, color="red")
# plt.plot(x_line, y_line, "-v",)
plt.show()


b = 10 ** lin_mod.coef_[0]
a = 10 ** lin_mod.intercept_
print(f"log10(t) = {a:.3e} * {b:.3f} ^ n")
