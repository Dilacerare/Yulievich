import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

f = pd.read_csv('data.csv', header=0, encoding='windows-1251', sep=';')

print(f.head(6))

coeffs = np.polyfit(f.X, f.Y, 1)

x_line = np.linspace(f.X.min(),
                     f.X.max(), 10)
y_line = coeffs[0] * x_line + coeffs[1]

print(coeffs)

plt.scatter(f.X, f.Y, marker='^', c=f.S)
plt.plot(x_line, y_line, "-v",)
plt.show()
