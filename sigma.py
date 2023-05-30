import pandas as pd
import statsmodels.api as sm
import numpy as np

x = np.random.uniform(150, 200, 10)
y = x - 100 + np.random.normal(0, 3, 10)

data = pd.DataFrame({'x': x, 'y': y})

model = sm.OLS(data['y'], data['x'], intercept=True)

data = sm.add_constant(data)

model = sm.OLS(data['y'], data[['const', 'x']]).fit()

print(model.summary())

# print(data)

# df = pd.read_csv('data.csv', header=0, encoding='windows-1251', sep=';')

# y = df['dependent_var']
# x = df['independent_var']
#
# model = sm.OLS(y, x).fit()
#
# std_dev = model.bse
