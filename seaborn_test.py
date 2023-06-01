import seaborn as sns
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

x = np.random.uniform(150, 200, 10)
y = x - 100 + np.random.normal(0, 3, 10)

sns.regplot(x=x, y=y, ci=99.7)

plt.show()
