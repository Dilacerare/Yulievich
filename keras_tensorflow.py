import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
import tensorflow
from sklearn.model_selection import train_test_split

url = 'https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt'
df = pd.read_csv(url, header=0, sep='\t')

model_bmi = Sequential()
model_bmi.add(Dense(1, input_dim=1, activation='linear'))
model_bmi.compile(loss='mean_squared_error', optimizer='adam')

model_bmi.fit(df.BMI, df.Y, epochs=1000)

keras_bmi_pridict = model_bmi.predict(df.BMI)

plt.scatter(df.BMI, df.Y)
plt.plot(df.BMI, keras_bmi_pridict, color='red')
plt.show()

history = model_bmi.fit(df.BMI, df.Y, epochs=2000)

print(type(history))

print(history)

print(type(history.history))

print(history.history.keys())

print(history.history['loss'])

plt.plot(history.history['loss'])
# plt.ylim('Функция потерь')
# plt.xlim('Эпохи')
plt.show()

X = np.array(df.BMI).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, df.Y, test_size=0.2, random_state=42)

history = model_bmi.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test))

print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
# plt.ylim('Функция потерь')
# plt.xlim('Эпохи')
plt.show()

weights = model_bmi.get_weights()[0]
bias = model_bmi.get_weights()[-1]

print(weights, bias)

print(model_bmi.weights)

xx = np.array([20, 40])
yy = bias[0] + xx * weights[0][0]

keras_bmi_pridict = model_bmi.predict(X)
plt.scatter(X, df.Y)
plt.plot(X, keras_bmi_pridict, color="red")
plt.plot(xx, yy, color="green")
plt.show()