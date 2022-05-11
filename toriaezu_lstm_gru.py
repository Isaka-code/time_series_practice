# 参考サイト　https://kagglenote.com/ml-tips/timeseries-nn-sin/

# setup dataset
# generate sin curve
import numpy as np

data_num = 500
num = np.linspace(0, 4*np.pi, data_num)
data = np.sin(num) #sin波作成

noise = np.random.normal(-0.1, 0.1, num.shape)
noise_data = data + noise #ノイズ有りsin波作成


import matplotlib.pyplot as plt

plt.plot(num, noise_data, label="noise sin")
plt.plot(num, data, label="sin")
plt.legend(bbox_to_anchor=(1, 1), loc='upper right')
plt.title("sin")
plt.show()


# make dataset
X_num = 50

X, y = [], []
for i in range(data_num-X_num):
    X_i = noise_data[i:i+X_num]
    y_i = noise_data[i+X_num]
    X.append(X_i)
    y.append(y_i)
    
X = np.array(X)
y = np.array(y)

X = X.reshape([X.shape[0], X.shape[1], 1])
y = y.reshape([y.shape[0], 1])


# make a lstm model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

model = Sequential()
model.add(LSTM(128, return_sequences=False, dropout=0.2))
model.add(Dense(1, activation="linear"))
model.compile(loss="mean_squared_error", optimizer="adam")

model.fit(X, y, batch_size=128, epochs=15)
y_predict = model.predict(X)
y_predict = y_predict.flatten()

plt.plot(num, noise_data, label="raw")
plt.plot(num[X_num:], y_predict, label="predict")
plt.legend(bbox_to_anchor=(1, 1), loc='upper right')
plt.title("lstm")
plt.show()


# make a gru model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import GRU
from keras.models import Sequential

model = Sequential()
model.add(GRU(128, return_sequences=False, dropout=0.2))
model.add(Dense(1, activation="linear"))
model.compile(loss="mean_squared_error", optimizer="adam")

model.fit(X, y, batch_size=128, epochs=15)
y_predict = model.predict(X)
y_predict = y_predict.flatten()

plt.plot(num, noise_data, label="raw")
plt.plot(num[X_num:], y_predict, label="predict")
plt.legend(bbox_to_anchor=(1, 1), loc='upper right')
plt.title("gru")
plt.show()