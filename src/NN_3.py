from Processing import *
import matplotlib.pyplot as plt
from testing_data import labeled_data
import numpy as np
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Dense

# x = np.array(([0,0],[0,1],[1,0],[1,1]))
# y = np.array(([0],[1],[1],[0]))
# model.add(Dense(2,activation='sigmoid',input_shape=(2,)))
# model.add(Dense(1))
# print(model.predict(x))

model = Sequential()
x_, y_ = labeled_data()

model.add(Dense(200, activation='relu', input_shape=(18,)))
model.add(Dense(500, activation='relu'))
model.add(Dense(9))

model.compile(loss='mse', optimizer='adam')

def run():
    x, y = shuffle(x_, y_)
    history = model.fit(x, y, epochs=3, batch_size=20, shuffle=True)
    f_out = np.vectorize(lambda x: 1 if x > .5 else 0)

    for idx, s in enumerate(x[0:50]):
        print(f"Piles: {bin_to_list(s)} "
              f"NN prediction: {process_move(f_out(model.predict(s[np.newaxis, :]))[0])} "
              f"Correct output: {process_move(y[idx])}")

    plt.plot(history.history['loss'])

if __name__ == '__main__':
    pass
