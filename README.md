# future
from keras import models
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import to_categorical

(tr_i,tr_l),(te_i,te_l)=mnist.load_data()
tr_i=tr_i.reshape((60000,28*28))
tr_i=tr_i.astype('float32')/255
print(te_i.shape)
te_i=te_i.reshape((10000,28*28))
te_i=te_i.astype('float32')/255

tr_l= to_categorical(tr_l)
te_l = to_categorical(te_l)

model=models.Sequential()
model.add(Dense(64,activation='relu', input_shape=(784,)))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

model.fit(tr_i,tr_l,epochs=20)
