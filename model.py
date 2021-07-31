import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist     #28 x 28 images of hand writtend digits 0-9


(x_train, y_train), (x_test, y_test)  = mnist.load_data()
#normalizzazione
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#creazione modello
model = tf.keras.models.Sequential()
#input layer
model.add(tf.keras.layers.Flatten())
#hidden layers
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#output layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#prova sempre a diminuire la loss
model.compile(optimizer= 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

#training
model.fit(x_train, y_train, epochs= 3) # great, but did it overfit?

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)


#saving model
model.save('predicting_numbers')
#load model
new_model = tf.keras.models.load_model('predicting_numbers')



predictions = new_model.predict([x_test])
plt.imshow(x_test[0])
plt.show()