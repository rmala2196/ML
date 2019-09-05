    //Machine learning model to recognize handwritten digits with Tensorflow using MNIST datasets.

from __future__ import absolute_import, division, print_function, unicode_literals
!pip install tensorflow==2.0.0-beta1

import tensorflow as tf

    //Import TensorFlow into your program.

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test /255.0

    //Load and prepare the MNIST dataset. Convert the samples from integers to floating-point numbers:

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


   //Build the tf.keras.Sequential model by stacking layers. Choose an optimizer and loss function for training:

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

  //Train and evaluate the model:
