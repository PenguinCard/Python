# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 01:40:05 2020

@author: user
"""

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

# seperate train set 6:4
train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])

(train_data, validation_data), test_data = tfds.load(
        name="imdb_reviews",
        split=(train_validation_split, tfds.Split.TEST),
        as_supervised=True)

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))

# movie review
print(train_examples_batch)

# movie review label
print(train_labels_batch)

embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)
print(hub_layer(train_examples_batch[:3]))

# model relu, sigmoid
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()

# model optimizer : adam, loss : binary crossentropy
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# mini-batch : 512 samples, epochs : 20, test_set : 10000
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20, validation_data=validation_data.batch(512),
                    verbose=1)

# model evaluation
results = model.evaluate(test_data.batch(512), verbose=2)
for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))
