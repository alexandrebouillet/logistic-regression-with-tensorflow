#!/usr/bin/python3
# -*- coding: utf-8 -*
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt

#Nombre d'observations du jeu de donnée
m = 1000
learning_rate = 0.01

X_moons, y_moons = make_moons(m, noise=0.1, random_state=42)

X_moons_with_bias = np.c_[np.ones((m, 1)), X_moons]
y_moons_column_vector = y_moons.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X_moons_with_bias, y_moons_column_vector , test_size=0.2)

def random_batch(X_train, y_train, batch_size):
    nb = np.random.randint(0, len(y_train), batch_size)
    X_batch = X_train[nb]
    y_batch = y_train[nb]
    return X_batch, y_batch
    
n_inputs = 2

X = tf.placeholder(tf.float32, shape=(None, n_inputs+1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1 ), name="y")
theta = tf.Variable(tf.random_uniform([n_inputs + 1, 1], -1.0, 1.0, seed=42), name="theta")
logits = tf.matmul(X, theta, name="logits")
y_proba = tf.sigmoid(logits)

loss = tf.losses.log_loss(y, y_proba)
    
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 1000
batch_size = 50
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = random_batch(X_train, y_train, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        loss_val = loss.eval({X: X_test, y: y_test})
        if epoch % 100 == 0:
            print("Epoch:", epoch, "\tLoss:", loss_val)
            saver.save(sess, "./make_moons.ckpt")
    
    y_proba_val = y_proba.eval(feed_dict={X: X_test, y: y_test})
    y_pred = (y_proba_val >=0.5 )
    pred = precision_score(y_test, y_pred)
    print("Model's precision:", pred)
    
    y_pred_idx = y_pred.reshape(-1) # a 1D array rather than a column vector
    plt.plot(X_test[y_pred_idx, 1], X_test[y_pred_idx, 2], 'go', label="Positive")
    plt.plot(X_test[~y_pred_idx, 1], X_test[~y_pred_idx, 2], 'r^', label="Negative")
    plt.legend()
    plt.show()
    
    
    saver.save(sess, "./make_moons.ckpt")