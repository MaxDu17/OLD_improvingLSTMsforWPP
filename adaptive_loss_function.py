import tensorflow as tf
import numpy as np

class My_Loss:
    def adaptive_loss(self, output, Y):
        squared = self.squared_loss(output, Y)
        absolute = self.abs_loss(output, Y)
        comparison = tf.greater(squared, absolute)
        with tf.Session() as s_:
            if s_.run(comparison):
                return squared
            else:
                return absolute

    def squared_loss(self, output, Y):
        loss = tf.square(tf.subtract(output, Y))
        loss = tf.reshape(loss, [])
        return loss

    def abs_loss(self, output, Y):
        loss = np.abs(tf.subtract(output, Y))
        loss = tf.reshape(loss, [])
        return loss