import tensorflow as tf
import numpy as np

class My_Loss:
    def adaptive_loss(self, output, Y):
        squared = self.squared_loss(output, Y)
        absolute = self.abs_loss(output, Y)

        if squared>absolute:
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