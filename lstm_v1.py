"""Maximilian Du 6-28-18
LSTM implementation with wind dataset"""
import tensorflow as tf
import numpy as np
from pipeline import SetMaker
from pipeline import Hyperparameters
sm = SetMaker()
hyp = Hyperparameters()

#constructing the big weight now
with tf.name_scope("weights and biases"):
    W_Concat = tf.Variable(tf.random_normal(shape = [4, hyp.hidden_dim+1]), name = "big_weight")
    B_Concat = tf.Variable(tf.zeros(shape=[4, hyp.cell_dim]), name = "big_bias")

#consruction of the placeholders now
with tf.name_scope("placeholders"):
    X = tf.placeholder(shape = [1,1], dtype =  tf.float32, name = "input_placeholder")
    Y = tf.placeholder(shape = [1,1], dtype = tf.float32, name = "label")
    H_last = tf.placeholder(shape = [1,hyp.hidden_dim], dtype = tf.float32, name = "last_hidden")
    C_last = tf.placeholder(shape= [1,hyp.cell_dim], dtype = tf.float32, name = "last_cell")

with tf.name_scope("to_gates"):
    tiling_tensor = tf.constant(shape = [4,1], dtype = tf.int32, name = "tiling_tensor_cst")
    input_data = tf.tile(X, tiling_tensor, name = "tiling_input")
    hidden = tf.tile(H_last, tiling_tensor, name = "tiling_hidden")
    concat_input = tf.stack