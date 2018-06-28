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
    W_Forget = tf.Variable(tf.random_normal(shape = [hyp.hidden_dim + 1,hyp.cell_dim]), name = "forget_weight")
    W_Output = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + 1,hyp.cell_dim]), name="output_weight")
    W_Gate = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + 1, hyp.cell_dim]), name="gate_weight")
    W_Input = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + 1, hyp.cell_dim]), name="input_weight")

    B_Forget = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name = "forget_bias")
    B_Output = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="output_bias")
    B_Gate = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="gate_bias")
    B_Input = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="input_bias")
#consruction of the placeholders now
with tf.name_scope("placeholders"):
    X = tf.placeholder(shape = [1,1], dtype =  tf.float32, name = "input_placeholder")
    Y = tf.placeholder(shape = [1,1], dtype = tf.float32, name = "label")
    H_last = tf.placeholder(shape = [1,hyp.hidden_dim], dtype = tf.float32, name = "last_hidden")
    C_last = tf.placeholder(shape= [1,hyp.cell_dim], dtype = tf.float32, name = "last_cell")

with tf.name_scope("to_gates"):
    concat_input = tf.concat([X, H_last], axis =0, name = "input_concat")

    forget_gate = tf.add(tf.matmul(concat_input, W_Forget, name = "f_w_m"),B_Forget, name = "f_b_a") #decides which to drop from cell
    output_gate = tf.add(tf.matmul(concat_input, W_Output, name = "o_w_m"), B_Output, name = "o_b_a") #decides which to reveal to next_hidd/output
    gate_gate = tf.add(tf.matmul(concat_input, W_Gate, name = "g_w_m"), B_Gate, name = "g_b_a") #decides which things to change in cell state
    input_gate = tf.add(tf.matmul(concat_input, W_Input, name = "i_w_m"), B_Input, name = "i_b_a") #decides which of the changes to accept