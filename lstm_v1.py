"""Maximilian Du 6-28-18
LSTM implementation with wind data set"""
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

with tf.name_scope("placeholders"):
    X = tf.placeholder(shape = [1,1], dtype =  tf.float32, name = "input_placeholder") #waits for the prompt
    Y = tf.placeholder(shape = [1,1], dtype = tf.float32, name = "label") #not used until the last cycle
    H_last = tf.placeholder(shape = [1,hyp.hidden_dim], dtype = tf.float32, name = "last_hidden") #last hidden state (aka the "output")
    C_last = tf.placeholder(shape= [1,hyp.cell_dim], dtype = tf.float32, name = "last_cell") #last cell state

with tf.name_scope("to_gates"):
    concat_input = tf.concat([X, H_last], axis = 0, name = "input_concat") #concatenates the inputs to one vector

    forget_gate = tf.add(tf.matmul(concat_input, W_Forget, name = "f_w_m"),B_Forget, name = "f_b_a") #decides which to drop from cell
    output_gate = tf.add(tf.matmul(concat_input, W_Output, name = "o_w_m"), B_Output, name = "o_b_a") #decides which to reveal to next_hidd/output
    gate_gate = tf.add(tf.matmul(concat_input, W_Gate, name = "g_w_m"), B_Gate, name = "g_b_a") #decides which things to change in cell state
    input_gate = tf.add(tf.matmul(concat_input, W_Input, name = "i_w_m"), B_Input, name = "i_b_a") #decides which of the changes to accept

with tf.name_scope("non-linearity"): #makes the gates into what they should be
    forget_gate = tf.sigmoid(forget_gate, name = "sigmoid_forget")
    output_gate = tf.sigmoid(output_gate, name="sigmoid_output")
    input_gate = tf.sigmoid(input_gate, name="sigmoid_input")
    gate_gate = tf.tanh(gate_gate, name = "tanh_gate")

with tf.name_scope("forget_gate"): #forget gate values and propagate
    pass
with tf.name_scope("output_gate"): #output gate values to hidden
    pass

with tf.name_scope("suggestion_node"): #suggestion gate
    pass