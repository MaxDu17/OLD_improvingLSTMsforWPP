import tensorflow as tf
import numpy as np
from pipeline import SetMaker
from pipeline import Hyperparameters
import os
import csv

sm = SetMaker()
hyp = Hyperparameters()

#constructing the big weight now
with tf.name_scope("weights_and_biases"):
    W_Forget = tf.Variable(tf.random_normal(shape = [hyp.hidden_dim + 1,hyp.cell_dim]), name = "forget_weight")
    W_Output = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + 1,hyp.cell_dim]), name="output_weight")
    W_Gate = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + 1, hyp.cell_dim]), name="gate_weight")
    W_Input = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + 1, hyp.cell_dim]), name="input_weight")
    W_Hidden_to_Out = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim,1]), name = "outwards_propagating_weight")

    B_Forget = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name = "forget_bias")
    B_Output = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="output_bias")
    B_Gate = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="gate_bias")
    B_Input = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="input_bias")
    B_Hidden_to_Out = tf.Variable(tf.zeros(shape=[1,1]), name = "outwards_propagating_bias")

with tf.name_scope("placeholders"):
    X = tf.placeholder(shape = [1,1], dtype =  tf.float32, name = "input_placeholder") #waits for the prompt
    Y = tf.placeholder(shape = [1,1], dtype = tf.float32, name = "label") #not used until the last cycle
    H_last = tf.placeholder(shape = [1,hyp.hidden_dim], dtype = tf.float32, name = "last_hidden") #last hidden state (aka the "output")
    C_last = tf.placeholder(shape= [1,hyp.cell_dim], dtype = tf.float32, name = "last_cell") #last cell state

with tf.name_scope("to_gates"):
    X = tf.reshape(X, shape = [1,1])
    H_last_ = tf.reshape(H_last, shape = [hyp.hidden_dim,1])
    concat_input = tf.concat([X, H_last_], axis = 0, name = "input_concat") #concatenates the inputs to one vector
    concat_input = tf.transpose(concat_input)
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
    current_cell = tf.multiply(forget_gate, C_last, name = "forget_gating")

with tf.name_scope("suggestion_node"): #suggestion gate
    suggestion_box = tf.multiply(input_gate, gate_gate, name = "input_determiner")
    current_cell = tf.add(suggestion_box, current_cell, name = "input_and_gate_gating")

with tf.name_scope("output_gate"): #output gate values to hidden
    current_cell = tf.tanh(current_cell, name = "output_presquashing")
    current_hidden = tf.multiply(output_gate, current_cell, name = "next_hidden")
    output = tf.add(tf.matmul(current_hidden, W_Hidden_to_Out, name = "WHTO_w_m"), B_Hidden_to_Out, name = "BHTO_b_a")

with tf.name_scope("loss"):
    loss = tf.square(tf.subtract(output, Y))
    loss = tf.reshape(loss, [])

with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=hyp.LEARNING_RATE).minimize(loss)

with tf.name_scope("summaries_and_saver"):
    tf.summary.histogram("W_Forget", W_Forget)
    tf.summary.histogram("W_Input", W_Input)
    tf.summary.histogram("W_Output", W_Output)
    tf.summary.histogram("W_Gate", W_Gate)

    tf.summary.histogram("Cell_State", current_cell)

    tf.summary.histogram("B_Forget", B_Forget)
    tf.summary.histogram("B_Input", B_Input)
    tf.summary.histogram("B_Output", B_Output)
    tf.summary.histogram("B_Gate", B_Gate)

    tf.summary.scalar("Loss", loss)

    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()

with tf.Session() as sess:
    tf.train.write_graph(sess.graph_def, 'GraphV2/GRAPHS/', 'graph.pbtxt')