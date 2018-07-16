import tensorflow as tf
import numpy as np
from pipeline import SetMaker
from pipeline import Hyperparameters
from pipeline import Model2
import os
import csv

sm = SetMaker()
hyp = Hyperparameters()

layer_1 = Model2()
layer_2 = Model2()

with tf.name_scope("placeholders"):
    Y = tf.placeholder(shape=[1, 1], dtype=tf.float32, name="label")  # not used until the last cycle
    init_state_1 = tf.placeholder(shape = [2,1,hyp.cell_dim], dtype = tf.float32, name = "initial_states_1")
    init_state_2 = tf.placeholder(shape = [2,1,hyp.cell_dim], dtype = tf.float32, name = "initial_states_2")
    inputs = tf.placeholder(shape = [hyp.FOOTPRINT,1,1], dtype = tf.float32,  name = "input_data")

with tf.name_scope("aux_weights_and_biases"):
    W_Hidden_to_Out_1 = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim, 1]),
                                  name="outwards_propagating_weight_1")
    W_Hidden_to_Out_2 = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim, 1]),
                                  name="outwards_propagating_weight_2")
    B_Hidden_to_Out_1 = tf.Variable(tf.zeros(shape=[1, 1]), name="outwards_propagating_bias_1")
    B_Hidden_to_Out_2 = tf.Variable(tf.zeros(shape=[1, 1]), name="outwards_propagating_bias_2")

with tf.name_scope("layer_1_propagation"):
    states_1 = layer_1.create_graph(layer_number = 1, inputs = inputs, init_state = init_state_1)
    curr_state_1 = states_1[-1]
    pass_back_state_1 = tf.add([0], states_1[0], name = "pass_back_state_1")
    input_2 = list()
    for i in range(hyp.FOOTPRINT):
        _, current_hidden = tf.unstack(states_1[i])
        current_input_2 = tf.add(tf.matmul(current_hidden, W_Hidden_to_Out_1, name="WHTO_w_m"), B_Hidden_to_Out_1, name="BHTO_b_a")
        input_2.append(current_input_2)
    input_2 = tf.reshape(input_2, [hyp.FOOTPRINT,1,1])


with tf.name_scope("layer_2_propagation"):
    states_2 = layer_2.create_graph(layer_number = 2, inputs = input_2, init_state = init_state_2)
    curr_state_2 = states_2[-1]
    pass_back_state_2 = tf.add([0], states_2[0], name = "pass_back_state_2")
    _, current_hidden_2 = tf.unstack(curr_state_2)
    raw_output = tf.add(tf.matmul(current_hidden_2, W_Hidden_to_Out_2, name="WHTO_w_m_"), B_Hidden_to_Out_2, name="BHTO_b_a")
    output = tf.nn.relu(raw_output, name="output")

with tf.name_scope("loss"):
    loss = tf.square(tf.subtract(output, Y))
    loss = tf.reshape(loss, [], name = "loss")

with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=hyp.LEARNING_RATE).minimize(loss)

with tf.name_scope("summaries_and_saver"):
    tf.summary.scalar("Loss", loss)
    tf.summary.histogram("W_Hidden_to_Out", W_Hidden_to_Out_1)
    tf.summary.histogram("B_Hidden_to_Out", B_Hidden_to_Out_1)
    tf.summary.histogram("W_Hidden_to_Out", W_Hidden_to_Out_2)
    tf.summary.histogram("B_Hidden_to_Out", B_Hidden_to_Out_2)

    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
with tf.Session() as sess:
    tf.train.write_graph(sess.graph_def, '2012/v8/GRAPHS/', 'graph.pbtxt')