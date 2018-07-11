import tensorflow as tf
import numpy as np
from pipeline import SetMaker
from pipeline import Hyperparameters
from pipeline import My_Loss
import os
import csv

sm = SetMaker()
hyp = Hyperparameters()
ml = My_Loss()

class Model:

    with tf.name_scope("weights_and_biases"):
        W_Forget_and_Input = tf.Variable(tf.random_normal(shape = [hyp.hidden_dim + hyp.cell_dim + 1,hyp.cell_dim]), name = "forget_and_input_weight") #note that forget_and_input actually works for forget, and the input is the inverse
        W_Output = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + hyp.cell_dim + 1,hyp.cell_dim]), name="output_weight")
        W_Gate = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + hyp.cell_dim + 1,hyp.cell_dim]), name="gate_weight")

        W_Hidden_to_Out = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim,1]), name = "outwards_propagating_weight")

        B_Forget_and_Input = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name = "forget_and_input_bias")
        B_Output = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="output_bias")
        B_Gate = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="gate_bias")
        B_Hidden_to_Out = tf.Variable(tf.zeros(shape=[1,1]), name = "outwards_propagating_bias")

    with tf.name_scope("placeholders"):
        X = tf.placeholder(shape = [1,1], dtype =  tf.float32, name = "input_placeholder") #waits for the prompt
        Y = tf.placeholder(shape = [1,1], dtype = tf.float32, name = "label") #not used until the last cycle
        H_last = tf.placeholder(shape = [1,hyp.hidden_dim], dtype = tf.float32, name = "last_hidden") #last hidden state (aka the "output")
        C_last = tf.placeholder(shape= [1,hyp.cell_dim], dtype = tf.float32, name = "last_cell") #last cell state

    with tf.name_scope("to_gates"):
        concat_input = tf.concat([X, H_last, C_last], axis=1, name="input_concat")  #concatenates the inputs to one vector
        forget_gate = tf.add(tf.matmul(concat_input, W_Forget_and_Input, name = "f_w_m"),B_Forget_and_Input, name = "f_b_a") #decides which to drop from cell

        gate_gate = tf.add(tf.matmul(concat_input, W_Gate, name = "g_w_m"), B_Gate, name = "g_b_a") #decides which things to change in cell state


    with tf.name_scope("non-linearity"): #makes the gates into what they should be
        forget_gate = tf.sigmoid(forget_gate, name = "sigmoid_forget")

        forget_gate_negated = tf.scalar_mul(-1, forget_gate) #this has to be here because it is after the nonlin
        input_gate = tf.add(tf.ones([1, hyp.cell_dim]), forget_gate_negated, name="making_input_gate")
        input_gate = tf.sigmoid(input_gate, name="sigmoid_input")

        gate_gate = tf.tanh(gate_gate, name = "tanh_gate")

    with tf.name_scope("forget_gate"): #forget gate values and propagate

        current_cell = tf.multiply(forget_gate, C_last, name = "forget_gating")

    with tf.name_scope("suggestion_node"): #suggestion gate
        suggestion_box = tf.multiply(input_gate, gate_gate, name = "input_determiner")
        current_cell = tf.add(suggestion_box, current_cell, name = "input_and_gate_gating")

    with tf.name_scope("output_gate"): #output gate values to hidden

        concat_output_input = tf.concat([X, H_last, current_cell], axis=1,name="input_concat")  # concatenates the inputs to one vector #here, the processed current cell is concatenated and prepared for output
        output_gate = tf.add(tf.matmul(concat_output_input, W_Output, name="o_w_m"), B_Output,name="o_b_a")  # we are making the output gates now, with the peephole.
        output_gate = tf.sigmoid(output_gate,name="sigmoid_output")  # the gate is complete. Note that the two lines were supposed to be back in "to gates" and "non-linearity", but it is necessary to put it here
        current_cell = tf.tanh(current_cell,name="cell_squashing")  # squashing the current cell, branching off now. Note the underscore, means saving a copy.
        current_hidden = tf.multiply(output_gate, current_cell,name="next_hidden")  # we are making the hidden by element-wise multiply of the squashed states

        raw_output = tf.add(tf.matmul(current_hidden, W_Hidden_to_Out, name="WHTO_w_m"), B_Hidden_to_Out,name="BHTO_b_a")  # now, we are propagating outwards

        output = tf.nn.relu(raw_output, name="output")  # makes sure it is not zero.

    with tf.name_scope("loss"):
        loss = tf.square(tf.subtract(output, Y))
        loss = tf.reduce_sum(loss)

    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=hyp.LEARNING_RATE).minimize(loss)

    with tf.name_scope("summaries_and_saver"):
        tf.summary.histogram("W_Forget_and_Input", W_Forget_and_Input)
        tf.summary.histogram("W_Output", W_Output)
        tf.summary.histogram("W_Gate", W_Gate)
        tf.summary.histogram("W_Hidden_to_Out", W_Hidden_to_Out)

        tf.summary.histogram("Forget", forget_gate)
        tf.summary.histogram("Input", input_gate)
        tf.summary.histogram("Output", output_gate)
        tf.summary.histogram("Gate", gate_gate)

        tf.summary.histogram("Cell_State", current_cell)

        tf.summary.histogram("B_Forget_and_Input", B_Forget_and_Input)
        tf.summary.histogram("B_Output", B_Output)
        tf.summary.histogram("B_Gate", B_Gate)
        tf.summary.histogram("B_Hidden_to_Out", B_Hidden_to_Out)

        tf.summary.scalar("Loss", loss)

        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver()
