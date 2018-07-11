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

    def create_graph(self, layer_number):
        with tf.name_scope("layer_" + str(layer_number)):

            with tf.name_scope("weights_and_biases"):
                self.W_Forget_and_Input = tf.Variable(tf.random_normal(shape = [hyp.hidden_dim + hyp.cell_dim + 1,hyp.cell_dim]), name = "forget_and_input_weight") #note that forget_and_input actually works for forget, and the input is the inverse
                self.W_Output = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + hyp.cell_dim + 1,hyp.cell_dim]), name="output_weight")
                self.W_Gate = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + hyp.cell_dim + 1,hyp.cell_dim]), name="gate_weight")

                self.W_Hidden_to_Out = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim,1]), name = "outwards_propagating_weight")

                self.B_Forget_and_Input = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name = "forget_and_input_bias")
                self.B_Output = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="output_bias")
                self.B_Gate = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="gate_bias")
                self.B_Hidden_to_Out = tf.Variable(tf.zeros(shape=[1,1]), name = "outwards_propagating_bias")

            with tf.name_scope("placeholders"):
                self.X = tf.placeholder(shape = [1,1], dtype =  tf.float32, name = "input_placeholder") #waits for the prompt
                self.H_last = tf.placeholder(shape = [1,hyp.hidden_dim], dtype = tf.float32, name = "last_hidden") #last hidden state (aka the "output")
                self.C_last = tf.placeholder(shape= [1,hyp.cell_dim], dtype = tf.float32, name = "last_cell") #last cell state

            with tf.name_scope("to_gates"):
                self.concat_input = tf.concat([self.X, self.H_last, self.C_last], axis=1, name="input_concat")  #concatenates the inputs to one vector
                self.forget_gate = tf.add(tf.matmul(self.concat_input, self.W_Forget_and_Input, name = "f_w_m"),self.B_Forget_and_Input, name = "f_b_a") #decides which to drop from cell

                self.gate_gate = tf.add(tf.matmul(self.concat_input, self.W_Gate, name = "g_w_m"), self.B_Gate, name = "g_b_a") #decides which things to change in cell state


            with tf.name_scope("non-linearity"): #makes the gates into what they should be
                self.forget_gate = tf.sigmoid(self.forget_gate, name = "sigmoid_forget")
                self.input_gate = tf.subtract(tf.ones([1, hyp.cell_dim]), self.forget_gate, name="making_input_gate")
                self.input_gate = tf.sigmoid(self.input_gate, name="sigmoid_input")

                self.gate_gate = tf.tanh(self.gate_gate, name = "tanh_gate")

            with tf.name_scope("forget_gate"): #forget gate values and propagate

                self.current_cell = tf.multiply(self.forget_gate, self.C_last, name = "forget_gating")

            with tf.name_scope("suggestion_node"): #suggestion gate
                self.suggestion_box = tf.multiply(self.input_gate, self.gate_gate, name = "input_determiner")
                self.current_cell = tf.add(self.suggestion_box, self.current_cell, name = "input_and_gate_gating")

            with tf.name_scope("output_gate"): #output gate values to hidden

                self.concat_output_input = tf.concat([self.X, self.H_last, self.current_cell], axis=1,name="input_concat")  # concatenates the inputs to one vector #here, the processed current cell is concatenated and prepared for output
                self.output_gate = tf.add(tf.matmul(self.concat_output_input, self.W_Output, name="o_w_m"), self.B_Output,name="o_b_a")  # we are making the output gates now, with the peephole.
                self.output_gate = tf.sigmoid(self.output_gate,name="sigmoid_output")  # the gate is complete. Note that the two lines were supposed to be back in "to gates" and "non-linearity", but it is necessary to put it here
                self.current_cell = tf.tanh(self.current_cell,name="cell_squashing")  # squashing the current cell, branching off now. Note the underscore, means saving a copy.
                self.current_hidden = tf.multiply(self.output_gate, self.current_cell,name="next_hidden")  # we are making the hidden by element-wise multiply of the squashed states

                self.raw_output = tf.add(tf.matmul(self.current_hidden, self.W_Hidden_to_Out, name="WHTO_w_m"), self.B_Hidden_to_Out,name="BHTO_b_a")  # now, we are propagating outwards

                self.output = tf.nn.relu(self.raw_output, name="output")  # makes sure it is not zero.

            return self.output, self.current_cell, self.current_hidden


