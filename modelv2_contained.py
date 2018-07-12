import tensorflow as tf
from pipeline import SetMaker
from pipeline import Hyperparameters
from pipeline import My_Loss

sm = SetMaker()
hyp = Hyperparameters()
ml = My_Loss()

class Model2:

    def create_graph_first_layer(self):
        with tf.name_scope("layer_1"):

            with tf.name_scope("weights_and_biases"):
                self.W_Forget = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + 1, hyp.cell_dim]), name="forget_weight")
                self.W_Output = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + 1, hyp.cell_dim]), name="output_weight")
                self.W_Gate = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + 1, hyp.cell_dim]), name="gate_weight")
                self.W_Input = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + 1, hyp.cell_dim]), name="input_weight")
                self.W_Hidden_to_Out = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim, 1]),
                                              name="outwards_propagating_weight")

                self.B_Forget = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="forget_bias")
                self.B_Output = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="output_bias")
                self.B_Gate = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="gate_bias")
                self.B_Input = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="input_bias")
                self.B_Hidden_to_Out = tf.Variable(tf.zeros(shape=[1, 1]), name="outwards_propagating_bias")

            with tf.name_scope("placeholders"):
                self.X = tf.placeholder(shape=[1, 1], dtype=tf.float32, name="input_placeholder")  # waits for the prompt
                self.H_last = tf.placeholder(shape=[1, hyp.hidden_dim], dtype=tf.float32, name="last_hidden")  # last hidden state (aka the "output")
                self.C_last = tf.placeholder(shape=[1, hyp.cell_dim], dtype=tf.float32, name="last_cell")  # last cell state

            with tf.name_scope("to_gates"):
                self.concat_input = tf.concat([self.X, self.H_last], axis=1,
                                         name="input_concat")  # concatenates the inputs to one vector
                self.forget_gate = tf.add(tf.matmul(self.concat_input, self.W_Forget, name="f_w_m"), self.B_Forget,
                                     name="f_b_a")  # decides which to drop from cell
                self.output_gate = tf.add(tf.matmul(self.concat_input, self.W_Output, name="o_w_m"), self.B_Output,
                                     name="o_b_a")  # decides which to reveal to next_hidd/output
                self.gate_gate = tf.add(tf.matmul(self.concat_input, self.W_Gate, name="g_w_m"), self.B_Gate,
                                   name="g_b_a")  # decides which things to change in cell state
                self.input_gate = tf.add(tf.matmul(self.concat_input, self.W_Input, name="i_w_m"), self.B_Input,
                                    name="i_b_a")  # decides which of the changes to accept

            with tf.name_scope("non-linearity"):  # makes the gates into what they should be
                self.forget_gate = tf.sigmoid(self.forget_gate, name="sigmoid_forget")
                self.output_gate = tf.sigmoid(self.output_gate, name="sigmoid_output")
                self.input_gate = tf.sigmoid(self.input_gate, name="sigmoid_input")
                self.gate_gate = tf.tanh(self.gate_gate, name="tanh_gate")

            with tf.name_scope("forget_gate"):  # forget gate values and propagate
                self.current_cell = tf.multiply(self.forget_gate, self.C_last, name="forget_gating")

            with tf.name_scope("suggestion_node"):  # suggestion gate
                self.suggestion_box = tf.multiply(self.input_gate, self.gate_gate, name="input_determiner")
                self.current_cell = tf.add(self.suggestion_box, self.current_cell, name="input_and_gate_gating")

            with tf.name_scope("output_gate"):  # output gate values to hidden
                self.current_cell = tf.tanh(self.current_cell, name="output_presquashing")
                self.current_hidden = tf.multiply(self.output_gate, self.current_cell, name="next_hidden")
                self.raw_output = tf.add(tf.matmul(self.current_hidden, self.W_Hidden_to_Out, name="WHTO_w_m"), self.B_Hidden_to_Out,
                                    name="BHTO_b_a")
                self.output = tf.nn.relu(self.raw_output, name="output")

            with tf.name_scope("summaries_and_saver"):
                tf.summary.histogram("W_Forget", self.W_Forget)
                tf.summary.histogram("W_Input", self.W_Input)
                tf.summary.histogram("W_Output", self.W_Output)
                tf.summary.histogram("W_Gate", self.W_Gate)
                tf.summary.histogram("W_Hidden_to_Out", self.W_Hidden_to_Out)

                tf.summary.histogram("Cell_State", self.current_cell)

                tf.summary.histogram("B_Forget", self.B_Forget)
                tf.summary.histogram("B_Input", self.B_Input)
                tf.summary.histogram("B_Output", self.B_Output)
                tf.summary.histogram("B_Gate", self.B_Gate)
                tf.summary.histogram("B_Hidden_to_Out", self.B_Hidden_to_Out)

        return self.output, self.current_cell, self.current_hidden

    def create_graph(self, layer_number, last_output): #last_output links them together
            with tf.name_scope("layer_" + str(layer_number)):
                with tf.name_scope("weights_and_biases"):
                    self.W_Forget = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + 1, hyp.cell_dim]),
                                                name="forget_weight")
                    self.W_Output = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + 1, hyp.cell_dim]),
                                                name="output_weight")
                    self.W_Gate = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + 1, hyp.cell_dim]),
                                              name="gate_weight")
                    self.W_Input = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + 1, hyp.cell_dim]),
                                               name="input_weight")
                    self.W_Hidden_to_Out = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim, 1]),
                                                       name="outwards_propagating_weight")

                    self.B_Forget = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="forget_bias")
                    self.B_Output = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="output_bias")
                    self.B_Gate = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="gate_bias")
                    self.B_Input = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="input_bias")
                    self.B_Hidden_to_Out = tf.Variable(tf.zeros(shape=[1, 1]), name="outwards_propagating_bias")
                '''
                with tf.name_scope("starting_states"):
                    H_begin = tf.Variable(tf.random_normal(shape = [1, hyp.hidden_dim], name = "starting_hidd_val"))
                    C_begin = tf.Variable(tf.random_normal(shape=[1, hyp.cell_dim], name="starting_cell_val"))
                '''
                with tf.name_scope("placeholders"):
                    self.H_last = tf.placeholder(shape=[1, hyp.hidden_dim], dtype=tf.float32,
                                                 name="last_hidden")  # last hidden state (aka the "output")
                    self.C_last = tf.placeholder(shape=[1, hyp.cell_dim], dtype=tf.float32,
                                                 name="last_cell")  # last cell state

                with tf.name_scope("to_gates"):
                    self.concat_input = tf.concat([last_output, self.H_last], axis=1,
                                                  name="input_concat")  # concatenates the inputs to one vector
                    self.forget_gate = tf.add(tf.matmul(self.concat_input, self.W_Forget, name="f_w_m"), self.B_Forget,
                                              name="f_b_a")  # decides which to drop from cell
                    self.output_gate = tf.add(tf.matmul(self.concat_input, self.W_Output, name="o_w_m"), self.B_Output,
                                              name="o_b_a")  # decides which to reveal to next_hidd/output
                    self.gate_gate = tf.add(tf.matmul(self.concat_input, self.W_Gate, name="g_w_m"), self.B_Gate,
                                            name="g_b_a")  # decides which things to change in cell state
                    self.input_gate = tf.add(tf.matmul(self.concat_input, self.W_Input, name="i_w_m"), self.B_Input,
                                             name="i_b_a")  # decides which of the changes to accept

                with tf.name_scope("non-linearity"):  # makes the gates into what they should be
                    self.forget_gate = tf.sigmoid(self.forget_gate, name="sigmoid_forget")
                    self.output_gate = tf.sigmoid(self.output_gate, name="sigmoid_output")
                    self.input_gate = tf.sigmoid(self.input_gate, name="sigmoid_input")
                    self.gate_gate = tf.tanh(self.gate_gate, name="tanh_gate")

                with tf.name_scope("forget_gate"):  # forget gate values and propagate
                    self.current_cell = tf.multiply(self.forget_gate, self.C_last, name="forget_gating")

                with tf.name_scope("suggestion_node"):  # suggestion gate
                    self.suggestion_box = tf.multiply(self.input_gate, self.gate_gate, name="input_determiner")
                    self.current_cell = tf.add(self.suggestion_box, self.current_cell, name="input_and_gate_gating")

                with tf.name_scope("output_gate"):  # output gate values to hidden
                    self.current_cell = tf.tanh(self.current_cell, name="output_presquashing")
                    self.current_hidden = tf.multiply(self.output_gate, self.current_cell, name="next_hidden")
                    self.raw_output = tf.add(tf.matmul(self.current_hidden, self.W_Hidden_to_Out, name="WHTO_w_m"),
                                             self.B_Hidden_to_Out,
                                             name="BHTO_b_a")
                    self.output = tf.nn.relu(self.raw_output, name="output")

                with tf.name_scope("summaries_and_saver"):
                    tf.summary.histogram("W_Forget", self.W_Forget)
                    tf.summary.histogram("W_Input", self.W_Input)
                    tf.summary.histogram("W_Output", self.W_Output)
                    tf.summary.histogram("W_Gate", self.W_Gate)
                    tf.summary.histogram("W_Hidden_to_Out", self.W_Hidden_to_Out)

                    tf.summary.histogram("Cell_State", self.current_cell)

                    tf.summary.histogram("B_Forget", self.B_Forget)
                    tf.summary.histogram("B_Input", self.B_Input)
                    tf.summary.histogram("B_Output", self.B_Output)
                    tf.summary.histogram("B_Gate", self.B_Gate)
                    tf.summary.histogram("B_Hidden_to_Out", self.B_Hidden_to_Out)

            return self.output, self.current_cell, self.current_hidden