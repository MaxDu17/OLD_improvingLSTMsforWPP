import tensorflow as tf
from pipeline import SetMaker
from pipeline import Hyperparameters
from pipeline import My_Loss

sm = SetMaker()
hyp = Hyperparameters()
ml = My_Loss()

class Model2:

    def create_graph(self, inputs, layer_number):
        with tf.name_scope("layer_" + str(layer_number)):
            with tf.name_scope("weights_and_biases"):
                W_Forget = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + 1, hyp.cell_dim]), name="forget_weight")
                W_Output = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + 1, hyp.cell_dim]), name="output_weight")
                W_Gate = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + 1, hyp.cell_dim]), name="gate_weight")
                W_Input = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + 1, hyp.cell_dim]), name="input_weight")
                W_Hidden_to_Out = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim, 1]),
                                              name="outwards_propagating_weight")

                B_Forget = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="forget_bias")
                B_Output = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="output_bias")
                B_Gate = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="gate_bias")
                B_Input = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="input_bias")
                B_Hidden_to_Out = tf.Variable(tf.zeros(shape=[1, 1]), name="outwards_propagating_bias")

            with tf.name_scope("placeholders"):
                init_state = tf.placeholder(shape=[2, 1, hyp.cell_dim], dtype=tf.float32, name="initial_states")

            def step(last_state, X):
                with tf.name_scope("to_gates"):
                    C_last, H_last = tf.unstack(last_state)
                    concat_input = tf.concat([X, H_last], axis=1,
                                             name="input_concat")  # concatenates the inputs to one vector
                    forget_gate = tf.add(tf.matmul(concat_input, W_Forget, name="f_w_m"), B_Forget,
                                         name="f_b_a")  # decides which to drop from cell
                    output_gate = tf.add(tf.matmul(concat_input, W_Output, name="o_w_m"), B_Output,
                                         name="o_b_a")  # decides which to reveal to next_hidd/output
                    gate_gate = tf.add(tf.matmul(concat_input, W_Gate, name="g_w_m"), B_Gate,
                                       name="g_b_a")  # decides which things to change in cell state
                    input_gate = tf.add(tf.matmul(concat_input, W_Input, name="i_w_m"), B_Input,
                                        name="i_b_a")  # decides which of the changes to accept

                with tf.name_scope("non-linearity"):  # makes the gates into what they should be
                    forget_gate = tf.sigmoid(forget_gate, name="sigmoid_forget")
                    output_gate = tf.sigmoid(output_gate, name="sigmoid_output")
                    input_gate = tf.sigmoid(input_gate, name="sigmoid_input")
                    gate_gate = tf.tanh(gate_gate, name="tanh_gate")

                with tf.name_scope("forget_gate"):  # forget gate values and propagate
                    current_cell = tf.multiply(forget_gate, C_last, name="forget_gating")

                with tf.name_scope("suggestion_node"):  # suggestion gate
                    suggestion_box = tf.multiply(input_gate, gate_gate, name="input_determiner")
                    current_cell = tf.add(suggestion_box, current_cell, name="input_and_gate_gating")

                with tf.name_scope("output_gate"):  # output gate values to hidden
                    current_cell = tf.tanh(current_cell, name="output_presquashing")
                    current_hidden = tf.multiply(output_gate, current_cell, name="next_hidden")
                    states = tf.stack([current_cell, current_hidden])
                return states

            with tf.name_scope("forward_roll"):
                states_list = tf.scan(fn=step, elems=inputs, initializer=init_state, name="scan")

            with tf.name_scope("summaries_and_saver"):
                tf.summary.histogram("W_Forget", W_Forget)
                tf.summary.histogram("W_Input", W_Input)
                tf.summary.histogram("W_Output", W_Output)
                tf.summary.histogram("W_Gate", W_Gate)
                tf.summary.histogram("W_Hidden_to_Out", W_Hidden_to_Out)

                tf.summary.histogram("B_Forget", B_Forget)
                tf.summary.histogram("B_Input", B_Input)
                tf.summary.histogram("B_Output", B_Output)
                tf.summary.histogram("B_Gate", B_Gate)
                tf.summary.histogram("B_Hidden_to_Out", B_Hidden_to_Out)

        return states_list

