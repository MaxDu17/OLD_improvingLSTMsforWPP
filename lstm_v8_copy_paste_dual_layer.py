"""Maximilian Du 7-2-18
LSTM implementation with wind data set
Version 8 changes:
multi-layer LSTM!
"""
import tensorflow as tf
import numpy as np
from pipeline import SetMaker
from pipeline import Hyperparameters
from pipeline import Model
import os
import csv

sm = SetMaker()
hyp = Hyperparameters()

with tf.name_scope("layer_1"):
    with tf.name_scope("weights_and_biases"):
        W_Forget_and_Input_1 = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + hyp.cell_dim + 1, hyp.cell_dim]),
                                              name="forget_and_input_weight")  # note that forget_and_input actually works for forget, and the input is the inverse
        W_Output_1 = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + hyp.cell_dim + 1, hyp.cell_dim]),
                                    name="output_weight")
        W_Gate_1 = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + hyp.cell_dim + 1, hyp.cell_dim]),
                                  name="gate_weight")

        W_Hidden_to_Out_1 = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim, 1]),
                                           name="outwards_propagating_weight")

        B_Forget_and_Input_1 = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="forget_and_input_bias")
        B_Output_1 = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="output_bias")
        B_Gate_1 = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="gate_bias")
        B_Hidden_to_Out_1 = tf.Variable(tf.zeros(shape=[1, 1]), name="outwards_propagating_bias")

    with tf.name_scope("placeholders"):
        X_1 = tf.placeholder(shape=[1, 1], dtype=tf.float32, name="input_placeholder")  # waits for the prompt
        H_last_1 = tf.placeholder(shape=[1, hyp.hidden_dim], dtype=tf.float32,
                                     name="last_hidden")  # last hidden state (aka the "output")
        C_last_1 = tf.placeholder(shape=[1, hyp.cell_dim], dtype=tf.float32, name="last_cell")  # last cell state

    with tf.name_scope("to_gates"):
        concat_input_1 = tf.concat([X_1, H_last_1, C_last_1], axis=1,
                                      name="input_concat")  # concatenates the inputs to one vector
        forget_gate_1 = tf.add(tf.matmul(concat_input_1, W_Forget_and_Input_1, name="f_w_m"),
                                  B_Forget_and_Input_1, name="f_b_a")  # decides which to drop from cell

        gate_gate_1 = tf.add(tf.matmul(concat_input_1, W_Gate_1, name="g_w_m"), B_Gate_1,
                                name="g_b_a")  # decides which things to change in cell state

    with tf.name_scope("non-linearity"):  # makes the gates into what they should be
        forget_gate_1 = tf.sigmoid(forget_gate_1, name="sigmoid_forget")
        input_gate_1 = tf.subtract(tf.ones([1, hyp.cell_dim]), forget_gate_1, name="making_input_gate")
        input_gate_1 = tf.sigmoid(input_gate_1, name="sigmoid_input")

        gate_gate_1 = tf.tanh(gate_gate_1, name="tanh_gate")

    with tf.name_scope("forget_gate"):  # forget gate values and propagate

        current_cell_1 = tf.multiply(forget_gate_1, C_last_1, name="forget_gating")

    with tf.name_scope("suggestion_node"):  # suggestion gate
        suggestion_box_1 = tf.multiply(input_gate_1, gate_gate_1, name="input_determiner")
        current_cell_1 = tf.add(suggestion_box_1, current_cell_1, name="input_and_gate_gating")

    with tf.name_scope("output_gate"):  # output gate values to hidden

        concat_output_input_1 = tf.concat([X_1, H_last_1, current_cell_1], axis=1,
                                             name="input_concat")  # concatenates the inputs to one vector #here, the processed current cell is concatenated and prepared for output
        output_gate_1 = tf.add(tf.matmul(concat_output_input_1, W_Output_1, name="o_w_m"), B_Output_1,
                                  name="o_b_a")  # we are making the output gates now, with the peephole.
        output_gate_1 = tf.sigmoid(output_gate_1,
                                      name="sigmoid_output")  # the gate is complete. Note that the two lines were supposed to be back in "to gates" and "non-linearity", but it is necessary to put it here

        current_cell_1 = tf.tanh(current_cell_1,
                                    name="cell_squashing")  # squashing the current cell, branching off now. Note the underscore, means saving a copy.
        current_hidden_1 = tf.multiply(output_gate_1, current_cell_1,
                                          name="next_hidden")  # we are making the hidden by element-wise multiply of the squashed states

        raw_output_1 = tf.add(tf.matmul(current_hidden_1, W_Hidden_to_Out_1, name="WHTO_w_m"),
                                 B_Hidden_to_Out_1, name="BHTO_b_a")  # now, we are propagating outwards

        output_1 = tf.nn.relu(raw_output_1, name="output")  # makes sure it is not zero.

    with tf.name_scope("summaries_and_saver"):
        tf.summary.histogram("W_Forget_and_Input", W_Forget_and_Input_1)
        tf.summary.histogram("W_Output", W_Output_1)
        tf.summary.histogram("W_Gate", W_Gate_1)
        tf.summary.histogram("W_Hidden_to_Out", W_Hidden_to_Out_1)

        tf.summary.histogram("Forget", forget_gate_1)
        tf.summary.histogram("Input", input_gate_1)
        tf.summary.histogram("Output", output_gate_1)
        tf.summary.histogram("Gate", gate_gate_1)

        tf.summary.histogram("B_Forget_and_Input", B_Forget_and_Input_1)
        tf.summary.histogram("B_Output", B_Output_1)
        tf.summary.histogram("B_Gate", B_Gate_1)
        tf.summary.histogram("B_Hidden_to_Out", B_Hidden_to_Out_1)
with tf.name_scope("layer_2"):
    with tf.name_scope("weights_and_biases"):
        W_Forget_and_Input_2 = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + hyp.cell_dim + 1, hyp.cell_dim]),
                                              name="forget_and_input_weight")  # note that forget_and_input actually works for forget, and the input is the inverse
        W_Output_2 = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + hyp.cell_dim + 1, hyp.cell_dim]),
                                    name="output_weight")
        W_Gate_2 = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim + hyp.cell_dim + 1, hyp.cell_dim]),
                                  name="gate_weight")

        W_Hidden_to_Out_2 = tf.Variable(tf.random_normal(shape=[hyp.hidden_dim, 1]),
                                           name="outwards_propagating_weight")

        B_Forget_and_Input_2 = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="forget_and_input_bias")
        B_Output_2 = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="output_bias")
        B_Gate_2 = tf.Variable(tf.zeros(shape=[1, hyp.cell_dim]), name="gate_bias")
        B_Hidden_to_Out_2 = tf.Variable(tf.zeros(shape=[1, 1]), name="outwards_propagating_bias")

    with tf.name_scope("placeholders"):
        X_2 = tf.placeholder(shape=[1, 1], dtype=tf.float32, name="input_placeholder")  # waits for the prompt
        H_last_2 = tf.placeholder(shape=[1, hyp.hidden_dim], dtype=tf.float32,
                                     name="last_hidden")  # last hidden state (aka the "output")
        C_last_2 = tf.placeholder(shape=[1, hyp.cell_dim], dtype=tf.float32, name="last_cell")  # last cell state

    with tf.name_scope("to_gates"):
        concat_input_2 = tf.concat([X_2, H_last_2, C_last_2], axis=1,
                                      name="input_concat")  # concatenates the inputs to one vector
        forget_gate_2 = tf.add(tf.matmul(concat_input_2, W_Forget_and_Input_2, name="f_w_m"),
                                  B_Forget_and_Input_2, name="f_b_a")  # decides which to drop from cell

        gate_gate_2 = tf.add(tf.matmul(concat_input_2, W_Gate_2, name="g_w_m"), B_Gate_2,
                                name="g_b_a")  # decides which things to change in cell state

    with tf.name_scope("non-linearity"):  # makes the gates into what they should be
        forget_gate_2 = tf.sigmoid(forget_gate_2, name="sigmoid_forget")
        input_gate_2 = tf.subtract(tf.ones([1, hyp.cell_dim]), forget_gate_2, name="making_input_gate")
        input_gate_2 = tf.sigmoid(input_gate_2, name="sigmoid_input")

        gate_gate_2 = tf.tanh(gate_gate_2, name="tanh_gate")

    with tf.name_scope("forget_gate"):  # forget gate values and propagate

        current_cell_2 = tf.multiply(forget_gate_2, C_last_2, name="forget_gating")

    with tf.name_scope("suggestion_node"):  # suggestion gate
        suggestion_box_2 = tf.multiply(input_gate_2, gate_gate_2, name="input_determiner")
        current_cell_2 = tf.add(suggestion_box_2, current_cell_2, name="input_and_gate_gating")

    with tf.name_scope("output_gate"):  # output gate values to hidden

        concat_output_input_2 = tf.concat([X_2, H_last_2, current_cell_2], axis=1,
                                             name="input_concat")  # concatenates the inputs to one vector #here, the processed current cell is concatenated and prepared for output
        output_gate_2 = tf.add(tf.matmul(concat_output_input_2, W_Output_2, name="o_w_m"), B_Output_2,
                                  name="o_b_a")  # we are making the output gates now, with the peephole.
        output_gate_2 = tf.sigmoid(output_gate_2,
                                      name="sigmoid_output")  # the gate is complete. Note that the two lines were supposed to be back in "to gates" and "non-linearity", but it is necessary to put it here

        current_cell_2 = tf.tanh(current_cell_2,
                                    name="cell_squashing")  # squashing the current cell, branching off now. Note the underscore, means saving a copy.
        current_hidden_2 = tf.multiply(output_gate_2, current_cell_2,
                                          name="next_hidden")  # we are making the hidden by element-wise multiply of the squashed states

        raw_output_2 = tf.add(tf.matmul(current_hidden_2, W_Hidden_to_Out_2, name="WHTO_w_m"),
                                 B_Hidden_to_Out_2, name="BHTO_b_a")  # now, we are propagating outwards

        output_2 = tf.nn.relu(raw_output_2, name="output")  # makes sure it is not zero.

    with tf.name_scope("summaries_and_saver"):
        tf.summary.histogram("W_Forget_and_Input", W_Forget_and_Input_2)
        tf.summary.histogram("W_Output", W_Output_2)
        tf.summary.histogram("W_Gate", W_Gate_2)
        tf.summary.histogram("W_Hidden_to_Out", W_Hidden_to_Out_2)

        tf.summary.histogram("Forget", forget_gate_2)
        tf.summary.histogram("Input", input_gate_2)
        tf.summary.histogram("Output", output_gate_2)
        tf.summary.histogram("Gate", gate_gate_2)

        tf.summary.histogram("B_Forget_and_Input", B_Forget_and_Input_2)
        tf.summary.histogram("B_Output", B_Output_2)
        tf.summary.histogram("B_Gate", B_Gate_2)
        tf.summary.histogram("B_Hidden_to_Out", B_Hidden_to_Out_2)

input_1, input_2 = list()
feed_1 ={X_1: input_1, H_last_1: next_hidd_1, C_last_1: next_cell_1}
feed_2 ={X_2:input_2, H_last_2:next_hidd_2, C_last_2:next_cell_2}


def zero_states():
    global next_cell_1, next_cell_2, next_hidd_1, next_hidd_2
    next_cell_1 = np.zeros(shape=[1, hyp.cell_dim])
    next_hidd_1 = np.zeros(shape=[1, hyp.hidden_dim])
    next_cell_2 = np.zeros(shape=[1, hyp.cell_dim])
    next_hidd_2 = np.zeros(shape=[1, hyp.hidden_dim])

with tf.name_scope("placeholders"):
    Y = tf.placeholder(shape=[1, 1], dtype=tf.float32, name="label")  # not used until the last cycle

with tf.name_scope("loss"):
    loss = tf.square(tf.subtract(output_2, Y))
    loss = tf.reduce_sum(loss)

with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=hyp.LEARNING_RATE).minimize(loss)

with tf.name_scope("summaries_and_saver"):
    tf.summary.histogram("Cell_State", current_cell_1)
    tf.summary.histogram("Cell_State", current_cell_2)

    tf.summary.scalar("Loss", loss)

    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('2012/v8/models/'))
    if ckpt and ckpt.model_checkpoint_path:
        query = input("checkpoint detected! Would you like to restore from <" + ckpt.model_checkpoint_path + "> ?(y or n)\n")
        if query == 'y':
            saver.restore(sess, ckpt.model_checkpoint_path)
            if np.sum(B_Forget_and_Input_1.eval()) != 0:
                print("session restored!")
        else:
            print("session discarded!")


    sm.create_training_set()
    log_loss = open("2012/v8/GRAPHS/LOSS.csv", "w")
    validation = open("2012/v8/GRAPHS/VALIDATION.csv", "w")
    test = open("2012/v8/GRAPHS/TEST.csv", "w")

    logger = csv.writer(log_loss, lineterminator="\n")
    validation_logger = csv.writer(validation, lineterminator="\n")
    test_logger = csv.writer(test, lineterminator="\n")

    sess.run(tf.global_variables_initializer())

    tf.train.write_graph(sess.graph_def, '2012/v8/GRAPHS/', 'graph.pbtxt')
    writer = tf.summary.FileWriter("2012/v8/GRAPHS/", sess.graph)

    summary = None

    zero_states()

    for epoch in range(hyp.EPOCHS):

        reset = sm.next_epoch()
        label = sm.get_label()
        label = np.reshape(label, [1, 1])
        loss_ = 0
        
        if reset: #this allows for hidden states to reset after the training set loops back around
            zero_states()

        for counter in range(hyp.FOOTPRINT):
            input_1 = sm.next_sample()
            input_1 = np.reshape(input_1, [1,1])
            if counter < hyp.FOOTPRINT-1:

                next_cell_1, next_hidd_1, output_1 = sess.run(
                    [current_cell_1, current_hidden_1, output_1],
                    feed_dict=feed_1)

                input_2 = output_1

                next_cell_2, next_hidd_2 = sess.run([current_cell_2, current_hidden_2],
                                                    feed_dict=feed_2)

            else:
                next_cell_1, next_hidd_1, output_1 = sess.run(
                    [current_cell_1, current_hidden_1, output_1],
                    feed_dict=feed_1)

                input_2 = output_1

                next_cell_2, next_hidd_2, output_2  = sess.run([current_cell_2, current_hidden_2, loss, summary_op, optimizer],
                                                    feed_dict=feed_2)
                loss_, summary, _ = sess.run([loss, summary_op, optimizer], feed_dict = {Y:label})
        logger.writerow([loss_])

        if epoch%50 == 0:
            writer.add_summary(summary, global_step=epoch)
            print("I finished epoch ", epoch, " out of ", hyp.EPOCHS, " epochs")
            print("The absolute value loss for this sample is ", np.sqrt(loss_))
            print("predicted number: ", output_2, ", real number: ", label)

############################################################### validation code here
        if epoch%2000 == 0 and epoch>498:
            saver.save(sess, "2012/v8/models/LSTMv8", global_step=epoch)
            print("saved model")

            hidden_saver_1, hidden_saver_2, cell_saver_1, cell_saver_2 = list() #initializing stuff

            next_cell_hold_1 = next_cell_1
            next_hidd_hold_1 = next_hidd_1
            next_hidd_hold_2 = next_hidd_2
            next_cell_hold_2 = next_cell_2

            sm.create_validation_set()

            RMS_loss = 0.0

            zero_states()

            for i in range(hyp.VALIDATION_NUMBER):
                sm.next_epoch_valid()
                label_ = sm.get_label()
                label = np.reshape(label_, [1, 1])
                # this gets each 10th
                for counter in range(hyp.FOOTPRINT):
                    data = sm.next_sample()
                    input_1 = np.reshape(data, [1, 1])
                    if counter < hyp.FOOTPRINT - 1:

                        next_cell_1, next_hidd_1, output_1 = sess.run(
                            [current_cell_1, current_hidden_1, output_1],
                            feed_dict=feed_1)

                        input_2 = output_1

                        next_cell_2, next_hidd_2 = sess.run([current_cell_2, current_hidden_2],
                                                            feed_dict=feed_2)
                        if counter == 0:
                            hidden_saver_1 = next_hidd_1  # saves THIS state for the next round
                            hidden_saver_2 = next_hidd_2
                            cell_saver_1 = next_cell_1
                            cell_saver_2 = next_cell_2


                    else:

                        next_cell_1, next_hidd_1, output_1 = sess.run(
                            [current_cell_1, current_hidden_1, output_1],
                            feed_dict=feed_1)

                        input_2 = output_1

                        next_cell_2, next_hidd_2, output_2, loss_, summary, _ = sess.run(
                            [current_cell_2, current_hidden_2, loss, summary_op, optimizer],
                            feed_dict=feed_2)

                next_cell_1 = cell_saver_1
                next_cell_2 = cell_saver_2
                next_hidd_1 = hidden_saver_1
                RMS_loss += np.sqrt(loss_)
                sm.clear_valid_counter()

            RMS_loss = RMS_loss/hyp.VALIDATION_NUMBER
            print("validation: RMS loss is ", RMS_loss)
            validation_logger.writerow([epoch,RMS_loss])

            next_cell_1 = next_cell_hold_1
            next_hidd_1 = next_hidd_hold_1
            next_hidd_2 = next_hidd_hold_2
            next_cell_2 = next_cell_hold_2

############################################################## below here is the test code

    RMS_loss = 0.0
    zero_states()
    hidden_saver_1, hidden_saver_2, cell_saver_1, cell_saver_2 = list()  # initializing stuff

    for test in range(hyp.Info.TEST_SIZE): #this will be replaced later

        sm.next_epoch_test_single_shift()
        label_ = sm.get_label()
        label = np.reshape(label_, [1, 1])
        # this gets each 10th
        for counter in range(hyp.FOOTPRINT):
            data = sm.next_sample()
            input_1 = np.reshape(data, [1, 1])
            if counter < hyp.FOOTPRINT - 1:

                next_cell_1, next_hidd_1, output_1 = sess.run(
                    [current_cell_1, current_hidden_1, output_1],
                    feed_dict=feed_1)

                input_2 = output_1

                next_cell_2, next_hidd_2 = sess.run([current_cell_2, current_hidden_2],
                                                    feed_dict=feed_2)
                if counter == 0:
                    hidden_saver_1 = next_hidd_1  # saves THIS state for the next round
                    hidden_saver_2 = next_hidd_2
                    cell_saver_1 = next_cell_1
                    cell_saver_2 = next_cell_2


            else:

                next_cell_1, next_hidd_1, output_1 = sess.run(
                    [current_cell_1, current_hidden_1, output_1],
                    feed_dict=feed_1)

                input_2 = output_1

                next_cell_2, next_hidd_2, output_2, loss_, summary, _ = sess.run(
                    [current_cell_2, current_hidden_2, loss, summary_op, optimizer],
                    feed_dict=feed_2)

                carrier = [label_, output_2[0][0], np.sqrt(loss_)]
                test_logger.writerow(carrier)
        next_cell_1 = cell_saver_1
        next_cell_2 = cell_saver_2
        next_hidd_1 = hidden_saver_1
        next_hidd_2 = hidden_saver_2
        RMS_loss += np.sqrt(loss_)
    RMS_loss = RMS_loss / hyp.Info.TEST_SIZE
    print("test: rms loss is ", RMS_loss)
    test_logger.writerow(["final adaptive loss average", RMS_loss])