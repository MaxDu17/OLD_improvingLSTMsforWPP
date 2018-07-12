"""Maximilian Du 7-2-18
LSTM implementation with wind data set
Version 2 changes:
-relu at the end (whoops! Negative wind!)
-continuous thread
-more markups

"""
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
    last_state = tf.placeholder(shape = [2,1,hyp.cell_dim], dtype = tf.float32, name = "last_states")

with tf.name_scope("to_gates"):
    C_last, H_last = tf.unstack(last_state)
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
    current_hidden = tf.multiply(output_gate, current_cell, name="next_hidden")
    raw_output = tf.add(tf.matmul(current_hidden, W_Hidden_to_Out, name = "WHTO_w_m"), B_Hidden_to_Out, name = "BHTO_b_a")
    output = tf.nn.relu(raw_output, name = "output")
    states = tf.stack([current_cell, current_hidden])

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
    tf.summary.histogram("W_Hidden_to_Out", W_Hidden_to_Out)

    tf.summary.histogram("Cell_State", current_cell)

    tf.summary.histogram("B_Forget", B_Forget)
    tf.summary.histogram("B_Input", B_Input)
    tf.summary.histogram("B_Output", B_Output)
    tf.summary.histogram("B_Gate", B_Gate)
    tf.summary.histogram("B_Hidden_to_Out", B_Hidden_to_Out)

    tf.summary.scalar("Loss", loss)

    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('2012/v2/models_CONTAINED/'))
    if ckpt and ckpt.model_checkpoint_path:
        query = input("checkpoint detected! Would you like to restore from <" + ckpt.model_checkpoint_path + "> ?(y or n)\n")
        if query == 'y':
            saver.restore(sess, ckpt.model_checkpoint_path)
            if np.sum(B_Forget.eval()) != 0:
                print("session restored!")
        else:
            print("session discarded!")


    sm.create_training_set()
    log_loss = open("2012/v2/GRAPHS_CONTAINED/LOSS.csv", "w")
    validation = open("2012/v2/GRAPHS_CONTAINED/VALIDATION.csv", "w")
    test = open("2012/v2/GRAPHS_CONTAINED/TEST.csv", "w")

    logger = csv.writer(log_loss, lineterminator="\n")
    validation_logger = csv.writer(validation, lineterminator="\n")
    test_logger = csv.writer(test, lineterminator="\n")

    sess.run(tf.global_variables_initializer())

    tf.train.write_graph(sess.graph_def, 'v2/GRAPHS_CONTAINED/', 'graph.pbtxt')
    writer = tf.summary.FileWriter("v2/GRAPHS_CONAINED/", sess.graph)

    summary = None
    next_state = np.zeros(shape=[2,1,hyp.cell_dim])
    
    for epoch in range(hyp.EPOCHS):
        sm.next_epoch()
        label = sm.get_label()
        label = np.reshape(label, [1, 1])
        loss_ = 0
        for counter in range(hyp.FOOTPRINT):
            data = sm.next_sample()
            data = np.reshape(data, [1,1])
            if counter < hyp.FOOTPRINT-1:
                next_state = sess.run(states,feed_dict= {X:data, last_state:next_state})
            else:
                next_state, output_, loss_, summary, _ = sess.run([states, output, loss, summary_op, optimizer],
                                                feed_dict={X:data, Y:label, last_state:next_state})

        logger.writerow([loss_])

        if epoch % 50 == 0:
            writer.add_summary(summary, global_step=epoch)
            print("I finished epoch ", epoch, " out of ", hyp.EPOCHS, " epochs")
            print("The absolute value loss for this sample is ", np.sqrt(loss_))
            print("predicted number: ", output_, ", real number: ", label)

        if epoch % 2000 == 0 and epoch > 498:
            saver.save(sess, "2012/v2/models/LSTMv2", global_step=epoch)
            print("saved model")

            next_state_hold = next_state
            sm.create_validation_set()
            average_rms_loss = 0.0
            next_state = np.zeros(shape=[2, 1, hyp.cell_dim])
            for i in range(hyp.VALIDATION_NUMBER):
                sm.next_epoch_valid()
                label_ = sm.get_label()
                label = np.reshape(label_, [1, 1])
                # this gets each 10th
                for counter in range(hyp.FOOTPRINT):
                    data = sm.next_sample()
                    data = np.reshape(data, [1, 1])
                    if counter < hyp.FOOTPRINT - 1:

                        next_state = sess.run(states, feed_dict={X: data, last_state: next_state})
                        if counter == 0:
                            state_saver = next_state
                    else:
                        next_state, output_, loss_ = sess.run(
                            [states, output, loss],
                            feed_dict={X: data, Y: label, last_state: next_state})

                next_state = state_saver

                average_rms_loss += np.sqrt(loss_)
                sm.clear_valid_counter()

            average_rms_loss = average_rms_loss / hyp.VALIDATION_NUMBER
            print("validation: RMS loss is ", average_rms_loss)
            validation_logger.writerow([epoch, average_rms_loss])

            next_state = next_state_hold

    RMS_loss = 0.0
    next_state = np.zeros(shape=[2, 1, hyp.cell_dim])
    for test in range(hyp.Info.TEST_SIZE):  # this will be replaced later

        sm.next_epoch_test_single_shift()
        label_ = sm.get_label()
        label = np.reshape(label_, [1, 1])
        # this gets each 10th
        for counter in range(hyp.FOOTPRINT):
            data = sm.next_sample()
            data = np.reshape(data, [1, 1])
            if counter < hyp.FOOTPRINT - 1:

                next_state = sess.run([states],
                                      feed_dict={X: data, last_state: next_state})
                if counter == 0:
                    state_saver = next_state
            else:
                next_state, output_, loss_ = sess.run(
                    [states, output, loss],
                    feed_dict={X: data, Y: label, last_state: next_state})

                carrier = [label_, output_[0][0], np.sqrt(loss_)]
                test_logger.writerow(carrier)

        next_state = state_saver
        RMS_loss += np.sqrt(loss_)
    RMS_loss = RMS_loss / hyp.Info.TEST_SIZE
    print("test: rms loss is ", RMS_loss)
    test_logger.writerow(["final adaptive loss average", RMS_loss])