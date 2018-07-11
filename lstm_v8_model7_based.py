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
model_layer_1 = Model()
model_layer_2 = Model()

output1, current_cell_1, current_hidden_1 = model_layer_1.create_graph(layer_number = 1)
output2, current_cell_2, current_hidden_2 = model_layer_2.create_graph(layer_number = 2)

with tf.name_scope("placeholders"):
    Y = tf.placeholder(shape=[1, 1], dtype=tf.float32, name="label")  # not used until the last cycle

with tf.name_scope("loss"):
    loss = tf.square(tf.subtract(output2, Y))
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
            if np.sum(model.B_Forget_and_Input.eval()) != 0:
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

    next_cell_1 = np.zeros(shape=[1, hyp.cell_dim])
    next_hidd_1 = np.zeros(shape=[1, hyp.hidden_dim])
    next_cell_2 = np.zeros(shape=[1, hyp.cell_dim])
    next_hidd_2 = np.zeros(shape=[1, hyp.hidden_dim])

    for epoch in range(hyp.EPOCHS):

        reset = sm.next_epoch()
        label = sm.get_label()
        label = np.reshape(label, [1, 1])
        loss_ = 0
        if reset: #this allows for hidden states to reset after the training set loops back around
            next_cell_1 = np.zeros(shape=[1, hyp.cell_dim])
            next_hidd_1 = np.zeros(shape=[1, hyp.hidden_dim])
            next_cell_2 = np.zeros(shape=[1, hyp.cell_dim])
            next_hidd_2 = np.zeros(shape=[1, hyp.hidden_dim])

        for counter in range(hyp.FOOTPRINT):
            input_1 = sm.next_sample()
            input_1 = np.reshape(input_1, [1,1])
            if counter < hyp.FOOTPRINT-1:

                next_cell, next_hidd, output_1 = sess.run(
                    [current_cell_1, current_hidden_1, output1],
                    feed_dict={model_layer_1.X: input_1, model_layer_1.H_last: next_hidd_1, model_layer_1.C_last: next_cell_1})

                input_2 = output_1

                next_cell_2, next_hidd_2 = sess.run([current_cell_2, current_hidden_2],
                                                    feed_dict={model_layer_2.X:input_2, model_layer_2.H_last:next_hidd_1,
                                                               model_layer_2.C_lat:next_cell_1})

            else:
                pass

        logger.writerow([loss_])

        if epoch%50 == 0:
            writer.add_summary(summary, global_step=epoch)
            print("I finished epoch ", epoch, " out of ", hyp.EPOCHS, " epochs")
            print("The absolute value loss for this sample is ", np.sqrt(loss_))
            print("predicted number: ", output_2, ", real number: ", label)

        if epoch%2000 == 0 and epoch>498:
            saver.save(sess, "2012/v7/models_CLASS/LSTMv7", global_step=epoch)
            print("saved model")

            next_cell_hold = next_cell
            next_hidd_hold = next_hidd
            sm.create_validation_set()
            average_rms_loss = 0.0
            next_cell = np.zeros(shape=[1, hyp.cell_dim])
            next_hidd = np.zeros(shape=[1, hyp.hidden_dim])
            for i in range(hyp.VALIDATION_NUMBER):

                sm.next_epoch_valid()
                label_ = sm.get_label()
                label = np.reshape(label_, [1, 1])
                # this gets each 10th
                for counter in range(hyp.FOOTPRINT):
                    data = sm.next_sample()
                    data = np.reshape(data, [1, 1])
                    if counter < hyp.FOOTPRINT - 1:

                        next_cell, next_hidd = sess.run([current_cell_1, current_hidden_1],
                                                        feed_dict={model.X: data, model.H_last: next_hidd, model.C_last: next_cell})
                        if counter == 0:
                            hidden_saver = next_hidd  # saves THIS state for the next round
                            cell_saver = next_cell
                    else:
                        next_cell, next_hidd, output_, loss_ = sess.run(
                            [current_cell, current_hidden, output, loss],
                            feed_dict={model.X: data, Y: label, model.H_last: next_hidd, model.C_last: next_cell})


                next_cell = cell_saver
                next_hidden = hidden_saver
                average_rms_loss += np.sqrt(loss_)
                sm.clear_valid_counter()

            average_rms_loss = average_rms_loss/hyp.VALIDATION_NUMBER
            print("validation: RMS loss is ", average_rms_loss)
            validation_logger.writerow([epoch, average_rms_loss])

            next_cell = next_cell_hold
            next_hidd = next_hidd_hold
    RMS_loss = 0.0
    next_cell = np.zeros(shape=[1, hyp.cell_dim])
    next_hidd = np.zeros(shape=[1, hyp.hidden_dim])
    for test in range(hyp.Info.TEST_SIZE): #this will be replaced later

        sm.next_epoch_test_single_shift()
        label_ = sm.get_label()
        label = np.reshape(label_, [1, 1])
        # this gets each 10th
        for counter in range(hyp.FOOTPRINT):
            data = sm.next_sample()
            data = np.reshape(data, [1, 1])
            if counter < hyp.FOOTPRINT - 1:

                next_cell, next_hidd = sess.run([current_cell, current_hidden],
                                                feed_dict={model.X: data, model.H_last: next_hidd, model.C_last: next_cell})
                if counter == 0:
                    hidden_saver = next_hidd  # saves THIS state for the next round
                    cell_saver = next_cell
            else:
                next_cell, next_hidd, output_, loss_ = sess.run(
                    [current_cell, current_hidden, output, loss],
                    feed_dict={model.X: data, Y: label, model.H_last: next_hidd, model.C_last: next_cell})

                carrier = [label_, output_[0][0], np.sqrt(loss_)]
                test_logger.writerow(carrier)
        next_cell = cell_saver
        next_hidden = hidden_saver
        RMS_loss += np.sqrt(loss_)
    RMS_loss = RMS_loss / hyp.Info.TEST_SIZE
    print("test: rms loss is ", RMS_loss)
    test_logger.writerow(["final adaptive loss average", RMS_loss])