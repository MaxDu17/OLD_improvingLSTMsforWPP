import tensorflow as tf
from pipeline import SetMaker
from pipeline import Hyperparameters
import numpy as np
import csv

hyp = Hyperparameters()
sm = SetMaker()
pbfilename = "2012/v2/models/LSTM_v2_frozen.pb"


with tf.gfile.GFile(pbfilename, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def,
                        input_map = None,
                        return_elements = None,
                        name = "")
    input = graph.get_tensor_by_name("placeholders/input_placeholder:0")
    output = graph.get_tensor_by_name("output_gate/output:0")
    H_last = graph.get_tensor_by_name("placeholders/last_hidden:0")
    current_hidden = graph.get_tensor_by_name("output_gate/next_hidden:0")
    C_last = graph.get_tensor_by_name("placeholders/last_cell:0")
    current_cell = graph.get_tensor_by_name("output_gate/output_presquashing:0")

with tf.Session(graph=graph) as sess:
    sm.create_training_set()
    test = open("2012/v2/GRAPHS/ONE-ONE-FEED.csv", "w")
    test_logger = csv.writer(test, lineterminator="\n")
    carrier = ["true_values", "predicted_values", "abs_error"]
    test_logger.writerow(carrier)
    RMS_loss = 0.0
    next_cell = np.zeros(shape=[1, hyp.cell_dim])
    next_hidd = np.zeros(shape=[1, hyp.hidden_dim])

    for counter in range(hyp.FOOTPRINT): #prompts the network
        print("building up the states" + str(counter))
        data, _ = sm.next_epoch_test_pair() #label is ignored in the state-building
        data = np.reshape(data, [1, 1])
        next_cell, next_hidd = sess.run([current_cell, current_hidden],
                                        feed_dict={input: data, H_last: next_hidd, C_last: next_cell})

    for test in range(hyp.Info.EVAULATE_TEST_SIZE): #now for every feed, we ask for a result
        print(test)
        data, label = sm.next_epoch_test_pair() #this gives you a single value, and the label for it

        input_data = np.reshape(data,[1,1])
        #this gets each 10th
        next_cell, next_hidd, output_ = sess.run(
            [current_cell, current_hidden, output],
            feed_dict={input: input_data, H_last: next_hidd, C_last: next_cell})

        carrier = [label, output_[0][0], np.sqrt(np.square((label - output_)[0][0]))]
        test_logger.writerow(carrier)

