import tensorflow as tf
from pipeline import SetMaker
from pipeline import Hyperparameters
import numpy as np
import csv

hyp = Hyperparameters()
sm = SetMaker()
pbfilename = "v2/models/LSTM_v2_frozen.pb"


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
    sm.create_training_set() #this is just to allow the math to work
    test = open("v2/GRAPHS/RUN_TEST.csv", "w")
    test_logger = csv.writer(test, lineterminator="\n")
    carrier = ["true_values", "predicted_values", "abs_error"]
    test_logger.writerow(carrier)
    RMS_loss = 0.0
    next_cell = np.zeros(shape=[1, hyp.cell_dim])
    next_hidd = np.zeros(shape=[1, hyp.hidden_dim])
    labels, prompt = sm.return_split_lists()
    counter = 0
    for value in prompt:
        value_ = np.reshape(value, [1, 1])
        next_cell, next_hidd, output_ = sess.run(
            [current_cell, current_hidden, output],
            feed_dict={input: value_, H_last: next_hidd, C_last: next_cell}) #discard outputs until the last cycle

    for test in labels: #this will be replaced later
        counter +=1
        print(counter)
        output_ = np.reshape(output_, [1, 1])
        test_ = np.reshape(test, [1,1])
        next_cell, next_hidd, output_ = sess.run(
            [current_cell, current_hidden, output],
            feed_dict={input: test_, H_last: next_hidd, C_last: next_cell})
        carrier = [test, output_[0][0], np.sqrt(np.square((test - output_)[0][0]))]
        test_logger.writerow(carrier)

