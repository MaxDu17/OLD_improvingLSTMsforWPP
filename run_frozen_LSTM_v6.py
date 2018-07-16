import tensorflow as tf
from pipeline import SetMaker
from pipeline import Hyperparameters
import numpy as np
import csv

hyp = Hyperparameters()
sm = SetMaker()
pbfilename = "2012/v2/models_CONTAINED/LSTM_v2_frozen_CONTAINED.pb"


with tf.gfile.GFile(pbfilename, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def,
                        input_map = None,
                        return_elements = None,
                        name = "")
    input = graph.get_tensor_by_name("placeholders/input_data:0")
    init_state_1 = graph.get_tensor_by_name("placeholders/initial_states_1:0")
    init_state_2 = graph.get_tensor_by_name("placeholders/initial_states_2:0")
    output = graph.get_tensor_by_name("layer_2_propagation/output:0")
    states_list = graph.get_tensor_by_name("forward_roll/scan/TensorArrayStack/TensorArrayGatherV3:0")


with tf.Session(graph=graph) as sess:
    sm.create_training_set()
    test = open("2012/v2/GRAPHS_CONTAINED/EVALUATE_TEST.csv", "w")
    test_logger = csv.writer(test, lineterminator="\n")
    carrier = ["true_values", "predicted_values", "abs_error"]
    test_logger.writerow(carrier)
    RMS_loss = 0.0
    next_state_ = np.zeros(shape=[2, 1, hyp.cell_dim])
    for test in range(hyp.Info.TEST_SIZE):  # this will be replaced later

        data = sm.next_epoch_test_waterfall()
        label_ = sm.get_label()
        label = np.reshape(label_, [1, 1])
        data = np.reshape(data, [hyp.FOOTPRINT, 1, 1])

        init_state_1_, init_state_2_, output_, loss_ = sess.run([pass_back_state_1, pass_back_state_2, output, loss], # why passback? Because we only shift by one!
                                                       feed_dict={Y: label, init_state_1: init_state_1_,
                                                                  init_state_2: init_state_2_, inputs: data})
        RMS_loss += np.sqrt(loss_)
        carrier = [label_, output_[0][0], np.sqrt(loss_)]
        test_logger.writerow(carrier)
    RMS_loss = RMS_loss / hyp.Info.TEST_SIZE
    print("test: rms loss is ", RMS_loss)
    test_logger.writerow(["final adaptive loss average", RMS_loss])
