import tensorflow as tf
from pipeline import SetMaker
from pipeline import Hyperparameters
import numpy as np

hyp = Hyperparameters()
sm = SetMaker()
pbfilename = "v1/models/LSTM_v1_frozen.pb"


with tf.gfile.GFile(pbfilename, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def,
                        input_map = None,
                        return_elements = None,
                        name = "")
    input = graph.get_tensor_by_name("placeholders/input_placeholder:0")
    output = graph.get_tensor_by_name("output_gate/BHTO_b_a:0")
    H_last = graph.get_tensor_by_name("placeholders/last_hidden:0")
    H_next = graph.get_tensor_by_name("output_gate/hidden_layer_propagation:0")

with tf.Session(graph=graph) as sess:

    output_prediction_ = []
    counter = 0
    first = True
    input_array= set_maker.load_blind(name = file_name)
    for slice in input_array:
        slice = np.reshape(slice, [1, HYP.INPUT_LAYER])
        if counter == 15:
             output_prediction_ = sess.run(output, feed_dict=
            {
                input: slice,
                last_hidd: prev_hidd_layer_
            })
        else:
            if (first):
                prev_hidd_layer_ = np.zeros(shape=HYP.HIDDEN_LAYER)
                prev_hidd_layer_ = np.reshape(prev_hidd_layer_, [1, HYP.HIDDEN_LAYER])
                first = False

            next_hidd_layer_ = sess.run(next_hidd_layer, feed_dict=
            {
                input: slice,
                last_hidd: prev_hidd_layer_
            })
            prev_hidd_layer_ = next_hidd_layer_
            counter += 1

    print("this is the predicted matrix: ", output_prediction_)
    print("inhale: ", output_prediction_[0][0])
    print("exhale: ", output_prediction_[0][1])
    print("unknown: ", output_prediction_[0][2])
    print("winning prediction: ", prediction_dictionary[np.argmax(output_prediction_)])
