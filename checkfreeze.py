import tensorflow as tf
g = tf.GraphDef()
g.ParseFromString(open("2012/v8/models/LSTM_v8_frozen.pb", "rb").read())

k = [n for n in g.node if n.name.find("layer_1_propagation/pass_back_state_1") != -1] # for output or any other node you want to make sure is ok
#k = [n for n in g.node if n.name.find("layer_2_propagation/layer_2/forward_roll/scan/TensorArrayStack/TensorArrayGatherV3") != -1] # for output or any other node you want to make sure is ok

p = g.node
print(k)