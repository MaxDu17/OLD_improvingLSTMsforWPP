import tensorflow as tf
g = tf.GraphDef()
g.ParseFromString(open("2012/v2/models_CONTAINED/LSTM_v2_frozen_CONTAINED.pb", "rb").read())
k = [n for n in g.node if n.name.find("loss") != -1] # same for output or any other node you want to make sure is ok
p = g.node
print(k)