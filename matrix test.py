import numpy as np
import tensorflow as tf
x = [1,2,3,4,5,6,7]
y = [2,2,2,2,2,2,2]

k = tf.Variable(tf.zeros(shape = [1,1]))
d = tf.Variable(tf.zeros(shape = [25,1]))
m = tf.concat([k, d],axis = 0)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    l = sess.run(m, feed_dict = {})
    print(l)
#print(np.matmul(x,np.transpose(y)))
#print(np.concatenate([x,y], axis = 0))
'''
print(np.multiply(x,y))

for i in range(10):
    print(i)
'''