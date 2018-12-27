import tt
import tensorflow as tf


a = tt.Net()
b = tt.Net()
init = tf.global_variables_initializer()
tt.sess.run(init)
print(tt.sess.run(a.v))
print(tt.sess.run(a.v))