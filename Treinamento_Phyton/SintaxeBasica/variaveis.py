# COMO FUNCIONA EM OUTRAS LINGUAGENS:

x = 35
y = x + 35

print(y)

# Como deve ser feito no TensorFlow

import tensorflow as tf

valor1 = tf.constant(15, name = 'valor1')
print(valor1)

soma = tf.Variable(valor1 + 5, name = 'valor1')

print(soma)
type(soma)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    s = sess.run(soma)
    
print(s)

