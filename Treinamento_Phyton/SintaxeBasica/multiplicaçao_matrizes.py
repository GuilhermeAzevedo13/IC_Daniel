import tensorflow as tf

a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[-1, 3], [4, 2]])

multiplicacao = tf.matmul(a, b)

with tf.Session() as sess:
    print(sess.run(a))
    print('\n')
    print(sess.run(b))
    print('\n')
    print(sess.run(multiplicacao))
    
multiplicao2 = tf.matmul(b, a)

with tf.Session() as sess:
    print('\n')
    print(sess.run(multiplicao2))
    
a1 = tf.constant([[2, 3], [0, 1], [-1, 4]])
b1 = tf.constant([[1,2,3], [-2,0,4]])
multiplicacao3 = tf.matmul(a1, b1)

with tf.Session() as sess:
    print('\n')
    print(sess.run(a1))
    print('\n')
    print(sess.run(b1))
    print('\n')
    print(sess.run(multiplicacao3))

