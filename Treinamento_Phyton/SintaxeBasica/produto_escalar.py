import tensorflow as tf

a = tf.constant([[-1.0, 7.0, 5.0]], name = 'entradas')
b = tf.constant([[0.8, 0.1, 0.0]], name = 'pesos')
multiplicacao = tf.multiply(a,b) # Realizar uma multiplicaçao simples
soma = tf.reduce_sum(multiplicacao)

with tf.Session() as sess:
    print(sess.run(a))
    print('\n')
    print(sess.run(b))
    print('\n')
    print(sess.run(multiplicacao))
    print('\n')
    print(sess.run(soma))