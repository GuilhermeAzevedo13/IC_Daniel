import tensorflow as tf
import os

os.system('cls') #Retire isso se for usar no linux ou substitua o cls por clear
a = tf.constant([9, 8, 7], name = 'a')
b = tf.constant([1, 2, 3], name = 'b')

soma = a + b

type(a)

print(a)

with tf.Session() as sess:
    print(sess.run(soma))
    
a1 = tf.constant([[1, 2, 3], [4, 5, 6]], name = 'a1')

type(a1)

print(a1)

b1 = tf.constant([[1, 2, 3], [4, 5, 6]], name = 'b1')

soma1 = tf.add(a1, b1)

with tf.Session() as sess:
    print(sess.run(a1))
    print('\n')
    print(sess.run(b1))
    print('\n')
    print(sess.run(soma1))
    print('\n')
    
a2 = tf.constant([[1, 2, 3], [4, 5, 6]])
b2 = tf.constant([[1], [2]])
soma2 = tf.add(a2, b2)

with tf.Session() as sess:
    print(sess.run(a2))
    print('\n')
    print(sess.run(b2))
    print('\n')
    print(sess.run(soma2))